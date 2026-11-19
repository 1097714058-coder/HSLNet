from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list,compute_ranked_list_cm
from clustercontrast.cam_associate_loss_fused import get_proxy_associate_loss_fused
import numpy as np
import gc

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)

part=5
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

def compute_cross_agreement_dd(features_g, features_p,features_g_s,features_p_s, k=20, search_option=3):
    print("Compute cross agreement score...")
    N, D = features_p.size()
    M, D = features_g.size()
    score = torch.FloatTensor()
    end = time.time()
    ranked_list_g = compute_ranked_list_cm(features_g,features_p, k=k, search_option=search_option, verbose=False)
    ranked_list_p_i = compute_ranked_list_cm(features_p_s,features_p_s, k=k, search_option=search_option, verbose=False)
    score_all =[]
    for i in range(M):
        intersect_i = torch.FloatTensor(
            [len(np.intersect1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        union_i = torch.FloatTensor(
            [len(np.union1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score_all.append(score_i)
    score = torch.cat(score_all, dim=0)
    # print(score_i.size())
    # print("Cross agreement score time cost: {}".format(time.time() - end))
    return score_i

def get_proxy(pseudo_labels, cams):
    """根据伪标签和相机信息创建代理标签"""
    proxy_labels = -1 * torch.ones(len(pseudo_labels), dtype=torch.int).cuda()
    count = 0
    
    # 过滤掉 -1 标签，只处理有效的伪标签
    valid_labels = pseudo_labels[pseudo_labels >= 0]
    if len(valid_labels) == 0:
        return proxy_labels, 0
    
    for i in range(int(valid_labels.max()) + 1):
        inds = torch.nonzero(pseudo_labels == i).squeeze()
        if inds.numel() == 0:
            continue
        if inds.dim() == 0:
            inds = inds.unsqueeze(0)
        temp_cams = cams[inds]
        for cam in torch.unique(temp_cams):
            mask = (temp_cams == cam)
            proxy_labels[inds[mask]] = count
            count += 1
    num_proxies = len(torch.unique(proxy_labels)) - (1 if -1 in proxy_labels else 0)
    return proxy_labels, num_proxies

def creat_dic(pseudo_labels, proxy_labels, unique_cameras, cams):
    # 过滤掉 -1 标签
    valid_mask = (pseudo_labels >= 0) & (proxy_labels >= 0)
    pseudo_labels = pseudo_labels[valid_mask]
    proxy_labels = proxy_labels[valid_mask]
    cams = cams[valid_mask]
    
    num_labels = int(pseudo_labels.max()) + 1
    # 使用实际的代理数量，而不是最大索引+1
    num_proxies = int(proxy_labels.max()) + 1 if proxy_labels.numel() > 0 else 0
    
    # 创建相机ID到连续索引的映射
    cam_id_to_idx = {int(cam_id): idx for idx, cam_id in enumerate(unique_cameras)}
    
    # 使用列表存储，避免创建大型密集张量
    label2proxy = [[] for _ in range(num_labels)]
    proxy2label = torch.zeros(num_proxies, dtype=torch.int)
    proxy2cam = torch.zeros(num_proxies, dtype=torch.int)
    cam2proxy_list = [[] for _ in range(len(unique_cameras))]
    
    for i in range(len(pseudo_labels)):
        label = int(pseudo_labels[i])
        proxy = int(proxy_labels[i])
        cam = int(cams[i])
        cam_idx = cam_id_to_idx[cam]  # 使用连续索引
        
        label2proxy[label].append(proxy)
        proxy2label[proxy] = label
        proxy2cam[proxy] = cam
        cam2proxy_list[cam_idx].append(proxy)
    
    # 将列表转换为张量（填充-1）
    max_proxies = max(len(lst) for lst in cam2proxy_list) if cam2proxy_list else 0
    cam2proxy = torch.full((len(unique_cameras), max_proxies), -1, dtype=torch.int)
    for cam_idx, proxy_list in enumerate(cam2proxy_list):
        if proxy_list:
            cam2proxy[cam_idx, :len(proxy_list)] = torch.tensor(proxy_list, dtype=torch.int)
    
    return label2proxy, proxy2label, proxy2cam, cam2proxy

class ClusterContrastTrainer_HSDN(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_HSDN, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.memory_ir_s = memory
        self.memory_rgb_s = memory
        self.wise_memory_ir_s =  memory
        self.wise_memory_rgb_s =  memory
        self.shared_memory =  memory
        self.shared_memory_s =  memory
        self.htsd=0

        self.hm=0
        self.ht=0
        self.proxy_features = None  # 代理特征向量
        self.cam2proxy = None       # 相机到代理的映射
        self.proxy2cam = None       # 代理到相机的映射
        self.proxy_labels = None    # 代理标签
        self.num_proxies = 0
        self.bg_knn = 5             # 负样本数量（平衡为3，兼顾稳定性和效果）
        self.temperature = 0.05     # 温度参数，控制 softmax 分布的尖锐度

    def update_proxy(self, features, pseudo_labels, cams):
        """更新代理中心和映射"""
        proxy_labels, self.num_proxies = get_proxy(pseudo_labels, cams)
        
        # 过滤掉 -1 标签，只使用有效样本
        valid_mask = (proxy_labels >= 0)
        valid_features = features[valid_mask]
        valid_proxy_labels = proxy_labels[valid_mask]
        
        self.proxy_features = torch.zeros(self.num_proxies, features.size(1)).cuda()
        for i in range(self.num_proxies):
            mask = (valid_proxy_labels == i)
            if mask.sum() > 0:
                self.proxy_features[i] = valid_features[mask].mean(dim=0)
        
        self.proxy_features = F.normalize(self.proxy_features, dim=1)
        
        self.unique_cameras = torch.unique(cams)
        _, self.proxy2label, self.proxy2cam, self.cam2proxy = creat_dic(
            pseudo_labels, proxy_labels, self.unique_cameras, cams
        )
        
        # 创建代理查找字典，加速查找
        self.proxy_lookup = {}
        for i in range(self.num_proxies):
            label = int(self.proxy2label[i])
            cam = int(self.proxy2cam[i])
            self.proxy_lookup[(label, cam)] = i
        
        gc.collect()

    def _get_sample_proxy_index(self, labels, cams):
        """获取每个样本在各相机下的代理索引"""
        batch_size = labels.size(0)
        num_cams = len(self.unique_cameras)
        
        s_proxies = -torch.ones(batch_size, num_cams, dtype=torch.long).cuda()
        
        # 创建相机ID到连续索引的映射
        cam_id_to_idx = {int(cam_id): idx for idx, cam_id in enumerate(self.unique_cameras)}
        
        for i in range(batch_size):
            label = int(labels[i])
            cam = int(cams[i])
            cam_idx = cam_id_to_idx.get(cam, -1)
            if cam_idx >= 0:
                proxy_idx = self._find_proxy_index(label, cam)
                if proxy_idx >= 0:
                    s_proxies[i, cam_idx] = proxy_idx
        
        return s_proxies

    def _find_proxy_index(self, label, cam):
        """查找指定标签和相机对应的代理索引"""
        if self.proxy_features is None:
            return -1
        # 使用字典查找，O(1) 复杂度
        return self.proxy_lookup.get((label, cam), -1)

    def _update_proxy_features(self, features, labels, cams, momentum=0.9, normalize=True):
        """使用动量更新代理特征"""
        if self.proxy_features is None:
            return
            
        for i in range(features.size(0)):
            label = int(labels[i])
            cam = int(cams[i])
            proxy_idx = self._find_proxy_index(label, cam)
            
            if proxy_idx >= 0:
                # 动量更新，使用no_grad避免梯度累积
                with torch.no_grad():
                    self.proxy_features[proxy_idx] = (
                        momentum * self.proxy_features[proxy_idx] + 
                        (1 - momentum) * features[i]
                    )
        
        # 降低归一化频率，避免抵消更新效果
        if normalize:
            with torch.no_grad():
                self.proxy_features = F.normalize(self.proxy_features, dim=1)

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400, args=None):
        self.encoder.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()


        lamda_s_neibor=0.5
        lamda_d_neibor=1


        lamda_sd = 1

        # 代理特征更新频率控制
        proxy_update_freq = 1
        proxy_update_counter = 0

        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)


            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            
            targets_soft_rgb = []
            targets_soft_ir = []
            weights_rgb = []
            weights_ir = []
            

            for idx in indexes_rgb:
                targets_soft_rgb.append(self.correlations_rgb[idx.item()])
                weights_rgb.append(self.weights_rgb[idx.item()])
            for idx in indexes_ir:
                targets_soft_ir.append(self.correlations_ir[idx.item()])
                weights_ir.append(self.weights_ir[idx.item()])

            weights_rgb = torch.tensor(weights_rgb).cuda()
            weights_ir = torch.tensor(weights_ir).cuda()



            targets_soft_rgb = torch.stack(targets_soft_rgb, dim=0)
            targets_soft_ir = torch.stack(targets_soft_ir, dim=0)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            targets_soft_rgb = torch.cat((targets_soft_rgb, targets_soft_rgb), 0)
            weights_rgb = torch.cat((weights_rgb, weights_rgb), -1)
 
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # cids_ir =  torch.cat((cids_ir,cids_ir),-1)
    

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb])
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1).cuda()

            _,f_out_rgb,f_out_ir,f_out_rgb_s,f_out_ir_s,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            
            feats_ir = self.memory_ir(f_out_ir, labels_ir, return_out=True)
            feats_rgb = self.memory_rgb(f_out_rgb, labels_rgb, return_out=True)


            if feats_rgb.size(1) < targets_soft_rgb.size(1):
                feats_rgb = F.pad(feats_rgb, (0, targets_soft_rgb.size(1) - feats_rgb.size(1)))
            elif feats_rgb.size(1) > targets_soft_rgb.size(1):
                targets_soft_rgb = F.pad(targets_soft_rgb, (0, feats_rgb.size(1) - targets_soft_rgb.size(1)))
                
            if feats_ir.size(1) < targets_soft_ir.size(1):
                feats_ir = F.pad(feats_ir, (0, targets_soft_ir.size(1) - feats_ir.size(1)))
            elif feats_ir.size(1) > targets_soft_ir.size(1):
                targets_soft_ir = F.pad(targets_soft_ir, (0, feats_ir.size(1) - targets_soft_ir.size(1)))
                
            targets_soft_rgb = (self.balance * torch.zeros(feats_rgb.size()).scatter_(1, labels_rgb.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_rgb).cuda()
            targets_soft_ir = (self.balance * torch.zeros(feats_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_ir).cuda()
            
            log_probs_ir = nn.LogSoftmax(dim=1)(feats_ir)
            log_probs_rgb = nn.LogSoftmax(dim=1)(feats_rgb)



            loss_ir_s = torch.tensor([0.]).cuda()
            loss_rgb_s = torch.tensor([0.]).cuda()
            loss_ir = torch.tensor([0.]).cuda()
            loss_rgb = torch.tensor([0.]).cuda()
            ir_rgb_loss = torch.tensor([0.]).cuda()
            rgb_ir_loss = torch.tensor([0.]).cuda()
            rgb_rgb_loss = torch.tensor([0.]).cuda()
            ir_ir_loss = torch.tensor([0.]).cuda()

            ir_rgb_loss_s = torch.tensor([0.]).cuda()
            rgb_ir_loss_s = torch.tensor([0.]).cuda()
            rgb_rgb_loss_s = torch.tensor([0.]).cuda()
            ir_ir_loss_s = torch.tensor([0.]).cuda()
            loss_shared = torch.tensor([0.]).cuda()
            loss_shared_s = torch.tensor([0.]).cuda()


            loss_ir_s = lamda_sd* self.memory_ir_s(f_out_ir_s, labels_ir)
            loss_rgb_s = lamda_sd* self.memory_rgb_s(f_out_rgb_s, labels_rgb)
            if epoch>=self.cmlabel:
                loss_ir = ((- targets_soft_ir * log_probs_ir).sum(1) * weights_ir).mean()
                loss_rgb = ((- targets_soft_rgb * log_probs_rgb).sum(1) * weights_rgb).mean()
            else:
                loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            loss_cam = torch.tensor([0.]).cuda()
            loss_cam_rgb_log = torch.tensor([0.]).cuda()
            loss_cam_ir_log = torch.tensor([0.]).cuda()

            if self.proxy_features is not None:
                scores_proxy_rgb = f_out_rgb.mm(self.proxy_features.t())
                scores_proxy_ir = f_out_ir.mm(self.proxy_features.t())
                s_proxies_rgb = self._get_sample_proxy_index(labels_rgb, cids_rgb)
                s_proxies_ir = self._get_sample_proxy_index(labels_ir, cids_ir)
                
                # 计算聚类中心
                cluster_centers_rgb, cluster_nums_rgb = self.generate_cluster_features(labels_rgb, f_out_rgb)
                cluster_centers_ir, cluster_nums_ir = self.generate_cluster_features(labels_ir, f_out_ir)
                
                loss_cam_rgb, loss_cam_ir, loss_cross, loss_cam = get_proxy_associate_loss_fused(
                    cam2proxy_rgb=self.cam2proxy,
                    cam2proxy_ir=self.cam2proxy,
                    s_proxies_rgb=s_proxies_rgb,
                    s_proxies_ir=s_proxies_ir,
                    scores_proxy_rgb=scores_proxy_rgb,
                    scores_proxy_ir=scores_proxy_ir,
                    bg_knn=self.bg_knn,
                    temperature=0.1,
                    epoch=epoch,
                    total_epochs=args.epochs if args is not None else 40,
                    use_focal=True,
                    use_hard_mining=(epoch > (args.epochs if args is not None else 40) * 0.2),
                    hard_mining_ratio=0.3,
                    cross_modal_weight=0.3,
                    proxy_features_rgb=self.proxy_features,
                    proxy_features_ir=self.proxy_features,
                    cluster_centers_rgb=cluster_centers_rgb,
                    cluster_centers_ir=cluster_centers_ir
                )
                
                loss_camera_rgb_log.update(loss_cam_rgb.item())
                loss_camera_ir_log.update(loss_cam_ir.item())

            thresh=0.9
            hm_thresh = 0.9

################cpsrefine
# #############v2
            if epoch>=self.hm:#self.cmlabel:
                if epoch>=self.ht:#self.cmlabel:
##########################V4 wise neighbor
                    if epoch %2 ==0:
                        with torch.no_grad():
                            sim_prob_rgb_ir = self.wise_memory_rgb_s.features.detach()[index_rgb].mm(self.wise_memory_ir_s.features.detach().clone().t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                            sim_rgb_ir = self.wise_memory_rgb.features.detach()[index_rgb].mm(self.wise_memory_ir.features.detach().clone().t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)

                            nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                            nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                            mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh).detach().clone()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1


                        sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().clone().t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
                        sim_prob_rgb_ir = F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().clone().t())
                        sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                        # print('score_intra',score_intra)
                        score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                        # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                        rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                        rgb_ir_loss = 0.1*lamda_d_neibor*rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(mask_neighbor_intra_soft) ##.mul(rgb_ir_ca)
                        # print('rgb_ir_loss',rgb_ir_loss)
                        sim_prob_rgb_ir_exp =sim_prob_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir_s =   F.softmax(sim_prob_rgb_ir_exp,dim=1)
                        score_intra_rgb_ir_s = score_intra_rgb_ir_s.clamp_min(1e-8)

                        rgb_ir_loss_s = -score_intra_rgb_ir_s.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)

                        rgb_ir_loss_s = 0.1*lamda_s_neibor*rgb_ir_loss_s.div(num_neighbor_rgb_ir).mean()#.mul(rgb_ir_ca).mul(rgb_ir_ca) ##.mul(rgb_ir_ca)


                    else:
                        with torch.no_grad():
                            sim_prob_rgb_ir = self.wise_memory_ir_s.features[index_ir] @ self.wise_memory_rgb_s.features.t()
                            sim_rgb_ir = self.wise_memory_ir.features[index_ir] @ self.wise_memory_rgb.features.t()

                            nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                            nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                            mask_neighbor_rgb_ir = (sim_rgb_ir > nearest_rgb_ir * thresh).detach()
                            mask_neighbor_prob_rgb_ir = (sim_prob_rgb_ir > nearest_prob_rgb_ir * thresh)
                            num_neighbor_rgb_ir = (mask_neighbor_rgb_ir & mask_neighbor_prob_rgb_ir).sum(dim=1) + 1

                        sim_prob_rgb_ir = F.normalize(f_out_ir_s, dim=1) @ self.wise_memory_rgb_s.features.t()
                        sim_rgb_ir = F.normalize(f_out_ir, dim=1) @ self.wise_memory_rgb.features.t()
                        sim_rgb_ir_exp = sim_rgb_ir / 0.05
                        score_intra_rgb_ir = F.softmax(sim_rgb_ir_exp, dim=1).clamp_min(1e-8)
                        
                        combined_mask = mask_neighbor_rgb_ir & mask_neighbor_prob_rgb_ir
                        ir_rgb_loss = -score_intra_rgb_ir.log() * combined_mask.float()
                        ir_rgb_loss = lamda_d_neibor * ir_rgb_loss.sum(dim=1) / num_neighbor_rgb_ir.float()
                        ir_rgb_loss = ir_rgb_loss.mean()

                        sim_prob_rgb_ir_exp = sim_prob_rgb_ir / 0.05
                        score_intra_rgb_ir_s = F.softmax(sim_prob_rgb_ir_exp, dim=1).clamp_min(1e-8)
                        
                        ir_rgb_loss_s = -score_intra_rgb_ir_s.log() * combined_mask.float()
                        ir_rgb_loss_s = lamda_s_neibor * ir_rgb_loss_s.sum(dim=1) / num_neighbor_rgb_ir.float()
                        ir_rgb_loss_s = ir_rgb_loss_s.mean()


                    # else:
                with torch.no_grad():
                    sim_prob_rgb_rgb = self.wise_memory_rgb_s.features[index_rgb] @ self.wise_memory_rgb_s.features.t()
                    sim_rgb_rgb = self.wise_memory_rgb.features[index_rgb] @ self.wise_memory_rgb.features.t()

                    nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
                    nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
                    mask_neighbor_rgb_rgb = (sim_rgb_rgb > nearest_rgb_rgb * hm_thresh).detach()
                    mask_neighbor_prob_rgb_rgb = (sim_prob_rgb_rgb > nearest_prob_rgb_rgb * hm_thresh)
                    num_neighbor_rgb_rgb = (mask_neighbor_rgb_rgb & mask_neighbor_prob_rgb_rgb).sum(dim=1) + 1

                sim_prob_rgb_rgb = F.normalize(f_out_rgb_s, dim=1) @ self.wise_memory_rgb_s.features.t()
                sim_rgb_rgb = F.normalize(f_out_rgb, dim=1) @ self.wise_memory_rgb.features.t()
                sim_rgb_rgb_exp = sim_rgb_rgb / 0.05
                score_intra_rgb_rgb = F.softmax(sim_rgb_rgb_exp, dim=1).clamp_min(1e-8)
                
                combined_mask_rgb_rgb = mask_neighbor_rgb_rgb & mask_neighbor_prob_rgb_rgb
                rgb_rgb_loss = -score_intra_rgb_rgb.log() * combined_mask_rgb_rgb.float()
                rgb_rgb_loss = lamda_d_neibor * rgb_rgb_loss.sum(dim=1) / num_neighbor_rgb_rgb.float()
                rgb_rgb_loss = rgb_rgb_loss.mean()

                sim_prob_rgb_rgb_exp = sim_prob_rgb_rgb / 0.05
                score_intra_rgb_rgb_s = F.softmax(sim_prob_rgb_rgb_exp, dim=1).clamp_min(1e-8)
                
                rgb_rgb_loss_s = -score_intra_rgb_rgb_s.log() * combined_mask_rgb_rgb.float()
                rgb_rgb_loss_s = lamda_s_neibor * rgb_rgb_loss_s.sum(dim=1) / num_neighbor_rgb_rgb.float()
                rgb_rgb_loss_s = rgb_rgb_loss_s.mean()

                # #################ir-ir

                with torch.no_grad():
                    sim_prob_ir_ir = self.wise_memory_ir_s.features[index_ir] @ self.wise_memory_ir_s.features.t()
                    sim_ir_ir = self.wise_memory_ir.features[index_ir] @ self.wise_memory_ir.features.t()

                    nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
                    nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_ir_ir = (sim_prob_ir_ir > nearest_prob_ir_ir * hm_thresh)
                    mask_neighbor_ir_ir = (sim_ir_ir > nearest_ir_ir * hm_thresh).detach()
                    num_neighbor_ir_ir = (mask_neighbor_ir_ir & mask_neighbor_prob_ir_ir).sum(dim=1) + 1

                sim_prob_ir_ir = F.normalize(f_out_ir_s, dim=1) @ self.wise_memory_ir_s.features.t()
                sim_ir_ir = F.normalize(f_out_ir, dim=1) @ self.wise_memory_ir.features.t()
                sim_ir_ir_exp = sim_ir_ir / 0.05
                score_intra_ir_ir = F.softmax(sim_ir_ir_exp, dim=1).clamp_min(1e-8)
                
                combined_mask_ir_ir = mask_neighbor_ir_ir & mask_neighbor_prob_ir_ir
                ir_ir_loss = -score_intra_ir_ir.log() * combined_mask_ir_ir.float()
                ir_ir_loss = lamda_d_neibor * ir_ir_loss.sum(dim=1) / num_neighbor_ir_ir.float()
                ir_ir_loss = ir_ir_loss.mean()

                sim_prob_ir_ir_exp = sim_prob_ir_ir / 0.05
                score_intra_ir_ir_s = F.softmax(sim_prob_ir_ir_exp, dim=1).clamp_min(1e-8)
                
                ir_ir_loss_s = -score_intra_ir_ir_s.log() * combined_mask_ir_ir.float()
                ir_ir_loss_s = lamda_s_neibor * ir_ir_loss_s.sum(dim=1) / num_neighbor_ir_ir.float()
                ir_ir_loss_s = ir_ir_loss_s.mean()



            if epoch >= self.cmlabel:
                loss=(loss_ir+loss_rgb+loss_rgb_s+loss_ir_s)+0.1*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.1*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_s+ir_rgb_loss_s)+0.1*loss_cam
            elif epoch >= self.ht:
                loss=(loss_ir+loss_rgb+loss_rgb_s+loss_ir_s)+0.5*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.5*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_s+ir_rgb_loss_s)+0.1*loss_cam
            else:
                loss=(loss_ir+loss_rgb+loss_rgb_s+loss_ir_s)+1.0*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.2*loss_cam
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.wise_memory_ir.updateEM(f_out_ir, index_ir)
                self.wise_memory_rgb.updateEM(f_out_rgb, index_rgb)
                self.wise_memory_ir_s.updateEM(f_out_ir_s, index_ir)
                self.wise_memory_rgb_s.updateEM(f_out_rgb_s, index_rgb)

                # 在反向传播之后更新代理特征，不使用 detach() 让梯度能够传播
                if self.proxy_features is not None:
                    proxy_update_counter += 1
                    if proxy_update_counter % proxy_update_freq == 0:
                        self._update_proxy_features(f_out_rgb, labels_rgb, cids_rgb, momentum=0.7, normalize=True)
                        self._update_proxy_features(f_out_ir, labels_ir, cids_ir, momentum=0.7, normalize=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())

            ir_rgb_loss_log.update(ir_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]	'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
                print('loss_ir_s,loss_rgb_s',loss_ir_s.item(),loss_rgb_s.item())
                print('loss_shared_s,loss_shared',loss_shared_s.item(),loss_shared.item())
                print('ir_rgb_loss_s,rgb_ir_loss_s',ir_rgb_loss_s.item(),rgb_ir_loss_s.item())
                print('ir_ir_loss_s,rgb_rgb_loss_s',ir_ir_loss_s.item(),rgb_rgb_loss_s.item())
                print('camera proxy loss - rgb: {:.4f}'.format(
                    loss_camera_rgb_log.val))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name



    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            # print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self, labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])
        
        cluster_nums = [len(centers[idx]) for idx in sorted(centers.keys())]
        
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        
        if len(centers) > 0:
            centers = torch.stack(centers, dim=0)
        else:
            centers = torch.zeros(0, features.size(1), device=features.device)
        
        return centers, cluster_nums

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        
        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C

        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x









