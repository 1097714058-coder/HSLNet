# from model.make_model import EDITOR
# from dataloader import SYSUData, RegDBData,TestData, GenIdx, IdentitySampler,LLCMData
from cluster_contrast_vit_sysu import process_gallery_sysu, process_query_sysu,TestData
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
# import numpy as np
# import random
# import torch
# import transforms
from config.defaults import _C
import torch.utils.data as data
# from torch.autograd import Variable
# import torch
# from torch.cuda import amp
# import torch.nn as nn
# import os.path as osp
# import os
# from model.make_model import EDITOR
# import time
# import optimizer
# from scheduler import create_scheduler
# from loss.Triplet import TripletLoss,QuarCenterTripletLoss
# from loss.MSEL import MSEL
# from loss.DCL import DCL
# from utils import *
from clustercontrast.utils.data import transforms
from torch.utils.data import DataLoader
# import sys
# from optimizer import make_optimizer
# from config.config import cfg
# from eval_metrics import eval_sysu, eval_regdb, eval_llcm
import argparse
# import torch.nn.functional as F
# from tqdm import tqdm
# #设置随机种子
# seed = 10
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
 
# # ResNet101 网络提取图片特征
parser = argparse.ArgumentParser(description="PMT Training")
parser.add_argument("--config_file", default="vit_base_ics_288.yml",
                    help="path to config file", type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', type=str,
                        default="/root/autodl-tmp/project/logs/sysu_train/model_best.pth.tar",
                        metavar='PATH')
parser.add_argument('--model_path', default='/root/autodl-tmp/project/logs/sysu_train/model_best.pth.tar',
                    help='model save path', type=str)
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--num_workers', default=0,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=64,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()

# from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris,load_digits
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # 加载数据集

# def plot_embedding_2d(X, y, title=None):
#     """Plot an embedding X with the class label y colored by the domain d."""
    
#     X = X.reshape(X.shape[0], -1) 
#     X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X)
#     X_pca = PCA(n_components=2).fit_transform(X)

#     ckpt_dir="images"
#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     col=[]
#     color=[0,4,5,3,8,9,1,7,6,2,17,16,14,12,13,11,18,15,17,10]
#     # color=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
#     #        'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
#     #        'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
#     #        'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
#     #        'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 
#     #        'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
#     #        'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
#     #        'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 
#     #        'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
#     #        'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
#     #        'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
#     #        'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
#     #        'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 
#     #        'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 
#     #        'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
#     #        'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
#     #        'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter', 
#     #        'winter_r']

#     for i in range(0,len(y)):
#         col.append(color[y[i]%20])
        
#     # XX = np.arange(-5, 5, 0.25)
#     # YY = np.arange(-5, 5, 0.25)
#     # XX, YY = np.meshgrid(XX, YY)
#     # R = np.sqrt(XX**2 + YY**2)
#     # Z = np.sin(R)  
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=col,colormap='jet',label="t-SNE",marker='4')#hsv_r
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=col,cmap='tab20_r',label="t-SNE",marker='4',linewidth=1.1)
#     plt.legend()
#     # plt.subplot(122)
#     # #plt.scatter(X_pca[:, 0], X_pca[:, 1], color=col,colormap='jet',label="PCA",marker='4')
#     # plt.scatter(X_pca[:, 0], X_tsne[:, 1], c=col,cmap='jet',label="t-SNE",marker='4')
#     # plt.legend()
#     plt.savefig('images/digits_tsne-pca.png', dpi=1000, format='png')
#     plt.show()

# if __name__ == "__main__":
#     # 加载你的数据集
#     X = np.load('/home/ljb/code/PMT-main/Pytorch-PMT-VI-ReID/SYSU-MM01/train_rgb_resized_img.npy')  # 特征矩阵
#     y = np.load('/home/ljb/code/PMT-main/Pytorch-PMT-VI-ReID/SYSU-MM01/train_rgb_resized_label.npy')  # 标签数组

#     # 调用可视化函数
#     plot_embedding_2d(X, y, title="Your Dataset Visualization")
    
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from clustercontrast.model_vit_cmrefine.make_model import make_model
# 假设你的模型已经加载并设置为评估模式

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.transform_test
# 加载数据集

data_path = '/root/autodl-tmp/project/data/sysu'
query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(144, 288))

gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0)
gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(144, 288))

# Test loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,drop_last=True)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,drop_last=True)

model = make_model(_C, num_class=0, camera_num=0, view_num = 0)
weights_path = '/root/autodl-tmp/project/logs/sysu_train/model_best.pth.tar'
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
model.eval()
# 从 DataLoader 中提取数据
# def extract_data_from_loader(query_loader):
#     loader = query_loader
#     features = []
#     labels = []
#     model.eval()  # 设置为评估模式
#     with torch.no_grad():
#         for batch_idx, (input, label) in enumerate(loader):
#         #     # if len(features) >= num_samples:
#         #     #     break
#             outputs = model.forward(input, input,modal = 1)
#             features.append(outputs.cpu().numpy())
#             labels.append(label.cpu().numpy())
#     features = np.concatenate(features, axis=0)
#     labels = np.concatenate(labels, axis=0)
#     return features, labels
def extract_data_from_loader(query_loader):
    loader = query_loader
    features = []
    labels = []
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            outputs = model.forward(input, input, modal=1)
            # 从模型输出中提取特征张量
            if isinstance(outputs, tuple):
                # 通常第一个元素是特征(features)，但需要根据实际模型调整
                feat = outputs[0]  # 尝试第一个元素
                # 如果第一个不是特征，可能需要改为 outputs[1] 或其他索引
            else:
                feat = outputs
                
            
            features.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())

    # 合并所有batch的特征和标签
    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels
    else:
        raise ValueError("未能提取有效特征 - 请检查模型输出结构")
features, labels = extract_data_from_loader(query_loader)

def visualize_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab20', marker='o')
    plt.xlim(-20, 20) # 根据实际数据调整范围
    plt.ylim(-20, 20)  # 根据实际数据调整范围
    # plt.title('t-SNE Visualization of SMFC')

    # plt.xlabel('(b) t-SNE Visualization of DMFCNet', fontsize=18, labelpad=10)

    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')

    plt.savefig(save_path)
    plt.show()

# 调用可视化函数
save_path = 'tsne_visualizatiokong.png'
visualize_tsne(features, labels, save_path)
    

