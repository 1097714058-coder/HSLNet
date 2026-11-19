import torch
import torch.nn.functional as F

def get_proxy_associate_loss(cam2proxy_rgb, cam2proxy_ir,
                                   s_proxies_rgb, s_proxies_ir,
                                   scores_proxy_rgb, scores_proxy_ir,
                                   bg_knn, temperature=0.05,
                                   epoch=0, total_epochs=40,
                                   use_focal=True, use_contrastive=True,
                                   use_hard_mining=True, hard_mining_ratio=0.3,
                                   margin=0.5, cross_modal_weight=0.5):
    """
    融合版 Camera Proxy 关联损失
    
    融合策略：
    1. 自适应温度：根据训练进度动态调整
    2. Focal Loss：对难样本给予更多关注
    3. 对比学习：学习模态内和跨模态的相似性
    4. 难样本挖掘：只学习难样本
    5. 跨模态对齐：分别学习RGB和IR的相机代理
    
    Args:
        cam2proxy_rgb: RGB相机到代理的映射 [num_cams, num_proxies_per_cam]
        cam2proxy_ir: IR相机到代理的映射 [num_cams, num_proxies_per_cam]
        s_proxies_rgb: RGB样本的代理索引 [batch_size, num_cam]
        s_proxies_ir: IR样本的代理索引 [batch_size, num_cam]
        scores_proxy_rgb: RGB样本与代理的相似度分数 [batch_size, num_proxy]
        scores_proxy_ir: IR样本与代理的相似度分数 [batch_size, num_proxy]
        bg_knn: 背景邻居数量
        temperature: 初始温度参数
        epoch: 当前训练轮次
        total_epochs: 总训练轮次
        use_focal: 是否使用Focal Loss
        use_contrastive: 是否使用对比学习
        use_hard_mining: 是否使用难样本挖掘
        hard_mining_ratio: 难样本挖掘比例
        margin: 对比学习的margin参数
        cross_modal_weight: 跨模态对齐的权重
    
    Returns:
        loss_rgb: RGB模态损失
        loss_ir: IR模态损失
        loss_cross: 跨模态对齐损失
    """
    device = scores_proxy_rgb.device
    
    loss_rgb = compute_single_modal_loss(
        cam2proxy_rgb, s_proxies_rgb, scores_proxy_rgb, bg_knn,
        temperature, epoch, total_epochs, use_focal, use_contrastive,
        use_hard_mining, hard_mining_ratio, margin, 'rgb'
    )
    
    loss_ir = compute_single_modal_loss(
        cam2proxy_ir, s_proxies_ir, scores_proxy_ir, bg_knn,
        temperature, epoch, total_epochs, use_focal, use_contrastive,
        use_hard_mining, hard_mining_ratio, margin, 'ir'
    )
    
    loss_cross = compute_cross_modal_alignment(
        scores_proxy_rgb, scores_proxy_ir,
        temperature, epoch, total_epochs
    )
    
    total_loss = loss_rgb + loss_ir + cross_modal_weight * loss_cross
    
    return loss_rgb, loss_ir, loss_cross, total_loss


def compute_single_modal_loss(cam2proxy, s_proxies, scores_proxy, bg_knn,
                             temperature, epoch, total_epochs,
                             use_focal, use_contrastive,
                             use_hard_mining, hard_mining_ratio, margin,
                             modality):
    """
    计算单一模态的损失（RGB或IR）
    
    融合策略：
    1. 自适应温度
    2. Focal Loss
    3. 对比学习
    4. 难样本挖掘
    """
    device = scores_proxy.device
    
    cam2proxy = cam2proxy.to(device)
    
    batch_size, num_proxy = scores_proxy.shape
    num_cam = s_proxies.shape[1]
    num_cams = cam2proxy.shape[0]
    
    cam2proxy_mask = (cam2proxy != -1) & (cam2proxy >= 0) & (cam2proxy < num_proxy)

    per_label_proxies = s_proxies.clone()
    per_label_proxies[per_label_proxies == -1] = num_proxy
    per_label_proxies[per_label_proxies >= num_proxy] = num_proxy

    temp_sim = scores_proxy.clone()
    valid_mask = (per_label_proxies != num_proxy)

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_cam)
    valid_batch_indices = batch_indices[valid_mask]
    valid_proxies = per_label_proxies[valid_mask]

    valid_proxy_mask = valid_proxies < num_proxy
    valid_proxies = valid_proxies[valid_proxy_mask]
    valid_batch_indices = valid_batch_indices[valid_proxy_mask]

    if valid_proxies.numel() > 0 and valid_batch_indices.numel() > 0:
        temp_sim[valid_batch_indices, valid_proxies] = -10000.0

    loss = torch.zeros((batch_size, num_cam), device=device)
    sample_indices = []

    for cam in range(min(num_cam, num_cams)):
        samples_with_cam = torch.where(valid_mask[:, cam])[0]

        if samples_with_cam.numel() == 0:
            continue

        per_cam_proxies = torch.where(cam2proxy_mask[cam])[0]

        if per_cam_proxies.numel() == 0:
            continue

        per_cam_proxies = per_cam_proxies[per_cam_proxies < num_proxy]
        
        if per_cam_proxies.numel() == 0:
            continue

        per_cam_sim = scores_proxy[samples_with_cam][:, per_cam_proxies]
        per_cam_temp_sim = temp_sim[samples_with_cam][:, per_cam_proxies]

        actual_bg_knn = min(bg_knn + 1, per_cam_proxies.numel())
        
        topk_values, topk_indices = torch.topk(per_cam_temp_sim, actual_bg_knn, dim=1)
        select_sim = torch.gather(per_cam_sim, dim=1, index=topk_indices)

        select_target = torch.zeros_like(select_sim)
        select_target[:, -1] = 1.0

        adaptive_temp = temperature * (1.0 - 0.8 * epoch / total_epochs)
        log_probs = F.log_softmax(select_sim / adaptive_temp, dim=1)
        
        focal_weight = 1.0
        if use_focal:
            focal_weight = (1.0 - select_target) ** 2.0
        
        ce_loss = -1.0 * focal_weight * (log_probs * select_target).sum(dim=1)
        
        contrastive_loss = 0.0
        if use_contrastive:
            adaptive_margin = margin * (1.0 + 0.5 * epoch / total_epochs)
            pos_prob = select_sim[:, -1] / adaptive_temp
            neg_prob = select_sim[:, :-1] / adaptive_temp
            contrastive_loss = F.relu(adaptive_margin - pos_prob.unsqueeze(1) + neg_prob).mean(dim=1)
        
        total_sample_loss = ce_loss + 0.1 * contrastive_loss
        loss[samples_with_cam, cam] = total_sample_loss
        sample_indices.append(samples_with_cam)

    per_sample_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze()
    per_sample_mean = loss.sum(dim=1) / per_sample_counts

    if use_hard_mining and epoch > total_epochs * 0.3:
        all_samples = torch.cat(sample_indices) if len(sample_indices) > 0 else torch.arange(batch_size, device=device)
        all_losses = per_sample_mean[all_samples]
        
        num_hard = max(1, int(len(all_losses) * hard_mining_ratio))
        hard_indices = torch.topk(all_losses, num_hard)[1]
        hard_samples = all_samples[hard_indices]
        
        final_loss = per_sample_mean[hard_samples].mean()
    else:
        final_loss = per_sample_mean.mean()

    return final_loss


def compute_cross_modal_alignment(scores_proxy_rgb, scores_proxy_ir,
                              temperature, epoch, total_epochs):
    """
    计算跨模态对齐损失
    
    策略：
    1. 对齐RGB和IR的特征空间
    2. 使用对比学习框架
    3. 自适应温度
    """
    device = scores_proxy_rgb.device
    
    batch_size = scores_proxy_rgb.shape[0]
    
    adaptive_temp = temperature * (1.0 - 0.8 * epoch / total_epochs)
    
    rgb_features = F.normalize(scores_proxy_rgb, dim=1)
    ir_features = F.normalize(scores_proxy_ir, dim=1)
    
    similarity_matrix = rgb_features @ ir_features.t()
    
    pos_mask = torch.eye(batch_size, device=device).bool()
    neg_mask = ~pos_mask
    
    pos_sim = similarity_matrix[pos_mask].mean()
    neg_sim = similarity_matrix[neg_mask].mean()
    
    alignment_loss = F.relu(0.5 - pos_sim) + F.relu(neg_sim - 0.3)
    
    return alignment_loss