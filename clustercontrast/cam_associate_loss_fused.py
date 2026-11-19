import torch
import torch.nn.functional as F

def get_proxy_associate_loss_fused(cam2proxy_rgb, cam2proxy_ir,
                                   s_proxies_rgb, s_proxies_ir,
                                   scores_proxy_rgb, scores_proxy_ir,
                                   bg_knn, temperature=0.15,
                                   epoch=0, total_epochs=40,
                                   use_focal=True,
                                   use_hard_mining=True, hard_mining_ratio=0.3,
                                   cross_modal_weight=0.5,
                                   modality_weights=None,
                                   proxy_features_rgb=None,
                                   proxy_features_ir=None,
                                   cluster_centers_rgb=None,
                                   cluster_centers_ir=None):
    """
    1. 模态感知的自适应温度：根据训练进度和模态动态调整
    2. Focal Loss：对难样本给予更多关注
    3. 难样本挖掘：只学习难样本
    4. 跨模态对齐：使用特征分布对齐方法
    5. 模态权重平衡：平衡不同模态的学习难度
    6. 代理质量感知：基于特征多样性计算代理质量
    
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
        use_hard_mining: 是否使用难样本挖掘
        hard_mining_ratio: 难样本挖掘比例
        cross_modal_weight: 跨模态对齐的权重
        modality_weights: 模态权重字典 {'rgb': weight_rgb, 'ir': weight_ir}
        proxy_features_rgb: RGB代理特征 [num_proxies, feature_dim]
        proxy_features_ir: IR代理特征 [num_proxies, feature_dim]
        cluster_centers_rgb: RGB聚类中心 [num_clusters, feature_dim]
        cluster_centers_ir: IR聚类中心 [num_clusters, feature_dim]
    
    Returns:
        loss_rgb: RGB模态损失
        loss_ir: IR模态损失
        loss_cross: 跨模态对齐损失
        total_loss: 总损失
    """
    device = scores_proxy_rgb.device
    
    if modality_weights is None:
        modality_weights = {'rgb': 1.0, 'ir': 1.0}
    
    loss_rgb = compute_single_modal_loss(
        cam2proxy_rgb, s_proxies_rgb, scores_proxy_rgb, bg_knn,
        temperature, epoch, total_epochs, use_focal,
        use_hard_mining, hard_mining_ratio, 'rgb',
        proxy_features_rgb, cluster_centers_rgb
    )
    
    loss_ir = compute_single_modal_loss(
        cam2proxy_ir, s_proxies_ir, scores_proxy_ir, bg_knn,
        temperature, epoch, total_epochs, use_focal,
        use_hard_mining, hard_mining_ratio, 'ir',
        proxy_features_ir, cluster_centers_ir
    )
    
    loss_cross = compute_cross_modal_alignment(
        scores_proxy_rgb, scores_proxy_ir,
        temperature, epoch, total_epochs,
        s_proxies_rgb, s_proxies_ir
    )
    
    intra_modal_loss = modality_weights['rgb'] * loss_rgb + modality_weights['ir'] * loss_ir
    total_loss = intra_modal_loss + cross_modal_weight * loss_cross
    
    return loss_rgb, loss_ir, loss_cross, total_loss


def compute_single_modal_loss(cam2proxy, s_proxies, scores_proxy, bg_knn,
                             temperature, epoch, total_epochs,
                             use_focal,
                             use_hard_mining, hard_mining_ratio,
                             modality,
                             proxy_features=None,
                             cluster_centers=None):
    """
    计算单一模态的损失（RGB或IR）
    
    融合策略：
    1. 模态感知的自适应温度
    2. Focal Loss（代理质量感知）
    3. 难样本挖掘
    4. 基于特征多样性的代理质量计算
    
    Args:
        cam2proxy: 相机到代理的映射 [num_cams, num_proxies_per_cam]
        s_proxies: 样本的代理索引 [batch_size, num_cam]
        scores_proxy: 样本与代理的相似度分数 [batch_size, num_proxy]
        bg_knn: 背景邻居数量
        temperature: 初始温度参数
        epoch: 当前训练轮次
        total_epochs: 总训练轮次
        use_focal: 是否使用Focal Loss
        use_hard_mining: 是否使用难样本挖掘
        hard_mining_ratio: 难样本挖掘比例
        modality: 模态类型 ('rgb' 或 'ir')
        proxy_features: 代理特征 [num_proxies, feature_dim]
        cluster_centers: 聚类中心 [num_clusters, feature_dim]
    
    Returns:
        final_loss: 最终损失值
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
    
    progress = epoch / max(1.0, total_epochs)
    
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
        
        if modality == 'ir':
            adaptive_temp = temperature * (1.2 - 0.9 * progress)
        else:
            adaptive_temp = temperature * (1.0 - 0.8 * progress)
        
        adaptive_temp = max(0.01, min(1.0, adaptive_temp))

        log_probs = F.log_softmax(select_sim / adaptive_temp, dim=1)
        probs = F.softmax(select_sim / adaptive_temp, dim=1)
        
        pos_probs = probs[:, -1]
        
        focal_weight = 1.0
        if use_focal:
            if epoch < 10:
                use_focal_current = False
                focal_weight = 1.0
            elif epoch < 20:
                focal_progress = (epoch - 10) / 10
                focal_gamma = 1.0 + focal_progress * 1.0
                focal_weight = (1.0 - pos_probs) ** focal_gamma + 0.2
            else:
                focal_gamma = 2.0 if modality == 'ir' else 1.5
                
                # 基于特征多样性计算代理质量
                if proxy_features is not None:
                    proxy_qualities = compute_proxy_quality_by_diversity(proxy_features, cluster_centers)
                    # 对于每个样本，获取其目标代理的质量
                    target_proxy_indices = s_proxies[samples_with_cam, cam]
                    valid_proxy_mask = (target_proxy_indices >= 0) & (target_proxy_indices < num_proxy)
                    target_proxy_indices = target_proxy_indices[valid_proxy_mask]
                    
                    if len(target_proxy_indices) > 0:
                        proxy_quality = proxy_qualities[target_proxy_indices]
                        adaptive_gamma = focal_gamma * (2.0 - proxy_quality)
                    else:
                        proxy_quality = topk_values[:, -1] / (topk_values[:, -1].max() + 1e-8)
                        adaptive_gamma = focal_gamma * (2.0 - proxy_quality)
                else:
                    proxy_quality = topk_values[:, -1] / (topk_values[:, -1].max() + 1e-8)
                    adaptive_gamma = focal_gamma * (2.0 - proxy_quality)
                
                focal_weight = (1.0 - pos_probs) ** adaptive_gamma
                focal_weight = torch.clamp(focal_weight, min=0.1, max=2.0)
        
        ce_loss = -1.0 * focal_weight * log_probs[:, -1]
        
        loss[samples_with_cam, cam] = ce_loss
        sample_indices.append(samples_with_cam)

    per_sample_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze()
    per_sample_mean = loss.sum(dim=1) / per_sample_counts

    if use_hard_mining and epoch > total_epochs * 0.3:
        all_samples = torch.cat(sample_indices) if len(sample_indices) > 0 else torch.arange(batch_size, device=device)
        all_losses = per_sample_mean[all_samples]
        
        if modality == 'ir':
            effective_ratio = min(0.5, hard_mining_ratio * 1.3)
        else:
            effective_ratio = hard_mining_ratio
        
        num_hard = max(1, int(len(all_losses) * effective_ratio))
        hard_indices = torch.topk(all_losses, num_hard)[1]
        hard_samples = all_samples[hard_indices]
        
        final_loss = per_sample_mean[hard_samples].mean()
    else:
        final_loss = per_sample_mean.mean()
    
    final_loss = torch.clamp(final_loss, min=0.0, max=10.0)

    return final_loss


def compute_cross_modal_alignment(scores_proxy_rgb, scores_proxy_ir,
                              temperature, epoch, total_epochs,
                              s_proxies_rgb=None, s_proxies_ir=None):
    """
    计算跨模态对齐损失（相机感知版）
    
    策略：
    1. 对齐RGB和IR的特征分布
    2. 使用简化的分布对齐方法（均值对齐）
    3. 添加特征相似度约束
    4. 相机感知：按相机分组计算对齐损失
    """
    device = scores_proxy_rgb.device
    
    batch_size_rgb = scores_proxy_rgb.shape[0]
    batch_size_ir = scores_proxy_ir.shape[0]
    min_batch_size = min(batch_size_rgb, batch_size_ir)
    
    rgb_features = F.normalize(scores_proxy_rgb[:min_batch_size], dim=1)
    ir_features = F.normalize(scores_proxy_ir[:min_batch_size], dim=1)
    
    mean_rgb = rgb_features.mean(dim=0)
    mean_ir = ir_features.mean(dim=0)
    
    mean_alignment_loss = F.mse_loss(mean_rgb, mean_ir)
    
    similarity_matrix = rgb_features @ ir_features.t()
    diagonal_sim = torch.diag(similarity_matrix)
    
    similarity_loss = torch.mean((diagonal_sim - 1.0) ** 2)
    
    off_diagonal_sim = similarity_matrix - torch.eye(min_batch_size, device=device)
    regularization_loss = torch.mean(off_diagonal_sim ** 2)
    
    total_alignment_loss = mean_alignment_loss + 0.5 * similarity_loss + 0.1 * regularization_loss
    
    if s_proxies_rgb is not None and s_proxies_ir is not None:
        camera_aware_loss = compute_camera_aware_alignment(
            rgb_features, ir_features,
            s_proxies_rgb[:min_batch_size], s_proxies_ir[:min_batch_size]
        )
        total_alignment_loss = total_alignment_loss + 0.3 * camera_aware_loss
    
    return total_alignment_loss


def compute_camera_aware_alignment(rgb_features, ir_features, s_proxies_rgb, s_proxies_ir):
    """
    计算相机感知的对齐损失
    
    策略：
    1. 按相机分组计算特征均值
    2. 对齐相同相机ID的RGB和IR特征
    3. 添加跨相机的正则化
    """
    device = rgb_features.device
    batch_size, num_cam = s_proxies_rgb.shape
    
    camera_alignment_losses = []
    
    for cam in range(num_cam):
        rgb_mask = s_proxies_rgb[:, cam] >= 0
        ir_mask = s_proxies_ir[:, cam] >= 0
        
        if rgb_mask.sum() == 0 or ir_mask.sum() == 0:
            continue
        
        rgb_cam_features = rgb_features[rgb_mask]
        ir_cam_features = ir_features[ir_mask]
        
        if rgb_cam_features.shape[0] == 0 or ir_cam_features.shape[0] == 0:
            continue
        
        mean_rgb_cam = rgb_cam_features.mean(dim=0)
        mean_ir_cam = ir_cam_features.mean(dim=0)
        
        cam_alignment_loss = F.mse_loss(mean_rgb_cam, mean_ir_cam)
        camera_alignment_losses.append(cam_alignment_loss)
    
    if len(camera_alignment_losses) > 0:
        camera_aware_loss = torch.stack(camera_alignment_losses).mean()
    else:
        camera_aware_loss = torch.tensor(0.0, device=device)
    
    return camera_aware_loss


def compute_proxy_quality_by_diversity(proxy_features, cluster_centers=None):
    """
    基于特征空间中的多样性计算代理质量
    
    思想：好的代理应该既具有代表性又具有区分性
    
    Args:
        proxy_features: [num_proxies, feature_dim] 代理特征
        cluster_centers: [num_clusters, feature_dim] 聚类中心（可选）
    
    Returns:
        quality: [num_proxies] 每个代理的质量分数 [0.1, 1.0]
    """
    device = proxy_features.device
    num_proxies = proxy_features.shape[0]
    
    # 1. 代表性：代理特征与最近的聚类中心的距离
    if cluster_centers is not None:
        # 计算每个代理到所有聚类中心的距离
        dists = torch.cdist(proxy_features, cluster_centers)  # [num_proxies, num_clusters]
        min_dists, _ = dists.min(dim=1)  # 到最近聚类中心的距离
        # 距离越小 → 代表性越好 → 质量越高
        representativeness = torch.exp(-min_dists / (min_dists.mean() + 1e-8))
    else:
        representativeness = torch.ones(num_proxies, device=device)
    
    # 2. 区分性：代理之间的相互距离
    # 计算代理间的成对距离
    proxy_distances = torch.cdist(proxy_features, proxy_features)  # [num_proxies, num_proxies]
    # 排除自身距离
    proxy_distances = proxy_distances + torch.eye(num_proxies, device=device) * 1e6
    # 到最近其他代理的距离
    min_inter_distance, _ = proxy_distances.min(dim=1)
    # 距离越大 → 区分性越好 → 质量越高
    distinctiveness = torch.sigmoid(min_inter_distance / (min_inter_distance.mean() + 1e-8))
    
    # 3. 特征范数：特征强度
    feature_norms = torch.norm(proxy_features, p=2, dim=1)  # [num_proxies]
    strength = feature_norms / (feature_norms.max() + 1e-8)
    
    # 综合质量
    quality = 0.4 * representativeness + 0.3 * distinctiveness + 0.3 * strength
    
    return quality.clamp(0.1, 1.0)