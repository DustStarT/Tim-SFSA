import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _safe_fallback_loss(tensor, scale=1e-3):
    """当检测到数值异常时，返回与模型输出相关的一个小的可反向传播损失，避免静默返回0导致训练停滞。"""
    try:
        tensor = tensor.to(dtype=torch.float32)
        return scale * torch.mean(tensor**2)
    except Exception:
        # 最后手段：返回一个小的数字常量（在device上）
        return torch.tensor(1e-6, dtype=torch.float32)

def cox_loss_deepsurv(log_risk, durations, events, weights=None):
    """
    DeepSurv的Cox比例风险损失函数
    
    Args:
        log_risk: 模型输出的对数风险分数 (batch_size,)
        durations: 持续时间 (batch_size,)
        events: 事件指示器 (batch_size,)
        weights: 样本权重 (batch_size,)
        
    Returns:
        损失值
    """
    # 确保输入是张量
    if not isinstance(log_risk, torch.Tensor):
        log_risk = torch.tensor(log_risk, dtype=torch.float32)
    if not isinstance(durations, torch.Tensor):
        durations = torch.tensor(durations, dtype=torch.float32)
    if not isinstance(events, torch.Tensor):
        events = torch.tensor(events, dtype=torch.float32)
    
    # 移动到同一设备
    device = log_risk.device
    durations = durations.to(device)
    events = events.to(device)
    
    # 如果没有提供权重，使用全1权重
    if weights is None:
        weights = torch.ones_like(events, dtype=torch.float32)
    else:
        weights = weights.to(device)
    
    # 计算风险集合
    # 对于每个事件样本，找到所有持续时间大于等于它的样本
    # 确保所有输入都是1D张量
    if durations.dim() > 1:
        durations_flat = durations.squeeze()
    else:
        durations_flat = durations
        
    if log_risk.dim() > 1:
        log_risk_flat = log_risk.squeeze()
    else:
        log_risk_flat = log_risk
        
    if events.dim() > 1:
        events_flat = events.squeeze()
    else:
        events_flat = events
        
    if weights is not None and weights.dim() > 1:
        weights_flat = weights.squeeze()
    else:
        weights_flat = weights
    
    # 创建风险集合矩阵
    batch_size = len(durations_flat)
    risk_set = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=durations_flat.device)
    
    # 计算每个样本的风险集合
    for i in range(batch_size):
        risk_set[i] = durations_flat >= durations_flat[i]
    
    # 计算对数似然
    # 对于每个事件样本，计算其风险分数与风险集合中所有样本风险分数之和的比值
    loss = 0.0
    n_events = 0
    
    for i in range(batch_size):
        if events_flat[i] == 1:  # 只考虑事件样本
            # 当前样本的风险分数
            current_risk = log_risk_flat[i]
            
            # 风险集合中所有样本的风险分数
            risk_set_scores = log_risk_flat[risk_set[i]]
            
            # 计算风险集合的对数和
            if risk_set_scores.numel() > 0:
                risk_set_sum = torch.logsumexp(risk_set_scores, dim=0)
                
                # 计算负对数似然
                if weights_flat is not None:
                    sample_loss = -(current_risk - risk_set_sum) * weights_flat[i]
                else:
                    sample_loss = -(current_risk - risk_set_sum)
                loss += sample_loss
                n_events += 1
    
    # 如果没有事件样本，返回一个与log_risk有关的小的回退损失（避免静默0）
    if n_events == 0:
        logger.warning("cox_loss_deepsurv: no events in batch, returning safe fallback loss")
        return _safe_fallback_loss(log_risk_flat)
    
    # 返回平均损失
    return loss / n_events

def focal_loss_deepsurv(log_risk, durations, events, alpha=0.25, gamma=2.0, weights=None):
    """
    Focal loss for DeepSurv，用于处理类别不平衡
    
    Args:
        log_risk: 模型输出的对数风险分数 (batch_size,)
        durations: 持续时间 (batch_size,)
        events: 事件指示器 (batch_size,)
        alpha: 平衡参数
        gamma: 聚焦参数
        weights: 样本权重 (batch_size,)
        
    Returns:
        损失值
    """
    # 首先计算Cox损失
    cox_loss = cox_loss_deepsurv(log_risk, durations, events, weights)
    
    # 计算事件比例
    event_ratio = torch.mean(events.float())
    
    # 应用focal loss调整
    if event_ratio < 0.5:  # 如果事件比例小于0.5，说明事件样本较少
        # 对事件样本增加权重
        focal_weight = alpha * (1 - event_ratio) ** gamma
        cox_loss = cox_loss * focal_weight
    
    return cox_loss

def ranking_loss_deepsurv(log_risk, durations, events, margin=1.0, weights=None):
    """
    排序损失，确保高风险样本的持续时间更短
    
    Args:
        log_risk: 模型输出的对数风险分数 (batch_size,)
        durations: 持续时间 (batch_size,)
        events: 事件指示器 (batch_size,)
        margin: 排序边界
        weights: 样本权重 (batch_size,)
        
    Returns:
        损失值
    """
    device = log_risk.device
    
    # 如果没有提供权重，使用全1权重
    if weights is None:
        weights = torch.ones_like(events, dtype=torch.float32)
    else:
        weights = weights.to(device)
    
    loss = 0.0
    n_pairs = 0
    
    # 对每对样本计算排序损失
    for i in range(len(durations)):
        for j in range(i + 1, len(durations)):
            # 如果两个样本都是事件样本，或者一个是事件样本一个是删失样本
            if events[i] == 1 or events[j] == 1:
                # 如果样本i的持续时间更短，那么它的风险分数应该更高
                if durations[i] < durations[j]:
                    # 样本i的风险分数应该大于样本j的风险分数
                    risk_diff = log_risk[j] - log_risk[i] + margin
                    if risk_diff > 0:
                        loss += F.relu(risk_diff) * weights[i] * weights[j]
                elif durations[j] < durations[i]:
                    # 样本j的风险分数应该大于样本i的风险分数
                    risk_diff = log_risk[i] - log_risk[j] + margin
                    if risk_diff > 0:
                        loss += F.relu(risk_diff) * weights[i] * weights[j]
                
                n_pairs += 1
    
    # 如果没有有效对，返回零损失
    if n_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss / n_pairs

def _compute_ipcw_weights(durations: torch.Tensor, events: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    简单的IPCW估计：用Kaplan-Meier对删失分布进行估计，得到 G(t)=P(C>t)。
    返回 w_i = 1 / max(G(t_i), eps)。
    说明：为保持轻量与稳定，此实现使用批内KM近似，适合训练期pairwise加权；
    更精确的实现可在全数据/验证阶段离线计算并缓存。
    """
    device = durations.device
    # 排序
    order = torch.argsort(durations)
    t = durations[order]
    e = events[order]
    n = t.shape[0]
    # KM on censoring: treat censor=1-e
    censor = 1 - e
    # distinct times
    uniq, inverse = torch.unique_consecutive(t, return_inverse=True)
    # risk set counts at each time
    # For stability: compute at each unique time the number at risk and number censored
    G = torch.ones_like(uniq, dtype=torch.float32, device=device)
    at_risk = torch.empty_like(uniq, dtype=torch.float32)
    cens_at_time = torch.zeros_like(uniq, dtype=torch.float32)
    # counts
    for idx in range(uniq.shape[0]):
        mask_risk = t >= uniq[idx]
        at_risk[idx] = torch.sum(mask_risk).float()
        mask_time = (t == uniq[idx])
        cens_at_time[idx] = torch.sum(censor[mask_time]).float()
    # KM stepwise product: G(t_k) = Π_{j<=k} (1 - d_cens_j / n_risk_j)
    hazard_cens = torch.where(at_risk > 0, cens_at_time / at_risk, torch.zeros_like(at_risk))
    step = 1.0 - hazard_cens
    step = torch.clamp(step, min=0.0, max=1.0)
    G = torch.cumprod(step, dim=0)
    # map back to sample times
    G_at_i = G[inverse]
    # un-sort back
    G_unsort = torch.empty_like(G_at_i)
    G_unsort[order] = G_at_i
    w = 1.0 / torch.clamp(G_unsort, min=eps)
    return w

def ranking_loss_ipcw_pairwise(log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor,
                               weights: torch.Tensor = None, logistic: bool = True, sigma: float = 1.0) -> torch.Tensor:
    """
    IPCW加权的pairwise排序损失：只对可比较对(i,j)（t_i < t_j 且 e_i=1）计入，
    损失为 logistic: log(1 + exp(-(r_i - r_j)/sigma)) 或 hinge: relu(margin - (r_i - r_j))。
    """
    device = log_risk.device
    n = log_risk.shape[0]
    # compute IPCW per sample
    ipcw = _compute_ipcw_weights(durations.detach(), events.detach())
    if weights is not None:
        ipcw = ipcw.to(device) * weights.to(device)
    else:
        ipcw = ipcw.to(device)
    # build comparable mask
    # i earlier than j, and i is event
    ti = durations.view(-1, 1)
    tj = durations.view(1, -1)
    comp = (ti < tj) & (events.view(-1, 1) > 0.5)
    if not torch.any(comp):
        return torch.sum(log_risk * 0.0)
    ri = log_risk.view(-1, 1).expand(n, n)
    rj = log_risk.view(1, -1).expand(n, n)
    diff = ri - rj
    if logistic:
        per_pair = torch.nn.functional.softplus(-diff / max(sigma, 1e-6))
    else:
        per_pair = torch.relu(1.0 - diff)  # simple hinge with margin=1
    w_pairs = ipcw.view(-1, 1).expand(n, n)
    loss = (per_pair * comp.float() * w_pairs).sum() / (comp.float() * w_pairs + 1e-8).sum()
    return loss

def combined_loss_deepsurv(log_risk, durations, events, weights=None, 
                          cox_weight=1.0, focal_weight=0.3, ranking_weight=0.2,
                          ranking_variant: str = 'ipcw_pairwise', ranking_margin: float = 0.0):
    """
    组合损失函数，结合Cox损失、Focal损失和排序损失
    
    Args:
        log_risk: 模型输出的对数风险分数 (batch_size,)
        durations: 持续时间 (batch_size,)
        events: 事件指示器 (batch_size,)
        weights: 样本权重 (batch_size,)
        cox_weight: Cox损失权重
        focal_weight: Focal损失权重
        ranking_weight: 排序损失权重
        
    Returns:
        组合损失值
    """
    # 计算各种损失
    cox_loss = cox_loss_deepsurv(log_risk, durations, events, weights)
    focal_loss = focal_loss_deepsurv(log_risk, durations, events, weights=weights)
    if str(ranking_variant).lower() == 'ipcw_pairwise':
        ranking_loss = ranking_loss_ipcw_pairwise(log_risk, durations, events, weights=weights, logistic=True)
    elif str(ranking_variant).lower() == 'hinge':
        ranking_loss = ranking_loss_deepsurv(log_risk, durations, events, margin=ranking_margin, weights=weights)
    else:
        # fallback to logistic pairwise without IPCW
        ranking_loss = ranking_loss_deepsurv(log_risk, durations, events, margin=ranking_margin, weights=weights)
    
    # 组合损失
    total_loss = (cox_weight * cox_loss + 
                  focal_weight * focal_loss + 
                  ranking_weight * ranking_loss)
    
    return total_loss

def l2_regularization_loss(model, l2_lambda=0.001):
    """
    L2正则化损失
    
    Args:
        model: 模型
        l2_lambda: L2正则化系数
        
    Returns:
        L2正则化损失
    """
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2) ** 2
    return l2_lambda * l2_loss
