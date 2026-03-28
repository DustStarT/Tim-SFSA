import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _safe_fallback_loss(tensor, scale=1e-3):
    """Return a small MSE-based loss tied to tensor to keep graph for backprop when encountering numeric issues."""
    try:
        tensor = tensor.to(dtype=torch.float32)
        return scale * torch.mean(tensor**2)
    except Exception:
        return torch.tensor(1e-6, dtype=torch.float32)

def cox_ph_loss_stable(risk_scores, y_batch, sample_weights=None, model=None):
    """
    计算Cox比例风险损失，使用更稳定的数值方法。
    这是从根源解决NaN问题的版本。

    Args:
        risk_scores (torch.Tensor): 模型的风险预测，形状为 (batch_size,)。
        y_batch (torch.Tensor): 标签，形状为 (batch_size, 2)，其中 y_batch[:, 0] 是时间，y_batch[:, 1] 是事件指示器。
        sample_weights (torch.Tensor, optional): 样本权重，形状为 (batch_size,)。默认为 None。
        model (nn.Module, optional): 模型实例，用于计算正则化损失。

    Returns:
        torch.Tensor: 计算出的Cox损失（标量）。
    """
    # 确保输入是1D张量
    risk_scores = risk_scores.view(-1)
    if sample_weights is not None:
        sample_weights = sample_weights.view(-1)

    # 检查输入是否包含NaN或Inf
    if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
        logger.warning("cox_ph_loss_stable: NaN/Inf detected in risk_scores, using fallback loss")
        return _safe_fallback_loss(risk_scores)

    # 提取时间和事件信息
    durations = y_batch[:, 0]
    events = y_batch[:, 1]
    
    # 检查标签是否包含NaN或Inf
    if torch.isnan(durations).any() or torch.isinf(durations).any():
        logger.warning("cox_ph_loss_stable: NaN/Inf detected in durations, using fallback loss")
        return _safe_fallback_loss(risk_scores)
    
    if torch.isnan(events).any():
        logger.warning("cox_ph_loss_stable: NaN detected in events, using fallback loss")
        return _safe_fallback_loss(risk_scores)
    
    # 对风险分数进行排序（按时间降序）
    sorted_indices = torch.argsort(durations, descending=True)
    risk_scores_sorted = risk_scores[sorted_indices]
    events_sorted = events[sorted_indices]
    
    # 使用更稳定的log-sum-exp技巧
    # 1. 找到风险分数的最大值
    risk_max = torch.max(risk_scores_sorted)
    
    # 2. 如果最大值过大，进行缩放
    if risk_max > 20.0:
        scale_factor = 20.0 / risk_max
        risk_scores_sorted = risk_scores_sorted * scale_factor
        risk_max = 20.0
    
    # 3. 计算稳定的exp
    risk_scores_stabilized = risk_scores_sorted - risk_max
    exp_risk = torch.exp(risk_scores_stabilized)
    
    # 4. 检查exp是否包含NaN或Inf
    if torch.isnan(exp_risk).any() or torch.isinf(exp_risk).any():
        # 如果exp溢出，使用更保守的方法
        exp_risk = torch.clamp(exp_risk, 0.0, 1e6)
    
    # 5. 计算累积和
    cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
    
    # 6. 添加小的epsilon防止log(0)
    epsilon = 1e-8
    log_risk = torch.log(cumsum_exp_risk + epsilon) + risk_max
    
    # 7. 只考虑发生事件的样本
    # 使用 >0.5 来避免浮点比较带来的边界问题（events 可能是 float 张量）
    event_mask = (events_sorted > 0.5)
    
    # 如果没有事件，返回一个与模型输出有关的小回退损失
    if not torch.any(event_mask):
        logger.warning("cox_ph_loss_stable: no events in batch, returning fallback loss")
        return _safe_fallback_loss(risk_scores)
    
    # 8. 计算Cox偏对数似然损失
    # 损失 = log(sum(exp(risk_j))) - risk_i
    uncensored_losses = log_risk[event_mask] - risk_scores_sorted[event_mask]
    
    # 9. 检查损失是否包含NaN或Inf
    if torch.isnan(uncensored_losses).any() or torch.isinf(uncensored_losses).any():
        logger.warning("cox_ph_loss_stable: NaN/Inf detected in uncensored_losses, using fallback")
        return _safe_fallback_loss(risk_scores)
    
    # 10. 应用样本权重
    if sample_weights is not None:
        sorted_weights = sample_weights[sorted_indices]
        event_weights = sorted_weights[event_mask]
        uncensored_losses = uncensored_losses * event_weights
    
    # 11. 计算平均损失
    cox_loss = uncensored_losses.mean()
    
    # 12. 最终检查
    if torch.isnan(cox_loss) or torch.isinf(cox_loss):
        logger.warning("cox_ph_loss_stable: NaN/Inf detected in cox_loss, using fallback")
        return _safe_fallback_loss(risk_scores)
    
    # 13. 添加正则化损失（如果有）
    total_loss = cox_loss
    if model is not None and hasattr(model, 'get_regularization_loss'):
        try:
            reg_loss = model.get_regularization_loss()
            if not torch.isnan(reg_loss) and not torch.isinf(reg_loss):
                total_loss = cox_loss + reg_loss
        except:
            pass
    
    # 14. 最终检查总损失
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning("cox_ph_loss_stable: NaN/Inf detected in total_loss, using fallback")
        return _safe_fallback_loss(risk_scores)
    
    return total_loss

def monotonic_gradient_penalty(model, input_tensor: torch.Tensor, feature_indices=None, directions=None, lambda_penalty: float = 0.05) -> torch.Tensor:
    """
    对模型输出相对于指定输入特征的偏导数施加单调性约束：
    - 若 direction=+1 => 约束 ∂f/∂x_k >= 0;
    - 若 direction=-1 => 约束 ∂f/∂x_k <= 0;
    - input_tensor: 训练时喂给模型的张量（若使用LSTM，建议为最后时间步的输入）。
    """
    if not feature_indices:
        return torch.tensor(0.0, device=input_tensor.device)
    # ensure graph for input grads
    x = input_tensor.detach().clone()
    x.requires_grad_(True)
    try:
        out = model(x)
    except Exception:
        # 如果模型forward需要不同形状，直接返回零罚项
        return torch.tensor(0.0, device=input_tensor.device)
    # 对标量和向量输出进行统一处理
    if out.ndim > 1:
        out_sum = out.sum()
    else:
        out_sum = out.sum()
    grads = torch.autograd.grad(out_sum, x, create_graph=True, retain_graph=True)[0]
    penalty = 0.0
    if directions is None or len(directions) == 0:
        directions = [1 for _ in feature_indices]
    for idx, dir_sign in zip(feature_indices, directions):
        # 对于越大风险越高的特征，dir_sign=+1；若相反则 -1
        gk = grads[..., idx]
        if dir_sign >= 0:
            penalty = penalty + torch.relu(-gk).mean()
        else:
            penalty = penalty + torch.relu(gk).mean()
    return lambda_penalty * penalty

def deephit_loss_improved(logits, time_bins, events, alpha=1.0, sigma=1.0, device='cuda'):
    """
    改进的DeepHit损失函数
    
    Args:
        logits: 模型输出 [batch_size, num_events, num_time_bins]
        time_bins: 时间区间索引 [batch_size]
        events: 事件指示器 [batch_size]
        alpha: 排序损失权重
        sigma: 排序损失的高斯核带宽
        device: 计算设备
        
    Returns:
        total_loss: 总损失
    """
    # 检查输入是否包含NaN或Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.warning("deephit_loss_improved: NaN/Inf detected in logits at entry, using fallback")
        return _safe_fallback_loss(logits)
    
    batch_size = logits.shape[0]
    num_events = logits.shape[1]
    num_time_bins = logits.shape[2]
    
    # 确保logits是2D的 [batch_size * num_events, num_time_bins]
    if logits.ndim == 3:
        logits = logits.view(-1, num_time_bins)
    
    # 数值稳定性：减去最大值
    logits_max = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - logits_max
    
    # 检查logits是否仍然包含NaN或Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.warning("deephit_loss_improved: NaN/Inf detected in logits after stabilization, using fallback")
        return _safe_fallback_loss(logits)
    
    # 计算概率分布
    probs = F.softmax(logits, dim=1)  # [batch_size * num_events, num_time_bins]
    
    # 检查概率是否包含NaN或Inf
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        logger.warning("deephit_loss_improved: NaN/Inf detected in softmax probs, using fallback")
        return _safe_fallback_loss(logits)
    
    # 计算生存函数
    survival = 1 - torch.cumsum(probs, dim=1)  # [batch_size * num_events, num_time_bins]
    survival = torch.clamp(survival, 1e-8, 1.0)
    
    # 检查生存函数是否包含NaN或Inf
    if torch.isnan(survival).any() or torch.isinf(survival).any():
        logger.warning("deephit_loss_improved: NaN/Inf detected in survival, using fallback")
        return _safe_fallback_loss(logits)
    
    # 重塑回原始形状
    probs = probs.view(batch_size, num_events, num_time_bins)
    survival = survival.view(batch_size, num_events, num_time_bins)
    
    # 1. 对数似然损失
    ll_loss = 0.0
    
    for i in range(batch_size):
        event_time = time_bins[i]
        event_type = events[i]
        
        if event_type > 0:  # 有事件发生
            # 事件样本：-log(概率密度)
            prob_at_event = probs[i, event_type-1, event_time]
            # 使用更稳定的log计算
            prob_at_event = torch.clamp(prob_at_event, 1e-8, 1.0)
            ll_loss += -torch.log(prob_at_event)
        else:  # 删失样本
            # 删失样本：-log(生存概率)
            surv_at_event = survival[i, 0, event_time]  # 假设第一个事件类型
            # 使用更稳定的log计算
            surv_at_event = torch.clamp(surv_at_event, 1e-8, 1.0)
            ll_loss += -torch.log(surv_at_event)
    
    ll_loss = ll_loss / batch_size
    
    # 检查对数似然损失是否包含NaN或Inf
    if torch.isnan(ll_loss) or torch.isinf(ll_loss):
        # 创建一个与模型参数相关的损失，而不是独立的张量
        return torch.sum(logits * 0.0)
    
    # 2. 排序损失
    ranking_loss = 0.0
    
    # 计算风险分数（期望生存时间）
    time_points = torch.arange(num_time_bins, device=device, dtype=torch.float32)
    risk_scores = torch.sum(probs * time_points.unsqueeze(0).unsqueeze(0), dim=2)  # [batch_size, num_events]
    
    # 检查风险分数是否包含NaN或Inf
    if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
        logger.warning("deephit_loss_improved: NaN/Inf detected in risk_scores, using fallback")
        return _safe_fallback_loss(logits)
    
    # 计算样本对
    valid_pairs = 0
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
                
            # 检查是否i在j之前发生事件
            if (events[i] > 0 and events[j] > 0 and 
                time_bins[i] < time_bins[j]):
                
                risk_diff = risk_scores[i, events[i]-1] - risk_scores[j, events[j]-1]
                # 使用更稳定的排序损失计算
                ranking_loss += torch.log(1 + torch.exp(risk_diff / sigma))
                valid_pairs += 1
    
    if valid_pairs > 0:
        ranking_loss = ranking_loss / valid_pairs
    else:
        logger.warning("deephit_loss_improved: no valid pairs for ranking loss, using fallback zero-like loss")
        ranking_loss = _safe_fallback_loss(logits)
    
    # 检查排序损失是否包含NaN或Inf
    if torch.isnan(ranking_loss) or torch.isinf(ranking_loss):
        logger.warning("deephit_loss_improved: NaN/Inf detected in ranking_loss, using fallback")
        return _safe_fallback_loss(logits)
    
    # 总损失
    total_loss = ll_loss + alpha * ranking_loss
    
    # 最终检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error("deephit_loss_improved: NaN/Inf detected in total_loss, using fallback")
        return _safe_fallback_loss(logits)
    
    return total_loss

def deephit_loss(logits, time_bins, events, y_true, alpha, sigma, sample_weights=None, device='cuda'):
    """
    计算DeepHit模型的损失函数。
    该损失由两部分组成：对数似然损失和排序损失。

    参数:
    - logits (torch.Tensor): 模型的原始输出，形状为 [batch_size, num_time_bins]。
    - time_bins (torch.Tensor): 每个样本对应的离散化时间区间的索引，形状为 [batch_size]。
    - events (torch.Tensor): 事件指示器 (1 for event, 0 for censored)，形状为 [batch_size]。
    - y_true (torch.Tensor): 原始标签，形状为 [batch_size, 2]，y_true[:, 0]是连续时间。
    - alpha (float): 排序损失的权重。
    - sigma (float): 用于计算排序损失中高斯核的带宽。
    - sample_weights (torch.Tensor, optional): 样本权重，形状为 [batch_size]。默认为None。
    - device (str): 计算设备。

    返回:
    - torch.Tensor: 计算出的总损失。
    """
    if logits.ndim == 3:
        logits = logits.squeeze(1) # [batch, 1, bins] -> [batch, bins]

    # --- 核心数值稳定性修复 ---
    # 在计算exp之前，减去每个logit向量中的最大值，防止溢出
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    
    # 1. 对数似然损失 (Log-likelihood loss)
    # 预测的累积生存函数
    s_J = F.softmax(logits, dim=1)
    S_J = 1 - torch.cumsum(s_J, dim=1)
    
    # 防止S_J为0或1导致log计算出inf
    S_J = torch.clamp(S_J, 1e-8, 1 - 1e-8)

    # --- 核心修复：使用gather来替代布尔索引，确保维度正确 ---
    time_bins_long = time_bins.to(torch.int64).view(-1, 1)

    # 提取每个样本在其事件/审查时间点的概率密度和生存概率
    # prob_at_t 的形状是 [batch_size]
    prob_at_t = torch.gather(s_J, 1, time_bins_long).squeeze(1)
    # surv_at_t 的形状是 [batch_size]
    surv_at_t = torch.gather(S_J, 1, time_bins_long).squeeze(1)
    
    # 分别计算两种情况的损失 (log-likelihood)
    # 对于事件样本，损失是 -log(概率密度)
    # 对于审查样本，损失是 -log(生存概率)
    log_likelihood_uncensored = -torch.log(torch.clamp(prob_at_t, min=1e-8))
    log_likelihood_censored = -torch.log(surv_at_t)
    
    # 根据 event 指示器 (1=事件, 0=审查) 选择正确的损失
    log_likelihood_loss = \
        events * log_likelihood_uncensored + \
        (1 - events) * log_likelihood_censored
    
    # 2. 排序损失 (Ranking loss)
    # 预测的风险分数 (这里使用logits的加权和作为一种简单的风险代理)
    time_bins_float = torch.arange(logits.shape[1], device=device, dtype=torch.float32)
    predicted_risk = torch.sum(s_J * time_bins_float, dim=1)

    # 计算样本对
    # 如果样本i在样本j之前发生事件 (ti < tj)，那么预测的风险应该更高 (r(xi) > r(xj))
    n = logits.shape[0]
    
    # 识别有效的样本对
    # 1. 样本i发生了事件 (event_i = 1)
    # 2. 样本i的事件时间早于样本j的事件时间 (time_i < time_j)
    is_event_i = events == 1
    # --- 核心修复：使用原始连续时间y_true来构建比较对 ---
    # 之前使用离散的time_bins导致了错误的比较
    continuous_time = y_true[:, 0]
    is_comparable = continuous_time.unsqueeze(1) < continuous_time.unsqueeze(0)
    
    mask = is_event_i.unsqueeze(1) & is_comparable
    
    # 如果没有可比较的对，则排序损失为0
    if not torch.any(mask):
        # 创建一个与模型参数相关的损失，而不是独立的张量
        ranking_loss = torch.sum(logits * 0.0)  # 创建一个与logits相关的零损失
    else:
        # 计算风险差异
        risk_i = predicted_risk.unsqueeze(1).expand(-1, n)
        risk_j = predicted_risk.unsqueeze(0).expand(n, -1)
        risk_diff = risk_i - risk_j
        
        # 使用高斯核函数加权差异
        # exp(- (risk_diff)^2 / sigma)
        exp_risk_diff = torch.exp(-torch.pow(risk_diff, 2) / sigma)
        
        # 计算排序损失，只对有效对进行
        ranking_loss = torch.sum(mask * exp_risk_diff) / (torch.sum(mask) + 1e-8)

    # 应用样本权重
    if sample_weights is not None:
        log_likelihood_loss = log_likelihood_loss * sample_weights

    total_loss = torch.mean(log_likelihood_loss) + alpha * ranking_loss
    
    # 如果损失为NaN或inf，则记录诊断并返回回退损失
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error("deephit_loss: NaN/Inf in total_loss, logging diagnostics and returning fallback")
        try:
            logger.error(f"Log-likelihood mean: {torch.mean(log_likelihood_loss).item()}")
            logger.error(f"Ranking loss: {ranking_loss.item()}")
            logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, has_nan={torch.isnan(logits).any()}")
            logger.error(f"s_J stats: min={s_J.min().item()}, max={s_J.max().item()}, sum_mean={s_J.sum(dim=1).mean().item()}, has_nan={torch.isnan(s_J).any()}")
        except Exception:
            logger.exception("deephit_loss: failed to log numeric diagnostics")
        return _safe_fallback_loss(logits)

    return total_loss