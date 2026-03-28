import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DeepHitImproved(nn.Module):
    """
    改进的DeepHit模型实现，基于开源实现和论文。
    参考: https://github.com/chl8856/DeepHit
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_features = config['n_features']
        self.num_time_bins = config['num_time_bins']
        self.num_events = config['num_events']
        self.dropout = config.get('dropout', 0.3)
        
        # 共享网络
        shared_layers = config.get('shared_layers', [64, 32])
        self.shared_network = self._build_mlp(
            input_dim=self.num_features,
            hidden_dims=shared_layers,
            dropout=self.dropout
        )
        
        # 获取共享网络的输出维度
        shared_output_dim = shared_layers[-1]
        
        # 为每个事件类型创建特定的输出网络
        risk_specific_layers = config.get('risk_specific_layers', [32])
        self.risk_specific_networks = nn.ModuleList()
        for _ in range(self.num_events):
            self.risk_specific_networks.append(
                self._build_mlp(
                    input_dim=shared_output_dim,
                    hidden_dims=risk_specific_layers,
                    dropout=self.dropout
                )
            )
        
        # 最终输出层
        risk_output_dim = risk_specific_layers[-1]
        self.output_layers = nn.ModuleList([
            nn.Linear(risk_output_dim, self.num_time_bins) 
            for _ in range(self.num_events)
        ])
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _build_mlp(self, input_dim, hidden_dims, dropout=0.3):
        """构建多层感知机"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, num_features]
            
        Returns:
            logits: 原始logits [batch_size, num_events, num_time_bins]
        """
        # 共享网络
        shared_out = self.shared_network(x)
        
        # 事件特定网络
        all_logits = []
        for i in range(self.num_events):
            risk_specific_out = self.risk_specific_networks[i](shared_out)
            logits_i = self.output_layers[i](risk_specific_out)
            all_logits.append(logits_i)
        
        # 堆叠所有事件的logits
        logits = torch.stack(all_logits, dim=1)  # [batch_size, num_events, num_time_bins]
        
        return logits

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
    # 统一数值清理：将 NaN/Inf 替换为 0，避免后续提前返回0损失
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 兼容 2D ([B, T]) 与 3D ([B, E, T]) logits
    if logits.ndim == 3:
        batch_size, num_events, num_time_bins = logits.shape
        logits = logits.reshape(-1, num_time_bins)
    elif logits.ndim == 2:
        batch_size, num_time_bins = logits.shape
        num_events = 1
        # logits 已为 [B, T]
    else:
        # 意外维度，返回与参数相关的零损失，避免梯度中断
        return torch.sum(logits * 0.0)
    
    # 数值稳定性：减去最大值
    logits_max = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - logits_max
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 计算概率分布
    probs = F.softmax(logits, dim=1)  # [batch_size * num_events, num_time_bins] 或 [batch_size, num_time_bins]
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 计算生存函数
    survival = 1 - torch.cumsum(probs, dim=1)  # [batch_size * num_events, num_time_bins]
    survival = torch.clamp(survival, 1e-8, 1.0)
    survival = torch.nan_to_num(survival, nan=1.0, posinf=1.0, neginf=1e-8)
    
    # 重塑回原始形状
    probs = probs.view(batch_size, num_events, num_time_bins)
    survival = survival.view(batch_size, num_events, num_time_bins)
    
    # 1. 对数似然损失
    ll_loss = 0.0
    
    for i in range(batch_size):
        # 将索引安全地转换为 Python int，并做边界裁剪
        try:
            etype_val = events[i]
            etype = int(etype_val.item() if hasattr(etype_val, 'item') else int(etype_val))
        except Exception:
            etype = int(etype_val) if not isinstance(etype_val, (list, tuple)) else 0
        try:
            etime_val = time_bins[i]
            etime = int(etime_val.item() if hasattr(etime_val, 'item') else int(etime_val))
        except Exception:
            etime = int(etime_val) if not isinstance(etime_val, (list, tuple)) else 0

        # clamp 索引到合法范围
        if num_events <= 0:
            num_events = 1
        etime = max(0, min(int(etime), int(num_time_bins) - 1))
        # 事件类型按 (1..num_events)，转为 [0..num_events-1]
        etype_idx = max(0, min(int(etype) - 1, int(num_events) - 1))

        if etype > 0:  # 有事件发生
            # 事件样本：-log(概率密度)
            prob_at_event = probs[i, etype_idx, etime]
            prob_at_event = torch.clamp(prob_at_event, 1e-8, 1.0)
            ll_loss += -torch.log(prob_at_event)
        else:  # 删失样本
            # 删失样本：-log(生存概率)（使用事件通道0）
            surv_at_event = survival[i, 0, etime]
            surv_at_event = torch.clamp(surv_at_event, 1e-8, 1.0)
            ll_loss += -torch.log(surv_at_event)
    
    ll_loss = ll_loss / batch_size
    
    # 数值健壮：避免 NaN/Inf
    ll_loss = torch.nan_to_num(ll_loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. 排序损失
    ranking_loss = 0.0
    
    # 计算风险分数（期望生存时间）
    time_points = torch.arange(num_time_bins, device=device, dtype=torch.float32)
    risk_scores = torch.sum(probs * time_points.unsqueeze(0).unsqueeze(0), dim=2)  # [batch_size, num_events]
    
    # 检查风险分数是否包含NaN或Inf
    if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
        # 创建一个与模型参数相关的损失，而不是独立的张量
        return torch.sum(logits * 0.0)
    
    # 计算样本对 - 更高效的实现
    valid_pairs = 0
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
                
            # 检查是否i在j之前发生事件，且都是事件样本
            try:
                ei = int(events[i].item() if hasattr(events[i], 'item') else int(events[i]))
                ej = int(events[j].item() if hasattr(events[j], 'item') else int(events[j]))
                ti = int(time_bins[i].item() if hasattr(time_bins[i], 'item') else int(time_bins[i]))
                tj = int(time_bins[j].item() if hasattr(time_bins[j], 'item') else int(time_bins[j]))
            except Exception:
                continue

            if (ei > 0 and ej > 0 and ti < tj):
                
                # 获取对应事件类型的风险分数
                event_i = max(0, min(ei - 1, int(num_events) - 1))
                event_j = max(0, min(ej - 1, int(num_events) - 1))
                
                risk_diff = risk_scores[i, event_i] - risk_scores[j, event_j]
                
                # 使用更稳定的排序损失：softplus
                ranking_loss += F.softplus(risk_diff / sigma)
                valid_pairs += 1
    
    if valid_pairs > 0:
        ranking_loss = ranking_loss / valid_pairs
    else:
        # 无有效样本对，排名损失置零
        ranking_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    
    # 数值健壮
    ranking_loss = torch.nan_to_num(ranking_loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 总损失
    total_loss = ll_loss + alpha * ranking_loss
    
    # 最终数值健壮
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 添加调试信息
    if torch.rand(1).item() < 0.01:  # 1%的概率输出调试信息
        logger.debug(f"DeepHit损失 - LL: {ll_loss:.4f}, Ranking: {ranking_loss:.4f}, Total: {total_loss:.4f}")
        logger.debug(f"有效样本对数量: {valid_pairs}")
    
    return total_loss

def predict_survival_improved(model, x, time_points=None):
    """
    预测生存函数
    
    Args:
        model: 训练好的DeepHit模型
        x: 输入特征 [batch_size, num_features]
        time_points: 时间点列表，如果为None则使用默认时间点
        
    Returns:
        survival_probs: 生存概率 [batch_size, num_time_points]
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)  # [batch_size, num_events, num_time_bins]
        
        # 计算概率分布
        probs = F.softmax(logits, dim=2)  # [batch_size, num_events, num_time_bins]
        
        # 计算生存函数
        survival = 1 - torch.cumsum(probs, dim=2)  # [batch_size, num_events, num_time_bins]
        
        # 如果指定了时间点，进行插值
        if time_points is not None:
            # 这里需要实现时间点插值
            # 简化版本：返回原始时间点的生存概率
            pass
        
        return survival[:, 0, :]  # 返回第一个事件类型的生存概率

def predict_risk_improved(model, x):
    """
    预测风险分数
    
    Args:
        model: 训练好的DeepHit模型
        x: 输入特征 [batch_size, num_features]
        
    Returns:
        risk_scores: 风险分数 [batch_size]
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)  # [batch_size, num_events, num_time_bins]
        
        # 计算概率分布
        probs = F.softmax(logits, dim=2)  # [batch_size, num_events, num_time_bins]
        
        # 计算期望生存时间作为风险分数（时间越短风险越高）
        num_time_bins = logits.shape[2]
        time_points = torch.arange(num_time_bins, device=logits.device, dtype=torch.float32)
        
        # 期望生存时间
        expected_time = torch.sum(probs * time_points.unsqueeze(0).unsqueeze(0), dim=2)
        
        # 转换为风险分数（时间越短风险越高）
        risk_scores = num_time_bins - expected_time
        
        return risk_scores[:, 0]  # 返回第一个事件类型的风险分数 