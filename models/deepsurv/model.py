import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DeepSurvModel(nn.Module):
    """
    DeepSurv模型实现
    基于深度神经网络的Cox比例风险模型
    参考论文: "DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network"
    """
    
    def __init__(self, config, n_features, device):
        super(DeepSurvModel, self).__init__()
        
        self.config = config
        self.n_features = n_features
        self.device = device
        
        # 获取DeepSurv配置
        deepsurv_config = getattr(config.model, 'deepsurv', {})
        
        # 网络架构配置
        self.hidden_layers = deepsurv_config.get('hidden_layers', [64, 32, 16])
        self.dropout_rate = deepsurv_config.get('dropout_rate', 0.3)
        self.batch_norm = deepsurv_config.get('batch_norm', True)
        self.activation = deepsurv_config.get('activation', 'relu')
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层
        input_dim = n_features
        for hidden_dim in self.hidden_layers:
            # 线性层
            layer = nn.Linear(input_dim, hidden_dim)
            
            # 批归一化（如果启用）
            if self.batch_norm:
                layer = nn.Sequential(
                    layer,
                    nn.BatchNorm1d(hidden_dim),
                    self._get_activation(),
                    nn.Dropout(self.dropout_rate)
                )
            else:
                layer = nn.Sequential(
                    layer,
                    self._get_activation(),
                    nn.Dropout(self.dropout_rate)
                )
            
            self.layers.append(layer)
            input_dim = hidden_dim
        
        # 输出层 - 单个神经元，输出对数风险分数
        self.output_layer = nn.Linear(input_dim, 1)
        
        # 初始化权重
        self._init_weights()

        # 尝试将模型移动到指定device（如果在初始化阶段可用）
        try:
            if device is not None:
                self.to(device)
        except Exception:
            # 仅在device不可用或环境不支持cuda时忽略
            pass

        logging.info(f"DeepSurv模型创建完成: 输入维度={n_features}, 隐藏层={self.hidden_layers}")
    
    def _get_activation(self):
        """获取激活函数"""
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'tanh':
            return nn.Tanh()
        elif self.activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.activation.lower() == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, n_features) 或 (batch_size, seq_len, n_features)
            
        Returns:
            对数风险分数 (batch_size, 1)
        """
        # 处理时序输入 - 如果输入是3D的，取最后一个时间步
        if x.dim() == 3:
            x = x[:, -1, :]  # (batch_size, n_features)

        # 数值稳定性：禁用 AMP，统一在 FP32 下执行，并在各层之间清洗 NaN/Inf 与裁剪
        try:
            from torch.cuda.amp import autocast
            autocast_ctx = autocast(enabled=False)
        except Exception:
            class _Dummy:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
            autocast_ctx = _Dummy()

        # 读取裁剪阈值
        sentinel = 1e6
        try:
            sentinel = float(getattr(self.config.data, 'sentinel_threshold', 1e6))
        except Exception:
            pass

        with autocast_ctx:
            # 输入清洗（仅替换非有限值，不再裁剪，以保留分布幅度）
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # 前向传播（移除层间的数值清洗与裁剪，以保留可学习分布）
            for layer in self.layers:
                x = layer(x)

            # 输出层
            log_risk = self.output_layer(x)
            if not torch.isfinite(log_risk).all():
                log_risk = torch.nan_to_num(log_risk, nan=0.0, posinf=0.0, neginf=0.0)

        # 统一输出为一维向量 (batch_size,) 以便与训练循环和评估函数兼容
        if log_risk.dim() == 2 and log_risk.size(1) == 1:
            return log_risk.squeeze(-1)
        return log_risk
    
    def predict_risk(self, x):
        """
        预测风险分数
        
        Args:
            x: 输入张量
            
        Returns:
            风险分数
        """
        self.eval()
        with torch.no_grad():
            log_risk = self.forward(x)
            # DeepSurv语义：输出log风险；若需要非负风险分数，取exp
            return torch.exp(log_risk).squeeze(-1)
    
    def get_risk_scores(self, x):
        """
        获取风险分数（用于评估）
        
        Args:
            x: 输入张量
            
        Returns:
            风险分数
        """
        return self.predict_risk(x)
