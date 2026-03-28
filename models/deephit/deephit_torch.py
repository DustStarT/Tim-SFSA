import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_mlp(input_dim, layers_dims, dropout):
    """一个辅助函数，用于快速创建多层感知机（MLP）。"""
    layers = []
    for i, dim in enumerate(layers_dims):
        layers.append(nn.Linear(input_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        input_dim = dim
    return nn.Sequential(*layers)

class DeepHitTorch(nn.Module):
    """
    DeepHit模型的PyTorch实现，支持竞争风险。
    """
    def __init__(self, config):
        """
        初始化DeepHit模型。

        Args:
            config (dict): 包含模型配置的字典。
                - n_features (int): 输入特征维度。
                - shared_layers (list): 共享MLP的层维度。
                - risk_specific_layers (list): 特定风险MLP的层维度。
                - num_time_bins (int): 离散化的时间区间数量。
                - num_events (int): 竞争风险事件的数量。
                - dropout (float): dropout比率。
        """
        super().__init__()
        self.num_time_bins = config['num_time_bins']
        self.num_events = config['num_events']
        
        # 创建共享网络
        self.shared_network = create_mlp(
            input_dim=config['n_features'],
            layers_dims=config['shared_layers'],
            dropout=config['dropout']
        )
        
        # 获取共享网络的输出维度
        shared_output_dim = config['shared_layers'][-1]
        
        # 为每个风险事件创建特定的输出网络
        self.risk_specific_networks = nn.ModuleList()
        for _ in range(self.num_events):
            self.risk_specific_networks.append(
                create_mlp(
            input_dim=shared_output_dim,
            layers_dims=config['risk_specific_layers'],
            dropout=config['dropout']
                )
        )
        
        # 为每个风险创建最终的输出层
        risk_output_dim = config['risk_specific_layers'][-1]
        self.output_layers = nn.ModuleList(
            [nn.Linear(risk_output_dim, self.num_time_bins) for _ in range(self.num_events)]
        )

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, n_features)。

        Returns:
            tuple: (y_pred, logits)
                - y_pred (torch.Tensor): 每个时间区间和事件的联合概率，
                                         形状为 (batch_size, num_events, num_time_bins)。
                - logits (torch.Tensor): 每个时间区间和事件的原始分数（softmax之前），
                                         形状为 (batch_size, num_events, num_time_bins)。
        """
        # 共享网络
        shared_out = self.shared_network(x)
        
        # 特定风险网络
        all_logits = []
        for i in range(self.num_events):
            risk_specific_out = self.risk_specific_networks[i](shared_out)
            logits_i = self.output_layers[i](risk_specific_out)
            all_logits.append(logits_i)
            
        # 将所有风险的logits堆叠起来
        # logits -> (batch_size, num_events, num_time_bins)
        logits = torch.stack(all_logits, dim=1)

        # --- 最终修复 ---
        # 根据DeepHit的原始思想和参考实现，模型本身不应包含softmax。
        # Softmax应在似然损失函数中应用，而排序损失则直接使用原始logits。
        # 因此，模型只返回原始的logits。
        
        # y_pred (概率) 将在损失函数中计算。
        # 我们返回一个元组，以保持与下游代码的兼容性，
        # 但两个元素都是logits。下游代码应该知道如何处理。
        # 或者，更清晰的是，只返回logits，并修改下游代码。
        # 我们选择后者。
        
        return logits

# --- 已删除旧的、不正确的deephit_loss_function ---
# 正确的损失函数现在位于 utils/loss_functions.py 