"""
CoxKAN model wrapper.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
import os
from lifelines.fitters.breslow_fleming_harrington_fitter import BreslowFlemingHarringtonFitter
from lifelines import CoxPHFitter

from models.base_model import BaseModel
# 重命名导入的CoxKAN，避免与本类名冲突
from .core import CoxKAN as CoreCoxKAN
from models.kan import KAN

class CoxKANModel(BaseModel):
    """
    一个基于Kolmogorov-Arnold Network (KAN)的Cox比例风险模型。
    该模型接收一个扁平化的2D特征张量 (batch, features)，并输出一个风险分数。
    """
    def __init__(self, config, input_dim, device):
        super().__init__()
        self.config = config
        self.device = device
        # CoxKAN使用model.kan配置
        kan_cfg = config.model.kan
        kan_layers = [input_dim] + kan_cfg['layers'] + [1]
        
        # 存储dropout参数
        self.dropout_rate = kan_cfg.get('dropout', 0.0)
        
        # 直接使用CoreCoxKAN，不再通过KAN类间接实现
        self.kan = CoreCoxKAN(
            width=kan_layers,
            grid=kan_cfg.get('grid_size', 3),
            k=kan_cfg.get('spline_order', 2),
            noise_scale=kan_cfg.get('scale_noise', 0.01),
            noise_scale_base=kan_cfg.get('scale_base', 0.01),
            base_fun=torch.nn.SiLU(),
            device=device
        )

    def forward(self, x):
        """
        前向传播。
        """
        # 应用dropout（如果配置中设置了）
        if self.dropout_rate > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.kan(x)
        
    def predict(self, X):
        """
        预测风险得分。
        输入: X (batch_size, features)
        输出: risk_scores (batch_size, )
        """
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            risk_scores = self(X_tensor)
        
        return risk_scores.cpu().numpy().flatten()
