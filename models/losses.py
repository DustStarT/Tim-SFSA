import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """标准二分类 Focal Loss 实现。"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(f"不支持的 reduction: {reduction}")
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

