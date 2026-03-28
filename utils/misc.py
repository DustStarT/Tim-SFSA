import logging
import os
import json
from datetime import datetime
import sys
import torch
import numpy as np
from .config_validator import validate_config

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(log_dir):
    """Initializes logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'experiment.log')
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"日志文件位于: {log_file}")
    
def save_config(config, output_dir):
    """将配置保存到指定目录的config.json文件中。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config_path = os.path.join(output_dir, "config.json")
    
    # EasyDict is compatible with dict, so we can use json.dump
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def calculate_sample_weights(y_train):
    """
    根据类别不平衡计算样本权重。

    Args:
        y_train (pd.DataFrame): 训练标签，包含 'event' 列。

    Returns:
        torch.Tensor: 每个样本的权重张量。
    """
    events = y_train['event'].to_numpy()
    total_samples = len(events)
    n_events = np.sum(events == 1)
    n_censored = np.sum(events == 0)

    # 如果任一类别样本数为0，则返回均匀权重以避免除以0
    if n_events == 0 or n_censored == 0:
        return torch.ones(total_samples, dtype=torch.float32)

    # 计算权重：直接使用频率倒数，并以审查样本为基准(权重=1)
    # 这会更激进地放大事件样本的权重。
    weight_event = float(n_censored) / float(n_events)
    weight_censored = 1.0
    
    weights = np.where(events == 1, weight_event, weight_censored)
    
    logging.info(f"计算样本权重: 事件权重={weight_event:.2f}, 审查权重={weight_censored:.2f}")
    
    return torch.from_numpy(weights).float()

# 新增的 EarlyStopper 类
class EarlyStopper:
    """在验证分数不再提升时提前停止训练。"""
    def __init__(self, patience=5, delta=0, logger=None, model=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger or logging.getLogger(__name__)
        self.model = model
        self.model_state_dict = None

    def step(self, score):
        """
        根据验证分数决定是否停止。
        如果分数没有提升，则增加计数器。如果分数提升，则重置计数器并保存模型。
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
            self.logger.info(f"验证分数提升 ({self.best_score:.4f})。保存模型...")
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint()
            self.logger.info(f"验证分数提升 ({self.best_score:.4f})。保存模型...")
        else:
            self.counter += 1
            self.logger.info(f"早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self):
        """Saves model when validation score improves."""
        if self.model:
            self.logger.info(f"验证分数提升 ({self.best_score:.4f})。保存模型...")
            self.model_state_dict = self.model.state_dict()

    def load_best_weights(self):
        """加载在验证集上表现最好的模型权重。"""
        if self.model and self.model_state_dict:
            self.logger.info("加载早停找到的最佳模型权重。")
            self.model.load_state_dict(self.model_state_dict)
        else:
            self.logger.warning("没有可供加载的最佳模型状态。")

def init_weights_stable(m):
    """
    使用更稳定的初始化方法来初始化模型的所有层权重。
    专门为CoxKAN等复杂模型设计，防止数值不稳定。
    """
    if isinstance(m, torch.nn.Linear):
        # 线性层使用更保守的初始化
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)  # 使用较小的gain值
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LSTM):
        # LSTM层使用更稳定的初始化
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
    elif isinstance(m, torch.nn.LSTMCell):
        # LSTM单元使用更稳定的初始化
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
    elif isinstance(m, torch.nn.Conv1d):
        # 1D卷积层使用更保守的初始化
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        # 2D卷积层使用更保守的初始化
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        # 批归一化层
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        # 批归一化层
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Dropout):
        # Dropout层不需要初始化
        pass
    else:
        # 对于其他类型的层，尝试使用默认初始化
        try:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        except:
            pass  # 如果初始化失败，跳过

def init_weights(m):
    """
    使用合适的初始化方法来初始化模型的所有层权重。
    这有助于防止梯度消失或爆炸，确保训练稳定性。
    """
    if isinstance(m, torch.nn.Linear):
        # 线性层使用Kaiming初始化
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LSTM):
        # LSTM层使用Xavier初始化
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
    elif isinstance(m, torch.nn.LSTMCell):
        # LSTM单元使用Xavier初始化
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
    elif isinstance(m, torch.nn.Conv1d):
        # 1D卷积层使用Kaiming初始化
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        # 2D卷积层使用Kaiming初始化
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        # 批归一化层
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        # 批归一化层
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Dropout):
        # Dropout层不需要初始化
        pass 