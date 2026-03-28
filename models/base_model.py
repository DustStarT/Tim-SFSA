"""
模型部分的抽象基类
定义所有生存分析模型应遵循的统一接口。
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    所有模型的抽象基类。
    提供了一个统一的接口，但允许子类有不同的实现。
    """
    def __init__(self, device=None):
        super(BaseModel, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None # 子类应该在这里初始化它们的具体模型
        self.model_name = self.__class__.__name__

    def forward(self, x):
        """
        定义前向传播。如果模型可以直接调用，子类应覆盖此方法。
        """
        if self.model:
            return self.model(x)
        raise NotImplementedError("Forward pass is not implemented for this model.")

    def train_epoch(self, data_loader):
        """训练模型一个 epoch。"""
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练整个模型。"""
        pass

    def predict(self, X):
        """进行预测。"""
        pass

    def save_model(self, path):
        """保存模型状态。"""
        # 默认实现可以保存 PyTorch 模型
        if isinstance(self, nn.Module):
            torch.save(self.state_dict(), path)
        else:
            print(f"Warning: save_model not implemented for {self.model_name}")

    def load_model(self, path):
        """加载模型状态。"""
        # 默认实现可以加载 PyTorch 模型
        if isinstance(self, nn.Module):
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Warning: load_model not implemented for {self.model_name}")

    def get_name(self):
        """返回模型的名称。"""
        return self.model_name

    def predict_risk(self, data):
        """
        预测风险得分。默认情况下，它调用 predict()。
        如果模型的 predict() 返回的不是风险，则应重写此方法。
        """
        return self.predict(data)

    def get_feature_importance(self):
        """
        获取特征重要性。并非所有模型都支持此功能。
        """
        return None
