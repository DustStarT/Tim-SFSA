"""
lifelines CoxPH 模型的封装器。
"""
import pandas as pd
from lifelines import CoxPHFitter
import joblib
import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from models.base_model import BaseModel

class CoxPHModel(BaseModel):
    """
    对 lifelines.CoxPHFitter 的封装，以遵循 BaseModel 接口。
    """
    def __init__(self, config):
        """
        初始化 CoxPH 模型封装器。

        Args:
            config (dict): 配置字典。
        """
        super().__init__(config)
        self.model = CoxPHFitter(**self.config.get('fitter_params', {}))
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.fc = nn.Linear(config['input_dim'], 1)
        self.tanh = nn.Tanh()

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        """
        使用DataFrame进行训练。CoxPH是一次性训练。
        """
        # 1. 记录特征名称
        self.feature_names_ = X_train.columns.tolist()
        
        # 2. 特征缩放
        X_scaled = self.scaler.fit_transform(X_train)
        
        # 3. 准备lifelines的DataFrame
        train_df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X_train.index)
        
        # 确保y_train的索引与X_train对齐
        y_train = y_train.set_index(X_train.index)
        # defensive: canonicalize duration to hours to avoid mixed-units issues
        try:
            import numpy as _np
            from evaluation.plotting import _ensure_durations_in_hours
            if 'duration' in y_train.columns:
                try:
                    y_train['duration'] = _ensure_durations_in_hours(_np.asarray(y_train['duration'].values.astype(float)), cfg=None, name='coxph_fit_y_train')
                except Exception:
                    # leave original if conversion fails
                    pass
        except Exception:
            # best-effort import/conversion; non-fatal
            pass

        train_df = pd.concat([train_df, y_train], axis=1)

        # 4. 训练模型
        try:
            self.model.fit(train_df, 'duration', event_col='event')
            c_index = self.model.concordance_index_
            return {'loss': -c_index, 'c_index': c_index}
        except Exception as e:
            logging.error(f"CoxPH模型训练失败: {e}", exc_info=True)
            return {'loss': float('inf'), 'c_index': 0.0}

    def train_epoch(self, X_train, y_train):
        """对于CoxPH模型，此方法是无操作的，因为训练在fit中一次性完成。"""
        # 保持接口一致性，但不执行任何操作
        return {'loss': 0.0, 'c_index': self.model.concordance_index_ if self.model else 0.0}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测风险得分。
        """
        if self.feature_names_ is None:
            raise RuntimeError("模型尚未训练，无法获取特征名称。")

        # 确保输入是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        # 特征缩放
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
        
        # 预测
        return self.model.predict_partial_hazard(X_scaled_df).values.flatten()

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def get_feature_importance(self):
        if self.model and hasattr(self.model, 'params_'):
            return pd.Series(self.model.params_.values, index=self.model.params_.index)
        return None

    def save_model(self, file_path: str):
        """保存模型和scaler"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names_
        }
        joblib.dump(state, file_path)
        logging.info(f"CoxPHModel state saved to {file_path}")

    def load_model(self, file_path: str):
        """加载模型和scaler"""
        state = joblib.load(file_path)
        self.model = state['model']
        self.scaler = state['scaler']
        self.feature_names_ = state['feature_names']
        logging.info(f"CoxPHModel state loaded from {file_path}")

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入张量, shape (batch_size, input_dim)。
        Returns:
            torch.Tensor: 风险分数, shape (batch_size, 1)。
        """
        # 通过tanh限制输出范围，增强稳定性
        return self.tanh(self.fc(x))
