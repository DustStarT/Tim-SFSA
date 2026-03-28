"""
异常值处理模块
提供多种异常值检测和处理策略
"""
import numpy as np
import pandas as pd
from scipy import stats
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class OutlierHandler:
    """
    一个用于检测和处理数据中异常值的类。
    """
    def __init__(self, method='robust', strategy='clip', **kwargs):
        """
        初始化异常值处理器。

        Args:
            method (str): 检测异常值的方法 ('robust', 'zscore', 'iforest')。
            strategy (str): 处理异常值的策略 ('clip', 'remove', 'median')。
            **kwargs: 检测方法的参数 (例如 iqr_multiplier=1.5, zscore_threshold=3.0)。
        """
        self.method = method
        self.strategy = strategy
        self.params = kwargs
        self.lower_bound = None
        self.upper_bound = None
        self.model = None
        self.feature_bounds = {}
        self.feature_medians = {}
        
    def fit(self, X):
        """
        根据提供的数据拟合异常值检测模型。

        Args:
            X (pd.DataFrame): 用于拟合的输入数据。
        """
        self.feature_bounds = {}
        self.feature_medians = {}
        
        # 处理缺失值
        X_clean = X.copy()
        missing_ratio = X_clean.isnull().sum() / len(X_clean)
        logger.info(f"缺失值比例: {missing_ratio.to_dict()}")
        
        # 对于缺失值比例高的特征，使用中位数填充
        for col in X_clean.columns:
            if missing_ratio[col] > 0.1:  # 超过10%缺失
                logger.warning(f"特征 {col} 缺失值比例: {missing_ratio[col]:.2%}")
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            elif missing_ratio[col] > 0:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        if self.method == 'robust':
            Q1 = X_clean.quantile(0.25)
            Q3 = X_clean.quantile(0.75)
            IQR = Q3 - Q1
            iqr_multiplier = self.params.get('iqr_multiplier', 2.0)  # 放宽到2.0
            self.lower_bound = Q1 - iqr_multiplier * IQR
            self.upper_bound = Q3 + iqr_multiplier * IQR
            logger.info(f"使用稳健的IQR方法拟合异常值检测器，倍数: {iqr_multiplier}")
        
        elif self.method == 'zscore':
            mean_vals = X_clean.mean()
            std_vals = X_clean.std()
            zscore_threshold = self.params.get('zscore_threshold', 3.0)
            self.lower_bound = mean_vals - zscore_threshold * std_vals
            self.upper_bound = mean_vals + zscore_threshold * std_vals
            logger.info(f"使用Z-score方法拟合异常值检测器，阈值: {zscore_threshold}")
        
        elif self.method == 'iforest':
            self.model = IsolationForest(**self.params)
            self.model.fit(X_clean)
            logger.info("使用孤立森林拟合异常值检测器。")
        
        # 记录每个特征的中位数，用于填充策略
        for column in X_clean.columns:
            self.feature_medians[column] = X_clean[column].median()
            if self.method in ['robust', 'zscore']:
                self.feature_bounds[column] = {
                    'lower': self.lower_bound[column],
                    'upper': self.upper_bound[column],
                    'median': self.feature_medians[column]
                }
            
        return self

    def transform(self, X):
        """
        使用已拟合的模型处理数据中的异常值。

        Args:
            X (pd.DataFrame): 要处理的数据。

        Returns:
            pd.DataFrame: 处理异常值后的数据。
        """
        X_copy = X.copy()
        outlier_mask = self._detect(X_copy)

        if self.strategy == 'clip':
            if self.method == 'robust':
                clipped_X = X_copy.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)
                return clipped_X
            else:
                logger.warning("裁剪策略仅支持'robust'方法。")
                return X_copy
        
        elif self.strategy == 'remove':
            # `remove` 将删除任何特征中存在异常值的整行
            rows_with_outliers = outlier_mask.any(axis=1)
            logger.info(f"移除了 {rows_with_outliers.sum()} 行含有异常值的记录。")
            return X_copy[~rows_with_outliers]
            
        return X_copy

    def fit_transform(self, X):
        """
        拟合并转换数据。
        """
        self.fit(X)
        return self.transform(X)

    def _detect(self, X):
        """
        检测异常值并返回一个布尔掩码。
        """
        if self.method == 'robust':
            if self.lower_bound is None or self.upper_bound is None:
                raise RuntimeError("必须先调用 'fit' 方法。")
            # Use DataFrame methods to avoid FutureWarning
            return X.lt(self.lower_bound, axis=1) | X.gt(self.upper_bound, axis=1)
            
        elif self.method == 'iforest':
            if self.model is None:
                raise RuntimeError("必须先调用 'fit' 方法。")
            # -1 表示异常值
            return self.model.predict(X) == -1
            
        else:
            raise ValueError(f"未知的异常值检测方法: {self.method}")
        
    def get_feature_bounds(self):
        """获取特征边界值"""
        return self.feature_bounds

    def _handle_zscore(self, series: pd.Series) -> pd.Series:
        # ...
        
        clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
        
        num_clipped = (series != clipped_series).sum()
        # if num_clipped > 0:
        #     logger.info(f"将 {num_clipped} 个异常值点裁剪到Z-score边界。")
            
        return clipped_series
