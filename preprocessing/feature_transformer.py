import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

class FeatureTransformer:
    """
    一个用于对特征进行log1p变换和标准化的类。
    它可以学习哪些列适合进行变换（所有值非负），并应用该变换。
    """
    def __init__(self, enabled=True, normalize=True):
        self.log_transform_cols = []
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        if not self.enabled:
            self.logger.info("FeatureTransformer 已被禁用。")
        if not self.normalize:
            self.logger.info("标准化 已被禁用。")

    def fit(self, X: pd.DataFrame):
        """
        根据输入数据帧确定哪些列可以进行对数变换，并拟合标准化器。

        Args:
            X (pd.DataFrame): 用于拟合的输入数据。
        """
        if not self.enabled:
            self.log_transform_cols = []
            return self
            
        # 只选择所有值都大于等于0的列进行对数变换
        self.log_transform_cols = [col for col in X.columns if (X[col] >= 0).all()]
        self.logger.info(f"FeatureTransformer已拟合。将对以下特征应用log1p变换: {self.log_transform_cols}")
        
        # 拟合标准化器
        if self.normalize and self.scaler is not None:
            self.scaler.fit(X)
            self.logger.info("标准化器已拟合。")
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对数据帧应用已学习的对数变换和标准化。
        仅对在fit阶段确定的、且所有值都为非负的列进行变换。
        """
        if not self.enabled:
            return X

        if not self.log_transform_cols and not self.normalize:
            self.logger.warning("FeatureTransformer尚未拟合或没有可变换的列，将返回原始数据。")
            return X
        
        X_copy = X.copy()
        
        # 应用log变换
        if self.log_transform_cols:
            # 仅对拟合时确定的、且当前DataFrame中存在的列进行变换
            transform_cols_exist = [col for col in self.log_transform_cols if col in X_copy.columns]
            
            if transform_cols_exist:
                # 再次检查，以防transform的数据和fit的数据分布不同
                # 我们只变换那些在当前数据(X_copy)中也全是非负的列
                cols_to_transform_safely = [
                    col for col in transform_cols_exist if (X_copy[col] >= 0).all()
                ]
                
                # 记录那些因为包含负值而未被变换的列
                skipped_cols = set(transform_cols_exist) - set(cols_to_transform_safely)
                if skipped_cols:
                    self.logger.info(f"在transform阶段，以下特征因包含负值而未进行log1p变换: {list(skipped_cols)}")

                if cols_to_transform_safely:
                    X_copy[cols_to_transform_safely] = np.log1p(X_copy[cols_to_transform_safely])
        
        # 应用标准化
        if self.normalize and self.scaler is not None:
            X_copy = pd.DataFrame(
                self.scaler.transform(X_copy),
                columns=X_copy.columns,
                index=X_copy.index
            )
            self.logger.info("已应用标准化。")
            
        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并应用变换。

        Args:
            X (pd.DataFrame): 输入数据。

        Returns:
            pd.DataFrame: 经过变换的数据。
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 反向标准化
        if self.normalize and self.scaler is not None:
            df = pd.DataFrame(
                self.scaler.inverse_transform(df),
                columns=df.columns,
                index=df.index
            )
        
        # 反向log变换
        if self.log_transform_cols:
            df_copy = df.copy()
            transform_cols_exist = [col for col in self.log_transform_cols if col in df_copy.columns]
            if transform_cols_exist:
                df_copy[transform_cols_exist] = np.expm1(df_copy[transform_cols_exist])
            return df_copy

        return df 