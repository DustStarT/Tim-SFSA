"""
特征选择工具
提供不同的策略来从预处理数据中选择最重要的特征。
"""
import logging
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    一个多阶段的特征选择器，结合了基于分数的选择和基于VIF的共线性消除。
    """
    def __init__(self, n_features=10, vif_threshold=10.0):
        """
        初始化特征选择器。

        Args:
            n_features (int): 最终要选择的特征数量。
            vif_threshold (float): VIF（方差膨胀因子）的阈值。
        """
        self.n_features = n_features
        self.vif_threshold = vif_threshold
        self.log_transformed_cols = []

    def select_features(self, X, y):
        """
        执行多阶段特征选择。

        1. 对数变换处理数据倾斜。
        2. 为保证数值稳定性，对数据进行临时缩放。
        3. 使用基于单变量Cox回归的初步筛选。
        4. 在初步筛选出的特征上，使用VIF消除多重共线性。
        5. 从剩余的特征中，根据Cox回归分数选出最终的n_features个。
        """
        if X.empty or y.empty:
            logger.warning("输入数据为空，无法执行特征选择。")
            return []

        # 步骤 1: 数据已经被预处理和缩放，这里直接使用
        X_scaled = X.copy()
        
        # 填充可能在之前步骤中产生的任何NaN/inf值
        X_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X_scaled.isnull().values.any():
            logger.warning("在特征选择的输入中发现NaN值，将用0填充。")
            X_scaled.fillna(0, inplace=True)
        
        # 步骤 2: 基于单变量Cox回归的初步筛选
        initial_k = min(self.n_features * 2, X_scaled.shape[1])
        if initial_k <= 0:
            logger.warning("没有可供选择的特征。")
            return []

        logger.info(f"进行基于单变量CoxPH的初步筛选，目标是 {initial_k} 个特征...")
        
        feature_performance = {}
        for feature in X_scaled.columns:
            # lifelines期望一个包含duration和event的DataFrame
            data_for_fit = pd.DataFrame({
                'feature': X_scaled[feature],
                'duration': y['duration'].values,
                'event': y['event'].values
            })
            # 移除duration或feature无效的行
            data_for_fit.dropna(subset=['duration', 'event', 'feature'], inplace=True)
            
            if data_for_fit.empty or data_for_fit['event'].sum() < 2: # 需要至少2个事件才能计算C-index
                feature_performance[feature] = {'p_value': 1.0, 'c_index': 0.5}
                continue

            cph = CoxPHFitter()
            try:
                cph.fit(data_for_fit, duration_col='duration', event_col='event', formula="feature")
                p_value = cph.summary.p.feature if 'feature' in cph.summary.p else 1.0
                c_index = cph.concordance_index_ if hasattr(cph, 'concordance_index_') else 0.5
                
                feature_performance[feature] = {'p_value': p_value, 'c_index': c_index}
                
            except Exception as e:
                logger.warning(f"为特征 '{feature}' 拟合CoxPH失败: {e}. 性能记为基线。")
                feature_performance[feature] = {'p_value': 1.0, 'c_index': 0.5}

        # 打印所有特征的性能
        logger.info("--- 单变量CoxPH特征性能评估 ---")
        perf_df = pd.DataFrame.from_dict(feature_performance, orient='index').sort_values(by='p_value')
        logger.info(f"\\n{perf_df.to_string()}")
        logger.info("------------------------------------")

        # 使用-log10(p_value)作为分数进行排序
        scores = {k: -np.log10(v['p_value']) if v['p_value'] > 0 else 300 for k, v in feature_performance.items()}
        scores = pd.Series(scores)
        
        prelim_features = scores.sort_values(ascending=False).index[:initial_k].tolist()
        logger.info(f"初步筛选出的特征: {prelim_features}")

        # 步骤 3: 在初步筛选的特征上处理共线性
        logger.info("在初步筛选的特征上处理多重共线性 (VIF)...")
        final_feature_candidates = self._handle_collinearity(X_scaled[prelim_features])
        logger.info(f"VIF处理后剩余的候选特征: {final_feature_candidates}")
        
        if not final_feature_candidates:
             logger.warning("VIF处理后没有剩余特征。")
             return []

        # 步骤 4: 从剩余的候选特征中选出最终的Top N
        # 我们使用原始的Cox模型分数进行排序
        final_scores = scores[final_feature_candidates]
        final_features = final_scores.sort_values(ascending=False).index[:self.n_features].tolist()
        
        logger.info(f"最终选择的 {len(final_features)} 个特征: {final_features}")
        
        return final_features

    def _handle_collinearity(self, X):
        features = X.columns.tolist()
        while len(features) > 1:
            X_subset = X[features].astype(np.float64).replace([np.inf, -np.inf], 0).fillna(0)
            
            try:
                vif_data = pd.DataFrame({
                    "feature": features,
                    "VIF": [variance_inflation_factor(X_subset.values, i) for i in range(len(features))]
                })
            except Exception as e:
                logger.error(f"计算VIF时出错: {e}. 将返回当前特征集。")
                return features

            max_vif = vif_data['VIF'].max()
            if max_vif > self.vif_threshold:
                feature_to_remove = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
                features.remove(feature_to_remove)
                logger.warning(f"因VIF过高({max_vif:.2f})而移除特征: '{feature_to_remove}'")
            else:
                break
        return features

    def _apply_log_transform(self, X):
        """
        仅对全非负的列应用log1p变换。
        """
        X_transformed = X.copy()
        log_cols = []
        no_log_cols = []

        for col in X.columns:
            if (X[col] >= 0).all():
                X_transformed[col] = np.log1p(X[col])
                log_cols.append(col)
            else:
                no_log_cols.append(col)
        
        if log_cols:
            logger.info(f"在特征选择过程中，对 {len(log_cols)} 个非负特征应用了Log1p变换。")
        if no_log_cols:
            logger.info(f"在特征选择过程中，{len(no_log_cols)} 个特征因包含负值未应用Log1p变换: {no_log_cols}")

        return X_transformed
