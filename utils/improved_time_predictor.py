"""
改进的生存时间预测校准方法
解决Transformer模型预测偏差过大的问题
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pickle
import os

logger = logging.getLogger(__name__)

class CalibratedTimePredictor:
    """
    校准的生存时间预测器
    基于历史预测误差分析进行校准
    """
    
    def __init__(self, calibration_factor: float = 0.41):
        """
        初始化校准预测器
        
        Args:
            calibration_factor: 校准因子，基于历史分析得出
        """
        self.calibration_factor = calibration_factor
        self.is_fitted = False
        self.linear_model = None
        
    def fit(self, predicted_times: np.ndarray, true_times: np.ndarray, 
            events: np.ndarray) -> Dict[str, float]:
        """
        拟合校准模型
        
        Args:
            predicted_times: 原始预测时间
            true_times: 真实生存时间
            events: 事件指示器
            
        Returns:
            校准统计信息
        """
        # 只使用事件样本进行校准
        event_mask = events.astype(bool)
        pred_event = predicted_times[event_mask]
        true_event = true_times[event_mask]
        
        if len(pred_event) < 10:
            logger.warning("事件样本数量不足，使用默认校准因子")
            return {"calibration_factor": self.calibration_factor}
        
        # 计算偏差
        bias = np.mean(pred_event - true_event)
        mae = mean_absolute_error(true_event, pred_event)
        
        logger.info(f"校准前 - 偏差: {bias:.2f}h, MAE: {mae:.2f}h")
        
        # 使用线性回归进行校准
        X = pred_event.reshape(-1, 1)
        y = true_event
        
        self.linear_model = LinearRegression()
        self.linear_model.fit(X, y)
        
        # 计算校准后的性能
        calibrated_pred = self.predict(pred_event)
        calibrated_bias = np.mean(calibrated_pred - true_event)
        calibrated_mae = mean_absolute_error(true_event, calibrated_pred)
        
        logger.info(f"校准后 - 偏差: {calibrated_bias:.2f}h, MAE: {calibrated_mae:.2f}h")
        
        self.is_fitted = True
        
        return {
            "calibration_factor": self.calibration_factor,
            "original_bias": bias,
            "calibrated_bias": calibrated_bias,
            "original_mae": mae,
            "calibrated_mae": calibrated_mae,
            "improvement": (mae - calibrated_mae) / mae * 100
        }
    
    def predict(self, predicted_times: np.ndarray) -> np.ndarray:
        """
        使用校准模型进行预测
        
        Args:
            predicted_times: 原始预测时间
            
        Returns:
            校准后的预测时间
        """
        if self.is_fitted and self.linear_model is not None:
            # 使用线性模型校准
            X = predicted_times.reshape(-1, 1)
            calibrated = self.linear_model.predict(X)
        else:
            # 使用简单校准因子
            calibrated = predicted_times * self.calibration_factor
        
        # 确保预测时间在合理范围内
        calibrated = np.clip(calibrated, 0.1, 48.0)
        
        return calibrated

class ImprovedSurvivalTimePredictor:
    """
    改进的生存时间预测器
    集成多种预测方法
    """
    
    def __init__(self, meta_model: str = 'ridge', ridge_alpha: float = 1.0, max_method_weight: float = 0.7):
        self.calibrated_predictor = CalibratedTimePredictor()
        self.method_weights = {
            'adaptive_threshold': 0.4,
            'dynamic_bagging': 0.4,
            'ensemble': 0.2
        }
        # 保存最近一次调优的权重和统计信息
        self.last_tuning = {
            'weights': dict(self.method_weights),
            'method_mae': {}
        }
        # 用于元回归校准（将多方法预测作为特征 -> 真实时间）
        self.meta_linear_model = None
        # 元回归器配置
        self.meta_model_type = meta_model  # 'linear' or 'ridge'
        self.ridge_alpha = ridge_alpha
        self.max_method_weight = max_method_weight
        # 保存最近一次调优的权重和统计信息
        self.last_tuning = {
            'weights': dict(self.method_weights),
            'method_mae': {}
        }
        # 用于元回归校准（将多方法预测作为特征 -> 真实时间）
        self.meta_linear_model = None
    
    def adaptive_threshold_method(self, survival_funcs_df: pd.DataFrame) -> np.ndarray:
        """
        自适应阈值方法
        """
        times = survival_funcs_df.columns.astype(float)
        predicted_times = []
        
        for idx in range(len(survival_funcs_df)):
            surv_curve = survival_funcs_df.iloc[idx].values
            
            # 计算曲线特征
            curve_slope = np.mean(np.diff(survival_funcs_df.iloc[idx].values))
            curve_variance = np.var(survival_funcs_df.iloc[idx].values)
            
            # 根据曲线特征选择阈值
            if curve_slope < -0.1:  # 急剧下降
                threshold = 0.3
            elif curve_variance > 0.1:  # 高变异性
                threshold = 0.5
            else:  # 平缓曲线
                threshold = 0.7
            
            # 找到阈值对应的时间
            try:
                pred_time = times[np.where(survival_funcs_df.iloc[idx].values <= threshold)[0][0]]
            except IndexError:
                pred_time = times[-1]  # 使用最后一个时间点
            
            predicted_times.append(pred_time)
        
        return np.array(predicted_times)
    
        """
        使用Weibull分布对生存曲线进行拟合并预测中位生存时间
        基于线性化变换: log(-log(S)) = k * log(t) - k * log(lambda)
        """
        times = survival_funcs_df.columns.astype(float)
        predicted_times = []

        for idx in range(len(survival_funcs_df)):
            surv_curve = survival_funcs_df.iloc[idx].values

            # 需要 t>0 且 0<S<1
            valid_mask = (times > 0) & np.isfinite(surv_curve) & (surv_curve > 0) & (surv_curve < 1)
            if np.sum(valid_mask) < 3:
                predicted_times.append(np.median(times))
                continue

            t = times[valid_mask]
            S = surv_curve[valid_mask]

            # 变换
            try:
                y = np.log(-np.log(S))
                x = np.log(t)
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    predicted_times.append(np.median(t))
                    continue

                reg = LinearRegression()
                reg.fit(x.reshape(-1, 1), y)
                k = reg.coef_[0]
                intercept = reg.intercept_

                # lambda计算
                if k == 0:
                    predicted_times.append(np.median(t))
                    continue

                lam = math.exp(-intercept / k)

                # 计算中位时间 t_median where S=0.5 => t = lambda * ( -ln(0.5) )^(1/k)
                tm = lam * ((-math.log(0.5)) ** (1.0 / k))
                if not np.isfinite(tm) or tm <= 0:
                    tm = np.median(t)
                tm = np.clip(tm, 0.1, 48.0)
                predicted_times.append(tm)
            except Exception:
                predicted_times.append(np.median(t))

        return np.array(predicted_times)
    
        predicted_times = 0.5 + (48 - 0.5) * sigmoid_risk
        
        return predicted_times
    
    def predict_survival_times(self, survival_funcs_df: pd.DataFrame, 
                              risk_scores: np.ndarray,
                              true_times: Optional[np.ndarray] = None,
                              events: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        集成多种方法预测生存时间
        
        Args:
            survival_funcs_df: 生存函数数据框
            risk_scores: 风险分数
            true_times: 真实时间（用于校准）
            events: 事件指示器（用于校准）
            
        Returns:
            包含各种方法预测结果的字典
        """
        results = {}

        # 自适应阈值方法
        results['adaptive_threshold'] = self.adaptive_threshold_method(survival_funcs_df)

        # 动态装袋方法（需要实现或调用外部方法）
        # 暂时跳过，专注于核心方法
        
        # 如果提供了真实时间，进行校准（综合所有方法）
        if true_times is not None and events is not None:
            # 校验输入长度是否和 survival_funcs_df 对齐
            n_rows = len(survival_funcs_df)
            if len(true_times) != n_rows or len(events) != n_rows:
                logger.warning(
                    f"提供的 true_times/events 长度 ({len(true_times)},{len(events)}) 与 survival_funcs_df 行数 ({n_rows}) 不匹配，跳过基于标签的元回归校准。"
                )
                true_times = None
                events = None
        
        if true_times is not None and events is not None:
            # 使用所有可用方法的预测作为特征，训练一个元回归器将多方法结果映射到真实时间（仅事件样本）
            method_feat_keys = [k for k in ['adaptive_threshold'] if k in results]
            X_all = np.vstack([results[k] for k in method_feat_keys]).T  # shape (n_samples, n_methods)
            event_mask = events.astype(bool)
            X_event = X_all[event_mask]
            y_event = true_times[event_mask]

            if X_event.shape[0] < 10:
                # 若事件样本过少，退回到单一方法校准（原来的 calibrated predictor）
                base_predictions = results.get('adaptive_threshold', results.get('curve_fitting', np.median(list(results.values())[0])))
                calibration_stats = self.calibrated_predictor.fit(base_predictions, true_times, events)
                calib_pred = self.calibrated_predictor.predict(base_predictions)
                results['calibrated'] = calib_pred
                logger.warning("事件样本过少，退回到单方法校准")
                logger.info(f"校准统计: {calibration_stats}")
            else:
                # 拟合元回归（默认使用 Ridge 正则化以减少过拟合）
                try:
                    if self.meta_model_type == 'ridge':
                        meta_reg = Ridge(alpha=self.ridge_alpha)
                    else:
                        meta_reg = LinearRegression()

                    meta_reg.fit(X_event, y_event)
                    self.meta_linear_model = meta_reg
                    calib_full = meta_reg.predict(X_all)
                    calib_full = np.clip(calib_full, 0.1, 48.0)
                    results['calibrated'] = calib_full
                    # 记录校准统计
                    mae_before = mean_absolute_error(y_event, np.mean(X_event, axis=1))
                    mae_after = mean_absolute_error(y_event, calib_full[event_mask])
                    logger.info(f"Meta校准完成: MAE_before(avg_methods)={mae_before:.2f}h, MAE_after={mae_after:.2f}h")
                    calibration_stats = {'meta_mae_before': mae_before, 'meta_mae_after': mae_after, 'methods_used': method_feat_keys, 'meta_model': self.meta_model_type}
                except Exception as e:
                    logger.exception(f"元回归校准失败，退回到单方法校准: {e}")
                    base_predictions = results.get('adaptive_threshold', results.get('curve_fitting'))
                    calibration_stats = self.calibrated_predictor.fit(base_predictions, true_times, events)
                    results['calibrated'] = self.calibrated_predictor.predict(base_predictions)
                    logger.info(f"校准统计: {calibration_stats}")
        else:
            # 未提供真实标签时，使用简单的加权集成作为校准近似（用当前 method_weights）
            norm_weights = {m: w for m, w in self.method_weights.items() if m in results}
            s = sum(norm_weights.values()) if sum(norm_weights.values()) > 0 else 1.0
            ensemble_w = {m: float(w / s) for m, w in norm_weights.items()}
            approx = np.zeros_like(list(results.values())[0])
            for m, w in ensemble_w.items():
                approx += w * results[m]
            results['calibrated'] = np.clip(approx, 0.1, 48.0)

        # 如果提供了真实时间/事件，则评估各子方法并可选进行权重调优
        tuning_info = None
        if true_times is not None and events is not None:
            eval_stats = self.evaluate_predictions({k: v for k, v in results.items()}, true_times, events)
            # 记录每法的MAE，只考虑基础方法（不包含 ensemble/calibrated）
            base_methods = [k for k in ['adaptive_threshold', 'curve_fitting', 'curve_fitting_weibull', 'risk_based'] if k in results]
            method_mae = {m: eval_stats.get(m, {}).get('MAE', float('inf')) for m in base_methods}
            tuning_info = {'method_mae': method_mae}

            # 基于MAE反向加权（MAE越小权重越大），并平滑处理
            inv = {}
            eps = 1e-6
            for m, mae in method_mae.items():
                inv[m] = 1.0 / (mae + eps)

            total_inv = sum(inv.values()) if sum(inv.values()) > 0 else 1.0
            new_weights = {m: float(inv[m] / total_inv) for m in inv}

            # 限制单一方法的最大权重以防过拟合
            for m in new_weights:
                if new_weights[m] > self.max_method_weight:
                    new_weights[m] = self.max_method_weight

            # 重新归一化
            s = sum(new_weights.values())
            if s > 0:
                new_weights = {m: float(w / s) for m, w in new_weights.items()}

            # 将调优结果写回到 self.method_weights（仅替换存在的基础方法权重）
            for m in base_methods:
                if m in new_weights:
                    self.method_weights[m] = new_weights[m]

            self.last_tuning = {'weights': dict(self.method_weights), 'method_mae': method_mae}

        # 集成预测
        ensemble_pred = np.zeros_like(results['adaptive_threshold'])
        for method, weight in self.method_weights.items():
            if method in results:
                ensemble_pred += weight * results[method]

        results['ensemble'] = ensemble_pred
        if tuning_info is not None:
            results['tuning'] = self.last_tuning

        return results

    def save(self, path: str):
        """
        将整个 ImprovedSurvivalTimePredictor 对象持久化到磁盘（pickle）。
        """
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Saved ImprovedSurvivalTimePredictor to {path}")
        except Exception as e:
            logger.exception(f"Failed to save ImprovedSurvivalTimePredictor to {path}: {e}")

    @staticmethod
    def load(path: str):
        """
        从磁盘加载 ImprovedSurvivalTimePredictor（pickle）。
        返回对象实例，失败时抛出异常。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    def evaluate_predictions(self, predictions: Dict[str, np.ndarray], 
                           true_times: np.ndarray, events: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        评估各种预测方法的性能
        
        Args:
            predictions: 各种方法的预测结果
            true_times: 真实生存时间
            events: 事件指示器
            
        Returns:
            性能评估结果
        """
        # 只评估事件样本
        event_mask = events.astype(bool)
        true_event = true_times[event_mask]
        
        results = {}
        for method, pred in predictions.items():
            pred_event = pred[event_mask]
            
            mae = mean_absolute_error(true_event, pred_event)
            rmse = np.sqrt(mean_squared_error(true_event, pred_event))
            bias = np.mean(pred_event - true_event)
            
            # 计算MAPE（避免除零）
            mape = np.mean(np.abs((pred_event - true_event) / (true_event + 1e-8))) * 100
            
            results[method] = {
                'MAE': mae,
                'RMSE': rmse,
                'Bias': bias,
                'MAPE': mape
            }
        
        return results

def create_improved_time_predictor():
    """
    创建改进的时间预测器实例
    """
    return ImprovedSurvivalTimePredictor()
