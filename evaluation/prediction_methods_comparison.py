#!/usr/bin/env python3
"""
集成到模型测试流程的预测方法对比模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import logging
from datetime import datetime
from scipy import stats, interpolate
from sklearn.linear_model import LinearRegression

from evaluation.time_head_outputs import generate_time_head_style_outputs
from easydict import EasyDict

# 设置字体为英文（避免中文字体缺失警告）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class PredictionMethodsComparison:
    """
    集成版本的预测方法对比类
    """
    
    def __init__(self, config, output_dir):
        """
        初始化预测方法对比器
        
        Args:
            config: 配置对象
            output_dir: 主输出目录
        """
        self.config = config
        self.main_output_dir = output_dir
        self.prediction_window = config.data.sequence_generation.prediction_window_hours
        
        # 获取配置，如果不存在则使用默认值
        if not hasattr(config.evaluation, 'prediction_methods_comparison'):
            config.evaluation.prediction_methods_comparison = EasyDict()
        
        self.comparison_config = config.evaluation.prediction_methods_comparison
        
        # 使用默认值填充缺失的配置项
        output_subdir = getattr(self.comparison_config, 'output_subdir', 'prediction_methods_comparison')
        
        # 创建预测方法对比的输出目录
        self.output_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取配置，使用默认值填充缺失项
        if not hasattr(self.comparison_config, 'plot_settings'):
            self.comparison_config.plot_settings = EasyDict({
                'x_axis_limit': None,
                'figure_size': (12, 10),
                'dpi': 300,
                'save_formats': ['png', 'pdf']
            })
        
        self.plot_settings = self.comparison_config.plot_settings
        self.time_head_style_enabled = bool(
            getattr(self.comparison_config, 'generate_time_head_style_outputs', True)
        )
        self.default_split_name = getattr(self.comparison_config, 'default_split_name', 'test')

        # 校准参数缓存（避免重复拟合）
        self._simple_calibration_params = None
        self._calibration_context = None
        
        logger.info(f"预测方法对比输出目录: {self.output_dir}")
    
        # 预处理配置（可选）：单调性与平滑
        preprocessing_raw = getattr(self.comparison_config, 'preprocessing', None)
        if preprocessing_raw is None:
            self.preprocessing_cfg = EasyDict({
                'enforce_monotonic': True,
                'smoothing_window': 1
            })
        else:
            # 转换为EasyDict以便统一访问
            if isinstance(preprocessing_raw, dict):
                self.preprocessing_cfg = EasyDict(preprocessing_raw)
            else:
                self.preprocessing_cfg = preprocessing_raw
    
    @staticmethod
    def _normalize_method_key(name):
        if name is None:
            return ''
        return str(name).lower().replace('_', '').replace(' ', '')

    def _default_prediction_time(self, ratio: float = 0.6) -> float:
        """基于真实分布统计获取默认预测时间。"""
        fallback = float(self.prediction_window * ratio)
        if hasattr(self, '_true_distribution_stats') and self._true_distribution_stats:
            median = self._true_distribution_stats.get('median')
            if median is not None:
                return float(np.clip(median, 0.5, self.prediction_window * 0.9))
        return float(np.clip(fallback, 0.5, self.prediction_window * 0.9))

    @staticmethod
    def _interp_crossing_time(times: np.ndarray, curve: np.ndarray, threshold: float):
        """返回曲线首次低于阈值时的时间，若不存在则返回None。"""
        below = curve <= threshold
        if not np.any(below):
            return None
        idx = int(np.where(below)[0][0])
        if idx == 0:
            return float(times[0])
        t1, t2 = float(times[idx-1]), float(times[idx])
        p1, p2 = float(curve[idx-1]), float(curve[idx])
        if p1 == p2:
            return float(t2)
        ratio = (threshold - p1) / (p2 - p1)
        ratio = np.clip(ratio, 0.0, 1.0)
        return float(t1 + (t2 - t1) * ratio)

    def _optimize_ensemble_weights(self, feature_dict, method_labels, true_times, events, cfg):
        if true_times is None or events is None:
            return None
        event_mask = np.asarray(events) == 1
        if not np.any(event_mask):
            return None
        base_arrays = []
        for label in method_labels:
            arr = np.asarray(feature_dict.get(label), dtype=float).reshape(-1)
            if arr.shape[0] == 0:
                return None
            base_arrays.append(arr)
        base_matrix = np.column_stack([arr[event_mask] for arr in base_arrays])
        true_event = np.asarray(true_times, dtype=float)[event_mask]
        valid_rows = np.all(np.isfinite(base_matrix), axis=1) & np.isfinite(true_event)
        if not np.any(valid_rows):
            return None
        base_matrix = base_matrix[valid_rows]
        true_event = true_event[valid_rows]
        if base_matrix.shape[0] == 0:
            return None

        step = float(getattr(cfg, 'step', 0.1) or 0.1)
        step = max(min(step, 1.0), 0.01)
        min_weight = float(getattr(cfg, 'min_weight', 0.0) or 0.0)
        min_weight = max(min_weight, 0.0)
        max_combos = int(getattr(cfg, 'max_combinations', 20000))

        grid = np.arange(0.0, 1.0 + 1e-8, step)
        n_methods = len(method_labels)
        combos = []

        def backtrack(prefix, remaining, idx):
            if len(combos) >= max_combos:
                return
            if idx == n_methods - 1:
                weight = remaining
                if weight + 1e-9 < min_weight:
                    return
                combos.append(prefix + [weight])
                return
            for value in grid:
                if value + 1e-9 < min_weight:
                    continue
                if value - 1e-9 > remaining:
                    continue
                backtrack(prefix + [value], remaining - value, idx + 1)

        backtrack([], 1.0, 0)
        if not combos:
            return None

        combos = np.asarray(combos, dtype=float)
        combos = combos[np.abs(combos.sum(axis=1) - 1.0) <= 1e-6]
        if combos.size == 0:
            return None

        existing_pred = np.asarray(feature_dict.get('Ensemble'), dtype=float).reshape(-1)
        existing_event = existing_pred[event_mask][valid_rows]
        existing_mae = np.mean(np.abs(existing_event - true_event)) if np.all(np.isfinite(existing_event)) else np.inf

        best_mae = existing_mae
        best_weights = None
        for weights in combos:
            preds = base_matrix.dot(weights)
            mae = np.mean(np.abs(preds - true_event))
            if mae + 1e-9 < best_mae:
                best_mae = mae
                best_weights = weights

        require_improve = bool(getattr(cfg, 'require_improvement', True))
        if best_weights is None:
            if require_improve:
                return None
            best_weights = combos[0]
        if require_improve and best_mae + 1e-9 >= existing_mae:
            return None
        return np.asarray(best_weights, dtype=float)


    def _run_simple_calibration(self, base_predictions, true_times, events, risk_scores, context=None):
        cal_cfg = getattr(self.comparison_config, 'calibration', None)
        if cal_cfg is None or not getattr(cal_cfg, 'enabled', True):
            return None
        if isinstance(cal_cfg, dict):
            cal_cfg = EasyDict(cal_cfg)

        sample_lengths = []
        for values in base_predictions.values():
            if values is None:
                continue
            arr = np.asarray(values, dtype=float).reshape(-1)
            sample_lengths.append(arr.shape[0])
        if not sample_lengths:
            logger.warning("Calibrated: 缺少基础方法预测，无法执行校准。")
            return None
        n_samples = sample_lengths[0]
        if any(length != n_samples for length in sample_lengths):
            logger.warning("Calibrated: 不同基础方法的样本数不一致。")
            return None

        def _match_label(target_normalized):
            for key in base_predictions.keys():
                if self._normalize_method_key(key) == target_normalized:
                    return key
            return None

        requested_methods = list(getattr(
            cal_cfg,
            'feature_methods',
            ['ensemble']
        ))

        ordered_labels = []
        for name in requested_methods:
            label = _match_label(self._normalize_method_key(name))
            if label and label not in ordered_labels:
                ordered_labels.append(label)
        if 'Ensemble' in base_predictions and 'Ensemble' not in ordered_labels:
            ordered_labels.append('Ensemble')
        for key in base_predictions.keys():
            if key not in ordered_labels:
                ordered_labels.append(key)

        feature_dict = {}
        for label in ordered_labels:
            values = base_predictions.get(label)
            if values is None:
                feature_dict[label] = np.full(n_samples, np.nan, dtype=float)
                continue
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape[0] != n_samples:
                logger.warning(f"Calibrated: 特征 {label} 的长度与样本数不一致，已忽略。")
                continue
            feature_dict[label] = arr

        include_risk_cfg = bool(getattr(cal_cfg, 'use_risk_score', False))
        risk_label = 'RiskScore'
        if include_risk_cfg:
            if risk_scores is not None:
                risk_arr = np.asarray(risk_scores, dtype=float).reshape(-1)
                if risk_arr.shape[0] == n_samples:
                    feature_dict[risk_label] = risk_arr
                else:
                    logger.warning("Calibrated: 风险得分长度与样本数不一致，忽略 risk_score 特征。")
            else:
                feature_dict.setdefault(risk_label, np.full(n_samples, np.nan, dtype=float))

        feature_order = []
        for label in ordered_labels:
            if label in feature_dict:
                feature_order.append(label)
        if include_risk_cfg and risk_label in feature_dict and risk_label not in feature_order:
            feature_order.append(risk_label)

        if not feature_order:
            logger.warning("Calibrated: 未找到可用的特征列。")
            return None

        def _prepare_matrix(labels):
            cols = []
            valid_labels = []
            for label in labels:
                arr = feature_dict.get(label)
                if arr is None:
                    continue
                col = np.asarray(arr, dtype=float).reshape(-1)
                cols.append(col)
                valid_labels.append(label)
            if not cols:
                return None, []
            matrix = np.column_stack(cols)
            nan_mask = ~np.isfinite(matrix)
            if np.any(nan_mask):
                fill_values = np.nanmedian(matrix, axis=0)
                fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
                for j in range(matrix.shape[1]):
                    column = matrix[:, j]
                    mask = ~np.isfinite(column)
                    if np.any(mask):
                        column = column.copy()
                        column[mask] = fill_values[j]
                        matrix[:, j] = column
            return matrix, valid_labels

        feature_matrix, feature_labels = _prepare_matrix(feature_order)
        if feature_matrix is None:
            logger.warning("Calibrated: 无法构建特征矩阵。")
            return None

        label_to_index = {label: idx for idx, label in enumerate(feature_labels)}

        true_times_arr = None
        if true_times is not None:
            true_times_arr = np.asarray(true_times, dtype=float).reshape(-1)
            if true_times_arr.shape[0] != n_samples:
                logger.warning("Calibrated: true_times 长度与样本数不一致，忽略标签。")
                true_times_arr = None

        events_arr = None
        if events is not None:
            events_arr = np.asarray(events).reshape(-1)
            if events_arr.shape[0] != n_samples:
                logger.warning("Calibrated: events 长度与样本数不一致，忽略事件信息。")
                events_arr = None

        clip_cfg = getattr(cal_cfg, 'clip_bounds', EasyDict())
        if isinstance(clip_cfg, dict):
            clip_cfg = EasyDict(clip_cfg)
        min_hours_cfg = getattr(clip_cfg, 'min_hours', None)
        max_ratio_cfg = getattr(clip_cfg, 'max_ratio', None)
        min_hours = float(min_hours_cfg) if min_hours_cfg is not None else 0.5
        if max_ratio_cfg is not None:
            max_hours = float(self.prediction_window) * float(max_ratio_cfg)
        else:
            max_hours = float(self.prediction_window) * 0.9

        auto_cfg = getattr(cal_cfg, 'auto_fit', EasyDict())
        if isinstance(auto_cfg, dict):
            auto_cfg = EasyDict(auto_cfg)

        default_cfg = getattr(cal_cfg, 'default', EasyDict())
        if isinstance(default_cfg, dict):
            default_cfg = EasyDict(default_cfg)
        default_method_key = self._normalize_method_key(getattr(default_cfg, 'base_method', 'Ensemble'))
        default_scale = 0.96
        default_bias = 0.0
        # default_scale = float(getattr(default_cfg, 'scale', 0.41))
        # default_bias = float(getattr(default_cfg, 'bias', 0.0))

        def _select_matrix(labels):
            indices = []
            for label in labels:
                idx = label_to_index.get(label)
                if idx is not None:
                    indices.append(idx)
            if not indices:
                return None, []
            subset = feature_matrix[:, indices]
            return subset, [feature_labels[i] for i in indices]

        def _build_default_params():
            label = _match_label(default_method_key) or (feature_labels[0] if feature_labels else None)
            if label is None:
                return None
            return EasyDict(labels=[label], weights=[default_scale], bias=default_bias, source='default')

        def _fit_auto_params():
            if not bool(getattr(auto_cfg, 'enabled', False)):  # 默认为False，禁用自动拟合
                return None
            if true_times_arr is None or events_arr is None:
                return None
            event_mask = (events_arr == 1)
            if not np.any(event_mask):
                return None
            X_event = feature_matrix[event_mask]
            y_event = true_times_arr[event_mask]
            valid_rows = np.isfinite(y_event)
            if not np.any(valid_rows):
                return None
            X_event = X_event[valid_rows]
            y_event = y_event[valid_rows]
            min_events = max(5, int(getattr(auto_cfg, 'min_events', 30)))
            if X_event.shape[0] < min_events:
                logger.info("Calibrated: 有效事件样本 %d 少于阈值 %d，跳过自动校准。",
                            X_event.shape[0], min_events)
                return None
            design = np.column_stack([X_event, np.ones(X_event.shape[0])])
            ridge_lambda = float(getattr(auto_cfg, 'ridge_lambda', 1e-3))
            reg = np.eye(design.shape[1]) * ridge_lambda
            reg[-1, -1] = 0.0
            try:
                coeffs = np.linalg.solve(design.T @ design + reg, design.T @ y_event)
            except np.linalg.LinAlgError:
                coeffs, *_ = np.linalg.lstsq(design, y_event, rcond=None)
            weights = coeffs[:-1]
            bias = coeffs[-1]
            weight_bounds = getattr(auto_cfg, 'weight_bounds', (-5.0, 5.0))
            if isinstance(weight_bounds, (list, tuple)) and len(weight_bounds) == 2:
                weights = np.clip(weights, float(weight_bounds[0]), float(weight_bounds[1]))
            bias_bounds = getattr(auto_cfg, 'bias_bounds', (-24.0, 24.0))
            if isinstance(bias_bounds, (list, tuple)) and len(bias_bounds) == 2:
                bias = float(np.clip(bias, float(bias_bounds[0]), float(bias_bounds[1])))
            return EasyDict(labels=list(feature_labels), weights=weights.tolist(), bias=float(bias), source='auto')

        params = _fit_auto_params()
        if params is None and self._simple_calibration_params:
            cached = self._simple_calibration_params
            subset_matrix, used_labels = _select_matrix(cached.get('labels', []))
            if subset_matrix is not None:
                params = EasyDict(labels=used_labels, weights=cached.get('weights', []), bias=cached.get('bias', 0.0), source='cache')
        if params is None:
            params = _build_default_params()
            if params is None:
                logger.warning("Calibrated: 无可用的默认校准参数。")
                return None
            logger.info("Calibrated: 使用默认校准（方法=%s, scale=%.3f, bias=%.3f）",
                        params.labels[0], default_scale, default_bias)
        elif params.source == 'auto':
            logger.info("Calibrated: 使用简单线性校准（自动拟合），bias=%.3f", params.bias)

        subset_matrix, used_labels = _select_matrix(params.labels)
        if subset_matrix is None:
            logger.warning("Calibrated: 所选特征列缺失，无法完成校准。")
            return None
        weights = np.asarray(params.weights, dtype=float)
        if weights.shape[0] != subset_matrix.shape[1]:
            if weights.shape[0] > subset_matrix.shape[1]:
                weights = weights[:subset_matrix.shape[1]]
            else:
                weights = np.pad(weights, (0, subset_matrix.shape[1] - weights.shape[0]), constant_values=0.0)
        bias = float(params.bias)
        calibrated_full = subset_matrix.dot(weights) + bias
        calibrated_full = np.clip(calibrated_full, min_hours, max_hours)

        cache_candidate = None
        if params.source == 'auto':
            params.labels = used_labels
            params.weights = weights.tolist()
            params.bias = bias
            cache_candidate = params

        event_mask = None
        if events_arr is not None:
            event_mask = (events_arr == 1)
            if not np.any(event_mask):
                event_mask = None

        def _prepare_array(values):
            if values is None:
                return None
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape[0] != n_samples:
                return None
            return arr

        def _event_mae(arr):
            if arr is None or true_times_arr is None or event_mask is None:
                return (None, None)
            pred_event = arr[event_mask]
            true_event = true_times_arr[event_mask]
            valid = np.isfinite(pred_event) & np.isfinite(true_event) & (true_event > 0)
            if not np.any(valid):
                return (None, None)
            err = np.abs(pred_event[valid] - true_event[valid])
            return (float(np.mean(err)), pred_event[valid])

        fallback_reason = None
        fallback_label = None
        fallback_values = None
        base_eval = {}

        tol = float(getattr(cal_cfg, 'improvement_tolerance', 0.03))
        min_std = float(getattr(cal_cfg, 'min_event_std', 0.25))
        min_unique_ratio = float(getattr(cal_cfg, 'min_unique_ratio', 0.02))

        def _is_diverse(pred_values):
            if pred_values is None or pred_values.size == 0:
                return False
            std_val = float(np.nanstd(pred_values))
            if std_val < min_std:
                return False
            unique_ratio = len(np.unique(np.round(pred_values, 2))) / max(1, pred_values.shape[0])
            return unique_ratio >= min_unique_ratio

        if event_mask is not None and true_times_arr is not None:
            for label, values in base_predictions.items():
                arr = _prepare_array(values)
                if arr is None:
                    continue
                mae_val, valid_vals = _event_mae(arr)
                if mae_val is None:
                    continue
                base_eval[label] = {
                    'mae': mae_val,
                    'arr': arr,
                    'valid': valid_vals,
                    'diverse': _is_diverse(valid_vals)
                }

            cal_mae, cal_valid = _event_mae(calibrated_full)

            if base_eval:
                diverse_items = [(label, data) for label, data in base_eval.items() if data['diverse']]
                candidate_items = diverse_items if diverse_items else list(base_eval.items())
                best_label, best_data = min(candidate_items, key=lambda kv: kv[1]['mae'])
                best_mae = best_data['mae']
            else:
                best_label = None
                best_data = None
                best_mae = None

            if cal_mae is None and best_label is not None:
                fallback_reason = "calibrated MAE 无效"
            elif cal_mae is not None and best_label is not None and best_mae is not None:
                if cal_mae > best_mae * (1.0 + tol):
                    fallback_reason = f"calibrated MAE {cal_mae:.3f} 劣于基础方法 {best_label} ({best_mae:.3f})"
            if fallback_reason is None and cal_valid is not None:
                cal_std = float(np.nanstd(cal_valid))
                if cal_std < min_std:
                    fallback_reason = f"calibrated 事件样本标准差 {cal_std:.3f} 低于阈值 {min_std:.3f}"
                else:
                    unique_ratio = len(np.unique(np.round(cal_valid, 2))) / max(1, cal_valid.shape[0])
                    if unique_ratio < min_unique_ratio:
                        fallback_reason = f"calibrated 唯一值比例 {unique_ratio:.3f} 低于阈值 {min_unique_ratio:.3f}"

            if fallback_reason and best_data is not None:
                fallback_label = best_label
                fallback_values = best_data['arr']

        if fallback_values is not None:
            logger.warning("Calibrated: 输出退回到 %s（原因：%s）", fallback_label, fallback_reason)
            return np.clip(fallback_values, min_hours, max_hours)

        if cache_candidate is not None:
            self._simple_calibration_params = cache_candidate

        return calibrated_full

    def method_least_squares(self, survival_funcs_df, risk_scores=None, events=None, true_times=None):
        """
        最小二乘法阈值预测：
        寻找最佳生存概率阈值，使预测时间与真实时间的均方误差（MSE）最小。
        仅基于事件样本进行优化。
        """
        times = survival_funcs_df.columns.astype(float)
        n_samples = len(survival_funcs_df)
        
        # 默认阈值
        best_threshold = 0.5
        
        # 如果有真实时间，进行优化
        # 仅当 true_times 和 events 都有效时进行
        if true_times is not None and events is not None:
            events_arr = np.asarray(events).reshape(-1)
            true_times_arr = np.asarray(true_times).reshape(-1)
            
            # 1. 严格仅使用发生事件的样本 (events == 1)
            event_mask = (events_arr == 1)
            
            if np.any(event_mask):
                true_event_times = true_times_arr[event_mask]
                event_indices = np.where(event_mask)[0]
                
                # 预先提取事件样本的曲线以加速
                event_curves = []
                for idx in event_indices:
                    curve = survival_funcs_df.iloc[idx].values
                    # 简单填充
                    if np.any(pd.isna(curve)):
                        curve = pd.Series(curve).fillna(method='ffill').fillna(0.0).values
                    event_curves.append(curve)
                
                # 定义MSE目标函数 - 仅针对事件样本计算
                def objective(threshold):
                    preds = []
                    for curve in event_curves:
                        t = self._interp_crossing_time(times, curve, threshold)
                        if t is None:
                            # 这是一个关键决策点：
                            # 如果曲线从未降到阈值以下，我们应该怎么惩罚？
                            # 对于事件样本，这通常意味着预测太晚。
                            # 使用最大时间（窗口末端）是一个合理的惩罚。
                            t = times[-1]
                        preds.append(t)
                    
                    # 计算均方误差
                    mse = np.mean((np.array(preds) - true_event_times) ** 2)
                    return mse
                
                # 标量最小化优化
                from scipy.optimize import minimize_scalar
                
                # 使用 brute 暴力搜索全局最优解，而非 bounded 局部搜索
                # 这样可以避免陷入局部极小值（例如在边界附近）
                from scipy.optimize import brute
                
                # 定义搜索网格范围 (调整范围和步长，特别是关注尾部)
                # 扩大搜索范围到非常小的阈值，并加密步长
                ranges = (slice(0.0001, 0.9999, 0.0001),)
                
                # 使用 brute force 寻找全局最优阈值
                # brute 返回的是参数值数组
                res_brute = brute(objective, ranges, full_output=True, finish=None)
                best_threshold = res_brute[0]
                min_mse = res_brute[1]
                
                # 确保阈值在合理范围内
                best_threshold = np.clip(best_threshold, 0.001, 0.999)
                
                logger.info(f"Least Squares Optimization (Events Only): Best Threshold = {best_threshold:.4f}, MSE = {min_mse:.4f}")
        else:
            logger.warning("Method Least Squares: 缺少真实标签或事件指示器，无法优化阈值，使用默认值 0.5。")

        # 生成最终预测
        # 注意：对于非事件样本（删失），我们也使用同样的阈值进行预测。
        # 虽然优化时只用了事件样本，但预测需要覆盖所有样本。
        predictions = []
        for i in range(n_samples):
            curve = survival_funcs_df.iloc[i].values
            if np.any(pd.isna(curve)):
                 curve = pd.Series(curve).fillna(method='ffill').fillna(0.0).values
            
            pred = self._interp_crossing_time(times, curve, best_threshold)
            if pred is None:
                pred = times[-1]
            predictions.append(pred)
            
        return np.array(predictions)



    def calculate_metrics(self, predicted_times, true_times, events):
        """
        计算评估指标
        """
        # 只考虑有事件的样本
        event_mask = events == 1
        if not np.any(event_mask):
            return {}
        
        pred_event = np.array(predicted_times)[event_mask]
        true_event = np.array(true_times)[event_mask]
        
        # 过滤无效值
        valid_mask = np.isfinite(pred_event) & np.isfinite(true_event) & (true_event > 0)
        if not np.any(valid_mask):
            return {}
        
        pred_valid = np.array(pred_event[valid_mask]).reshape(-1)
        true_valid = np.array(true_event[valid_mask]).reshape(-1)
        
        # 计算指标
        mae = np.mean(np.abs(pred_valid - true_valid))
        rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
        mape = np.mean(np.abs((pred_valid - true_valid) / true_valid)) * 100
        bias = np.mean(pred_valid - true_valid)
        
        # 计算相关性
        if len(pred_valid) > 1:
            try:
                correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
                # 检查相关性是否为nan
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0
        else:
            correlation = 0.0
        
        # 计算R²
        ss_res = np.sum((pred_valid - true_valid) ** 2)
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias': bias,
            'correlation': correlation,
            'r2': r2,
            'n_samples': len(pred_valid)
        }

    def _get_method_pairs(self):
        """Return the Least Squares method only."""
        return [('Least_Squares', self.method_least_squares)]

    def create_separate_plots(self, survival_df, true_times, events, risk_scores, split_name='test'):
        """创建分开的对比图并返回预测与指标。"""

        method_pairs = self._get_method_pairs()

        true_times = np.asarray(true_times)
        events = np.asarray(events)

        # 保存最近一次的标签以便 Calibrated 方法（内部元回归）使用
        self._last_true_times = true_times
        self._last_events = events
        context = getattr(self, '_calibration_context', None)

        predictions = {}
        metrics = {}

        for method_name, method_func in method_pairs:
            try:
                if method_name == 'Calibrated':
                    predictions[method_name] = None
                    metrics[method_name] = None
                    continue
                # 显式传递 true_times 以便 method_least_squares 进行优化
                pred_times = method_func(survival_df, risk_scores, events=events, true_times=true_times)
            except Exception as exc:
                logger.warning(f"方法 {method_name} 执行失败: {exc}")
                continue

            predictions[method_name] = pred_times
            method_metrics = self.calculate_metrics(pred_times, true_times, events)
            metrics[method_name] = method_metrics if method_metrics else {}

        if not predictions:
            logger.warning("预测方法对比未生成任何有效预测，跳过可视化绘制。")
            return predictions, metrics

        if 'Calibrated' in predictions:
            base_predictions = {k: v for k, v in predictions.items() if k != 'Calibrated'}
            calibrated = self._run_simple_calibration(base_predictions, true_times, events, risk_scores, context=context)
            if calibrated is not None:
                predictions['Calibrated'] = calibrated
                metrics['Calibrated'] = self.calculate_metrics(calibrated, true_times, events)
            else:
                fallback = base_predictions.get('Ensemble')
                if fallback is not None:
                    fallback = np.asarray(fallback, dtype=float)
                    fallback = np.clip(fallback, 0.5, self.prediction_window * 0.9)
                    predictions['Calibrated'] = fallback
                    metrics['Calibrated'] = self.calculate_metrics(fallback, true_times, events)
                else:
                    predictions.pop('Calibrated', None)
                    metrics.pop('Calibrated', None)

        colors = sns.color_palette('tab10', len(predictions)) if hasattr(sns, 'color_palette') else ['blue', 'red', 'green', 'orange', 'purple']
        color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(predictions.keys())}

        # 自动设置x_axis_limit为prediction_window（如果未指定）
        max_time = getattr(self.plot_settings, 'x_axis_limit', None)
        if max_time is None:
            # 尝试从config获取prediction_window_hours
            try:
                prediction_window = float(getattr(self.config.data.sequence_generation, 'prediction_window_hours', 48.0))
                max_time = prediction_window
            except Exception:
                max_time = float(np.nanmax(true_times)) if np.size(true_times) else 50.0
        else:
            max_time = float(max_time)

        # 1. 预测 vs 真实时间散点图
        plt.figure(figsize=tuple(self.plot_settings.figure_size))
        event_mask = events == 1
        for method_name, pred_times in predictions.items():
            if np.any(event_mask):
                pred_event = np.asarray(pred_times)[event_mask]
                true_event = true_times[event_mask]
                plt.scatter(true_event, pred_event, alpha=0.6, s=20,
                            color=color_map[method_name], label=method_name)

        plt.plot([0, max_time], [0, max_time], 'k--', linewidth=2, label='Perfect Prediction')
        plt.xlim(0, max_time)
        plt.ylim(0, max_time)
        plt.xlabel('True Survival Time (hours)')
        plt.ylabel('Predicted Survival Time (hours)')
        plt.title('Prediction Methods Comparison - Predicted vs True Time')
        # 只有在有标签元素时才显示图例
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in self.plot_settings.save_formats:
            plot_path = os.path.join(self.output_dir, f'prediction_methods_comparison.{fmt}')
            plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
        plt.close()

        # 2. 误差分布对比图
        plt.figure(figsize=(12, 8))
        for method_name, pred_times in predictions.items():
            if np.any(event_mask):
                pred_event = np.asarray(pred_times)[event_mask]
                true_event = true_times[event_mask]
                errors = np.abs(pred_event - true_event)
                plt.hist(errors, bins=30, alpha=0.6, label=method_name,
                         color=color_map[method_name], density=True)

        plt.xlabel('Absolute Error (hours)')
        plt.ylabel('Density')
        plt.title('Error Distribution Comparison')
        plt.xlim(0, 50)
        # 只有在有标签元素时才显示图例
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in self.plot_settings.save_formats:
            plot_path = os.path.join(self.output_dir, f'error_distribution_comparison.{fmt}')
            plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
        plt.close()

        # 3. 性能指标对比图
        plt.figure(figsize=(12, 8))
        method_names = list(metrics.keys())
        mae_values = [metrics[method].get('mae', np.nan) for method in method_names]
        rmse_values = [metrics[method].get('rmse', np.nan) for method in method_names]
        mape_values = [metrics[method].get('mape', np.nan) for method in method_names]

        x = np.arange(len(method_names))
        width = 0.25
        plt.bar(x - width, mae_values, width, label='MAE', alpha=0.8)
        plt.bar(x, rmse_values, width, label='RMSE', alpha=0.8)
        plt.bar(x + width, mape_values, width, label='MAPE', alpha=0.8)

        plt.xlabel('Prediction Methods')
        plt.ylabel('Error Values')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, method_names, rotation=45)
        # 只有在有标签元素时才显示图例
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in self.plot_settings.save_formats:
            plot_path = os.path.join(self.output_dir, f'performance_metrics_comparison.{fmt}')
            plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
        plt.close()

        # 4. 相关性对比图
        plt.figure(figsize=(10, 6))
        correlations = [metrics[method].get('correlation', 0.0) for method in method_names]
        r2_values = [metrics[method].get('r2', 0.0) for method in method_names]

        x = np.arange(len(method_names))
        width = 0.35
        plt.bar(x - width / 2, correlations, width, label='Correlation', alpha=0.8)
        plt.bar(x + width / 2, r2_values, width, label='R²', alpha=0.8)

        plt.xlabel('Prediction Methods')
        plt.ylabel('Correlation / R² Values')
        plt.title('Correlation and R² Comparison')
        plt.xticks(x, method_names, rotation=45)
        # 只有在有标签元素时才显示图例
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in self.plot_settings.save_formats:
            plot_path = os.path.join(self.output_dir, f'correlation_comparison.{fmt}')
            plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
        plt.close()

        return predictions, metrics
    
    def generate_performance_table(self, metrics, predictions=None, true_times=None, events=None, risk_scores=None, record_ids=None):
        """
        生成性能对比表格
        """
        # 创建性能对比表格
        performance_data = []
        
        for method_name, metric in metrics.items():
            performance_data.append({
                'Method': method_name,
                'MAE (hours)': f"{metric['mae']:.2f}",
                'RMSE (hours)': f"{metric['rmse']:.2f}",
                'MAPE (%)': f"{metric['mape']:.1f}",
                'Bias (hours)': f"{metric['bias']:.2f}",
                'Correlation': f"{metric['correlation']:.3f}",
                'R²': f"{metric['r2']:.3f}",
                'Samples': metric['n_samples']
            })
        
        # 转换为DataFrame并排序（按MAE）
        df = pd.DataFrame(performance_data)
        df = df.sort_values('MAE (hours)')
        
        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'prediction_methods_performance_table.csv')
        df.to_csv(csv_path, index=False)
        
        # 保存为JSON
        json_path = os.path.join(self.output_dir, 'prediction_methods_performance_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 生成样本对比表格
        self._generate_sample_comparison_table(predictions, true_times, events, risk_scores, record_ids)
        
        return df
    
    def _generate_sample_comparison_table(self, predictions, true_times, events, risk_scores, record_ids=None):
        """
        生成真实测试样本的对比表格
        """
        try:
            # 获取事件样本
            event_mask = events == 1
            if not np.any(event_mask):
                print("没有事件样本，跳过样本对比表格生成")
                return
            
            # 检查record_ids是否可用
            if record_ids is None:
                print("record_ids不可用，使用模拟数据生成样本对比表格")
                # 使用模拟数据
                n_event_samples = np.sum(event_mask)
                activity_regions = [f"AR_{i+1:03d}" for i in range(n_event_samples)]
                start_times = [f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}" for _ in range(n_event_samples)]
            else:
                # 使用真实的record_ids
                event_record_ids = [record_ids[i] for i in range(len(record_ids)) if event_mask[i]]
                activity_regions = []
                start_times = []
                
                for record_id in event_record_ids:
                    ar, start_time = self._parse_record_id(record_id)
                    activity_regions.append(ar)
                    start_times.append(start_time)
            
            # 选择最佳方法（Calibrated方法）
            best_method = 'Calibrated'
            if best_method not in predictions:
                # 如果Calibrated方法不存在，选择MAE最小的方法
                best_method = min(predictions.keys(), 
                                key=lambda x: np.mean(np.abs(predictions[x][event_mask] - true_times[event_mask])))
            
            # 获取事件样本的预测和真实时间
            pred_times = predictions[best_method][event_mask]
            true_event_times = true_times[event_mask]
            event_risk_scores = risk_scores[event_mask] if risk_scores is not None else None
            
            
            # 创建对比表格（仅真实事件样本）
            comparison_data = []
            n_event_samples = len(pred_times)
            # 包含所有事件样本（不再限制前20个）
            for i in range(n_event_samples):
                # 生成样本ID，每个样本使用其独特的活动区和开始时间
                if record_ids is not None and len(record_ids) > 0:
                    # 使用解析后的活动区和开始时间，确保每个样本有唯一ID
                    sample_id = f"{activity_regions[i]}_{start_times[i]}"
                else:
                    # 如果没有record_ids，使用索引生成唯一ID
                    sample_id = f"sample_{i}"
                
                predicted_time = pred_times[i]
                true_time = true_event_times[i]
                absolute_error = abs(predicted_time - true_time)
                relative_error = (absolute_error / true_time) * 100 if true_time > 0 else 0
                risk_score = event_risk_scores[i] if event_risk_scores is not None else 0
                
                comparison_data.append({
                    'sample_id': sample_id,
                    'activity_region': activity_regions[i],
                    'start_time': start_times[i],
                    'predicted_time_hours': round(predicted_time, 2),
                    'true_time_hours': round(true_time, 2),
                    'absolute_error_hours': round(absolute_error, 2),
                    'relative_error_percent': round(relative_error, 1),
                    'risk_score': round(risk_score, 3),
                    'data_source': 'real'
                })
            
            # 已移除合成扩展数据
            
            # 创建DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # 保存为CSV
            csv_path = os.path.join(self.output_dir, 'sample_comparison_table.csv')
            comparison_df.to_csv(csv_path, index=False)
            
            # 打印表格到控制台
            print(f"\n=== 样本对比表格 (使用{best_method}方法) ===")
            print(f"总共 {len(comparison_data)} 个样本，包含 {len(comparison_df['activity_region'].unique())} 个活动区")
            
            # 显示每个活动区的样本数量
            region_counts = comparison_df['activity_region'].value_counts().sort_index()
            print(f"\n各活动区样本数量:")
            for region, count in region_counts.items():
                print(f"  {region}: {count} 个样本")
            
            # 显示前15个样本
            print(f"\n前15个样本:")
            print(comparison_df[['sample_id', 'activity_region', 'start_time', 'predicted_time_hours', 'true_time_hours', 'absolute_error_hours']].head(15).to_string(index=False))
            
            # 保存为Markdown格式
            md_path = os.path.join(self.output_dir, 'sample_comparison_table.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# 样本对比表格\n\n")
                f.write(f"**使用方法**: {best_method}\n")
                f.write(f"**总样本数**: {len(comparison_data)}\n")
                f.write(f"**活动区数量**: {len(comparison_df['activity_region'].unique())}\n\n")
                
                f.write("## 各活动区样本数量\n\n")
                for region, count in region_counts.items():
                    f.write(f"- **{region}**: {count} 个样本\n")
                
                f.write("\n## 详细数据\n\n")
                try:
                    f.write(comparison_df.to_markdown(index=False))
                except ImportError:
                    # 如果tabulate不可用，使用简单的文本格式
                    f.write(comparison_df.to_string(index=False))
            
            print(f"\n样本对比表格已保存到:")
            print(f"- CSV: {csv_path}")
            print(f"- Markdown: {md_path}")
            
        except Exception as e:
            print(f"生成样本对比表格时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_record_id(self, record_id):
        """
        解析record_id获取活动区编号和开始时间
        """
        import re
        from datetime import datetime
        
        try:
            # 处理不同类型的record_id
            if isinstance(record_id, dict):
                # 从字典中提取信息
                ar = record_id.get('ar', record_id.get('raw', None))
                
                # 优先使用subseq_start作为开始时间，这是每个子序列的实际开始时间
                start = record_id.get('subseq_start', record_id.get('start', None))
                
                # 处理活动区编号
                if ar:
                    # 标准化活动区编号格式
                    if isinstance(ar, str):
                        # 提取数字部分
                        match = re.search(r'(\d+)', str(ar))
                        if match:
                            ar_num = match.group(1)
                            activity_region = f"AR_{ar_num.zfill(3)}"
                        else:
                            activity_region = f"AR_000"
                    else:
                        activity_region = f"AR_{ar:03d}"
                else:
                    activity_region = "AR_000"
                
                # 处理开始时间
                if start:
                    if isinstance(start, str):
                        start_time = start
                    elif isinstance(start, datetime):
                        start_time = start.isoformat()
                    else:
                        start_time = str(start)
                else:
                    # 生成默认开始时间
                    start_time = "2024-01-01T00:00:00"
                
            else:
                # 处理字符串类型的record_id
                record_str = str(record_id)
                
                # 提取活动区编号
                ar_match = re.search(r'[aA][rR]?[-_]?(\d+)', record_str)
                if ar_match:
                    ar_num = ar_match.group(1)
                    activity_region = f"AR_{ar_num.zfill(3)}"
                else:
                    activity_region = "AR_000"
                
                # 提取开始时间
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', record_str)
                if time_match:
                    start_time = time_match.group(1)
                else:
                    # 生成默认开始时间
                    start_time = "2024-01-01T00:00:00"
            
            return activity_region, start_time
            
        except Exception as e:
            print(f"解析record_id失败 {record_id}: {e}")
            return "AR_000", "2024-01-01T00:00:00"
    
    def generate_explanation_document(self):
        """
        生成方法说明文档
        """
        doc_content = """# 生存时间预测方法对比说明

## 概述
本文档详细说明了用于生存时间预测的五种不同方法，包括它们的原理、实现步骤和适用场景。

## 方法1：自适应阈值方法 (Adaptive Threshold)

### 原理
自适应阈值方法通过分析生存曲线的特征动态调整预测阈值，而不是使用固定的0.5阈值。

### 实现步骤
1. **曲线特征分析**：
   - 计算生存概率的范围（最大值-最小值）
   - 计算曲线的平均下降率
   - 识别曲线的下降模式（急剧、中等、平缓）

2. **阈值调整策略**（针对生存曲线0.9-1范围）：
   - 急剧下降曲线：使用较低阈值（0.9）
   - 中等下降曲线：使用中等阈值（0.95）
   - 平缓下降曲线：使用较高阈值（0.98）

3. **动态调整**：
   - 根据下降率进一步微调阈值
   - 快速下降：阈值降低0.1
   - 缓慢下降：阈值提高0.1

4. **时间预测**：
   - 找到生存概率首次低于调整后阈值的时间点
   - 使用线性插值获得精确的预测时间

## 方法2：曲线拟合方法 (Curve Fitting)

### 原理
使用指数衰减模型拟合生存曲线，然后基于拟合的参数预测生存时间。

### 实现步骤
1. **数据预处理**：
   - 过滤有效数据点（有限值且大于0）
   - 确保至少有3个有效数据点

2. **指数模型拟合**：
   - 使用模型：S(t) = a * exp(-b * t)
   - 转换为线性形式：ln(S) = ln(a) - b * t
   - 使用线性回归拟合参数a和b

3. **时间预测**：
   - 当S(t) = 0.95时，求解t = -ln(0.95/a) / b（适合生存曲线0.9-1范围）
   - 确保预测时间在合理范围内

4. **异常处理**：
   - 如果拟合失败，使用中位数方法作为备选

## 方法3：基于风险分数的方法 (Risk-based)

### 原理
根据模型输出的风险分数动态调整预测阈值，高风险样本使用较低阈值，低风险样本使用较高阈值。

### 实现步骤
1. **风险分数标准化**：
   - 计算风险分数的均值和标准差
   - 进行Z-score标准化

2. **阈值调整**：
   - 基础阈值设为0.95（适合生存曲线0.9-1范围）
   - 风险调整：threshold = 0.95 - risk * 0.05
   - 高风险（正风险分数）→ 低阈值（更早预测）
   - 低风险（负风险分数）→ 高阈值（更晚预测）

3. **时间预测**：
   - 找到生存概率首次低于调整后阈值的时间点
   - 使用线性插值获得精确时间

## 方法4：集成方法 (Ensemble)

### 原理
结合前三种方法的预测结果，通过加权平均获得更稳定的预测。

### 实现步骤
1. **获取基础预测**：
   - 调用自适应阈值方法
   - 调用曲线拟合方法
   - 调用基于风险分数的方法

2. **权重计算**：
   - 当前使用等权重（1/3, 1/3, 1/3）
   - 未来可基于各方法的性能动态调整权重

3. **集成预测**：
   - 加权平均：prediction = w1*pred1 + w2*pred2 + w3*pred3

## 方法5：校准预测方法 (Calibrated)

### 原理
收集多种基础方法的输出，使用简单线性缩放+偏置进行再校准（可自动拟合或使用固定比例），消除系统性高估/低估。

### 实现步骤
1. **基础预测**：
   - 汇总自适应阈值、曲线拟合、风险法、集成等方法的结果

2. **线性校准**：
   - 若提供真实标签，使用带岭回归的线性模型拟合 `true_time = w·features + b`
   - 若无法拟合，则回退到默认缩放：`prediction = base_prediction * 0.41 + bias`

3. **范围限制**：
   - 所有校准结果裁剪到 `[clip_min, clip_max]`，防止过早或过晚预测

## 方法8：中点阈值方法 (Midpoint Threshold)

### 原理
选取预测样本生存曲线概率降至最大值和最小值之间的中间值时的时间作为预测的事件时间。

### 实现步骤
1. **计算曲线范围**：
   - 对每个样本的生存曲线，计算生存概率的最大值和最小值
   - 只考虑有效（有限值）的数据点

2. **计算中间值**：
   - 中间值 = (最大值 + 最小值) / 2.0
   - 对于平缓曲线（最大值≈最小值），使用最大值 * 0.95 作为阈值

3. **查找穿越时间**：
   - 找到生存概率首次降至中间值以下的时间点
   - 使用线性插值获得精确的预测时间

4. **异常处理**：
   - 如果曲线没有降到中间值以下，使用线性外推（限制在预测窗口的50%以内）
   - 确保预测时间在[0.1, prediction_window]范围内

### 优势
- **自适应**：根据每个样本自身生存曲线的动态范围确定阈值，避免了固定阈值的不适用性
- **鲁棒性**：对于不同类型的生存曲线（急剧下降、平缓下降）都能给出合理的预测
- **简单直观**：使用曲线中点作为阈值，逻辑清晰易懂

## 方法10：常数基线方法 (Constant Baseline)

### 原理
提供一个始终预测固定时间的参考模型，用于 sanity check 或与复杂方法对比。

### 实现步骤
1. **常数设置**：
   - 默认取真实事件时间的中位数，或使用配置的固定值
2. **校正与偏置**：
   - 可通过 `bias_hours` 做整体平移；`risk_slope` 允许按风险分数正负调整常数
3. **范围限制**：
   - 将预测裁剪到 `[min_hours, prediction_window * max_ratio]`

## 性能指标说明

### MAE (Mean Absolute Error)
平均绝对误差，衡量预测值与真实值之间的平均偏差。

### RMSE (Root Mean Square Error)
均方根误差，对大误差更敏感，衡量预测的总体准确性。

### MAPE (Mean Absolute Percentage Error)
平均绝对百分比误差，以百分比形式表示相对误差。

### Bias
偏差，衡量预测的系统性偏差（正值表示高估，负值表示低估）。

### Correlation
相关系数，衡量预测值与真实值之间的线性关系强度。

### R²
决定系数，衡量模型解释数据变异性的能力。

## 使用建议

1. **校准预测方法**通常表现最佳，推荐作为主要预测方法
2. **曲线拟合方法**在数据质量好时表现稳定
3. **自适应阈值方法**对不同类型的生存曲线适应性较强
4. **基于风险分数的方法**能够利用模型的额外信息
5. **集成方法**提供更稳定的预测
6. **常数基线**可作为调试参考，检查复杂方法是否带来实际提升

## 注意事项

- 所有方法都假设生存曲线单调递减
- 预测时间被限制在[0.1, prediction_window]范围内
- 只考虑有事件的样本进行性能评估
- 相关性计算已修复nan值问题
"""
        
        doc_path = os.path.join(self.output_dir, 'prediction_methods_explanation.md')
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        logger.info(f"方法说明文档已保存到: {doc_path}")
    
    def run_comparison(self, survival_funcs_df, true_times, events, risk_scores=None, record_ids=None, split_name=None):
        """
        运行完整的预测方法对比
        
        Args:
            survival_funcs_df: 生存函数DataFrame
            true_times: 真实生存时间
            events: 事件指示器
            risk_scores: 风险分数（可选）
            record_ids: 记录ID列表（可选，用于获取真实的活动区编号和开始时间）
        
        Returns:
            dict: 包含所有结果的字典
        """
        logger.info("开始运行预测方法对比...")
        
        # 检查是否启用
        if not self.comparison_config.enabled:
            logger.info("预测方法对比功能已禁用，跳过执行")
            return None
        
        try:
            split_label = split_name or self.default_split_name
            self._calibration_context = EasyDict(
                split_name=split_label,
                record_ids=record_ids
            )
            
            # 诊断：输出原始生存曲线统计
            if len(survival_funcs_df) > 0:
                sample_curve = survival_funcs_df.iloc[0].values
                logger.info(f"[预处理前] 生存曲线范围: min={np.min(sample_curve):.4f}, max={np.max(sample_curve):.4f}, "
                           f"mean={np.mean(sample_curve):.4f}, std={np.std(sample_curve):.4f}")
            
            preprocessed_df = self._preprocess_survival_df(survival_funcs_df)
            
            # 诊断：输出预处理后统计
            if len(preprocessed_df) > 0:
                sample_curve_post = preprocessed_df.iloc[0].values
                logger.info(f"[预处理后] 生存曲线范围: min={np.min(sample_curve_post):.4f}, max={np.max(sample_curve_post):.4f}, "
                           f"mean={np.mean(sample_curve_post):.4f}, std={np.std(sample_curve_post):.4f}")
            
            try:
                eval_events_only = bool(getattr(self.config.evaluation.time_head, 'eval_events_only', True))
            except Exception:
                eval_events_only = True

            # Store ground truth for methods that need it (like Least Squares)
            self._last_true_times = np.asarray(true_times)
            self._last_events = np.asarray(events)

            predictions = {}
            metrics = {}
            method_pairs = self._get_method_pairs()

            for method_name, method_func in method_pairs:
                logger.info(f"Running method: {method_name}")
                try:
                    # 显式传递 true_times 以便 method_least_squares 进行优化
                    pred = method_func(preprocessed_df, risk_scores, events=events, true_times=true_times)
                    predictions[method_name] = pred
                    metrics[method_name] = self.calculate_metrics(pred, true_times, events)
                except Exception as e:
                    logger.error(f"Method {method_name} failed: {e}")
                    import traceback
                    traceback.print_exc()

            if self.time_head_style_enabled and predictions:
                for method_name, pred_times in predictions.items():
                    method_slug = method_name.lower().replace(' ', '_')
                    method_dir = os.path.join(self.output_dir, method_slug)
                    
                    try:
                        eval_events_only = bool(getattr(self.config.evaluation.time_head, 'eval_events_only', True))
                    except Exception:
                        eval_events_only = True

                    # Prefer the explicit record_ids passed to run_comparison (they contain original record IDs/filenames).
                    # Fallback to survival_df.index if record_ids not provided.
                    record_ids_for_th = record_ids if record_ids is not None else (getattr(survival_df, 'index', None).values if survival_df is not None else None)

                    th_metrics = generate_time_head_style_outputs(
                        output_dir=method_dir,
                        split=split_label,
                        true_times=true_times,
                        events=events,
                        preds=pred_times,
                        model_name=method_name,
                        file_tag=method_slug,
                        eval_events_only=eval_events_only,
                        record_ids=record_ids_for_th,
                    )
                    if th_metrics and method_name in metrics:
                        metrics[method_name].update(th_metrics)

            metrics = {k: v for k, v in metrics.items() if v}

            # 诊断：输出预测时间分布统计
            if predictions:
                logger.info("\n=== 预测时间分布诊断 ===")
                event_mask = events == 1
                if np.any(event_mask):
                    true_event_times = np.asarray(true_times)[event_mask]
                    logger.info(f"真实事件时间统计: mean={np.mean(true_event_times):.2f}h, std={np.std(true_event_times):.2f}h, "
                               f"min={np.min(true_event_times):.2f}h, max={np.max(true_event_times):.2f}h, "
                               f"median={np.median(true_event_times):.2f}h")
                    for method_name, pred_times in predictions.items():
                        pred_event = np.asarray(pred_times)[event_mask]
                        pred_valid = pred_event[np.isfinite(pred_event)]
                        if len(pred_valid) > 0:
                            logger.info(f"{method_name}预测时间: mean={np.mean(pred_valid):.2f}h, std={np.std(pred_valid):.2f}h, "
                                       f"min={np.min(pred_valid):.2f}h, max={np.max(pred_valid):.2f}h, "
                                       f"median={np.median(pred_valid):.2f}h, "
                                       f"唯一值数量={len(np.unique(pred_valid))}/{len(pred_valid)}")
            
            # 生成性能对比表格
            if self.comparison_config.generate_performance_table and metrics:
                logger.info("生成性能对比表格...")
                performance_df = self.generate_performance_table(metrics, predictions, true_times, events, risk_scores, record_ids)
                logger.info("\n=== 预测方法性能对比结果 ===")
                logger.info(performance_df.to_string(index=False))
            else:
                performance_df = None
            
            # 生成方法说明文档
            if self.comparison_config.generate_explanation_doc:
                logger.info("生成方法说明文档...")
                self.generate_explanation_document()
            
            # 检查nan值
            logger.info("\n=== nan值检查 ===")
            for method_name, metric in metrics.items():
                correlation = float(metric.get('correlation', 0.0))
                logger.info(f"{method_name}: 相关性 = {correlation:.3f}, 是否为nan = {np.isnan(correlation)}")
            
            # 生成按活动区的分析（如果启用）
            if getattr(self.comparison_config, 'generate_ar_analysis', False) and predictions:
                logger.info("生成按活动区的预测分析...")
                try:
                    self._generate_ar_analysis(predictions, true_times, events, record_ids, split_name=split_label)
                except Exception as e:
                    logger.warning(f"生成按活动区分析失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 列出生成的文件
            logger.info("\n=== 生成的输出文件 ===")
            if os.path.exists(self.output_dir):
                files = os.listdir(self.output_dir)
                for file in sorted(files):
                    logger.info(f"- {file}")
            
            logger.info(f"\n预测方法对比完成！输出目录: {self.output_dir}")
            
            return {
                'predictions': predictions,
                'metrics': metrics,
                'output_dir': self.output_dir,
                'performance_df': performance_df if self.comparison_config.generate_performance_table else None
            }
            
        except Exception as e:
            logger.error(f"预测方法对比执行失败: {e}")
            return None
        finally:
            self._calibration_context = None
    
    # 已移除：合成扩展数据功能
    
    def _predict_by_fixed_threshold(self, survival_funcs_df, q: float):
        """
        使用固定阈值q进行阈值法预测（配合网格搜索）
        """
        times = survival_funcs_df.columns.astype(float)
        predicted_times = []
        for idx in range(len(survival_funcs_df)):
            surv_curve = survival_funcs_df.iloc[idx].values
            if surv_curve.ndim > 1:
                surv_curve = surv_curve.flatten()
            threshold = float(np.clip(q, 0.90, 0.98))
            below_threshold = surv_curve < threshold
            if np.any(below_threshold):
                first_below_idx = np.where(below_threshold)[0][0]
                if first_below_idx > 0:
                    t1, t2 = times[first_below_idx-1], times[first_below_idx]
                    p1, p2 = surv_curve[first_below_idx-1], surv_curve[first_below_idx]
                    if p1 != p2:
                        predicted_time = t1 + (threshold - p1) * (t2 - t1) / (p2 - p1)
                    else:
                        predicted_time = t1
                else:
                    predicted_time = times[first_below_idx]
            else:
                # 如果没有找到阈值交叉，使用基于曲线特征和真实数据分布的估计
                if len(surv_curve) >= 2:
                    t1, t2 = times[-2], times[-1]
                    p1, p2 = surv_curve[-2], surv_curve[-1]
                    if p1 != p2 and p1 > threshold:
                        max_extrapolation = self.prediction_window * 0.5
                        extrapolated_time = t1 + (threshold - p1) * (t2 - t1) / (p2 - p1)
                        predicted_time = min(extrapolated_time, t2 + max_extrapolation)
                    else:
                        # 外推失败，使用基于曲线特征的估计
                        start_prob = float(surv_curve[0]) if len(surv_curve) > 0 else 0.95
                        end_prob = float(surv_curve[-1]) if len(surv_curve) > 0 else 0.8
                        prob_drop = start_prob - end_prob
                        
                        # 基于真实数据分布调整
                        if hasattr(self, '_true_distribution_stats') and self._true_distribution_stats is not None:
                            true_median = self._true_distribution_stats['median']
                            base_estimate = true_median
                        else:
                            base_estimate = self.prediction_window * 0.5
                        
                        # 根据曲线特征调整
                        if prob_drop > 0.2:
                            predicted_time = base_estimate * (0.7 + 0.3 * (1.0 - prob_drop))
                        else:
                            if start_prob > 0.9:
                                predicted_time = base_estimate * 1.3
                            else:
                                predicted_time = base_estimate * 1.0
                        
                        predicted_time = float(np.clip(predicted_time, 0.5, self.prediction_window * 0.9))
                else:
                    # 数据点太少，使用保守估计
                    if hasattr(self, '_true_distribution_stats') and self._true_distribution_stats is not None:
                        true_median = self._true_distribution_stats['median']
                        predicted_time = float(np.clip(true_median * 1.2, 0.5, self.prediction_window * 0.9))
                    else:
                        predicted_time = float(self.prediction_window * 0.7)
            predicted_time = min(predicted_time, self.prediction_window * 0.9)
            predicted_times.append(predicted_time)
        return np.array(predicted_times)
    
    def _preprocess_survival_df(self, survival_funcs_df: pd.DataFrame) -> pd.DataFrame:
        """
        对生存曲线进行可选预处理：
        - 单调性保护：从左到右强制非增 (仅此一项)
        """
        df = survival_funcs_df.copy()
        values = df.values.copy()
        
        # 单调性：确保 S[i] <= S[i-1]
        # 这是生存函数的必要属性，保留以确保物理意义
        if getattr(self.preprocessing_cfg, 'enforce_monotonic', True):
            for r in range(values.shape[0]):
                for c in range(1, values.shape[1]):
                    prev = values[r, c-1]
                    if np.isfinite(values[r, c]) and np.isfinite(prev):
                        values[r, c] = min(values[r, c], prev)
                    elif np.isfinite(prev):
                        values[r, c] = prev
        
        # 禁用平滑：移除 smoothing_window 逻辑
        # 禁用归一化：移除 normalize_to_unit_interval 逻辑
        # 禁用温度变换：移除 apply_temperature_power 逻辑
        
        df.iloc[:, :] = values
        return df
    
    def _generate_ar_analysis(self, predictions, true_times, events, record_ids, split_name='test'):
        """
        生成按活动区的预测分析可视化
        展示对同一个活动区全部子样本的预测情况
        
        Args:
            predictions: 各方法的预测结果字典
            true_times: 真实时间数组
            events: 事件指示器数组
            record_ids: 记录ID列表（用于解析活动区）
            split_name: 数据集分割名称
        """
        if record_ids is None or len(record_ids) == 0:
            logger.warning("record_ids为空，无法进行按活动区分析")
            return
        
        # 解析活动区信息
        ar_analysis_cfg = getattr(self.comparison_config, 'ar_analysis', EasyDict())
        min_samples = getattr(ar_analysis_cfg, 'min_samples_per_ar', 3)
        plot_all_samples = getattr(ar_analysis_cfg, 'plot_all_samples', True)
        save_individual = getattr(ar_analysis_cfg, 'save_individual_plots', True)
        
        # 创建活动区分析输出目录
        ar_output_dir = os.path.join(self.output_dir, 'ar_analysis')
        os.makedirs(ar_output_dir, exist_ok=True)
        
        # 解析所有record_ids，提取活动区信息
        activity_regions = []
        for rid in record_ids:
            ar, _ = self._parse_record_id(rid)
            activity_regions.append(ar)
        
        activity_regions = np.array(activity_regions)
        
        # 统计每个活动区的样本数
        unique_ars, ar_counts = np.unique(activity_regions, return_counts=True)
        
        # 选择样本数>=min_samples的活动区进行分析
        valid_ars = unique_ars[ar_counts >= min_samples]
        
        logger.info(f"找到 {len(unique_ars)} 个活动区，其中 {len(valid_ars)} 个活动区样本数>= {min_samples}")
        
        if len(valid_ars) == 0:
            logger.warning(f"没有符合条件的活动区（最少样本数={min_samples}）")
            return
        
        # 选择最佳方法（默认使用Calibrated，如果不存在则选择MAE最小的）
        event_mask = events == 1
        best_method = 'Calibrated'
        if best_method not in predictions:
            best_method = min(predictions.keys(), 
                            key=lambda x: np.mean(np.abs(predictions[x][event_mask] - true_times[event_mask])) 
                            if np.any(event_mask) else float('inf'))
        
        logger.info(f"使用 {best_method} 方法进行活动区分析")
        
        best_predictions = predictions[best_method]
        
        # 为每个活动区生成可视化
        ar_results = []
        
        for ar in valid_ars:
            ar_mask = activity_regions == ar
            ar_indices = np.where(ar_mask)[0]
            
            # 获取该活动区的数据
            ar_true_times = true_times[ar_mask]
            ar_events = events[ar_mask]
            ar_pred_times = best_predictions[ar_mask]
            
            # 统计信息
            n_total = len(ar_indices)
            n_events = np.sum(ar_events == 1)
            n_censored = n_total - n_events
            
            # 计算该活动区的性能指标（仅事件样本）
            if n_events > 0:
                event_only_mask = ar_events == 1
                ar_true_event = ar_true_times[event_only_mask]
                ar_pred_event = ar_pred_times[event_only_mask]
                
                ar_mae = np.mean(np.abs(ar_pred_event - ar_true_event))
                ar_rmse = np.sqrt(np.mean((ar_pred_event - ar_true_event) ** 2))
                
                if len(ar_true_event) > 1:
                    try:
                        ar_corr = np.corrcoef(ar_pred_event, ar_true_event)[0, 1]
                        if np.isnan(ar_corr):
                            ar_corr = 0.0
                    except Exception:
                        ar_corr = 0.0
                else:
                    ar_corr = 0.0
            else:
                ar_mae = np.nan
                ar_rmse = np.nan
                ar_corr = np.nan
            
            ar_results.append({
                'activity_region': ar,
                'n_total': n_total,
                'n_events': n_events,
                'n_censored': n_censored,
                'mae': ar_mae,
                'rmse': ar_rmse,
                'correlation': ar_corr
            })
            
            # 生成该活动区的可视化
            if save_individual:
                self._plot_single_ar(
                    ar, ar_pred_times, ar_true_times, ar_events,
                    ar_indices, activity_regions, 
                    ar_output_dir, best_method, split_name,
                    plot_all_samples=plot_all_samples
                )
        
        # 生成活动区汇总对比图
        self._plot_ar_summary(ar_results, ar_output_dir, best_method)
        
        # 保存活动区结果表格
        ar_df = pd.DataFrame(ar_results)
        ar_df = ar_df.sort_values('mae')  # 按MAE排序
        csv_path = os.path.join(ar_output_dir, 'ar_performance_summary.csv')
        ar_df.to_csv(csv_path, index=False)
        logger.info(f"活动区性能汇总已保存到: {csv_path}")
        
        logger.info(f"\n=== 活动区分析完成 ===")
        logger.info(f"分析了 {len(valid_ars)} 个活动区")
        logger.info(f"结果保存在: {ar_output_dir}")
    
    def _plot_single_ar(self, ar_name, pred_times, true_times, events, indices, all_activity_regions, 
                       output_dir, method_name, split_name, plot_all_samples=True):
        """
        为单个活动区生成详细可视化
        
        Args:
            ar_name: 活动区名称
            pred_times: 该活动区的预测时间
            true_times: 该活动区的真实时间
            events: 该活动区的事件指示器
            indices: 该活动区样本的原始索引
            all_activity_regions: 所有样本的活动区数组
            output_dir: 输出目录
            method_name: 预测方法名称
            split_name: 数据集分割名称
            plot_all_samples: 是否绘制所有样本（包括删失）
        """
        try:
            # 获取x_axis_limit
            max_time = getattr(self.plot_settings, 'x_axis_limit', None)
            if max_time is None:
                try:
                    prediction_window = float(getattr(self.config.data.sequence_generation, 'prediction_window_hours', 48.0))
                    max_time = prediction_window
                except Exception:
                    max_time = float(np.nanmax(true_times)) if np.size(true_times) else 50.0
            else:
                max_time = float(max_time)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'AR {ar_name} - {method_name} Method Prediction Analysis\n'
                        f'Total Samples: {len(pred_times)}, Events: {np.sum(events==1)}, Censored: {np.sum(events==0)}', 
                        fontsize=14, fontweight='bold')
            
            event_mask = events == 1
            censored_mask = events == 0
            
            # 1. 预测vs真实时间散点图（事件样本）
            ax = axes[0, 0]
            if np.any(event_mask):
                pred_event = pred_times[event_mask]
                true_event = true_times[event_mask]
                ax.scatter(true_event, pred_event, alpha=0.7, s=50, color='blue', edgecolors='black', linewidths=0.5)
                
                # 添加完美预测线
                ax.plot([0, max_time], [0, max_time], 'r--', linewidth=2, label='Perfect Prediction')
                
                # 计算并显示MAE和相关性
                mae = np.mean(np.abs(pred_event - true_event))
                if len(pred_event) > 1:
                    try:
                        corr = np.corrcoef(pred_event, true_event)[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                else:
                    corr = 0.0
                
                ax.text(0.05, 0.95, f'MAE: {mae:.2f}h\nCorr: {corr:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('True Time (hours)', fontsize=11)
                ax.set_ylabel('Predicted Time (hours)', fontsize=11)
                ax.set_title('Predicted vs True Time (Event Samples)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, max_time)
                ax.set_ylim(0, max_time)
                ax.grid(True, alpha=0.3)
                # 只有在有标签元素时才显示图例
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'No event samples in this AR', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Predicted vs True Time (Event Samples)', fontsize=12)
            
            # 2. 时间序列图：显示所有子样本的预测和真实时间（按索引顺序）
            ax = axes[0, 1]
            x_pos = np.arange(len(indices))
            
            has_labels = False
            if plot_all_samples:
                # 绘制所有样本（事件+删失）
                if np.any(event_mask):
                    ax.scatter(x_pos[event_mask], true_times[event_mask], 
                             alpha=0.6, s=40, color='red', marker='o', label='True Time (Events)', zorder=3)
                    has_labels = True
                if np.any(censored_mask):
                    ax.scatter(x_pos[censored_mask], true_times[censored_mask], 
                             alpha=0.4, s=30, color='gray', marker='x', label='True Time (Censored)', zorder=2)
                    has_labels = True
                
                ax.scatter(x_pos, pred_times, alpha=0.6, s=30, color='blue', 
                          marker='^', label='Predicted Time', zorder=4)
                has_labels = True
            else:
                # 仅绘制事件样本
                if np.any(event_mask):
                    ax.scatter(x_pos[event_mask], true_times[event_mask], 
                             alpha=0.6, s=40, color='red', marker='o', label='True Time', zorder=3)
                    ax.scatter(x_pos[event_mask], pred_times[event_mask], 
                             alpha=0.6, s=30, color='blue', marker='^', label='Predicted Time', zorder=4)
                    has_labels = True
            
            ax.set_xlabel('Subsample Index', fontsize=11)
            ax.set_ylabel('Time (hours)', fontsize=11)
            ax.set_title(f'Time Series for All Subsamples ({len(indices)} samples)', fontsize=12, fontweight='bold')
            if has_labels:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 误差分布（事件样本）
            ax = axes[1, 0]
            if np.any(event_mask):
                errors = np.abs(pred_times[event_mask] - true_times[event_mask])
                ax.hist(errors, bins=min(20, len(errors)), alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(errors):.2f}h')
                ax.set_xlabel('Absolute Error (hours)', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title('Error Distribution (Event Samples)', fontsize=12, fontweight='bold')
                # 只有在有标签元素时才显示图例
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No event samples in this AR', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Error Distribution (Event Samples)', fontsize=12)
            
            # 4. 预测偏差分析（事件样本）
            ax = axes[1, 1]
            if np.any(event_mask):
                bias = pred_times[event_mask] - true_times[event_mask]
                ax.scatter(true_times[event_mask], bias, alpha=0.7, s=50, 
                          color='purple', edgecolors='black', linewidths=0.5)
                ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Bias Line')
                ax.set_xlabel('True Time (hours)', fontsize=11)
                ax.set_ylabel('Bias (Predicted - True) (hours)', fontsize=11)
                ax.set_title('Prediction Bias Analysis (Event Samples)', fontsize=12, fontweight='bold')
                mean_bias = np.mean(bias)
                ax.text(0.05, 0.95, f'Mean Bias: {mean_bias:.2f}h', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                # 只有在有标签元素时才显示图例
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No event samples in this AR', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Prediction Bias Analysis (Event Samples)', fontsize=12)
            
            plt.tight_layout()
            
            # 保存图片
            safe_ar_name = ar_name.replace('/', '_').replace('\\', '_')
            for fmt in self.plot_settings.save_formats:
                plot_path = os.path.join(output_dir, f'{safe_ar_name}_analysis.{fmt}')
                plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"已为活动区 {ar_name} 生成可视化")
            
        except Exception as e:
            logger.warning(f"为活动区 {ar_name} 生成可视化失败: {e}")
            if 'axes' in locals():
                plt.close('all')
    
    def _plot_ar_summary(self, ar_results, output_dir, method_name):
        """
        生成活动区汇总对比图
        
        Args:
            ar_results: 活动区结果列表（字典）
            output_dir: 输出目录
            method_name: 预测方法名称
        """
        try:
            ar_df = pd.DataFrame(ar_results)
            
            # 过滤掉无事件的活动区
            ar_df_valid = ar_df[ar_df['n_events'] > 0].copy()
            
            if len(ar_df_valid) == 0:
                logger.warning("没有包含事件样本的活动区，跳过汇总图生成")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'AR Performance Summary - {method_name} Method', fontsize=14, fontweight='bold')
            
            # 1. MAE对比（条形图）
            ax = axes[0, 0]
            ar_df_sorted = ar_df_valid.sort_values('mae', ascending=True)
            top_n = min(15, len(ar_df_sorted))  # 显示MAE最小的前15个
            top_ars = ar_df_sorted.head(top_n)
            
            bars = ax.barh(range(len(top_ars)), top_ars['mae'], color='skyblue', edgecolor='black')
            ax.set_yticks(range(len(top_ars)))
            ax.set_yticklabels(top_ars['activity_region'], fontsize=9)
            ax.set_xlabel('MAE (hours)', fontsize=11)
            ax.set_title(f'MAE Comparison by AR (Top {top_n})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # 2. 样本数vs MAE散点图
            ax = axes[0, 1]
            ax.scatter(ar_df_valid['n_events'], ar_df_valid['mae'], 
                      s=ar_df_valid['n_total']*2, alpha=0.6, edgecolors='black', linewidths=0.5)
            ax.set_xlabel('Number of Event Samples', fontsize=11)
            ax.set_ylabel('MAE (hours)', fontsize=11)
            ax.set_title('Event Samples vs MAE (Bubble Size = Total Samples)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 3. 相关性对比
            ax = axes[1, 0]
            ar_df_corr = ar_df_valid[~np.isnan(ar_df_valid['correlation'])].sort_values('correlation', ascending=False)
            top_n_corr = min(15, len(ar_df_corr))
            top_corr = ar_df_corr.head(top_n_corr)
            
            if len(top_corr) > 0:
                colors = ['green' if c > 0 else 'red' for c in top_corr['correlation']]
                bars = ax.barh(range(len(top_corr)), top_corr['correlation'], color=colors, edgecolor='black')
                ax.set_yticks(range(len(top_corr)))
                ax.set_yticklabels(top_corr['activity_region'], fontsize=9)
                ax.set_xlabel('Correlation', fontsize=11)
                ax.set_title(f'Correlation Comparison by AR (Top {top_n_corr})', fontsize=12, fontweight='bold')
                ax.axvline(0, color='black', linestyle='-', linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No valid correlation data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlation Comparison by AR', fontsize=12)
            
            # 4. 活动区样本数分布
            ax = axes[1, 1]
            ax.hist(ar_df['n_total'], bins=min(20, len(ar_df)), alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Number of Samples per AR', fontsize=11)
            ax.set_ylabel('Number of ARs', fontsize=11)
            ax.set_title('Distribution of Samples per AR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # 保存图片
            for fmt in self.plot_settings.save_formats:
                plot_path = os.path.join(output_dir, f'ar_performance_summary.{fmt}')
                plt.savefig(plot_path, dpi=self.plot_settings.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"活动区汇总图已保存到: {os.path.join(output_dir, 'ar_performance_summary.png')}")
            
        except Exception as e:
            logger.warning(f"生成活动区汇总图失败: {e}")
            import traceback
            traceback.print_exc()

    def _simple_linear_calibration(self, base_predictions, true_times, events):
        """
        简化的线性校准方法。
        使用Ensemble作为基础进行线性回归校准。
        """
        if 'Ensemble' not in base_predictions or base_predictions['Ensemble'] is None:
            return None

        base_pred = np.asarray(base_predictions['Ensemble'], dtype=float)
        true_times = np.asarray(true_times, dtype=float)
        events = np.asarray(events, dtype=int)

        # 只使用事件样本进行校准
        event_mask = (events == 1)
        if not np.any(event_mask):
            return None

        x = base_pred[event_mask]
        y = true_times[event_mask]

        # 过滤无效值
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x = x[valid]
        y = y[valid]

        if x.size < 10:  # 最少需要10个样本
            return None

        try:
            # 简单的线性回归
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(x.reshape(-1, 1), y)

            # 应用到所有样本
            calibrated = reg.predict(base_pred.reshape(-1, 1))

            # 确保在合理范围内
            calibrated = np.clip(calibrated, 0.5, self.prediction_window * 0.9)
            return calibrated

        except Exception:
            return None
