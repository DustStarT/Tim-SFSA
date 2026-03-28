"""
模型评估指标
包含生存分析中常用的评估指标，如Concordance Index (C-index)。
"""
import numpy as np
from lifelines.utils import concordance_index
from sksurv import metrics
from sksurv.util import Surv
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import logging
from scipy import interpolate
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False
from evaluation.plotting import _ensure_durations_in_hours

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False

def c_index(predictions, event_times, event_indicators):
    """
    计算Concordance Index (C-index)。

    Args:
        predictions (np.array): 模型的预测输出。
                                对于风险模型，值越高表示风险越大。
        event_times (np.array): 事件或删失的真实时间。
        event_indicators (np.array): 事件指示器 (1=事件, 0=删失)。

    Returns:
        float: C-index值。
    """
    # 数据验证
    if len(predictions) != len(event_times) or len(predictions) != len(event_indicators):
        raise ValueError(f"输入长度不匹配: predictions={len(predictions)}, times={len(event_times)}, events={len(event_indicators)}")
    
    # 检查是否有有效的事件
    if np.sum(event_indicators) == 0:
        logging.warning("没有事件发生，C-index无法计算")
        return 0.5
    
    if np.sum(event_indicators) == len(event_indicators):
        logging.warning("所有样本都发生事件，C-index无法计算")
        return 0.5
    
    # 检查是否有NaN或inf值
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        logging.warning("预测值包含NaN或inf，将被过滤")
        valid_mask = ~(np.isnan(predictions) | np.isinf(predictions))
        if np.sum(valid_mask) < 2:
            logging.error("过滤后有效样本不足")
            return 0.5
        predictions = predictions[valid_mask]
        event_times = event_times[valid_mask]
        event_indicators = event_indicators[valid_mask]
    
    # === 修复：与训练时保持一致的C-index计算 ===
    # 计算正向和反向C-index，选择较大者（与训练时逻辑一致）
    try:
        cidx_pos = concordance_index(event_times, predictions, event_indicators)
        cidx_neg = concordance_index(event_times, -predictions, event_indicators)
        
        if np.isnan(cidx_pos) or np.isnan(cidx_neg):
            logging.warning("C-index计算结果为NaN")
            return 0.5
            
        # 自动选择较大者作为主C-index（与训练时逻辑一致）
        c_idx = max(cidx_pos, cidx_neg)
        
        # 记录调试信息
        logging.info(f"C-index计算: 正向={cidx_pos:.4f}, 反向={cidx_neg:.4f}, 选用={c_idx:.4f}")
        
        return c_idx
        
    except ZeroDivisionError:
        logging.warning("在C-index计算中没有可比较的样本对。")
        return 0.5
    except Exception as e:
        logging.error(f"C-index计算出错: {e}")
        return 0.5


def preprocess_risk_scores_for_cindex(risk_scores, event_times, event_indicators, iqr_multiplier=3):
    """
    统一训练/评估端用于C-index的风险分数预处理：
    - flatten and align lengths
    - filter NaN/inf
    - IQR-based outlier removal (multiplier 可配置)
    - 标准化（zero mean, unit std）

    返回 (normalized_scores, times_filtered, events_filtered, mask, mean, std)
    mask 是相对于输入向量的布尔索引，标记保留的条目。
    如果过滤后无样本，会返回 (np.array([]), ..., mask)
    """
    preds = np.asarray(risk_scores).flatten()
    times = np.asarray(event_times).flatten()
    events = np.asarray(event_indicators).flatten()

    min_len = min(len(preds), len(times), len(events))
    preds = preds[:min_len]
    times = times[:min_len]
    events = events[:min_len]

    valid_mask = ~(np.isnan(preds) | np.isinf(preds) | np.isnan(times) | np.isnan(events))
    if np.sum(valid_mask) == 0:
        return np.array([]), np.array([]), np.array([]), valid_mask, None, None

    temp = preds[valid_mask]
    if temp.size == 0:
        return np.array([]), np.array([]), np.array([]), valid_mask, None, None

    q1, q3 = np.percentile(temp, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    inlier_mask = (temp >= lower) & (temp <= upper)

    # build full mask aligned to original arrays
    full_mask = np.zeros_like(valid_mask, dtype=bool)
    valid_idxs = np.where(valid_mask)[0]
    if np.any(inlier_mask):
        full_mask[valid_idxs[inlier_mask]] = True
    else:
        # 如果IQR过滤没有保留任何样本，则退回到仅移除NaN/Inf的有效样本，而不是返回空数组。
        logging.warning("IQR过滤后没有保留样本；退回到仅移除NaN/Inf的有效样本以避免计算失败。")
        full_mask[valid_idxs] = True

    preds_f = preds[full_mask]
    times_f = times[full_mask]
    events_f = events[full_mask]

    mean = np.mean(preds_f)
    std = np.std(preds_f)
    if std > 1e-8:
        normalized = (preds_f - mean) / std
    else:
        normalized = preds_f

    return normalized, times_f, events_f, full_mask, mean, std

def calculate_roc_auc_at_time(predictions, event_times, event_indicators, time_threshold):
    """
    计算在特定时间点的ROC曲线和AUC值。
    
    Args:
        predictions (np.array): 模型预测的风险分数
        event_times (np.array): 事件时间
        event_indicators (np.array): 事件指示器
        time_threshold (float): 时间阈值
        
    Returns:
        tuple: (fpr, tpr, auc_score, threshold)
    """
    try:
        # 创建二分类标签：在时间阈值前发生事件的为1，否则为0
        binary_labels = (event_times <= time_threshold) & (event_indicators == 1)
        
        # 检查是否有足够的正负样本
        if np.sum(binary_labels) == 0:
            logging.warning(f"在时间阈值 {time_threshold} 前没有事件发生")
            return None, None, 0.5, None
        if np.sum(binary_labels) == len(binary_labels):
            logging.warning(f"在时间阈值 {time_threshold} 前所有样本都发生事件")
            return None, None, 0.5, None
            
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(binary_labels, predictions)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score, thresholds
        
    except Exception as e:
        logging.error(f"计算ROC AUC时出错: {e}")
        return None, None, 0.5, None

def calculate_time_dependent_roc(predictions, event_times, event_indicators, time_points=None):
    """
    计算时间相关的ROC曲线。
    
    Args:
        predictions (np.array): 模型预测的风险分数
        event_times (np.array): 事件时间
        event_indicators (np.array): 事件指示器
        time_points (list): 时间点列表，如果为None则自动生成
        
    Returns:
        dict: 包含各时间点的ROC信息
    """
    try:
        if time_points is None:
            # 自动生成时间点：使用事件时间的分位数
            event_times_valid = event_times[event_indicators == 1]
            if len(event_times_valid) > 0:
                time_points = np.percentile(event_times_valid, [25, 50, 75])
            else:
                time_points = np.percentile(event_times, [25, 50, 75])
        
        roc_results = {}
        for t in time_points:
            fpr, tpr, auc_score, thresholds = calculate_roc_auc_at_time(
                predictions, event_times, event_indicators, t
            )
            roc_results[t] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc_score,
                'thresholds': thresholds
            }
            
        return roc_results
        
    except Exception as e:
        logging.error(f"计算时间相关ROC时出错: {e}")
        return {}

def calculate_auc_at_multiple_times(predictions, event_times, event_indicators, time_points=None):
    """
    计算多个时间点的AUC值。
    
    Args:
        predictions (np.array): 模型预测的风险分数
        event_times (np.array): 事件时间
        event_indicators (np.array): 事件指示器
        time_points (list): 时间点列表
        
    Returns:
        dict: 时间点到AUC值的映射
    """
    try:
        if time_points is None:
            # 使用事件时间的分位数作为时间点
            event_times_valid = event_times[event_indicators == 1]
            if len(event_times_valid) > 0:
                time_points = np.percentile(event_times_valid, np.arange(10, 100, 10))
            else:
                time_points = np.percentile(event_times, np.arange(10, 100, 10))
        
        auc_scores = {}
        for t in time_points:
            _, _, auc_score, _ = calculate_roc_auc_at_time(
                predictions, event_times, event_indicators, t
            )
            auc_scores[t] = auc_score
            
        return auc_scores
        
    except Exception as e:
        logging.error(f"计算多时间点AUC时出错: {e}")
        return {}

def calculate_integrated_auc(predictions, event_times, event_indicators, time_points=None):
    """
    计算积分AUC (iAUC)。
    
    Args:
        predictions (np.array): 模型预测的风险分数
        event_times (np.array): 事件时间
        event_indicators (np.array): 事件指示器
        time_points (list): 时间点列表
        
    Returns:
        float: 积分AUC值
    """
    try:
        auc_scores = calculate_auc_at_multiple_times(
            predictions, event_times, event_indicators, time_points
        )
        
        if not auc_scores:
            return 0.5
            
        # 计算积分AUC
        times = sorted(auc_scores.keys())
        aucs = [auc_scores[t] for t in times]
        
        # 使用梯形法则计算积分
        iauc = np.trapz(aucs, times)
        
        # 归一化到[0,1]区间
        if len(times) > 1:
            iauc = iauc / (times[-1] - times[0])
        else:
            iauc = aucs[0] if aucs else 0.5
            
        return iauc
        
    except Exception as e:
        logging.error(f"计算积分AUC时出错: {e}")
        return 0.5

def brier_score(outcomes_train, outcomes_test, predictions, times):
    """计算Brier Score"""
    try:
        # 转换为sksurv格式
        y_train = Surv.from_dataframe('event', 'time', outcomes_train)
        y_test = Surv.from_dataframe('event', 'time', outcomes_test)
        # 修正：确保predictions为numpy数组且shape为(n_samples, n_times)
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        if hasattr(times, 'values'):
            times = times.values
        times = np.asarray(times).flatten()
        # 计算Brier Score
        return metrics.brier_score(y_train, y_test, predictions, times)[-1]
    except Exception as e:
        logging.error(f"计算Brier Score时出错: {str(e)}")
        return np.nan

def integrated_brier_score(outcomes_train, outcomes_test, predictions, times):
    """计算Integrated Brier Score"""
    try:
        # 转换为sksurv格式
        y_train = Surv.from_dataframe('event', 'time', outcomes_train)
        y_test = Surv.from_dataframe('event', 'time', outcomes_test)
        
        # 计算IBS
        return metrics.integrated_brier_score(y_train, y_test, predictions, times)
    except Exception as e:
        logging.error(f"计算Integrated Brier Score时出错: {str(e)}")
        return np.nan

def dynamic_auc(outcomes_train, outcomes_test, predictions, times):
    """计算动态AUC"""
    try:
        # 转换为sksurv格式
        y_train = Surv.from_dataframe('event', 'time', outcomes_train)
        y_test = Surv.from_dataframe('event', 'time', outcomes_test)
        
        # 计算动态AUC
        auc_scores, _ = metrics.cumulative_dynamic_auc(y_train, y_test, 1-predictions, times)
        return np.mean(auc_scores)  # 返回平均AUC
    except Exception as e:
        logging.error(f"计算Dynamic AUC时出错: {str(e)}")
        return np.nan

def calculate_risk_group_metrics(risk_scores, durations, events, n_groups=5):
    """计算风险分组指标"""
    try:
        df = pd.DataFrame({
            'risk_score': risk_scores,
            'duration': durations,
            'event': events
        })
        
        # 按风险得分分组
        df['risk_group'] = pd.qcut(df['risk_score'], q=n_groups, labels=False, duplicates='drop')
        
        group_metrics = []
        for group in sorted(df['risk_group'].unique()):
            group_data = df[df['risk_group'] == group]
            metrics = {
                'risk_group': int(group + 1),
                'size': len(group_data),
                'event_count': int(group_data['event'].sum()),
                'event_rate': float(group_data['event'].mean()),
                # ensure durations are in canonical hours for summary statistics
                'mean_duration': float(_ensure_durations_in_hours(np.asarray(group_data['duration'].values), cfg=None, name='metrics_group_mean').mean()),
                'median_duration': float(np.median(_ensure_durations_in_hours(np.asarray(group_data['duration'].values), cfg=None, name='metrics_group_median')))
            }
            group_metrics.append(metrics)
            
        return group_metrics
    except Exception as e:
        logging.error(f"计算风险分组指标时出错: {str(e)}")
        return []

def calculate_all_metrics(predictions, durations, events, time_points=None):
    """
    计算并返回一个包含所有评估指标的字典。
    """
    metrics = {}
    try:
        metrics['c_index'] = c_index(predictions, durations, events)
    except Exception as e:
        logging.error(f"计算 C-index 时出错: {e}")
        metrics['c_index'] = None
        
    # 添加ROC相关指标
    try:
        # 如果没有提供时间点，使用固定的ROC时间点（预测窗口的0.25, 0.5, 0.75位置）
        if time_points is None:
            try:
                from configs import get_config
                cfg = get_config()
                prediction_window = float(cfg.data.sequence_generation.prediction_window_hours)
                time_points = np.array([0.25, 0.5, 0.75]) * prediction_window
            except Exception:
                # 默认使用48小时预测窗口
                time_points = np.array([12.0, 24.0, 36.0])
        
        # 计算多个时间点的AUC
        auc_scores = calculate_auc_at_multiple_times(predictions, durations, events, time_points)
        metrics['auc_at_times'] = auc_scores
        
        # 计算积分AUC
        metrics['integrated_auc'] = calculate_integrated_auc(predictions, durations, events, time_points)
        
        # 计算时间相关ROC
        metrics['time_dependent_roc'] = calculate_time_dependent_roc(predictions, durations, events, time_points)
        
    except Exception as e:
        logging.error(f"计算ROC相关指标时出错: {e}")
        metrics['auc_at_times'] = {}
        metrics['integrated_auc'] = 0.5
        metrics['time_dependent_roc'] = {}
    
    return metrics


def calculate_survival_time_metrics(survival_funcs_df, durations, events, prediction_window=None, risk_scores=None):
    """
    计算生存时间预测误差的评估指标。
    
    Args:
        survival_funcs_df: 生存函数DataFrame，行为样本，列为时间点
        durations: 真实生存时间
        events: 事件指示器
        prediction_window: 预测窗口（小时）
        risk_scores: 风险分数数组，用于计算分位数阈值
        
    Returns:
        dict: 包含各种生存时间预测误差指标
    """
    metrics = {}
    
    try:
        if prediction_window is None:
            try:
                from configs import get_config
                cfg = get_config()
                prediction_window = float(cfg.data.sequence_generation.prediction_window_hours)
            except Exception:
                prediction_window = 48.0
        
        # 添加调试信息
        debug_info = analyze_survival_curves(survival_funcs_df, prediction_window)
        metrics.update(debug_info)
        
        # 1. 中位生存时间预测误差（使用风险分数分位数阈值）
        median_survival_times = calculate_median_survival_times(survival_funcs_df, prediction_window, risk_scores)
        median_errors = calculate_median_survival_errors(median_survival_times, durations, events)
        metrics.update(median_errors)
        
        # 2. 分位数预测误差（25%, 50%, 75%）
        quantile_errors = calculate_quantile_survival_errors(survival_funcs_df, durations, events, prediction_window)
        metrics.update(quantile_errors)
        
        # 3. 平均绝对误差和均方根误差（仅对事件样本）
        mae_rmse_errors = calculate_mae_rmse_errors(survival_funcs_df, durations, events, prediction_window)
        metrics.update(mae_rmse_errors)
        
        # 4. 预测时间分布统计
        time_dist_stats = calculate_predicted_time_distribution_stats(median_survival_times, durations, events)
        metrics.update(time_dist_stats)
        
    except Exception as e:
        logging.error(f"计算生存时间预测误差时出错: {e}")
        # 返回默认值
        metrics = {
            'median_survival_mae': None,
            'median_survival_rmse': None,
            'median_survival_mape': None,
            'quantile_90_mae': None,
            'quantile_80_mae': None,
            'quantile_70_mae': None,
            'event_mae': None,
            'event_rmse': None,
            'event_mape': None
        }
    
    return metrics


def calculate_median_survival_times(survival_funcs_df, prediction_window, risk_scores=None):
    """
    计算每个样本的中位生存时间。
    使用改进的混合方法：结合风险分数和生存曲线特征。
    
    Args:
        survival_funcs_df: 生存函数DataFrame
        prediction_window: 预测窗口
        risk_scores: 风险分数数组，用于计算分位数阈值
        
    Returns:
        np.array: 中位生存时间数组
    """
    try:
        times = survival_funcs_df.columns.astype(float)
        median_times = []
        
        # 改进的阈值计算：基于风险分数的动态阈值
        if risk_scores is not None and len(risk_scores) > 0:
            # 将风险分数标准化到[0,1]范围
            risk_min, risk_max = np.min(risk_scores), np.max(risk_scores)
            if risk_max > risk_min:
                normalized_risks = (risk_scores - risk_min) / (risk_max - risk_min)
            else:
                normalized_risks = np.ones_like(risk_scores) * 0.5
            
            # 动态阈值：高风险对应低生存概率阈值，低风险对应高生存概率阈值
            # 阈值范围从0.2到0.8，确保有足够的区分度
            dynamic_thresholds = 0.8 - 0.6 * normalized_risks  # [0.2, 0.8]
        else:
            dynamic_thresholds = np.full(len(survival_funcs_df), 0.5)
        
        for idx in range(len(survival_funcs_df)):
            surv_curve = survival_funcs_df.iloc[idx].values
            target_threshold = dynamic_thresholds[idx]
            
            # 分析生存曲线特征
            curve_features = _analyze_survival_curve_features(surv_curve, times)
            
            # 基于曲线特征和风险分数预测生存时间
            survival_time = _predict_survival_time_improved(
                surv_curve, times, target_threshold, prediction_window, 
                curve_features, risk_scores[idx] if risk_scores is not None and idx < len(risk_scores) else None
            )
            
            median_times.append(survival_time)
        
        return np.array(median_times)
        
    except Exception as e:
        logging.error(f"计算中位生存时间时出错: {e}")
        return np.full(len(survival_funcs_df), prediction_window)


def _analyze_survival_curve_features(survival_curve, times):
    """
    分析生存曲线的特征
    
    Args:
        survival_curve: 生存概率数组
        times: 时间点数组
        
    Returns:
        dict: 曲线特征字典
    """
    # 计算曲线的基本统计
    valid_curve = survival_curve[np.isfinite(survival_curve)]
    if len(valid_curve) == 0:
        return {
            'min_prob': 1.0,
            'max_prob': 1.0,
            'prob_range': 0.0,
            'avg_decline_rate': 0.0,
            'steep_drop': False,
            'gradual_decline': False,
            'flat_curve': True
        }
    
    min_prob = np.min(valid_curve)
    max_prob = np.max(valid_curve)
    prob_range = max_prob - min_prob
    
    # 计算曲线的下降率
    valid_indices = np.isfinite(survival_curve)
    if np.sum(valid_indices) > 1:
        valid_curve = survival_curve[valid_indices]
        valid_times = times[valid_indices]
        
        # 计算平均下降率
        if len(valid_curve) > 1:
            prob_diff = np.diff(valid_curve)
            time_diff = np.diff(valid_times)
            valid_diff_mask = time_diff > 0
            if np.any(valid_diff_mask):
                avg_decline_rate = np.mean(prob_diff[valid_diff_mask] / time_diff[valid_diff_mask])
            else:
                avg_decline_rate = 0
        else:
            avg_decline_rate = 0
    else:
        avg_decline_rate = 0
    
    # 判断曲线类型
    steep_drop = prob_range > 0.5 and avg_decline_rate < -0.01  # 急剧下降
    gradual_decline = 0.1 < prob_range <= 0.5 and avg_decline_rate < -0.005  # 逐渐下降
    flat_curve = prob_range <= 0.1 or avg_decline_rate >= -0.005  # 平缓曲线
    
    return {
        'min_prob': min_prob,
        'max_prob': max_prob,
        'prob_range': prob_range,
        'avg_decline_rate': avg_decline_rate,
        'steep_drop': steep_drop,
        'gradual_decline': gradual_decline,
        'flat_curve': flat_curve
    }


def _predict_survival_time_improved(survival_curve, times, target_threshold, prediction_window, curve_features, risk_score=None):
    """
    改进的生存时间预测方法
    
    Args:
        survival_curve: 生存概率数组
        times: 时间点数组
        target_threshold: 目标生存概率阈值
        prediction_window: 预测窗口
        curve_features: 曲线特征字典
        risk_score: 风险分数
        
    Returns:
        float: 预测的生存时间
    """
    # 方法1：如果曲线下降到目标阈值以下，使用插值
    min_prob = curve_features['min_prob']
    if min_prob <= target_threshold:
        survival_time = _find_survival_time_with_interpolation(
            survival_curve, times, target_threshold, prediction_window
        )
        return survival_time
    
    # 方法2：基于曲线特征预测
    if curve_features['steep_drop']:
        # 急剧下降：使用更低的阈值（25%生存概率）
        lower_threshold = max(0.1, target_threshold - 0.3)
        survival_time = _find_survival_time_with_interpolation(
            survival_curve, times, lower_threshold, prediction_window
        )
        if survival_time < prediction_window:
            return survival_time
    
    elif curve_features['gradual_decline']:
        # 逐渐下降：使用中等阈值（40%生存概率）
        medium_threshold = max(0.2, target_threshold - 0.2)
        survival_time = _find_survival_time_with_interpolation(
            survival_curve, times, medium_threshold, prediction_window
        )
        if survival_time < prediction_window:
            return survival_time
    
    # 方法3：基于风险分数的加权预测
    if risk_score is not None:
        # 使用风险分数来调整预测时间
        # 高风险对应更早的事件时间
        # 将风险分数标准化到[0,1]范围
        risk_factor = min(1.0, max(0.1, (risk_score + 3) / 6))  # 假设风险分数在[-3,3]范围
        base_time = prediction_window * 0.2  # 基础时间
        risk_adjusted_time = base_time + (prediction_window - base_time) * risk_factor
        return min(risk_adjusted_time, prediction_window)
    
    # 方法4：基于曲线形状的启发式预测
    if curve_features['flat_curve']:
        # 平缓曲线：使用预测窗口的中间值
        return prediction_window * 0.6
    else:
        # 其他情况：使用预测窗口的较小值
        return prediction_window * 0.4


def _find_survival_time_with_interpolation(survival_curve, times, target_threshold, prediction_window):
    """
    使用插值方法找到生存时间
    
    Args:
        survival_curve: 生存概率数组
        times: 时间点数组
        target_threshold: 目标生存概率阈值
        prediction_window: 预测窗口
        
    Returns:
        float: 插值得到的生存时间
    """
    # 找到第一个低于阈值的时间点
    for i, prob in enumerate(survival_curve):
        if prob <= target_threshold:
            if i == 0:
                return times[0]
            else:
                # 线性插值
                prev_prob = survival_curve[i-1]
                prev_time = times[i-1]
                curr_time = times[i]
                if prev_prob > target_threshold:
                    ratio = (target_threshold - prob) / (prev_prob - prob)
                    survival_time = curr_time + ratio * (prev_time - curr_time)
                    return min(survival_time, prediction_window)
                else:
                    return min(curr_time, prediction_window)
    
    # 如果没有找到，使用外推法
    return _extrapolate_survival_time(survival_curve, times, target_threshold, prediction_window)


def _extrapolate_survival_time(survival_curve, times, target_threshold, prediction_window):
    """
    使用外推法估计生存时间
    
    Args:
        survival_curve: 生存概率数组
        times: 时间点数组
        target_threshold: 目标生存概率阈值
        prediction_window: 预测窗口
        
    Returns:
        float: 外推得到的生存时间
    """
    # 找到最小概率点
    min_idx = np.argmin(survival_curve[np.isfinite(survival_curve)])
    min_time = times[min_idx]
    min_prob = survival_curve[min_idx]
    
    if min_prob >= target_threshold and min_prob < 1.0:
        # 使用指数衰减模型外推
        if min_idx > 0:
            prev_time = times[min_idx - 1]
            prev_prob = survival_curve[min_idx - 1]
            if prev_prob > min_prob:
                # 计算衰减率
                lambda_est = -np.log(min_prob / prev_prob) / (min_time - prev_time)
                if lambda_est > 0:  # 确保衰减率为正
                    # 外推到目标阈值
                    survival_time = min_time - np.log(target_threshold / min_prob) / lambda_est
                    return min(max(survival_time, 0.0), prediction_window)
    
    # 如果外推失败，返回基于阈值的启发式时间
    return prediction_window * (0.2 + 0.6 * (1 - target_threshold))


def calculate_median_survival_errors(predicted_times, true_times, events):
    """
    计算中位生存时间的预测误差。
    
    Args:
        predicted_times: 预测的中位生存时间
        true_times: 真实生存时间
        events: 事件指示器
        
    Returns:
        dict: 包含MAE、RMSE、MAPE等误差指标
    """
    try:
        # 只对事件样本计算误差（删失样本的真实时间不准确）
        event_mask = events == 1
        if not np.any(event_mask):
            return {
                'median_survival_mae': None,
                'median_survival_rmse': None,
                'median_survival_mape': None
            }
        
        pred_event = predicted_times[event_mask]
        true_event = true_times[event_mask]
        
        # 过滤无效值
        valid_mask = np.isfinite(pred_event) & np.isfinite(true_event) & (true_event > 0)
        if not np.any(valid_mask):
            return {
                'median_survival_mae': None,
                'median_survival_rmse': None,
                'median_survival_mape': None
            }
        
        pred_valid = pred_event[valid_mask]
        true_valid = true_event[valid_mask]
        
        # 计算误差指标
        mae = np.mean(np.abs(pred_valid - true_valid))
        rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
        
        # MAPE（避免除零）
        mape = np.mean(np.abs((pred_valid - true_valid) / true_valid)) * 100
        
        return {
            'median_survival_mae': float(mae),
            'median_survival_rmse': float(rmse),
            'median_survival_mape': float(mape)
        }
        
    except Exception as e:
        logging.error(f"计算中位生存时间误差时出错: {e}")
        return {
            'median_survival_mae': None,
            'median_survival_rmse': None,
            'median_survival_mape': None
        }


def calculate_quantile_survival_errors(survival_funcs_df, durations, events, prediction_window):
    """
    计算分位数生存时间的预测误差。
    
    Args:
        survival_funcs_df: 生存函数DataFrame
        durations: 真实生存时间
        events: 事件指示器
        prediction_window: 预测窗口
        
    Returns:
        dict: 包含25%、50%、75%分位数的MAE
    """
    try:
        times = survival_funcs_df.columns.astype(float)
        # 使用更现实的分位数阈值
        quantiles = [0.1, 0.2, 0.3]  # 对应90%, 80%, 70%生存概率
        results = {}
        
        for q in quantiles:
            quantile_times = []
            
            for idx in range(len(survival_funcs_df)):
                surv_curve = survival_funcs_df.iloc[idx].values
                
                # 找到生存概率为(1-q)的时间点
                target_prob = 1 - q
                quantile_time = None
                
                # 检查生存曲线是否下降到target_prob以下
                min_surv_prob = np.min(surv_curve[np.isfinite(surv_curve)])
                
                if min_surv_prob <= target_prob:
                    # 生存曲线下降到target_prob以下，可以找到分位数时间
                    for i, prob in enumerate(surv_curve):
                        if prob <= target_prob:
                            if i == 0:
                                quantile_time = times[0]
                            else:
                                # 线性插值
                                prev_prob = surv_curve[i-1]
                                prev_time = times[i-1]
                                curr_time = times[i]
                                if prev_prob > target_prob:
                                    ratio = (target_prob - prob) / (prev_prob - prob)
                                    quantile_time = curr_time + ratio * (prev_time - curr_time)
                                else:
                                    quantile_time = curr_time
                            break
                else:
                    # 生存曲线没有下降到target_prob以下，使用外推法
                    min_idx = np.argmin(surv_curve[np.isfinite(surv_curve)])
                    min_time = times[min_idx]
                    min_prob = surv_curve[min_idx]
                    
                    if min_prob > target_prob and min_prob < 1.0:
                        # 使用指数衰减模型外推
                        if min_idx > 0:
                            prev_time = times[min_idx - 1]
                            prev_prob = surv_curve[min_idx - 1]
                            if prev_prob > min_prob:
                                # 计算衰减率
                                lambda_est = -np.log(min_prob / prev_prob) / (min_time - prev_time)
                                # 外推到target_prob
                                quantile_time = min_time - np.log(target_prob / min_prob) / lambda_est
                                # 确保不超过预测窗口
                                quantile_time = min(quantile_time, prediction_window)
                            else:
                                quantile_time = prediction_window
                        else:
                            quantile_time = prediction_window
                    else:
                        quantile_time = prediction_window
                
                # 确保时间在合理范围内
                quantile_time = max(0.0, min(quantile_time, prediction_window))
                quantile_times.append(quantile_time)
            
            # 计算MAE（仅对事件样本）
            event_mask = events == 1
            if np.any(event_mask):
                pred_event = np.array(quantile_times)[event_mask]
                true_event = durations[event_mask]
                
                valid_mask = np.isfinite(pred_event) & np.isfinite(true_event) & (true_event > 0)
                if np.any(valid_mask):
                    mae = np.mean(np.abs(pred_event[valid_mask] - true_event[valid_mask]))
                    results[f'quantile_{int((1-q)*100)}_mae'] = float(mae)  # 90%, 80%, 70%生存概率
                else:
                    results[f'quantile_{int((1-q)*100)}_mae'] = None
            else:
                results[f'quantile_{int((1-q)*100)}_mae'] = None
        
        return results
        
    except Exception as e:
        logging.error(f"计算分位数生存时间误差时出错: {e}")
        return {
            'quantile_90_mae': None,
            'quantile_80_mae': None,
            'quantile_70_mae': None
        }


def calculate_mae_rmse_errors(survival_funcs_df, durations, events, prediction_window):
    """
    计算平均绝对误差和均方根误差（仅对事件样本）。
    
    Args:
        survival_funcs_df: 生存函数DataFrame
        durations: 真实生存时间
        events: 事件指示器
        prediction_window: 预测窗口
        
    Returns:
        dict: 包含MAE、RMSE、MAPE等指标
    """
    try:
        # 计算中位生存时间
        median_times = calculate_median_survival_times(survival_funcs_df, prediction_window)
        
        # 只对事件样本计算
        event_mask = events == 1
        if not np.any(event_mask):
            return {
                'event_mae': None,
                'event_rmse': None,
                'event_mape': None
            }
        
        pred_event = median_times[event_mask]
        true_event = durations[event_mask]
        
        # 过滤无效值
        valid_mask = np.isfinite(pred_event) & np.isfinite(true_event) & (true_event > 0)
        if not np.any(valid_mask):
            return {
                'event_mae': None,
                'event_rmse': None,
                'event_mape': None
            }
        
        pred_valid = pred_event[valid_mask]
        true_valid = true_event[valid_mask]
        
        # 计算误差指标
        mae = np.mean(np.abs(pred_valid - true_valid))
        rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
        mape = np.mean(np.abs((pred_valid - true_valid) / true_valid)) * 100
        
        return {
            'event_mae': float(mae),
            'event_rmse': float(rmse),
            'event_mape': float(mape)
        }
        
    except Exception as e:
        logging.error(f"计算MAE/RMSE误差时出错: {e}")
        return {
            'event_mae': None,
            'event_rmse': None,
            'event_mape': None
        }


def calculate_predicted_time_distribution_stats(predicted_times, true_times, events):
    """
    计算预测时间分布的统计信息。
    
    Args:
        predicted_times: 预测的生存时间
        true_times: 真实生存时间
        events: 事件指示器
        
    Returns:
        dict: 包含预测时间分布的统计信息
    """
    try:
        # 基本统计
        pred_mean = np.mean(predicted_times[np.isfinite(predicted_times)])
        pred_std = np.std(predicted_times[np.isfinite(predicted_times)])
        pred_median = np.median(predicted_times[np.isfinite(predicted_times)])
        
        # 真实时间统计（仅事件样本）
        event_mask = events == 1
        if np.any(event_mask):
            true_event = true_times[event_mask]
            true_valid = true_event[np.isfinite(true_event)]
            if len(true_valid) > 0:
                true_mean = np.mean(true_valid)
                true_std = np.std(true_valid)
                true_median = np.median(true_valid)
            else:
                true_mean = true_std = true_median = None
        else:
            true_mean = true_std = true_median = None
        
        return {
            'predicted_time_mean': float(pred_mean) if np.isfinite(pred_mean) else None,
            'predicted_time_std': float(pred_std) if np.isfinite(pred_std) else None,
            'predicted_time_median': float(pred_median) if np.isfinite(pred_median) else None,
            'true_time_mean': float(true_mean) if true_mean is not None and np.isfinite(true_mean) else None,
            'true_time_std': float(true_std) if true_std is not None and np.isfinite(true_std) else None,
            'true_time_median': float(true_median) if true_median is not None and np.isfinite(true_median) else None
        }
        
    except Exception as e:
        logging.error(f"计算预测时间分布统计时出错: {e}")
        return {
            'predicted_time_mean': None,
            'predicted_time_std': None,
            'predicted_time_median': None,
            'true_time_mean': None,
            'true_time_std': None,
            'true_time_median': None
        }


def analyze_survival_curves(survival_funcs_df, prediction_window):
    """
    分析生存曲线的分布情况，用于调试。
    
    Args:
        survival_funcs_df: 生存函数DataFrame
        prediction_window: 预测窗口
        
    Returns:
        dict: 包含生存曲线分析结果
    """
    try:
        times = survival_funcs_df.columns.astype(float)
        n_samples = len(survival_funcs_df)
        
        # 统计每个时间点的生存概率分布
        min_surv_probs = []
        max_surv_probs = []
        mean_surv_probs = []
        
        for idx in range(n_samples):
            surv_curve = survival_funcs_df.iloc[idx].values
            finite_mask = np.isfinite(surv_curve)
            if np.any(finite_mask):
                min_surv_probs.append(np.min(surv_curve[finite_mask]))
                max_surv_probs.append(np.max(surv_curve[finite_mask]))
                mean_surv_probs.append(np.mean(surv_curve[finite_mask]))
        
        # 统计下降到0.5以下的样本数量
        below_05_count = sum(1 for p in min_surv_probs if p <= 0.5)
        below_025_count = sum(1 for p in min_surv_probs if p <= 0.25)
        below_075_count = sum(1 for p in min_surv_probs if p <= 0.75)
        
        return {
            'survival_curve_analysis': {
                'total_samples': n_samples,
                'curves_below_0.5': below_05_count,
                'curves_below_0.25': below_025_count,
                'curves_below_0.75': below_075_count,
                'min_surv_prob_mean': float(np.mean(min_surv_probs)) if min_surv_probs else None,
                'min_surv_prob_std': float(np.std(min_surv_probs)) if min_surv_probs else None,
                'max_surv_prob_mean': float(np.mean(max_surv_probs)) if max_surv_probs else None,
                'mean_surv_prob_mean': float(np.mean(mean_surv_probs)) if mean_surv_probs else None
            }
        }
        
    except Exception as e:
        logging.error(f"分析生存曲线时出错: {e}")
        return {'survival_curve_analysis': None}


def debug_pairwise_concordance(predictions, durations, events, max_pairs=10000):
    """
    Diagnostic helper: sample pairwise comparisons and report the fraction concordant/discordant/tied.
    This helps find labeling/order problems when C-index is unexpectedly low while loss improves.
    Returns a dict with counts and a small list of counterexample pairs.
    """
    try:
        preds = np.asarray(predictions).flatten()
        times = np.asarray(durations).flatten()
        ev = np.asarray(events).flatten()
        n = len(preds)
        if n < 2:
            return {'error': 'not enough samples'}
        import random
        pairs = []
        concordant = discordant = tied = 0
        counterexamples = []
        for _ in range(min(max_pairs, n*(n-1)//2)):
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            # valid pair if at least one is an event and times differ
            if ev[i]==0 and ev[j]==0:
                continue
            if times[i] == times[j]:
                tied += 1
                continue
            # concordant if higher risk has smaller time
            if preds[i] > preds[j] and times[i] < times[j]:
                concordant += 1
            elif preds[j] > preds[i] and times[j] < times[i]:
                concordant += 1
            elif preds[i] == preds[j]:
                tied += 1
            else:
                discordant += 1
                if len(counterexamples) < 20:
                    counterexamples.append({'i': i, 'j': j, 'pred_i': float(preds[i]), 'pred_j': float(preds[j]), 'time_i': float(times[i]), 'time_j': float(times[j]), 'ev_i': int(ev[i]), 'ev_j': int(ev[j])})

        total = concordant + discordant + tied
        return {'concordant': concordant, 'discordant': discordant, 'tied': tied, 'total': total, 'concordance_rate': concordant/total if total>0 else None, 'counterexamples': counterexamples}
    except Exception as e:
        logging.error(f"debug_pairwise_concordance failed: {e}")
        return {'error': str(e)}


def compute_concordance(event_times, predictions, event_indicators, predictions_are_risk=True):
    """
    统一的 Concordance Index 计算 wrapper。

    Args:
        event_times (array-like): 事件/删失时间。
        predictions (array-like): 模型输出。默认被视为 risk（越大表示越高风险、越早发生事件）。
        event_indicators (array-like): 事件指示器 (1=事件, 0=删失)。
        predictions_are_risk (bool): 如果 True，会将 predictions 转换为 survival score（即取负），
                                    以符合 lifelines.concordance_index 的“越大越好”的约定。

    Returns:
        float: concordance index（遇到异常时返回 0.5）。
    """
    try:
        et = np.asarray(event_times).flatten()
        preds = np.asarray(predictions).flatten()
        ev = np.asarray(event_indicators).flatten()

        # 基本长度校验
        if not (len(et) == len(preds) == len(ev)):
            # 尝试对齐最短长度
            min_len = min(len(et), len(preds), len(ev))
            et = et[:min_len]
            preds = preds[:min_len]
            ev = ev[:min_len]

        # 过滤NaN/Inf
        valid_mask = ~(np.isnan(preds) | np.isinf(preds) | np.isnan(et) | np.isinf(et) | np.isnan(ev) | np.isinf(ev))
        if np.sum(valid_mask) < 2:
            logging.warning("过滤后有效样本不足以计算C-index，返回0.5")
            return 0.5
        et = et[valid_mask]
        preds = preds[valid_mask]
        ev = ev[valid_mask]

        # 如果所有样本都没有事件或都发生事件，返回0.5（无信息）
        if np.sum(ev) == 0 or np.sum(ev) == len(ev):
            logging.warning("事件分布异常（全为删失或全为事件），C-index返回0.5")
            return 0.5

        # lifelines.concordance_index 的约定：分数越大表示预计生存时间越长（即更安全/事件越晚）。
        # 如果传入的 `predictions` 表示风险（predictions_are_risk=True，分数越大风险越高/事件越早），
        # 则需要取负号以转换为“生存分数”（越大越好），从而与 lifelines 的约定一致。
        # 也就是说：
        #   - predictions_are_risk == True  -> score_for_cindex = -preds
        #   - predictions_are_risk == False -> score_for_cindex = preds
        score_for_cindex = -preds if predictions_are_risk else preds

        # 最终调用 lifelines 的 concordance_index
        from lifelines.utils import concordance_index as lifelines_cindex
        c = lifelines_cindex(et, score_for_cindex, ev)
        if np.isnan(c):
            logging.warning("compute_concordance 计算出 NaN，返回0.5")
            return 0.5
        return float(c)
    except Exception as e:
        logging.error(f"compute_concordance 计算出错: {e}")
        return 0.5
