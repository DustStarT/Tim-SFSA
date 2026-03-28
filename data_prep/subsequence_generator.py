"""
子序列生成器
为时序生存模型（如LSTM-DeepHit）从合并后的长时序样本中生成子序列。
"""
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

def generate_subsequences(
    sample_dict,
    num_covariate_timesteps,
    prediction_window_hours,
    sub_sequence_step,
    active_feature_names
):
    """
    从单个长时序样本中生成子序列。

    Args:
        sample_dict (dict): 包含 'features', 'time', 'event' 和 'timestamps_list' 的字典。
        num_covariate_timesteps (int): 每个子序列应包含的时间步数。
        prediction_window_hours (int): 用于确定事件标签的预测窗口（小时）。
        sub_sequence_step (int): 创建子序列的步长（滑动窗口的移动距离）。
        active_feature_names (list): 要使用的特征的名称列表。

    Returns:
        list: 一个包含多个子序列字典的列表。
    """
    all_features_df = sample_dict['features']
    event_timestamp_str = sample_dict.get('event_time_raw')
    parent_event_occurred = sample_dict['event']
    timestamps = sample_dict['timestamps_list']
    
    # 核心修复: 只使用激活的特征
    if active_feature_names:
        features_df = all_features_df[active_feature_names]
    else:
        # 如果没有指定激活特征，则使用所有特征（作为后备）
        features_df = all_features_df

    # 关键修复: 重置DataFrame的索引，以确保它能与后续创建的
    # 布尔掩码(valid_mask)的索引对齐。
    features_df.reset_index(drop=True, inplace=True)

    subsequences = []
    
    # 2. 验证和处理时间戳
    # 显式创建Series以使用reset_index方法
    all_timestamps_dt = pd.Series(pd.to_datetime(timestamps, errors='coerce'))
    valid_mask = ~pd.isna(all_timestamps_dt)
    if not valid_mask.any():
        return subsequences
        
    features = features_df[valid_mask]
    timestamps = all_timestamps_dt[valid_mask].reset_index(drop=True)

    if len(features) < num_covariate_timesteps:
        return subsequences

    actual_event_time = pd.to_datetime(event_timestamp_str, errors='coerce') if parent_event_occurred and event_timestamp_str else pd.NaT

    # 3. 定义预测窗口并开始滑动
    prediction_window_timedelta = pd.Timedelta(hours=prediction_window_hours)
    
    # 包含端点：当 len(timestamps) == num_covariate_timesteps 时也应生成一个子序列
    iterator = range(0, len(timestamps) - num_covariate_timesteps + 1, sub_sequence_step)

    # 诊断性日志：如果正好能生成一个窗口，记录一条调试信息（便于追踪被丢弃的边界样本）
    try:
        logger = logging.getLogger(__name__)
        if len(timestamps) == num_covariate_timesteps:
            logger.debug(f"Subsequence generator: timestamps length == num_covariate_timesteps ({num_covariate_timesteps}); will generate exactly one subsequence for record_id={sample_dict.get('record_id')}")
    except Exception:
        pass

    def _build_record_metadata(sample):
        raw = sample.get('record_id')
        info = {'raw': raw, 'ar': None, 'start': None, 'end': None, 'flare_class': None, 'role': None}
        try:
            if raw is not None and isinstance(raw, str):
                # Support patterns like '@4629_ar1999' and 'AR1999' etc.
                # Extract the last occurrence of ar followed by digits and normalize to lowercase 'ar<digits>'.
                m_ar_all = re.findall(r'ar\D*?(\d+)', raw, flags=re.IGNORECASE)
                if m_ar_all:
                    # pick the last numeric group (most specific)
                    ar_num = m_ar_all[-1]
                    info['ar'] = f'ar{int(ar_num)}'
                else:
                    # fallback to older pattern
                    m_ar = re.search(r'\b([aA][rR]?\d+)\b', raw)
                    if m_ar:
                        info['ar'] = m_ar.group(1).lower()
                m_fc = re.search(r'\b([M|X|C|B]\d+(?:\.\d+)?)\b', raw, flags=re.IGNORECASE)
                if m_fc:
                    info['flare_class'] = m_fc.group(1)
                m_role = re.search(r'\b(Primary|Secondary)\b', raw, flags=re.IGNORECASE)
                if m_role:
                    info['role'] = m_role.group(1)
        except Exception:
            pass
        st = sample.get('start_time') or None
        ed = sample.get('end_time') or None
        if st is not None and not isinstance(st, datetime):
            try:
                st = pd.to_datetime(st)
            except Exception:
                st = None
        if ed is not None and not isinstance(ed, datetime):
            try:
                ed = pd.to_datetime(ed)
            except Exception:
                ed = None
        info['start'] = st
        info['end'] = ed
        return info

    for i in iterator:
        # 定义协变量窗口
        covariate_window_end_idx = i + num_covariate_timesteps
        covariate_features = features[i:covariate_window_end_idx]
        
        # 生存分析在协变量窗口结束后开始
        prediction_window_start_time = timestamps.iloc[covariate_window_end_idx - 1]
        prediction_window_end_time = prediction_window_start_time + prediction_window_timedelta

        # 确定此子序列的事件和生存/审查时间
        event = 0
        duration = float(prediction_window_hours) # 默认审查时间

        if pd.notna(actual_event_time):
            # 检查真实事件是否落入此子序列的预测窗口内
            if prediction_window_start_time < actual_event_time <= prediction_window_end_time:
                event = 1
                # 计算从窗口开始到事件的精确持续时间
                duration = (actual_event_time - prediction_window_start_time).total_seconds() / 3600.0
                duration = min(duration, float(prediction_window_hours))

        # 确保协变量窗口是完整的
        if covariate_features.shape[0] == num_covariate_timesteps:
            # 转为 numpy 并确保为 float32
            try:
                final_features = covariate_features.values.astype(np.float32)
            except Exception:
                # fallback: cast DataFrame directly
                final_features = np.array(covariate_features, dtype=np.float32)

            # 如果是 1D，reshape 为 (1, n_features)
            if final_features.ndim == 1:
                final_features = final_features.reshape(1, -1)
            # ========== 序列级聚合特征（可选） ==========
            try:
                # 只有当外部配置明确允许时才计算和附加聚合特征
                include_agg = False
                if 'sequence_generation' in sample_dict.get('config', {}):
                    include_agg = bool(sample_dict['config']['sequence_generation'].get('include_aggregated_features', False))
                # 兼容旧调用：如果没有在 sample_dict 中传递 config，对外部调用方传入的 active_feature_names 来判断（preprocessor 会传入一个 wrapper）
                if not include_agg and hasattr(generate_subsequences, '_include_agg'):
                    include_agg = bool(getattr(generate_subsequences, '_include_agg'))

                if include_agg:
                    ft = final_features  # shape: (T, n_features)
                    # 在统计前对极端值进行裁剪和 NaN 清洗
                    try:
                        SENTINEL = float(sample_dict.get('config', {}).get('data', {}).get('sentinel_threshold', 1e6))
                    except Exception:
                        SENTINEL = 1e6
                    # 替换 inf 为 NaN
                    ft = np.where(np.isfinite(ft), ft, np.nan).astype(np.float32)
                    # 裁剪极端值
                    try:
                        ft = np.clip(ft, -SENTINEL, SENTINEL)
                    except Exception:
                        pass
                    # 计算按列的统计量（NaN 安全）
                    mean_vals = np.nan_to_num(np.nanmean(ft, axis=0).astype(np.float32), nan=0.0)
                    var_vals = np.nan_to_num(np.nanvar(ft, axis=0).astype(np.float32), nan=0.0)
                    std_vals = np.nan_to_num(np.nanstd(ft, axis=0).astype(np.float32), nan=0.0)
                    max_vals = np.nan_to_num(np.nanmax(ft, axis=0).astype(np.float32), nan=0.0)
                    min_vals = np.nan_to_num(np.nanmin(ft, axis=0).astype(np.float32), nan=0.0)

                    # 带上自身（原始序列的最后时刻值）
                    self_vals = np.nan_to_num(ft[-1, :].astype(np.float32), nan=0.0)

                    # 拼接顺序：self, mean, var, std, max, min -> 共 6 倍每特征
                    agg_vec = np.concatenate([self_vals, mean_vals, var_vals, std_vals, max_vals, min_vals]).astype(np.float32)
                    # 广播到每个时间步（与现有下游一致的按时间步拼接）
                    agg_broadcast = np.repeat(agg_vec[None, :], ft.shape[0], axis=0)
                    final_features = np.concatenate([ft, agg_broadcast], axis=1)
                else:
                    # 保持原样
                    final_features = final_features
            except Exception:
                # 任何失败都回退到原始 final_features
                final_features = final_features

            # Debug: log the resulting feature width to help track mismatches
            try:
                logger = logging.getLogger(__name__)
                logger.debug(f"Generated subsequence: T={final_features.shape[0]}, n_features={final_features.shape[1]} (base_features={ft.shape[1]}) for sample={sample_dict.get('record_id')}")
            except Exception:
                pass

            # 构建记录元数据并附加 interval/subsequence 信息
            record_meta = _build_record_metadata(sample_dict)
            try:
                record_meta['interval_index'] = int(i)
            except Exception:
                record_meta['interval_index'] = i
            try:
                record_meta['subseq_start'] = timestamps.iloc[i]
            except Exception:
                record_meta['subseq_start'] = None
            try:
                record_meta['subseq_end'] = prediction_window_end_time
            except Exception:
                record_meta['subseq_end'] = None

            subseq = {
                'features': final_features,
                'duration': duration,
                'event': event,
                'record_id': record_meta
            }

            subsequences.append(subseq)

    return subsequences
