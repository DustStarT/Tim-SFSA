"""
独立的异常值分析模块
不依赖PyTorch，可以单独使用
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
import json

def analyze_outliers_detailed(X, y, logger, set_name, record_ids=None, feature_names=None):
    """
    详细分析异常值的分布情况，包括时间步和特征维度的分析。
    
    Args:
        X: 输入数据 (n_samples, n_timesteps, n_features)
        y: 标签数据 (n_samples, 2)
        logger: 日志记录器
        set_name: 数据集名称
        record_ids: 记录ID列表
        feature_names: 特征名称列表
    
    Returns:
        outlier_analysis: 异常值分析结果字典
    """
    logger.info(f"=== 开始详细异常值分析: {set_name} ===")
    
    original_shape = X.shape
    logger.info(f"原始形状: X={X.shape}, y={y.shape}")
    
    # 如果没有提供特征名称，使用默认名称
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[-1])]
    
    # 展平数据用于统计
    X_flat = X.reshape(-1, X.shape[-1])
    logger.info(f"展平后的特征形状: {X_flat.shape}")
    
    # 计算每个特征的统计信息
    feature_stats = {}
    for i, feature_name in enumerate(feature_names):
        feature_data = X_flat[:, i]
        feature_stats[feature_name] = {
            'max': np.max(feature_data),
            'min': np.min(feature_data),
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'median': np.median(feature_data),
            'q1': np.percentile(feature_data, 25),
            'q3': np.percentile(feature_data, 75),
            'iqr': np.percentile(feature_data, 75) - np.percentile(feature_data, 25),
            'upper_bound': np.percentile(feature_data, 99.5),
            'lower_bound': np.percentile(feature_data, 0.5),
            'upper_bound_iqr': np.percentile(feature_data, 75) + 2.0 * (np.percentile(feature_data, 75) - np.percentile(feature_data, 25)),
            'lower_bound_iqr': np.percentile(feature_data, 25) - 2.0 * (np.percentile(feature_data, 75) - np.percentile(feature_data, 25))
        }
        
        logger.info(f"  {feature_name}: max={feature_stats[feature_name]['max']:.4f}, "
                   f"min={feature_stats[feature_name]['min']:.4f}, "
                   f"mean={feature_stats[feature_name]['mean']:.4f}, "
                   f"std={feature_stats[feature_name]['std']:.4f}")
    
    # 分析标签
    duration, event = y[:, 0], y[:, 1]
    duration_stats = {
        'max': np.max(duration),
        'min': np.min(duration),
        'mean': np.mean(duration),
        'std': np.std(duration),
        'upper_bound': np.percentile(duration, 99.5),
        'lower_bound': np.percentile(duration, 0.5)
    }
    logger.info(f"  标签 'duration': max={duration_stats['max']:.4f}, "
               f"min={duration_stats['min']:.4f}, "
               f"mean={duration_stats['mean']:.4f}, "
               f"std={duration_stats['std']:.4f}")
    
    # 检测异常值
    outlier_analysis = {
        'feature_outliers': {},
        'time_step_outliers': {},
        'sample_outliers': {},
        'summary': {}
    }
    
    # 1. 按特征分析异常值
    for i, feature_name in enumerate(feature_names):
        feature_data = X[:, :, i]  # (n_samples, n_timesteps)
        upper_bound = feature_stats[feature_name]['upper_bound']
        lower_bound = feature_stats[feature_name]['lower_bound']
        
        # 找出异常值的位置
        outlier_mask = (feature_data > upper_bound) | (feature_data < lower_bound)
        outlier_positions = np.where(outlier_mask)
        
        outlier_analysis['feature_outliers'][feature_name] = {
            'total_outliers': np.sum(outlier_mask),
            'outlier_ratio': np.sum(outlier_mask) / feature_data.size,
            'sample_indices': outlier_positions[0].tolist(),
            'time_indices': outlier_positions[1].tolist(),
            'values': feature_data[outlier_mask].tolist(),
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        }
        
        logger.info(f"  {feature_name} 异常值: {np.sum(outlier_mask)} 个 "
                   f"({np.sum(outlier_mask)/feature_data.size:.2%})")
    
    # 2. 按时间步分析异常值
    for time_step in range(X.shape[1]):
        time_data = X[:, time_step, :]  # (n_samples, n_features)
        outlier_mask_per_feature = np.zeros_like(time_data, dtype=bool)
        
        for i, feature_name in enumerate(feature_names):
            upper_bound = feature_stats[feature_name]['upper_bound']
            lower_bound = feature_stats[feature_name]['lower_bound']
            outlier_mask_per_feature[:, i] = (time_data[:, i] > upper_bound) | (time_data[:, i] < lower_bound)
        
        # 任何特征有异常值的样本
        sample_outlier_mask = np.any(outlier_mask_per_feature, axis=1)
        
        outlier_analysis['time_step_outliers'][time_step] = {
            'total_outliers': np.sum(sample_outlier_mask),
            'outlier_ratio': np.sum(sample_outlier_mask) / len(sample_outlier_mask),
            'sample_indices': np.where(sample_outlier_mask)[0].tolist(),
            'feature_outlier_counts': np.sum(outlier_mask_per_feature, axis=0).tolist()
        }
    
    # 3. 按样本分析异常值
    for sample_idx in range(X.shape[0]):
        sample_data = X[sample_idx, :, :]  # (n_timesteps, n_features)
        outlier_mask_per_feature = np.zeros_like(sample_data, dtype=bool)
        
        for i, feature_name in enumerate(feature_names):
            upper_bound = feature_stats[feature_name]['upper_bound']
            lower_bound = feature_stats[feature_name]['lower_bound']
            outlier_mask_per_feature[:, i] = (sample_data[:, i] > upper_bound) | (sample_data[:, i] < lower_bound)
        
        # 任何时间步有异常值的特征
        feature_outlier_mask = np.any(outlier_mask_per_feature, axis=0)
        
        outlier_analysis['sample_outliers'][sample_idx] = {
            'total_outliers': np.sum(feature_outlier_mask),
            'outlier_features': np.where(feature_outlier_mask)[0].tolist(),
            'outlier_feature_names': [feature_names[i] for i in np.where(feature_outlier_mask)[0]],
            'max_outlier_ratio': np.max(np.sum(outlier_mask_per_feature, axis=0) / sample_data.shape[0])
        }
    
    # 4. 总结统计
    total_outliers_by_feature = [outlier_analysis['feature_outliers'][f]['total_outliers'] 
                                for f in feature_names]
    total_outliers_by_time = [outlier_analysis['time_step_outliers'][t]['total_outliers'] 
                             for t in range(X.shape[1])]
    
    outlier_analysis['summary'] = {
        'total_samples': X.shape[0],
        'total_timesteps': X.shape[1],
        'total_features': X.shape[2],
        'max_outliers_by_feature': max(total_outliers_by_feature),
        'max_outliers_by_time': max(total_outliers_by_time),
        'feature_with_most_outliers': feature_names[np.argmax(total_outliers_by_feature)],
        'time_with_most_outliers': np.argmax(total_outliers_by_time),
        'samples_with_outliers': sum(1 for s in outlier_analysis['sample_outliers'].values() if s['total_outliers'] > 0)
    }
    
    logger.info(f"=== 异常值分析总结 ===")
    logger.info(f"  总样本数: {outlier_analysis['summary']['total_samples']}")
    logger.info(f"  有异常值的样本数: {outlier_analysis['summary']['samples_with_outliers']}")
    logger.info(f"  异常值最多的特征: {outlier_analysis['summary']['feature_with_most_outliers']}")
    logger.info(f"  异常值最多的时间步: {outlier_analysis['summary']['time_with_most_outliers']}")
    
    return outlier_analysis

def clean_and_log_data(X, y, logger, set_name, record_ids=None, feature_names=None):
    """
    改进的数据清理函数，包含详细的异常值分析和智能处理策略。
    
    策略：
    1. 详细分析异常值的分布
    2. 对于耀斑事件相关的异常值，采用更保守的处理方式
    3. 使用多种异常值检测方法
    4. 提供异常值处理的选项
    """
    logger.info(f"=== 开始数据清理: {set_name} ===")
    
    # 如果没有提供特征名称，使用默认名称
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[-1])]
    
    # 详细分析异常值
    outlier_analysis = analyze_outliers_detailed(X, y, logger, set_name, record_ids, feature_names)
    
    # 智能异常值处理策略
    original_shape = X.shape
    X_cleaned = X.copy()
    y_cleaned = y.copy()
    record_ids_cleaned = record_ids.copy() if record_ids is not None else None
    
    # 策略1: 对于耀斑相关特征，使用更宽松的阈值
    flare_related_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 
                             'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45']
    
    # 策略2: 使用IQR方法进行更稳健的异常值检测
    X_flat = X.reshape(-1, X.shape[-1])
    outlier_mask_per_sample = np.zeros(X.shape[0], dtype=bool)
    
    for i, feature_name in enumerate(feature_names):
        feature_data = X[:, :, i]
        feature_flat = X_flat[:, i]
        
        # 计算IQR统计量
        q1 = np.percentile(feature_flat, 25)
        q3 = np.percentile(feature_flat, 75)
        iqr = q3 - q1
        
        # 根据特征类型选择不同的倍数
        if feature_name in flare_related_features:
            # 耀斑相关特征使用更宽松的阈值
            multiplier = 3.0
            logger.info(f"  耀斑相关特征 {feature_name} 使用宽松阈值 (IQR × {multiplier})")
        else:
            # 其他特征使用标准阈值
            multiplier = 2.0
            logger.info(f"  普通特征 {feature_name} 使用标准阈值 (IQR × {multiplier})")
        
        upper_bound = q3 + multiplier * iqr
        lower_bound = q1 - multiplier * iqr
        
        # 检测异常值
        outlier_mask = (feature_data > upper_bound) | (feature_data < lower_bound)
        
        # 只标记整个样本为异常，如果异常值比例过高
        sample_outlier_ratio = np.sum(outlier_mask, axis=1) / feature_data.shape[1]
        extreme_outlier_samples = sample_outlier_ratio > 0.5  # 超过50%的时间步都是异常值
        
        outlier_mask_per_sample |= extreme_outlier_samples
        
        logger.info(f"  {feature_name}: 异常值比例 >50% 的样本数: {np.sum(extreme_outlier_samples)}")
    
    # 策略3: 检查标签异常值
    duration, event = y[:, 0], y[:, 1]
    duration_q1 = np.percentile(duration, 25)
    duration_q3 = np.percentile(duration, 75)
    duration_iqr = duration_q3 - duration_q1
    duration_upper_bound = duration_q3 + 3.0 * duration_iqr
    duration_lower_bound = duration_q1 - 3.0 * duration_iqr
    
    label_outlier_mask = (duration > duration_upper_bound) | (duration < duration_lower_bound)
    outlier_mask_per_sample |= label_outlier_mask
    
    # 统计异常值
    num_outliers = np.sum(outlier_mask_per_sample)
    outlier_ratio = num_outliers / len(X)
    
    logger.info(f"=== 异常值处理结果 ===")
    logger.info(f"  检测到的异常样本数: {num_outliers} ({outlier_ratio:.2%})")
    
    if num_outliers > 0:
        # 分析被移除的样本
        removed_samples = X[outlier_mask_per_sample]
        removed_labels = y[outlier_mask_per_sample]
        
        logger.info(f"  被移除样本的标签统计:")
        logger.info(f"    事件比例: {np.mean(removed_labels[:, 1]):.3f} (vs 总体 {np.mean(y[:, 1]):.3f})")
        logger.info(f"    平均持续时间: {np.mean(removed_labels[:, 0]):.3f} (vs 总体 {np.mean(y[:, 0]):.3f})")
        
        # 移除异常样本
        X_cleaned = X[~outlier_mask_per_sample]
        y_cleaned = y[~outlier_mask_per_sample]
        if record_ids_cleaned is not None:
            record_ids_cleaned = record_ids_cleaned[~outlier_mask_per_sample]
        
        logger.warning(f"  已移除 {num_outliers} 个异常样本 (占总数的 {outlier_ratio:.2%})")
    else:
        logger.info("  未发现需要移除的极端异常值")
    
    logger.info(f"  清理后的形状: X={X_cleaned.shape}, y={y_cleaned.shape}")
    
    # 保存异常值分析结果
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_dir = f"outlier_analysis_{set_name}_{timestamp}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 保存JSON分析结果
        analysis_file = os.path.join(analysis_dir, "outlier_analysis.json")
        simplified_analysis = {
            'summary': outlier_analysis['summary'],
            'feature_outliers_summary': {
                feature: {
                    'total_outliers': data['total_outliers'],
                    'outlier_ratio': data['outlier_ratio'],
                    'upper_bound': data['upper_bound'],
                    'lower_bound': data['lower_bound']
                } for feature, data in outlier_analysis['feature_outliers'].items()
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(simplified_analysis, f, indent=2, default=str)
        
        logger.info(f"  异常值分析结果已保存到: {analysis_dir}")
    except Exception as e:
        logger.warning(f"  保存异常值分析结果失败: {e}")
        import traceback
        logger.warning(f"  错误详情: {traceback.format_exc()}")
    
    return X_cleaned, y_cleaned, record_ids_cleaned 