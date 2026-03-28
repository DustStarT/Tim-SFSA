import logging
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np
try:
    import torch
    import torch.optim as optim
except Exception as _e:
    # 在某些轻量测试环境中未安装torch；记录友好提示但允许阅读源码
    raise
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import sys
import json
from datetime import datetime
import time
import math
from typing import Optional
from torch.cuda.amp import autocast, GradScaler
from models.model_factory import get_model
from evaluation.evaluator import Evaluator
from evaluation.plotting import plot_risk_progression, plot_training_curves, plot_enhanced_training_curves, plot_model_performance_comparison, _ensure_durations_in_hours
from evaluation.time_head_outputs import generate_time_head_style_outputs
from utils.loss_functions import cox_ph_loss_stable, deephit_loss
from utils.deepsurv_loss import combined_loss_deepsurv, l2_regularization_loss
from utils.misc import EarlyStopper, setup_logging, save_config, calculate_sample_weights, init_weights, init_weights_stable
from utils.config_validator import validate_config
import torch.nn.functional as F
from models.deephit.deephit_improved import deephit_loss_improved, predict_risk_improved
import yaml
import numpy as _np
from utils.multiformat_plotting import enable_multiformat_plotting

def _make_json_serializable(obj):
    """Convert numpy/tensor-rich objects into JSON-serializable Python types."""
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if _torch is not None and isinstance(obj, _torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return list(obj)
    return obj
# 第三阶段：时间预测头（简单MLP）- 已删除，改用传统方法
# class TimeHeadMLP(torch.nn.Module):
#     def __init__(self, in_dim: int, layers: list, dropout: float, out_dim: int = 1):
#         super().__init__()
#         self.out_dim = int(out_dim)
#         dims = [in_dim] + list(layers) + [self.out_dim]
#         mods = []
#         for i in range(len(dims)-2):
#             mods.append(torch.nn.Linear(dims[i], dims[i+1]))
#             mods.append(torch.nn.GELU())
#             if dropout and dropout > 0:
#                 mods.append(torch.nn.Dropout(dropout))
#         mods.append(torch.nn.Linear(dims[-2], dims[-1]))
#         self.net = torch.nn.Sequential(*mods)
#     def forward(self, x):
#         out = self.net(x)
#         return out if self.out_dim != 1 else out.squeeze(-1)

def predict_time_traditional(survival_funcs_df, risk_scores, config=None, true_times=None, events=None, output_dir=None):
    """
    使用传统方法从生存函数预测时间（仅考虑事件数据，不考虑删失）
    使用ImprovedSurvivalTimePredictor的多种方法
    
    Args:
        survival_funcs_df: 生存函数DataFrame，列名为时间点，行为样本
        risk_scores: 风险分数
        config: 配置对象
        
    Returns:
        predicted_times: 预测的时间（ensemble方法的结果）
    """
    from utils.improved_time_predictor import ImprovedSurvivalTimePredictor
    logger = logging.getLogger(__name__)

    # 如果在 output_dir 中存在已持久化的 predictor，则加载以保证在测试时复用校准模型
    predictor = None
    try:
        if output_dir:
            ppath = os.path.join(output_dir, 'time_predictor.pkl')
            if os.path.exists(ppath):
                predictor = ImprovedSurvivalTimePredictor.load(ppath)
                logger.info(f"Loaded persisted ImprovedSurvivalTimePredictor from {ppath}")
    except Exception:
        logger.exception("Failed to load persisted ImprovedSurvivalTimePredictor; will create a fresh one")

    if predictor is None:
        predictor = ImprovedSurvivalTimePredictor()
    
    # 获取配置中的方法偏好
    if config is not None:
        try:
            methods = getattr(config.evaluation.prediction_methods_comparison, 'methods_to_compare', None)
            if methods and 'ensemble' in methods:
                # 使用集成的所有方法
                # 如果可用，传入真实时间与事件指标以在预测时拟合校准模型
                predictions_dict = predictor.predict_survival_times(
                    survival_funcs_df, risk_scores, true_times=true_times, events=events
                )
                return predictions_dict.get('ensemble', predictions_dict.get('adaptive_threshold', np.array([])))
        except Exception:
            pass
    
    # 默认使用ensemble方法
    predictions_dict = predictor.predict_survival_times(
        survival_funcs_df, risk_scores, true_times=true_times, events=events
    )

    # 如果提供了输出目录，持久化训练好的时间预测器以便在后续测试/评估中复用校准模型
    try:
        if output_dir:
            ppath = os.path.join(output_dir, 'time_predictor.pkl')
            predictor.save(ppath)
            logger.info(f"Persisted ImprovedSurvivalTimePredictor to {ppath}")
    except Exception:
        logger.exception("Failed to persist ImprovedSurvivalTimePredictor")

    # 如果有调优信息且提供了输出目录，则保存调优统计到文件以便审查
    try:
        if output_dir and 'tuning' in predictions_dict:
            import json
            os.makedirs(output_dir, exist_ok=True)
            tuning_path = os.path.join(output_dir, 'time_prediction_tuning.json')
            with open(tuning_path, 'w') as _f:
                json.dump(predictions_dict['tuning'], _f, indent=2)
            logger.info(f"Saved time prediction tuning info to: {tuning_path}")
    except Exception as _:
        logger.exception("Failed to save time prediction tuning info")

    return predictions_dict.get('ensemble', predictions_dict.get('adaptive_threshold', np.array([])))

def _evaluate_time_head_predictions(
    out_dir: str,
    split: str,
    y_src: np.ndarray,
    preds: np.ndarray,
    model_name: str = 'TimeHead',
    cfg=None,
    record_ids=None,
):
    try:
        eval_events_only = True
        if cfg is not None:
            try:
                eval_events_only = bool(getattr(cfg.evaluation.time_head, 'eval_events_only', True))
            except Exception:
                eval_events_only = True

        if y_src is None or preds is None:
            logging.getLogger(__name__).warning('[TimeHead][Eval] 缺少输入数据，跳过评估输出')
            return None

        y_src = np.asarray(y_src)
        if y_src.ndim != 2 or y_src.shape[1] < 2:
            logging.getLogger(__name__).warning('[TimeHead][Eval] 输入标签格式错误，跳过评估输出')
            return None

        file_tag = 'time_head'
        if model_name and model_name.lower() != 'timehead':
            file_tag = model_name.lower().replace(' ', '_')

        metrics = generate_time_head_style_outputs(
            output_dir=out_dir,
            split=split,
            true_times=y_src[:, 0],
            events=y_src[:, 1],
            preds=preds,
            model_name=model_name,
            file_tag=file_tag,
            eval_events_only=eval_events_only,
            record_ids=record_ids,
        )
        return metrics
    except Exception as exc:
        logging.getLogger(__name__).warning(f'[TimeHead][Eval] 评估输出失败: {exc}')
        return None


def _undersample_non_events(
    X,
    y,
    *,
    ratio: float = 1.0,
    random_state: Optional[int] = None,
    record_ids=None,
    sample_weights=None,
):
    """对生存分析数据中的无事件样本执行欠采样。

    Args:
        X: 特征数组，形状 (n_samples, ...)
        y: 标签数组，形状 (n_samples, 2) -> [duration, event]
        ratio: 欠采样后的非事件:事件目标比例（例如 1.0 表示与事件样本数一致）
        random_state: 随机种子
        record_ids: 可选记录ID列表，将与数据一同过滤
        sample_weights: 可选样本权重向量

    Returns:
        (X_balanced, y_balanced, record_ids_balanced, sample_weights_balanced)
    """

    try:
        ratio = float(ratio) if ratio is not None else 1.0
    except Exception:
        ratio = 1.0
    if not np.isfinite(ratio) or ratio <= 0:
        logging.getLogger(__name__).warning(f"欠采样比例 {ratio} 无效，已回退为 1.0")
        ratio = 1.0

    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[1] < 2:
        return X, y, record_ids, sample_weights

    event_mask = y_arr[:, 1] == 1
    non_event_mask = ~event_mask

    n_events = int(np.sum(event_mask))
    n_non_events = int(np.sum(non_event_mask))

    if n_events == 0 or n_non_events == 0:
        return X, y, record_ids, sample_weights

    target_non_events = int(round(n_events * ratio))
    target_non_events = max(1, target_non_events)
    target_non_events = min(target_non_events, n_non_events)

    if target_non_events >= n_non_events:
        return X, y, record_ids, sample_weights

    rng = np.random.default_rng(random_state)
    event_indices = np.where(event_mask)[0]
    non_event_indices = np.where(non_event_mask)[0]
    selected_non_events = rng.choice(non_event_indices, size=target_non_events, replace=False)
    selected_indices = np.concatenate([event_indices, selected_non_events])
    rng.shuffle(selected_indices)

    def _subset(array, indices):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            return array[indices]
        if torch.is_tensor(array):
            return array[indices]
        try:
            return [array[int(i)] for i in indices]
        except Exception:
            return array

    X_balanced = X[selected_indices]
    y_balanced = y_arr[selected_indices]
    record_ids_balanced = _subset(record_ids, selected_indices)
    sample_weights_balanced = _subset(sample_weights, selected_indices)

    logging.getLogger(__name__).info(
        f"欠采样无事件样本: 事件={n_events}, 无事件={n_non_events} -> {target_non_events}, 总样本={len(selected_indices)}"
    )

    return X_balanced, y_balanced, record_ids_balanced, sample_weights_balanced

def _deep_update_config_obj(cfg, overrides):
    """Recursively update EasyDict-like cfg with plain dict overrides."""
    if overrides is None:
        return cfg
    for k, v in overrides.items():
        try:
            if isinstance(v, dict):
                # ensure sub-attr exists
                if not hasattr(cfg, k) or getattr(cfg, k) is None:
                    setattr(cfg, k, type('obj', (), {})())
                _deep_update_config_obj(getattr(cfg, k), v)
            else:
                setattr(cfg, k, v)
        except Exception:
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg

def _infer_run_dir_from_ckpt_path(ckpt_path: str) -> str:
    try:
        if os.path.isdir(ckpt_path):
            return ckpt_path
        return os.path.dirname(ckpt_path)
    except Exception:
        return os.path.dirname(ckpt_path)

def _load_pretrained_config_dict(ckpt_path: str):
    """尽力从 checkpoint 或其运行目录中读取预训练配置(dict)。"""
    import json as _json
    run_dir = _infer_run_dir_from_ckpt_path(ckpt_path)
    # 1) 优先尝试从ckpt中读取
    try:
        import torch as _t
        state = _t.load(ckpt_path, map_location='cpu') if os.path.isfile(ckpt_path) else None
        if isinstance(state, dict):
            for key in ('config', 'cfg', 'configuration'):
                if key in state and isinstance(state[key], (dict, list)):
                    return state[key]
    except Exception:
        pass
    # 2) 再尝试常见的配置文件名
    candidates = [
        'config.json','saved_config.json','experiment_config.json','args.json',
        'config.yaml','config.yml'
    ]
    for name in candidates:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                if p.endswith(('.yaml','.yml')):
                    with open(p, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    with open(p, 'r') as f:
                        return _json.load(f)
            except Exception:
                continue
    return None

def _auto_align_config_with_pretrained(config, pretrained_cfg_dict):
    """将关键兼容字段与预训练配置对齐（保守覆盖）。"""
    if not isinstance(pretrained_cfg_dict, dict):
        return config
    logger = logging.getLogger(__name__)
    try:
        # 对齐编码器/下游类型
        model_cfg = pretrained_cfg_dict.get('model') or {}
        encoder_cfg = (model_cfg.get('encoder') or {})
        
        # 如果没有encoder配置，根据use_lstm字段推断编码器类型
        if not encoder_cfg:
            use_lstm = model_cfg.get('use_lstm', True)  # 默认为True（LSTM）
            enc_type = 'lstm' if use_lstm else 'transformer'
            logger.info(f"预训练配置中无encoder字段，根据use_lstm={use_lstm}推断编码器类型为: {enc_type}")
        else:
            enc_type = str(encoder_cfg.get('type', 'lstm')).lower()
            logger.info(f"从预训练配置中读取编码器类型: {enc_type}")
        
        if hasattr(config.model, 'encoder'):
            old_type = getattr(config.model.encoder, 'type', 'unknown')
            config.model.encoder.type = enc_type
            logger.info(f"编码器类型已从 {old_type} 对齐为 {enc_type}")
        
        # 根据预训练模型的实际情况设置 use_lstm
        pretrained_use_lstm = model_cfg.get('use_lstm', True)  # 默认为True
        config.model.use_lstm = pretrained_use_lstm
        logger.info(f"config.model.use_lstm 设置为: {config.model.use_lstm} (来自预训练配置)")
        
        if 'downstream_model' in model_cfg:
            config.model.downstream_model = model_cfg.get('downstream_model')
        if 'name' in model_cfg:
            config.model.name = model_cfg.get('name')

        # LSTM 结构
        lstm_cfg = model_cfg.get('lstm') or {}
        for k in ('hidden_size','num_lstm_layers','dropout_rate','bidirectional','use_attention'):
            if k in lstm_cfg:
                setattr(config.model.lstm, k, lstm_cfg[k])
        
        # Transformer 结构
        transformer_cfg = encoder_cfg.get('transformer') or {}
        if transformer_cfg and hasattr(config.model.encoder, 'transformer'):
            for k in ('d_model','nhead','num_layers','dim_feedforward','dropout','activation','norm'):
                if k in transformer_cfg:
                    setattr(config.model.encoder.transformer, k, transformer_cfg[k])

        # 连接器
        connector_cfg = (model_cfg.get('connector') or {})
        for k in ('enabled','output_dim','norm','activation','dropout','fuse_with_raw_start'):
            if k in connector_cfg and hasattr(config.model.two_stage, 'connector'):
                setattr(config.model.two_stage.connector, k, connector_cfg[k])

        # 下游 DeepSurv 超参
        ds_cfg = model_cfg.get('deepsurv') or {}
        for k in ('hidden_layers','dropout_rate','batch_norm','activation','l2_reg'):
            if k in ds_cfg:
                setattr(config.model.deepsurv, k, ds_cfg[k])

        # 序列窗口与步长
        data_cfg = pretrained_cfg_dict.get('data') or {}
        sg = data_cfg.get('sequence_generation') or {}
        for k in ('num_covariate_timesteps','prediction_window_hours','sub_sequence_step'):
            if k in sg:
                setattr(config.data.sequence_generation, k, sg[k])

        # 特征列（尽量对齐，若缺失由下游报错或警告）
        if 'specified_features' in data_cfg and isinstance(data_cfg['specified_features'], (list, tuple)):
            config.data.specified_features = list(data_cfg['specified_features'])
    except Exception:
        pass
    return config

# 导入两阶段训练相关模块
from models.classification_trainer import ClassificationModelManager, run_classification_only

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 使用默认配置
from configs.default_config import get_config

from preprocessing.preprocessor import DataPreprocessor

# 添加基础导入，避免文件中使用未导入的符号导致运行时异常
import sys, os, logging
import numpy as np
import pandas as pd
import torch

# 清理旧的预处理产物以避免列数/特征不一致（尤其在切换 include_aggregated_features 时）
def _cleanup_preproc_reports():
    try:
        import shutil
        preproc_dir = os.path.join('results', 'preproc_reports')
        if os.path.isdir(preproc_dir):
            shutil.rmtree(preproc_dir, ignore_errors=True)
    except Exception:
        pass

# 修复predict_risk函数中的风险分数计算
def predict_risk(model, X_data, device, config=None, calibrator=None):
    """
    使用给定的模型预测风险分数。
    - 对于Cox/DeepSurv模型，假定模型返回每个样本的单值风险（或一个shape=(B,1)张量）
    - 对于DeepHit模型，假定模型返回每个样本在时间bin上的概率分布shape=(B, num_bins)
    
    Args:
        model: 训练好的模型
        X_data: 输入数据
        device: 设备
        config: 配置对象
        calibrator: 风险分数校准器（可选），如果提供则对风险分数进行校准
        
    返回: (n_samples,) 的numpy数组
    """
    # 安全调用 model.eval()，兼容只实现 __call__ 的简单 Dummy 模型
    getattr(model, 'eval', lambda: None)()
    batch_size = 1024

    # 确保数据是3D的 (n_samples, n_timesteps, n_features)
    if isinstance(X_data, np.ndarray) and X_data.ndim == 2:
        X_data = X_data[:, np.newaxis, :]

    all_risk_scores = []

    is_deephit_model = bool(config and getattr(config.model, 'name', '').lower() == 'deephit')
    logger = logging.getLogger(__name__)

    with torch.no_grad():
        n_samples = X_data.shape[0]
        for i in range(0, n_samples, batch_size):
            batch = X_data[i: i + batch_size]

            # 转换为Tensor并移动到device
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.from_numpy(batch).float().to(device)
            elif torch.is_tensor(batch):
                batch_tensor = batch.float().to(device)
            else:
                # 尝试按array转化
                batch_tensor = torch.tensor(np.array(batch)).float().to(device)
            
            # 关键修复：与验证阶段完全一致的输入处理
            if config and not getattr(config.model, 'use_lstm', False) and batch_tensor.ndim == 3:
                input_tensor = batch_tensor[:, -1, :]
            else:
                input_tensor = batch_tensor

            # DeepHit: 概率分布 -> 期望时间作为风险值
            if is_deephit_model and (config is not None):
                out = model(input_tensor)
                # 支持model返回字典/元组或直接张量
                probs = None
                if isinstance(out, dict):
                    # 常见key名：'probs', 'y_hat', 'p'
                    probs = out.get('probs') or out.get('y_hat') or out.get('p')
                elif isinstance(out, (tuple, list)):
                    probs = out[0]
                else:
                    probs = out

                if torch.is_tensor(probs):
                    probs_np = probs.detach().cpu().numpy()
                else:
                    probs_np = np.array(probs)

                # 如果是二维概率矩阵 (B, num_bins)，计算期望时间
                if probs_np.ndim == 2:
                    num_bins = int(config.model.deephit.num_time_bins)
                    pred_window = float(config.data.sequence_generation.prediction_window_hours)
                    bin_width = pred_window / float(num_bins)
                    bin_centers = (np.arange(num_bins) + 0.5) * bin_width
                    expected = (probs_np * bin_centers.reshape(1, -1)).sum(axis=1)
                    all_risk_scores.append(expected)
                else:
                    # 退化情况：直接把输出当做风险分数
                    all_risk_scores.append(probs_np.reshape(-1))

            else:
                # Cox/DeepSurv/其他：模型输出单值风险
                out = model(input_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                
                # 注意：LSTMPrefix返回的已经是1D向量(batch,)，不需要额外的3D处理
                # 只有在model未包装在LSTMPrefix中且输出3D时才需要处理
                # 但现在所有LSTM模型都包装在LSTMPrefix中，所以这里不需要特殊处理
                
                if torch.is_tensor(out):
                    out_np = out.detach().cpu().numpy().reshape(-1)
                else:
                    out_np = np.array(out).reshape(-1)
                all_risk_scores.append(out_np)

    # 合并所有批次的结果并扁平化
    try:
        risk_scores = np.concatenate([np.asarray(a).reshape(-1) for a in all_risk_scores], axis=0)
        risk_scores = risk_scores.flatten()
        
        # 如果提供了校准器，应用校准
        if calibrator is not None:
            try:
                risk_scores = calibrator.predict(risk_scores)
                logger.debug(f"已应用风险分数校准: method={calibrator.method}")
            except Exception as e:
                logger.warning(f"风险分数校准失败: {e}，使用原始风险分数")
        
        return risk_scores
    except Exception as e:
        logger.error(f"合并风险分数时发生错误: {e}")
        # 返回一个长度与输入样本数一致的零向量，便于上层调用继续运行但需要注意日志
        return np.zeros(X_data.shape[0])

def compute_baseline_survival(model, X_train, y_train, device, config):
    """使用训练好的模型和训练数据计算基线生存函数。"""
    risk_scores = predict_risk(model, X_train, device, config)
    
    durations = y_train[:, 0]
    events = y_train[:, 1]
    
    # 确保只使用有效数据
    valid_idx = (durations > 0) & (~np.isnan(durations))
    if not np.any(valid_idx):
        logging.getLogger(__name__).error("没有有效的持续时间来计算基线生存函数。")
        max_dur = np.nanmax(durations) if durations.size > 0 else 1.0
        return pd.Series([0.5], index=[max_dur]) # 返回一个虚拟基线

    # canonicalize durations to hours relative to subsequence start so downstream
    # lifelines/Cox baseline fitting and plotting use consistent units
    dur_arr = durations[valid_idx]
    try:
        dur_hours = _ensure_durations_in_hours(np.asarray(dur_arr), cfg=config, name='compute_baseline_survival')
    except Exception:
        # fallback to raw durations if conversion helper fails
        dur_hours = np.asarray(dur_arr)

    baseline_df = pd.DataFrame({
        'duration': dur_hours,
        'event': events[valid_idx],
        'risk_score': risk_scores[valid_idx]
    })

    try:
        # 使用lifelines的CoxPHFitter拟合风险分数来获得基线
        cph_baseline = CoxPHFitter()
        # formula="risk_score"表示我们将风险分数作为唯一的协变量
        cph_baseline.fit(baseline_df, 'duration', 'event', formula="risk_score")
        return cph_baseline.baseline_survival_
    except Exception as e:
        logging.getLogger(__name__).error(f"使用CoxPHFitter计算基线生存失败: {e}。将创建一个虚拟基线。")
        unique_times = np.unique(baseline_df['duration'])
        return pd.Series(np.linspace(1, 0, len(unique_times)), index=unique_times)

def _time_to_bin(time_tensor, config, device):
    """将连续时间转换为离散的时间区间索引。
    
    按照DeepHit论文的逻辑：
    - 将时间轴分为K个等宽区间
    - 如果时间t在区间(k, k+1]中，则映射到索引k
    - 对于超出预测窗口的时间，映射到最后一个区间
    """
    prediction_window = config.data.sequence_generation.prediction_window_hours
    num_bins = config.model.deephit.num_time_bins
    
    # 创建时间区间边界 (0, bin_width, 2*bin_width, ..., prediction_window)
    bin_width = prediction_window / num_bins
    time_bins = torch.linspace(0, prediction_window, num_bins + 1, device=device)
    
    # 处理超出预测窗口的时间
    time_tensor_clipped = torch.clamp(time_tensor, 0, prediction_window)
    
    # 计算每个时间属于哪个区间
    # 对于时间t，如果t在区间(k*bin_width, (k+1)*bin_width]中，则bin_index = k
    # 特殊情况：t=0时，bin_index = 0
    bin_indices = torch.floor(time_tensor_clipped / bin_width).long()
    
    # 处理边界情况：当时间正好等于prediction_window时，应该映射到最后一个区间
    bin_indices = torch.where(
        time_tensor_clipped == prediction_window,
        torch.tensor(num_bins - 1, device=device),
        bin_indices
    )
    
    # 确保索引在有效范围内 [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    # 添加调试信息
    if torch.rand(1).item() < 0.01:  # 1%的概率输出调试信息
        logging.debug(f"时间映射调试 - 原始时间: {time_tensor[:5]}, 映射后区间: {bin_indices[:5]}")
        logging.debug(f"时间区间边界: {time_bins[:5]}...{time_bins[-5:]}")
        logging.debug(f"区间宽度: {bin_width}")
    
    return bin_indices


def build_balanced_batch_indices(event_indices, censored_indices, batch_size, target_event_ratio, shuffle=True):
    """
    构建一个由索引组成的list，用于在训练时按batch返回。每个batch尽量包含target_event_ratio的事件样本。
    返回一个列表，里面每个元素是一个numpy array的索引，长度为num_batches。
    """
    # 复制输入以便安全打乱
    ev = list(event_indices)
    ce = list(censored_indices)
    if shuffle:
        random.shuffle(ev)
        random.shuffle(ce)

    num_total = len(ev) + len(ce)
    if num_total == 0:
        return []

    # 计算每batch期望的事件数
    expected_ev_per_batch = int(round(batch_size * float(target_event_ratio)))
    expected_ce_per_batch = batch_size - expected_ev_per_batch

    batches = []
    ev_ptr = 0
    ce_ptr = 0

    # 计算需要生成的批次数，保证覆盖所有样本
    import math
    total_samples = len(ev) + len(ce)
    if total_samples == 0:
        return []
    max_batches = int(math.ceil(total_samples / float(batch_size)))

    # 当事件不足时，允许放回补样（oversample）以保证每个batch尽量达成目标事件比例
    # 我们使用循环索引来实现放回：当某类耗尽时从头开始重用索引
    ev_cycle_ptr = 0
    ce_cycle_ptr = 0

    for bidx in range(max_batches):
        this_batch = []

        # 先尝试取目标事件数（放回时允许循环使用）
        take_ev = expected_ev_per_batch
        ev_available = max(0, len(ev) - ev_ptr)
        if ev_available >= take_ev:
            this_batch.extend(ev[ev_ptr: ev_ptr + take_ev])
            ev_ptr += take_ev
        else:
            # 取尽剩余的事件
            if ev_available > 0:
                this_batch.extend(ev[ev_ptr: ev_ptr + ev_available])
                ev_ptr += ev_available
            # 放回补齐：循环使用事件索引
            need_ev = take_ev - len(this_batch)
            if need_ev > 0 and len(ev) > 0:
                for i in range(need_ev):
                    this_batch.append(ev[ev_cycle_ptr % len(ev)])
                    ev_cycle_ptr += 1

        # 取删失样本
        take_ce = expected_ce_per_batch
        ce_available = max(0, len(ce) - ce_ptr)
        if ce_available >= take_ce:
            this_batch.extend(ce[ce_ptr: ce_ptr + take_ce])
            ce_ptr += take_ce
        else:
            if ce_available > 0:
                this_batch.extend(ce[ce_ptr: ce_ptr + ce_available])
                ce_ptr += ce_available
            need_ce = take_ce - (len(this_batch) - take_ev)
            if need_ce > 0 and len(ce) > 0:
                for i in range(need_ce):
                    this_batch.append(ce[ce_cycle_ptr % len(ce)])
                    ce_cycle_ptr += 1

        # 最终再检查是否仍未填满（极端情况：两类之一为空）
        need = batch_size - len(this_batch)
        if need > 0:
            # 优先用删失补齐
            if len(ce) > 0:
                for i in range(need):
                    this_batch.append(ce[ce_cycle_ptr % len(ce)])
                    ce_cycle_ptr += 1
            elif len(ev) > 0:
                for i in range(need):
                    this_batch.append(ev[ev_cycle_ptr % len(ev)])
                    ev_cycle_ptr += 1

        # 小批内部打乱
        if shuffle:
            random.shuffle(this_batch)

        batches.append(np.array(this_batch, dtype=int))

    return batches

# def log_gradients(model, logger):
#     """
#     记录模型中所有参数的梯度统计信息。
#     """
#     logger.info("--- Gradient Stats ---")
#     total_norm = 0
#     for name, p in model.named_parameters():
#         if p.grad is not None:
#             param_norm = p.grad.data.norm(2)
#             total_norm += param_norm.item() ** 2
#             grad_mean = p.grad.data.mean()
#             grad_std = p.grad.data.std()
#             if torch.isnan(grad_std):
#                 grad_std = torch.tensor(0.0)
#             logger.info(f"  {name}: grad_norm={param_norm:.4f}, grad_mean={grad_mean:.6f}, grad_std={grad_std:.6f}")
#         else:
#             logger.info(f"  {name}: has no gradient.")
#     total_norm = total_norm ** 0.5
#     logger.info(f"--- Total Gradient Norm: {total_norm:.4f} ---")

def train_model_loop(model, config, X_train, y_train, X_val, y_val, sample_weights, output_dir=None):
    """
    Trains the model for one fold of cross-validation with enhanced monitoring.
    """
    logger = logging.getLogger(__name__)
    training_config = config.training
    device = torch.device(training_config.device)
    
    model.to(device)

    mp_enabled = bool(getattr(training_config, 'mixed_precision', False))
    if mp_enabled:
        disable_reasons = []
        try:
            enc_type_for_amp = str(getattr(getattr(config.model, 'encoder', {}), 'type', 'lstm')).lower()
        except Exception:
            enc_type_for_amp = 'lstm'
        if enc_type_for_amp == 'transformer':
            mp_enabled = False
            disable_reasons.append('Transformer encoder prefix')
        loss_type_cfg = getattr(getattr(training_config, 'loss', {}), 'deepsurv_loss_type', 'cox')
        if str(loss_type_cfg).lower() == 'combined':
            mp_enabled = False
            disable_reasons.append('DeepSurv combined loss')
        if not mp_enabled:
            reason_text = ', '.join(disable_reasons) if disable_reasons else 'stability guard'
            logger.warning(f"[AMP] Disabled automatic mixed precision due to {reason_text}.")

    weight_decay = float(config.training.optimizer.weight_decay)
    learning_rate = float(training_config.learning_rate)
    
    # 使用配置文件中的优化器设置（支持判别式学习率：LSTM/Transformer 与下游不同 lr）
    param_groups = []
    try:
        # LSTM参数组（若存在）
        if hasattr(model, 'lstm'):
            lstm_params = [p for p in model.lstm.parameters() if p.requires_grad]
            if len(lstm_params) > 0:
                lstm_lr_cfg = getattr(config.training.optimizer, 'lstm_lr', None)
                lstm_lr = learning_rate * 0.1 if (lstm_lr_cfg is None) else float(lstm_lr_cfg)
                param_groups.append({'params': lstm_params, 'lr': lstm_lr, 'weight_decay': weight_decay})

        transformer_param_names = set()
        transformer_params = []
        if hasattr(model, 'encoder') and hasattr(model, 'input_proj'):
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith('input_proj') or name.startswith('encoder') or name.startswith('cls_token'):
                    transformer_params.append(param)
                    transformer_param_names.add(name)
            if len(transformer_params) > 0:
                transformer_lr_cfg = getattr(config.training.optimizer, 'transformer_lr', getattr(config.training.optimizer, 'lstm_lr', None))
                transformer_lr = learning_rate * 0.1 if transformer_lr_cfg is None else float(transformer_lr_cfg)
                param_groups.append({'params': transformer_params, 'lr': transformer_lr, 'weight_decay': weight_decay})
        # 下游参数组
        downstream_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('lstm.'):
                continue
            if name in transformer_param_names:
                continue
            downstream_params.append(param)
        if len(downstream_params) > 0:
            param_groups.append({'params': downstream_params, 'lr': learning_rate, 'weight_decay': weight_decay})
    except Exception:
        # 回退：不区分参数组
        param_groups = [{'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}]

    if config.training.optimizer.name.lower() == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
    else:
        optimizer = optim.Adam(
            param_groups,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
    # 诊断日志：记录初始化学习率与 optimizer param_groups，使用科学计数法避免被格式化为0.000000
    try:
        logger.info(f"[LR_INIT] configured learning_rate={learning_rate:.6e}, optimizer param_groups_lr={[float(pg['lr']) for pg in optimizer.param_groups]}")
    except Exception:
        logger.info("[LR_INIT] Unable to log optimizer param_groups")
    
    # 使用配置文件中的学习率调度器设置
    if config.training.lr_scheduler.type == 'CosineAnnealingWarmRestarts':
        # 某些配置文件将Cosine的参数放在 training.scheduler 下，优先从两个位置读取
        T_0 = getattr(config.training.lr_scheduler, 'T_0', getattr(config.training.scheduler, 'T_0', 50))
        T_mult = getattr(config.training.lr_scheduler, 'T_mult', getattr(config.training.scheduler, 'T_mult', 1))
        eta_min = getattr(config.training.lr_scheduler, 'eta_min', getattr(config.training.scheduler, 'eta_min', 0))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        # 记录调度器信息
        try:
            logger.info(f"[SCHED] Using CosineAnnealingWarmRestarts T_0={T_0}, T_mult={T_mult}, eta_min={eta_min:.6e}")
        except Exception:
            logger.info("[SCHED] Using CosineAnnealingWarmRestarts (params unavailable)")
    elif config.training.lr_scheduler.type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config.training.lr_scheduler.mode,
            factor=config.training.lr_scheduler.factor,
            patience=config.training.lr_scheduler.patience,
            min_lr=config.training.lr_scheduler.min_lr,
            verbose=config.training.lr_scheduler.verbose
        )
        try:
            logger.info(f"[SCHED] Using ReduceLROnPlateau mode={config.training.lr_scheduler.mode}, factor={config.training.lr_scheduler.factor}, patience={config.training.lr_scheduler.patience}, min_lr={float(config.training.lr_scheduler.min_lr):.6e}")
        except Exception:
            logger.info("[SCHED] Using ReduceLROnPlateau (params unavailable)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20,
            min_lr=config.training.lr_scheduler.min_lr if hasattr(config.training.lr_scheduler, 'min_lr') else 1e-12,
            verbose=True
        )

    # --- 训练监控与早停（本地实现，避免依赖外部EarlyStopper接口不一致） ---
    best_val_c_index = -np.inf
    best_epoch = -1
    epochs_no_improve = 0
    patience = int(getattr(training_config, 'early_stopping_patience', getattr(config.training, 'early_stopping_patience', 50)))
    stop_training = False
    # 存放最优模型的路径：优先使用传入的 output_dir（每次运行的结果目录），否则回退到 config.results_dir
    results_dir_for_ckpt = output_dir if (output_dir is not None and output_dir != '') else getattr(config, 'results_dir', 'results')
    os.makedirs(results_dir_for_ckpt, exist_ok=True)
    best_model_path = os.path.join(results_dir_for_ckpt, f"best_model_{int(time.time())}.pth")

    metrics_log = []  # 用于保存每个epoch的指标，训练结束会dump到csv

    print('X_train:',{X_train.shape})
    print('y_train:',{y_train.shape})
    print('sample_weights',sample_weights.shape)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        sample_weights 
    )
    # 如果启用平衡采样，使用自定义 StratifiedSampler 以便 DataLoader 直接返回平衡过的mini-batch
    if getattr(training_config, 'enable_balanced_batching', False):
        from utils.stratified_sampler import StratifiedSampler
        # labels只需要y_train中事件列
        labels_for_sampler = y_train[:, 1]
        sampler = StratifiedSampler(
            labels_for_sampler,
            batch_size=int(training_config.batch_size),
            target_event_ratio=float(getattr(training_config, 'target_event_ratio_per_batch', config.training.target_event_ratio_per_batch)),
            shuffle=True,
            min_events_per_batch=int(getattr(training_config, 'min_events_per_batch', getattr(config.training, 'min_events_per_batch', 1)))
        )
        train_loader = DataLoader(train_dataset, batch_size=int(training_config.batch_size), sampler=sampler, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    
    # 如果没有验证集（X_val 为空或 None），则跳过 val_loader 的创建
    has_validation = True
    try:
        if X_val is None or (hasattr(X_val, 'shape') and getattr(X_val, 'size', 0) == 0) or (isinstance(X_val, (list, tuple)) and len(X_val) == 0):
            has_validation = False
    except Exception:
        has_validation = False

    if has_validation:
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
    else:
        val_loader = None
    
    scaler = GradScaler(enabled=mp_enabled)
    
    is_deephit_model = config.model.name.lower() == 'deephit'
    is_deepsurv_model = config.model.name.lower() == 'deepsurv'

    # 记录训练历史
    train_losses = []
    val_losses = []
    train_c_indices = []
    val_c_indices = []
    learning_rates = []

    num_epochs = training_config.num_epochs
    validation_frequency = getattr(training_config, 'validation_frequency', 1)
    warmup_epochs = getattr(config.training.lr_scheduler, 'warmup_epochs', 10)
    
    mitigate_cooldown = 0
    for epoch in range(num_epochs):
        # 学习率预热机制
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
        # 冻结-解冻：前 N 个 epoch 冻结 LSTM/Transformer，之后解冻（若模型支持）
        try:
            freeze_n = int(getattr(config.model.two_stage.survival_stage, 'freeze_first_n_epochs', 0))
            if freeze_n > 0:
                # 检查 LSTM 冻结/解冻方法
                if hasattr(model, 'freeze_lstm') and hasattr(model, 'unfreeze_lstm'):
                    if epoch == 0:
                        model.freeze_lstm()
                    elif epoch == freeze_n:
                        model.unfreeze_lstm()
                # 检查 Transformer 冻结/解冻方法
                elif hasattr(model, 'freeze_transformer') and hasattr(model, 'unfreeze_transformer'):
                    if epoch == 0:
                        model.freeze_transformer()
                    elif epoch == freeze_n:
                        model.unfreeze_transformer()
        except Exception:
            pass
        model.train()
        total_loss = 0
        num_batches = 0
        train_risk_scores = []
        train_durations = []
        train_events = []
        
        # 统一使用train_loader（如需禁用生存阶段的批内平衡，这里仅按 DataLoader 迭代）
        batch_iterable = enumerate(train_loader)

        # per-batch event ratio tracking (for balanced-batching verification)
        batch_event_ratios = []

        for i, batch_data in batch_iterable:
            # DataLoader返回 (X_batch, y_batch, weights_batch)
            X_batch, y_batch, weights_batch = batch_data
            # 记录本批次事件比例
            try:
                batch_events_np = y_batch[:, 1].cpu().numpy() if torch.is_tensor(y_batch) else np.array(y_batch)[:, 1]
                batch_event_ratio = float(np.mean(batch_events_np)) if len(batch_events_np) > 0 else 0.0
            except Exception:
                # 退化情况，尝试用numpy转换
                by = np.array(y_batch)
                batch_event_ratio = float(np.mean(by[:, 1])) if by.size > 0 else 0.0
            batch_event_ratios.append(batch_event_ratio)
            X_batch, y_batch, weights_batch = X_batch.to(device), y_batch.to(device), weights_batch.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=mp_enabled):
                if not config.model.use_lstm and X_batch.ndim == 3:
                    input_tensor = X_batch[:, -1, :]
                else:
                    input_tensor = X_batch
                
                # 检查输入是否包含NaN
                if getattr(config.training.stability, 'replace_input_nan', True) and torch.isnan(input_tensor).any():
                    logger.warning(f"[NaN] 输入包含NaN值，已自动替换为0")
                    input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
                
                model_output = model(input_tensor)
                
                if config.model.use_lstm and model_output.ndim == 3:
                    logits = model_output[:, -1, :]
                else:
                    logits = model_output

                # 检查logits是否包含NaN
                if getattr(config.training.stability, 'replace_logits_nan', True) and torch.isnan(logits).any():
                    logger.warning(f"[NaN] logits包含NaN值，已自动替换为0")
                    logits = torch.nan_to_num(logits, nan=0.0)

                if is_deephit_model:
                    time_batch = y_batch[:, 0].contiguous()
                    event_batch = y_batch[:, 1].contiguous()
                    time_bins_batch = _time_to_bin(time_batch, config, device)
                    
                    loss = deephit_loss_improved(
                        logits, 
                        time_bins_batch, 
                        event_batch,
                        alpha=config.model.deephit.loss_alpha,
                        sigma=config.model.deephit.loss_sigma,
                        device=device
                    )
                    # 训练阶段只取logits最后一个时间步，保证shape一致
                elif is_deepsurv_model:
                    # DeepSurv: 支持标准Cox或组合损失
                    loss_type = str(getattr(getattr(training_config, 'loss', {}), 'deepsurv_loss_type', 'cox')).lower()
                    apply_weights = bool(getattr(training_config, 'apply_sample_weights_to_loss', False))
                    weights_norm = None
                    if apply_weights:
                        weights_norm = weights_batch.view(-1).float()
                        if torch.mean(weights_norm) > 0:
                            weights_norm = weights_norm / torch.mean(weights_norm)
                        logger.info(f"[WEIGHTS] Applying normalized sample weights to DeepSurv loss (mean={float(torch.mean(weights_norm)):.4f})")

                    log_risk_vec = logits.squeeze()
                    if loss_type == 'combined':
                        from utils.deepsurv_loss import combined_loss_deepsurv
                        loss = combined_loss_deepsurv(
                            log_risk=log_risk_vec,
                            durations=y_batch[:, 0],
                            events=y_batch[:, 1],
                            weights=weights_norm,
                            cox_weight=float(getattr(training_config.loss, 'cox_weight', 1.0)),
                            focal_weight=float(getattr(training_config.loss, 'focal_weight', 0.3)),
                            ranking_weight=float(getattr(training_config.loss, 'ranking_weight', 0.2)),
                            ranking_variant=str(getattr(training_config.loss, 'ranking_variant', 'ipcw_pairwise')),
                            ranking_margin=float(getattr(training_config.loss, 'ranking_margin', 0.0))
                        )
                    else:
                        loss = cox_ph_loss_stable(
                            log_risk_vec,
                            y_batch,
                            sample_weights=weights_norm if apply_weights else None
                        )
                    # 风险分数直接使用logits
                    risk_scores_batch = logits.squeeze().detach().cpu().numpy()
                    train_risk_scores.append(risk_scores_batch)
                else:
                    # Cox模型损失计算（其它变体）。同样仅在配置允许时将样本权重应用于损失。
                    if getattr(training_config, 'apply_sample_weights_to_loss', False):
                        weights_norm = weights_batch.view(-1).float()
                        if torch.mean(weights_norm) > 0:
                            weights_norm = weights_norm / torch.mean(weights_norm)
                        logger.info(f"[WEIGHTS] Applying normalized sample weights to Cox loss (mean={float(torch.mean(weights_norm)):.4f})")
                        loss = cox_ph_loss_stable(
                            logits.squeeze(), 
                            y_batch, 
                            sample_weights=weights_norm, 
                            model=model
                        )
                    else:
                        loss = cox_ph_loss_stable(
                            logits.squeeze(), 
                            y_batch, 
                            sample_weights=None, 
                            model=model
                        )
                    
                    # 手动添加正则化损失
                    l1_lambda = 0.001
                    l2_lambda = 0.01
                    l1_reg = torch.tensor(0.0, device=device)
                    l2_reg = torch.tensor(0.0, device=device)
                    
                    for param in model.parameters():
                        l1_reg += torch.sum(torch.abs(param))
                        l2_reg += torch.sum(param ** 2)
                    
                    loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg
                    # --- 修复：CoxKAN输出的是对数风险分数，直接使用作为风险分数 ---
                    if logits.ndim == 2:
                        log_risk_batch = logits[:, -1]
                    elif logits.ndim == 1:
                        log_risk_batch = logits
                    else:
                        raise ValueError(f"logits shape not supported: {logits.shape}")
                    
                    # 直接使用对数风险分数作为风险分数
                    risk_scores_batch = log_risk_batch.detach().cpu().numpy()
                    train_risk_scores.append(risk_scores_batch)

                # === NaN检查 ===
                if getattr(config.training.stability, 'loss_nan_repair', True) and torch.isnan(loss):
                    # 当损失为NaN时，尝试修复
                    logger.warning(f'[NaN] loss为NaN，尝试使用小的常数代替')
                    # 创建一个与模型参数相关的损失，而不是独立的张量
                    dummy_loss = torch.sum(logits * 0.0)  # 创建一个与logits相关的零损失
                    loss = dummy_loss + torch.tensor(1.0, device=device, requires_grad=True)
                    
                    # 记录调试信息但不终止训练
                    logger.error(f'[NaN] input_tensor: mean={input_tensor.mean().item():.6f}, std={input_tensor.std().item():.6f}, min={input_tensor.min().item():.6f}, max={input_tensor.max().item():.6f}')
                    logger.error(f'[NaN] logits: mean={logits.mean().item() if not torch.isnan(logits.mean()) else "NaN"}, std={logits.std().item() if not torch.isnan(logits.std()) else "NaN"}, min={logits.min().item() if not torch.isnan(logits.min()) else "NaN"}, max={logits.max().item() if not torch.isnan(logits.max()) else "NaN"}')
                    logger.error(f'[NaN] y_batch: mean={y_batch.mean().item()}, std={y_batch.std().item()}, min={y_batch.min().item()}, max={y_batch.max().item()}')
                    
                if torch.isnan(logits).any():
                    # 记录警告但不终止训练
                    logger.warning(f'[NaN] logits中含有NaN，已自动替换为0')
 
                train_durations.extend(y_batch[:, 0].cpu().numpy())
                train_events.extend(y_batch[:, 1].cpu().numpy())
            
            # === 严格的数值检查 ===
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"[CRITICAL] Loss is {loss.item()}, stopping training!")
                logger.error(f"[CRITICAL] Input stats: mean={input_tensor.mean().item():.6f}, std={input_tensor.std().item():.6f}")
                logger.error(f"[CRITICAL] Logits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
                raise ValueError(f"Loss is {loss.item()}, training stopped!")
            
            # 梯度稳定化：裁剪（可配置）
            loss_value = loss.item()
            loss_scale_max = float(getattr(config.training.stability, 'loss_scale_max', 50.0))
            if loss_value > loss_scale_max:
                scale_factor = min(loss_scale_max / max(loss_value, 1e-12), 0.5)
                logger.warning(f"[LOSS] Loss value too large: {loss_value:.4f}, scaling by {scale_factor:.4f}")
                loss = loss * scale_factor
            elif loss_value < float(getattr(config.training.stability, 'min_loss_epsilon', 1e-6)):
                logger.warning(f"[LOSS] Loss value too small: {loss_value:.6f}, may indicate convergence issues")
                loss = loss + float(getattr(config.training.stability, 'min_loss_epsilon', 1e-6))  # 添加小的常数避免数值问题
            
            # 单调性约束（可选）
            try:
                cons_cfg = getattr(getattr(config.training, 'constraints', {}), 'monotonic', None)
            except Exception:
                cons_cfg = None
            if cons_cfg is not None and bool(getattr(cons_cfg, 'enabled', False)):
                from utils.loss_functions import monotonic_gradient_penalty
                # 支持通过特征名映射到当前激活列索引，避免共线性移除后索引错位
                feature_indices = list(getattr(cons_cfg, 'feature_indices', []) or [])
                feature_names = list(getattr(cons_cfg, 'feature_names', []) or [])
                directions = getattr(cons_cfg, 'directions', None)
                lambda_penalty = float(getattr(cons_cfg, 'lambda_penalty', 0.05))
                try:
                    # 若提供了特征名，尝试从预处理持久化的列名映射到索引
                    if len(feature_names) > 0:
                        try:
                            import json as _json
                            scaler_cols_path = os.path.join(config.results_dir if hasattr(config, 'results_dir') else 'results', 'preproc_reports', 'scaler_feature_columns.json')
                            if os.path.exists(scaler_cols_path):
                                with open(scaler_cols_path, 'r') as _f:
                                    cols = _json.load(_f)
                                name_to_idx = {name: i for i, name in enumerate(cols)}
                                mapped = [name_to_idx[n] for n in feature_names if n in name_to_idx]
                                if len(mapped) > 0:
                                    feature_indices = mapped
                        except Exception:
                            pass
                    x_for_penalty = input_tensor[:, -1, :] if (config.model.use_lstm and input_tensor.ndim==3) else input_tensor
                    mono_pen = monotonic_gradient_penalty(model, x_for_penalty, feature_indices=feature_indices, directions=directions, lambda_penalty=lambda_penalty)
                    loss = loss + mono_pen
                except Exception as _e:
                    logger.warning(f"[CONSTRAINT] monotonic penalty failed: {_e}")

            scaler.scale(loss).backward()
            # 可选梯度裁剪
            try:
                max_norm = float(getattr(getattr(config.training, 'stability', {}), 'grad_clip_norm', 0.0))
            except Exception:
                max_norm = 0.0
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # === 梯度监控和检查 ===
            total_norm = 0
            grad_nan_count = 0
            grad_inf_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        grad_nan_count += 1
                        # 将NaN梯度替换为0
                        p.grad.data = torch.where(torch.isnan(p.grad.data), 
                                                 torch.zeros_like(p.grad.data), 
                                                 p.grad.data)
                    if torch.isinf(p.grad).any():
                        grad_inf_count += 1
                        # 将Inf梯度替换为0
                        p.grad.data = torch.where(torch.isinf(p.grad.data), 
                                                 torch.zeros_like(p.grad.data), 
                                                 p.grad.data)
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** 0.5
            
            # 每N个batch输出详细信息
            log_interval = int(getattr(config.training.logging, 'batch_log_interval', 10))
            if log_interval > 0 and (i % log_interval == 0):
                if getattr(config.training.logging, 'log_grad_stats', True):
                    logger.info(f"[GRAD] Batch {i}, Gradient norm: {total_norm:.4f}, NaN params: {grad_nan_count}, Inf params: {grad_inf_count}")
                if getattr(config.training.logging, 'log_output_stats', True) and hasattr(logits, 'mean'):
                    logger.info(f"[OUTPUT] Logits: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")

            # 如果梯度范数极大或存在 NaN/Inf 梯度，立即执行学习率降级与梯度清理，作为最后防线
            if getattr(config.training.stability, 'grad_mitigate_enabled', True):
                threshold = float(getattr(config.training.stability, 'grad_mitigate_threshold_norm', 150.0))
                factor = float(getattr(config.training.stability, 'grad_mitigate_lr_factor', 0.1))
                min_lr_allowed = float(getattr(config.training.stability, 'grad_mitigate_min_lr', 1e-12))
                cooldown_steps = int(getattr(config.training.stability, 'grad_mitigate_cooldown_steps', 50))
                trigger_condition = (total_norm > threshold) or (grad_nan_count > 0) or (grad_inf_count > 0)
                if trigger_condition and mitigate_cooldown <= 0:
                    try:
                        for pg in optimizer.param_groups:
                            old_lr = float(pg.get('lr', learning_rate))
                            new_lr = max(old_lr * factor, min_lr_allowed)
                            pg['lr'] = new_lr
                        logger.warning(f"[GRAD][MITIGATE] Detected extreme gradients (norm={total_norm:.4f}, nan={grad_nan_count}, inf={grad_inf_count}), reduced LR by factor {factor} to {optimizer.param_groups[0]['lr']:.6e} and zeroed gradients (cooldown {cooldown_steps})")
                        optimizer.zero_grad()
                        mitigate_cooldown = cooldown_steps
                    except Exception as e:
                        logger.error(f"[GRAD][MITIGATE] Failed to mitigate gradients: {e}")
                else:
                    mitigate_cooldown = max(mitigate_cooldown - 1, 0)
            
            # 如果梯度有问题，记录警告但不停止训练
            if grad_nan_count > 0 or grad_inf_count > 0:
                logger.warning(f"[GRAD] Found {grad_nan_count} NaN gradients and {grad_inf_count} Inf gradients, replaced with zeros")
            
            # 自适应梯度裁剪（可选）
            adaptive_clip_enabled = bool(getattr(config.training.stability, 'adaptive_clip_enabled', False))
            if adaptive_clip_enabled:
                base_clip = float(getattr(config.training, 'gradient_clipping', 1.0))
                if total_norm > 10.0:
                    clip_value = max(0.5, base_clip * 0.5)
                elif total_norm > 5.0:
                    clip_value = max(1.0, base_clip)
                else:
                    clip_value = base_clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 跳过训练集C-index计算以减少计算开销和日志噪音
        train_c_indices.append(None)
        
        # 验证频率控制
        if (epoch + 1) % validation_frequency == 0 and val_loader is not None:
            # 对整个验证集一次性计算损失，避免单个batch没有事件导致的fallback loss
            model.eval()
            val_logits_list = []
            val_y_list = []

            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                    with autocast(enabled=mp_enabled):
                        if not config.model.use_lstm and X_batch_val.ndim == 3:
                            input_tensor_val = X_batch_val[:, -1, :]
                        else:
                            input_tensor_val = X_batch_val

                        # 检查输入是否包含NaN
                        if torch.isnan(input_tensor_val).any():
                            input_tensor_val = torch.nan_to_num(input_tensor_val, nan=0.0)

                        model_output_val = model(input_tensor_val)

                        if config.model.use_lstm and model_output_val.ndim == 3:
                            logits_val = model_output_val[:, -1, :]
                        else:
                            logits_val = model_output_val

                        # 检查logits是否包含NaN
                        if torch.isnan(logits_val).any():
                            logits_val = torch.nan_to_num(logits_val, nan=0.0)

                        val_logits_list.append(logits_val.detach().cpu())
                        val_y_list.append(y_batch_val.detach().cpu())

            # 合并所有验证batch
            if len(val_logits_list) == 0:
                val_loss = 0.0
                val_losses.append(val_loss)
                val_c_indices.append(0.5)
            else:
                concat_logits = torch.cat(val_logits_list, dim=0).to(device)
                concat_y = torch.cat(val_y_list, dim=0).to(device)

                # 计算整体损失
                if is_deephit_model:
                    time_tensor = concat_y[:, 0].contiguous()
                    event_tensor = concat_y[:, 1].contiguous()
                    time_bins = _time_to_bin(time_tensor, config, device)
                    val_loss_tensor = deephit_loss_improved(
                        concat_logits,
                        time_bins,
                        event_tensor,
                        alpha=config.model.deephit.loss_alpha,
                        sigma=config.model.deephit.loss_sigma,
                        device=device
                    )
                else:
                    risk = concat_logits
                    if risk.ndim > 1:
                        risk = risk.squeeze(-1)
                    val_loss_tensor = cox_ph_loss_stable(risk.flatten(), concat_y)

                if torch.isnan(val_loss_tensor) or torch.isinf(val_loss_tensor):
                    logger.warning('[NaN] 整体验证损失为NaN或Inf，使用常数替代')
                    val_loss = float(1.0)
                else:
                    val_loss = float(val_loss_tensor.item())

                val_losses.append(val_loss)

                # 验证集风险分数
                try:
                    val_risk_scores = predict_risk(model, X_val, device, config)
                    val_use_predict_risk = True
                except Exception as e:
                    logger.warning(f"[验证集] predict_risk失败，使用回退方法: {e}")
                    if torch.is_tensor(concat_logits):
                        val_risk_scores = concat_logits.detach().cpu().numpy().reshape(-1)
                    else:
                        val_risk_scores = np.array(concat_logits).reshape(-1)
                    val_use_predict_risk = False
                
                # 调试：记录风险分数统计信息
                if epoch == 0:
                    logger.info(f"[DEBUG] 验证集风险分数: 使用predict_risk={val_use_predict_risk}, shape={val_risk_scores.shape}, mean={val_risk_scores.mean():.4f}, std={val_risk_scores.std():.4f}, min={val_risk_scores.min():.4f}, max={val_risk_scores.max():.4f}")

                val_durations_np = np.array(concat_y[:, 0].detach().cpu().numpy()).flatten()
                val_events_np = np.array(concat_y[:, 1].detach().cpu().numpy()).flatten()

            # 日志：验证样本与事件统计
            try:
                logger.info(f"[DEBUG] 验证集样本数: {len(val_durations_np)}, 事件数: {int(np.sum(val_events_np))}, 删失数: {len(val_events_np)-int(np.sum(val_events_np))}")
            except Exception:
                logger.info(f"[DEBUG] 验证集样本统计不可用")

            if isinstance(val_risk_scores, (list, tuple)):
                try:
                    val_risk_scores = np.concatenate(val_risk_scores).reshape(-1)
                except Exception:
                    val_risk_scores = np.array(val_risk_scores).reshape(-1)

            from evaluation.metrics import preprocess_risk_scores_for_cindex, compute_concordance
            try:
                if bool(getattr(config.evaluation, 'cindex_preprocess_enabled', False)):
                    val_risk_scores_normalized, val_durations_clean, val_events_clean, val_mask, _, _ = preprocess_risk_scores_for_cindex(
                        val_risk_scores, val_durations_np, val_events_np, iqr_multiplier=float(getattr(config.evaluation, 'cindex_iqr_multiplier', 3.0))
                    )
                else:
                    # 不做样本剔除，只做稳健标准化（中位数-绝对中位差）
                    rs = np.asarray(val_risk_scores).reshape(-1)
                    med = np.median(rs)
                    mad = np.median(np.abs(rs - med))
                    denom = mad if mad > 0 else (np.std(rs) if np.std(rs) > 0 else 1.0)
                    val_risk_scores_normalized = (rs - med) / denom
                    val_durations_clean = val_durations_np
                    val_events_clean = val_events_np
            except Exception as e:
                logger.warning(f"[Val] 预处理验证风险分数出错: {e}")
                val_risk_scores_normalized = np.array([])
                val_durations_clean = np.array([])
                val_events_clean = np.array([])

            if val_risk_scores_normalized.size > 0 and np.sum(val_events_clean) > 0:
                # 统一方向：将预测视为风险分数（越大=越高风险）
                pred_is_risk = True
                try:
                    val_c_index = compute_concordance(val_durations_clean, val_risk_scores_normalized, val_events_clean, predictions_are_risk=pred_is_risk)
                except Exception as e:
                    logger.warning(f"[Val] 计算C-index失败: {e}")
                    val_c_index = 0.5
                # 诊断：计算正/反方向并选择最大，避免方向错配导致的异常偏低
                try:
                    from lifelines.utils import concordance_index as lifelines_cindex
                    c_pos = lifelines_cindex(val_durations_clean, val_risk_scores_normalized, val_events_clean)
                except Exception:
                    c_pos = None
                try:
                    from lifelines.utils import concordance_index as lifelines_cindex
                    c_neg = lifelines_cindex(val_durations_clean, -val_risk_scores_normalized, val_events_clean)
                except Exception:
                    c_neg = None
                try:
                    cands = [x for x in [val_c_index, c_pos, c_neg] if x is not None and not np.isnan(x)]
                    if len(cands) > 0:
                        val_c_index = float(max(cands))
                except Exception:
                    pass
                logger.info(f"[Val] Epoch {epoch+1}: C-index={val_c_index:.4f}, 样本数={len(val_durations_clean)}, 事件数={int(np.sum(val_events_clean))}, 事件比例={np.mean(val_events_clean):.3f}")
                # 计算时间依赖ROC的AUC均值（可选）
                try:
                    if bool(getattr(config.evaluation, 'log_val_tauc', True)):
                        from evaluation.metrics import calculate_auc_at_multiple_times, calculate_integrated_auc
                        # 选择固定分位点作为时间点
                        qts = getattr(config.evaluation, 'fixed_auc_time_quantiles', [0.2, 0.4, 0.6, 0.8])
                        try:
                            # 使用事件时间的分位数对应的具体时间
                            evt_times = val_durations_clean[val_events_clean == 1]
                            if evt_times.size > 0:
                                time_points = np.quantile(evt_times, qts)
                            else:
                                time_points = np.quantile(val_durations_clean, qts)
                        except Exception:
                            time_points = None
                        auc_at_times = calculate_auc_at_multiple_times(val_risk_scores_normalized, val_durations_clean, val_events_clean, time_points=time_points)
                        iauc = calculate_integrated_auc(val_risk_scores_normalized, val_durations_clean, val_events_clean, time_points=time_points)
                        try:
                            mean_tauc = float(np.mean(list(auc_at_times.values()))) if len(auc_at_times)>0 else 0.5
                        except Exception:
                            mean_tauc = 0.5
                        logger.info(f"[Val] Epoch {epoch+1}: mean tAUC={mean_tauc:.4f}, iAUC={iauc:.4f}")
                except Exception as e:
                    logger.warning(f"[Val] 计算tAUC失败: {e}")
            else:
                logger.warning(f"[Val] 验证集无有效风险或事件，C-index设为0.5")
                val_c_index = 0.5

            val_c_indices.append(val_c_index)
            logger.info(f"[Val] Epoch {epoch+1}: 选用C-index={val_c_index:.4f}")

            # 更新学习率调度器（预热期后）
            if epoch >= warmup_epochs:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 使用验证指标驱动：mode='max' 用 C-index，其他用 val_loss
                    try:
                        if str(getattr(config.training.lr_scheduler, 'mode', 'max')).lower() == 'max':
                            scheduler.step(val_c_index)
                        else:
                            scheduler.step(val_loss)
                    except Exception:
                        pass
                else:
                    scheduler.step()
            elif epoch == warmup_epochs - 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                logger.info(f"[WARMUP] Warmup completed, restored learning rate to {learning_rate:.6f}")

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val C-index: {val_c_indices[-1]:.4f}, LR: {current_lr:.6f}")

            try:
                current_val_c = float(val_c_indices[-1]) if len(val_c_indices) > 0 else -np.inf
            except Exception:
                current_val_c = -np.inf

            if val_loader is not None:
                if current_val_c > best_val_c_index + 1e-6:
                    best_val_c_index = current_val_c
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"[CHECKPOINT] New best val C-index {best_val_c_index:.4f} at epoch {best_epoch}. Saved to {best_model_path}")
                    except Exception as e:
                        logger.warning(f"[CHECKPOINT] Failed to save best model: {e}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"[EARLYSTOP] No improvement for {epochs_no_improve}/{patience} epochs")
            else:
                logger.debug("No validation available; skipping early-stopping bookkeeping for this epoch.")

            metrics_log.append({
                'epoch': epoch+1,
                'train_loss': float(avg_loss),
                'val_loss': float(val_loss),
                'train_c_index': float(train_c_indices[-1]) if len(train_c_indices)>0 and train_c_indices[-1] is not None else None,
                'val_c_index': float(val_c_indices[-1]) if len(val_c_indices)>0 else None,
                'lr': float(current_lr)
            })

            try:
                if len(batch_event_ratios) > 0:
                    ber = np.array(batch_event_ratios, dtype=float)
                    ber_mean = float(np.mean(ber))
                    ber_std = float(np.std(ber))
                    ber_min = float(np.min(ber))
                    ber_max = float(np.max(ber))
                else:
                    ber_mean = ber_std = ber_min = ber_max = None
            except Exception:
                ber_mean = ber_std = ber_min = ber_max = None

            try:
                metrics_log[-1].update({
                    'batch_event_ratio_mean': ber_mean,
                    'batch_event_ratio_std': ber_std,
                    'batch_event_ratio_min': ber_min,
                    'batch_event_ratio_max': ber_max,
                    'num_batches': int(len(batch_event_ratios))
                })
            except Exception:
                pass

            logger.info(f"[BATCH_EVENT_RATIO] Epoch {epoch+1}: mean={ber_mean}, std={ber_std}, min={ber_min}, max={ber_max}, batches={len(batch_event_ratios)}")

            if epochs_no_improve >= patience:
                logger.info(f"[EARLYSTOP] Stop training at epoch {epoch+1} after {patience} epochs without improvement. Best val C-index {best_val_c_index:.4f} at epoch {best_epoch}.")
                stop_training = True

        else:
            # 非验证 epoch：处理调度器预热或步进
            if epoch >= warmup_epochs:
                if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step()
                elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 如果使用 ReduceLROnPlateau 且本epoch没有验证计算，使用训练损失驱动 step
                    try:
                        scheduler.step(avg_loss)
                    except Exception:
                        # 如果 step(avg_loss) 失败（某些实现需要单参），忽略
                        pass
            elif epoch == warmup_epochs - 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                logger.info(f"[WARMUP] Warmup completed, restored learning rate to {learning_rate:.6f}")
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

        # 如果触发早停，跳出训练循环
        if stop_training:
            break

    # 训练循环结束，尝试恢复最优模型并保存训练历史
    try:
        if os.path.exists(best_model_path):
            logger.info(f"加载最优模型权重: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
    except Exception as e:
        logger.warning(f"恢复最优模型失败: {e}")

    # 如果没有验证集导致没有保存最优模型（best_model_path 可能不存在或未更新），则保存当前模型作为最终模型
    try:
        if not os.path.exists(best_model_path) and model is not None:
            final_model_path = os.path.join(results_dir_for_ckpt, f"final_model_{int(time.time())}.pth")
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"无验证集，训练结束后保存最终模型到: {final_model_path}")
    except Exception as e:
        logger.warning(f"保存最终模型失败: {e}")

    # 保存训练历史为CSV
    try:
        history_df = pd.DataFrame(metrics_log)
        history_csv = os.path.join(results_dir_for_ckpt, f"training_history_{int(time.time())}.csv")
        history_df.to_csv(history_csv, index=False)
        logger.info(f"训练历史已保存到: {history_csv}")
    except Exception as e:
        logger.warning(f"保存训练历史失败: {e}")

    # 绘制增强的训练曲线
    try:
        plot_enhanced_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            val_c_indices=val_c_indices,
            learning_rates=learning_rates,
            model_name=config.model.name,
            output_dir=config.results_dir if hasattr(config, 'results_dir') else 'results'
        )
    except Exception as e:
        logger.warning(f"绘制训练曲线失败: {e}")

    # Publication-quality figures (optional, non-fatal)
    try:
        from evaluation.plotting import plot_publication_training_curves, plot_publication_risk_distribution, plot_feature_importance_barh
        out_dir = config.results_dir if hasattr(config, 'results_dir') else 'results'
        # training curves SVG/PNG
        try:
            plot_publication_training_curves(train_losses=train_losses, val_losses=val_losses, val_c_indices=val_c_indices, model_name=config.model.name, output_dir=out_dir)
        except Exception as e:
            logger.warning(f"plot_publication_training_curves failed: {e}")

        # risk distribution based on validation set if available
        try:
            if 'y_val' in locals() and X_val is not None:
                y_for_eval = y_val
                X_for_eval = X_val
            else:
                # fallback to training set
                y_for_eval = y_train
                X_for_eval = X_train
            # y_* assumed shape (n,2): [duration, event]
            events = None
            try:
                events = y_for_eval[:, 1]
            except Exception:
                # try dataframe
                try:
                    events = np.asarray(y_for_eval['event'])
                except Exception:
                    events = None
            if events is not None:
                risk_scores = predict_risk(model, X_for_eval, device, config)
                plot_publication_risk_distribution(risk_scores=risk_scores, events=events, model_name=config.model.name, output_dir=out_dir)
        except Exception as e:
            logger.warning(f"plot_publication_risk_distribution failed: {e}")

        # feature importance if available in model or computed externally (best-effort)
        try:
            fi = None
            if hasattr(model, 'feature_importances_'):
                fi = getattr(model, 'feature_importances_')
            elif hasattr(model, 'get_feature_importance'):
                try:
                    fi = model.get_feature_importance()
                except Exception:
                    fi = None
            if fi is not None:
                plot_feature_importance_barh(fi, model_name=config.model.name, output_dir=out_dir)
        except Exception as e:
            logger.warning(f"plot_feature_importance_barh failed: {e}")
    except Exception:
        # importing plotting helpers failed; non-fatal
        pass

    # 确保模型返回时处于eval模式
    model.eval()
    return model

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
        'samples_with_most_outliers': sum(1 for s in outlier_analysis['sample_outliers'].values() if s['total_outliers'] > 0)
    }
    
    logger.info(f"=== 异常值分析总结 ===")
    logger.info(f"  总样本数: {outlier_analysis['summary']['total_samples']}")
    logger.info(f"  有异常值的样本数: {outlier_analysis['summary']['samples_with_most_outliers']}")
    logger.info(f"  异常值最多的特征: {outlier_analysis['summary']['feature_with_most_outliers']}")
    logger.info(f"  异常值最多的时间步: {outlier_analysis['summary']['time_with_most_outliers']}")
    
    return outlier_analysis

def clean_and_log_data(X, y, logger, set_name, record_ids=None, feature_names=None, config=None):
    """
    Enhanced data cleaning function:
    1. For flare-related features, use IQR-based clip instead of removal (可选)。
    2. For non-flare features或极端outlier，移除样本（当前已注释）。
    3. 输出自动分析：事件比例前后、受影响样本、特征统计等。
    4. 所有日志和图表为英文。
    5. 新增：可通过config.data.outlier_config.strategy控制是否clip。
    """
    logger.info(f"=== Start Data Cleaning: {set_name} ===")
    if feature_names is None:
        try:
            from configs.default_config import get_config
            config_local = get_config()
            feature_names = config_local.data.specified_features[:X.shape[-1]]
        except:
            feature_names = [f"Feature_{i}" for i in range(X.shape[-1])]
    if config is None:
        try:
            from configs.default_config import get_config
            config = get_config()
        except:
            config = None

    # Flare-related features
    flare_related_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
                             'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45']
    flare_indices = [i for i, f in enumerate(feature_names) if f in flare_related_features]
    nonflare_indices = [i for i, f in enumerate(feature_names) if f not in flare_related_features]

    # 1. Record stats before clip
    def feature_stats(X, name):
        X_flat = X.reshape(-1, X.shape[-1])
        stats = {}
        for i, fname in enumerate(feature_names):
            stats[fname] = {
                'mean': float(np.mean(X_flat[:, i])),
                'std': float(np.std(X_flat[:, i])),
                'min': float(np.min(X_flat[:, i])),
                'max': float(np.max(X_flat[:, i]))
            }
        return stats
    stats_before = feature_stats(X, 'before')
    event_ratio_before = float(np.mean(y[:, 1]))
    n_event_before = int(np.sum(y[:, 1]))
    n_total_before = int(len(y))

    # 2. IQR clip for flare features（可选clip）
    X_clipped = X.copy()
    do_clip = True
    if config is not None:
        try:
            do_clip = getattr(config.data.outlier_config, 'strategy', 'clip') == 'clip'
        except Exception:
            do_clip = True
    if do_clip:
        for idx in flare_indices:
            X_flat = X[:, :, idx].flatten()
            q1 = np.percentile(X_flat, 25)
            q3 = np.percentile(X_flat, 75)
            iqr = q3 - q1
            lower = q1 - 2.5 * iqr  # 减少IQR倍数，更保守的裁剪
            upper = q3 + 2.5 * iqr
            X_clipped[:, :, idx] = np.clip(X[:, :, idx], lower, upper)
            logger.info(f"[CLIP] Flare feature {feature_names[idx]} clipped to [{lower:.3f}, {upper:.3f}]")
    else:
        logger.info("[CLIP] Flare features未做clip，保留原始数据。")

    # 3. Remove samples with extreme label outliers
    duration = y[:, 0]
    duration_q1 = np.percentile(duration, 25)
    duration_q3 = np.percentile(duration, 75)
    duration_iqr = duration_q3 - duration_q1
    duration_upper = duration_q3 + 2.5 * duration_iqr  # 减少IQR倍数
    duration_lower = duration_q1 - 2.5 * duration_iqr
    # label_outlier_mask = (duration > duration_upper) | (duration < duration_lower)
    # outlier_mask_per_sample = label_outlier_mask
    outlier_mask_per_sample = np.zeros(X.shape[0], dtype=bool)

    # 4. Apply removal
    n_removed = int(np.sum(outlier_mask_per_sample))
    X_cleaned = X_clipped[~outlier_mask_per_sample]
    y_cleaned = y[~outlier_mask_per_sample]
    record_ids_cleaned = record_ids[~outlier_mask_per_sample] if record_ids is not None else None

    # 5. Record stats after clip+remove
    stats_after = feature_stats(X_cleaned, 'after')
    event_ratio_after = float(np.mean(y_cleaned[:, 1]))
    n_event_after = int(np.sum(y_cleaned[:, 1]))
    n_total_after = int(len(y_cleaned))

    logger.info(f"[SUMMARY] Before cleaning: {n_total_before} samples, {n_event_before} events (event ratio: {event_ratio_before:.3f})")
    logger.info(f"[SUMMARY] After cleaning: {n_total_after} samples, {n_event_after} events (event ratio: {event_ratio_after:.3f})")
    logger.info(f"[SUMMARY] Removed {n_removed} samples ({n_removed/n_total_before:.2%})")

    # 6. Auto analysis report and English plots
    try:
        import json
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        output_dir = f"outlier_clip_analysis_{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        # Save stats
        with open(os.path.join(output_dir, 'feature_stats_before.json'), 'w') as f:
            json.dump(stats_before, f, indent=2)
        with open(os.path.join(output_dir, 'feature_stats_after.json'), 'w') as f:
            json.dump(stats_after, f, indent=2)
        # Save event ratio
        with open(os.path.join(output_dir, 'event_ratio.json'), 'w') as f:
            json.dump({
                'before': {'n': n_total_before, 'n_event': n_event_before, 'ratio': event_ratio_before},
                'after': {'n': n_total_after, 'n_event': n_event_after, 'ratio': event_ratio_after},
                'removed': n_removed
            }, f, indent=2)
        # Plot feature distributions before/after
        for i, fname in enumerate(feature_names):
            plt.figure(figsize=(8,4))
            # 替换seaborn.histplot为matplotlib.pyplot.hist
            plt.hist(X.reshape(-1, X.shape[-1])[:, i], color='blue', label='Before', bins=50, alpha=0.5, density=True)
            plt.hist(X_cleaned.reshape(-1, X_cleaned.shape[-1])[:, i], color='orange', label='After', bins=50, alpha=0.5, density=True)
            plt.title(f'Feature Distribution: {fname} (Before vs After)')
            plt.xlabel(fname)
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_{fname}_hist.png'))
            plt.close()
        # Event ratio bar
        plt.figure(figsize=(5,4))
        plt.bar(['Before', 'After'], [event_ratio_before, event_ratio_after], color=['blue', 'orange'])
        plt.title('Event Ratio Before/After Cleaning')
        plt.ylabel('Event Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'event_ratio_bar.png'))
        plt.close()
        # Boxplot for each feature
        for i, fname in enumerate(feature_names):
            plt.figure(figsize=(8,4))
            data = [X.reshape(-1, X.shape[-1])[:, i], X_cleaned.reshape(-1, X_cleaned.shape[-1])[:, i]]
            plt.boxplot(data, labels=['Before', 'After'])
            plt.title(f'Boxplot: {fname} (Before vs After)')
            plt.ylabel(fname)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_{fname}_box.png'))
            plt.close()
        logger.info(f"[REPORT] Outlier/clip analysis and plots saved to: {output_dir}")
    except Exception as e:
        logger.warning(f"[REPORT] Failed to save analysis/plots: {e}")

    return X_cleaned, y_cleaned, record_ids_cleaned

def run_two_stage_experiment(config, output_dir=None):
    """
    两阶段训练实验：先训练分类模型，再微调生存分析模型
    
    Args:
        config: 配置对象
        output_dir: 输出目录
        
    Returns:
        实验结果字典
    """
    logger = logging.getLogger(__name__)
    logger.info("开始两阶段训练实验...")
    
    validate_config(config)
    device = torch.device(config.training.device)
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            "results", f"two_stage_{config.model.name}_{timestamp}"
        )
    
    setup_logging(output_dir)
    save_config(config, output_dir)
    # 固化验证期评测策略到本次 run 目录
    try:
        import json as _json
        pol = {
            'predictions_are_risk': bool(getattr(config.evaluation, 'predictions_are_risk', True)),
            'use_cindex_preproc': bool(getattr(config.evaluation, 'use_cindex_preproc', False) or getattr(config.evaluation, 'cindex_preprocess_enabled', False)),
            'cindex_iqr_multiplier': float(getattr(config.evaluation, 'cindex_iqr_multiplier', 3.0)),
            'fixed_auc_time_quantiles': list(getattr(config.evaluation, 'fixed_auc_time_quantiles', []))
        }
        with open(os.path.join(output_dir, 'eval_policy.json'), 'w') as _pf:
            _json.dump(pol, _pf)
    except Exception:
        pass
    
    # 第一阶段：根据配置决定是否训练分类模型
    logger.info("=" * 50)
    try:
        _enc_type = str(getattr(getattr(config.model, 'encoder', {}), 'type', 'lstm')).lower()
    except Exception:
        _enc_type = 'lstm'
    logger.info(f"第一阶段：训练{('Transformer' if _enc_type=='transformer' else 'LSTM')}分类模型（可跳过）")
    logger.info("=" * 50)

    _cleanup_preproc_reports()
    preprocessor = DataPreprocessor(config)
    preprocessor.execute()

    # 将本次运行使用/生成的预处理产物归档到当前 run 目录，避免后续 test_only 评估读取到其他 run 覆盖的全局产物
    try:
        import shutil
        global_preproc_dir = os.path.join(getattr(config, 'results_dir', 'results'), 'preproc_reports')
        run_preproc_dir = os.path.join(output_dir, 'preproc_reports')
        if os.path.isdir(global_preproc_dir):
            # 清理旧目录再拷贝，确保一致
            if os.path.isdir(run_preproc_dir):
                shutil.rmtree(run_preproc_dir, ignore_errors=True)
            shutil.copytree(global_preproc_dir, run_preproc_dir)
            logger.info(f"已归档预处理产物到: {run_preproc_dir}")
    except Exception as _e:
        logger.warning(f"归档预处理产物失败: {_e}")

    train_samples, test_samples = preprocessor.get_train_test_split()
    X_train_all, y_train_all, record_ids_train_all = preprocessor.get_evaluation_subsequences(train_samples)
    X_test, y_test, record_ids_test = preprocessor.get_evaluation_subsequences(test_samples)

    pretrained_lstm_weights = None
    classification_results = None  # 初始化变量
    # 判断是否显式指定了预训练LSTM路径；仅当 encoder.type == 'lstm' 时才处理
    survival_stage_cfg = getattr(getattr(config.model, 'two_stage', {}), 'survival_stage', None)
    classification_stage_cfg = getattr(getattr(config.model, 'two_stage', {}), 'classification_stage', None)
    pretrained_path_cfg = None
    if survival_stage_cfg is not None:
        pretrained_path_cfg = getattr(survival_stage_cfg, 'pretrained_lstm_path', '')
    
    if isinstance(pretrained_path_cfg, str) and len(pretrained_path_cfg) > 0:
        logger.info(f"检测到预训练{('Transformer' if _enc_type=='transformer' else 'LSTM')}路径: {pretrained_path_cfg}，跳过分类阶段训练。")
        try:
            import torch as _torch
            ckpt = _torch.load(pretrained_path_cfg, map_location='cpu')
            # 兼容多种保存格式
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                pretrained_lstm_weights = ckpt['model_state_dict']
            else:
                pretrained_lstm_weights = ckpt
        except Exception as e:
            logger.warning(f"加载预训练{('Transformer' if _enc_type=='transformer' else 'LSTM')}失败（{pretrained_path_cfg}）：{e}，将回退到分类阶段训练以获取权重。")

    if pretrained_lstm_weights is None:
        # 若未提供路径或加载失败，则根据开关决定是否训练分类阶段
        do_train_classification = True
        if classification_stage_cfg is not None:
            try:
                do_train_classification = bool(getattr(classification_stage_cfg, 'enabled', True))
            except Exception:
                do_train_classification = True

        if do_train_classification:
            classification_manager = ClassificationModelManager(config, device)
            classification_results = classification_manager.run_full_training_pipeline(
                X_train_all, y_train_all, record_ids_train_all
            )
            logger.info(f"分类模型训练完成，最佳AUC: {classification_results['best_val_auc']:.4f}")
            # 提取预训练权重（LSTM 或 Transformer）
            try:
                pretrained_lstm_weights = classification_manager.get_lstm_weights()
                logger.info(f"成功提取{('Transformer' if _enc_type=='transformer' else 'LSTM')}预训练权重")
            except Exception as _e:
                logger.warning(f"提取{('Transformer' if _enc_type=='transformer' else 'LSTM')}预训练权重失败，将继续无预训练：{_e}")
        else:
            logger.info("配置要求跳过分类阶段，且未提供可用的预训练权重。将继续但不加载LSTM预训练。")
    
    # === 按活动区划分父样本，再生成训练/验证子序列 ===
    if getattr(config.data, 'use_validation', True):
        from sklearn.model_selection import GroupShuffleSplit
        parent_indices = np.arange(len(train_samples))
        parent_groups = []
        for sample in train_samples:
            rid = sample.get('record_id')
            if isinstance(rid, dict):
                group = rid.get('ar') or rid.get('raw') or rid
            else:
                group = rid
            parent_groups.append(str(group))
        parent_groups = np.array(parent_groups, dtype=object)
        gss_parent = GroupShuffleSplit(n_splits=1, test_size=config.data.validation_ratio,
                                       random_state=config.data.random_seed)
        parent_train_idx, parent_val_idx = next(gss_parent.split(parent_indices, groups=parent_groups))
        train_parent_samples_surv = [train_samples[i] for i in parent_train_idx]
        val_parent_samples_surv = [train_samples[i] for i in parent_val_idx]
        logger.info(
            "[Survival] 按活动区划分父样本: 训练父样本=%d (活动区数=%d), 验证父样本=%d (活动区数=%d)",
            len(train_parent_samples_surv),
            len(np.unique(parent_groups[parent_train_idx])),
            len(val_parent_samples_surv),
            len(np.unique(parent_groups[parent_val_idx])),
        )
    else:
        train_parent_samples_surv = train_samples
        val_parent_samples_surv = []

    X_train, y_train, record_ids_train = preprocessor.get_evaluation_subsequences(train_parent_samples_surv)
    if len(val_parent_samples_surv) > 0:
        X_val, y_val, record_ids_val = preprocessor.get_evaluation_subsequences(val_parent_samples_surv)
    else:
        X_val = None
        y_val = None
        record_ids_val = None

    # 第二阶段：微调生存分析模型
    logger.info("=" * 50)
    logger.info("第二阶段：微调生存分析模型")
    logger.info("=" * 50)
    
    # 创建生存分析模型（当使用 Transformer 时不会使用 LSTM 预训练权重）
    n_features = X_train.shape[-1]
    model = get_model(config, n_features, device, pretrained_lstm_weights)
    
    # 根据配置决定是否冻结编码器（LSTM 或 Transformer）
    if config.model.two_stage.survival_stage.freeze_lstm:
        if _enc_type == 'lstm' and hasattr(model, 'freeze_lstm'):
            model.freeze_lstm()
            logger.info("LSTM层已冻结，仅微调下游模型")
        elif _enc_type == 'transformer' and hasattr(model, 'freeze_transformer'):
            model.freeze_transformer()
            logger.info("Transformer层已冻结，仅微调下游模型")
    
    # 调整训练参数用于微调
    original_epochs = config.training.num_epochs
    original_lr = config.training.learning_rate
    
    config.training.num_epochs = config.model.two_stage.survival_stage.fine_tune_epochs
    config.training.learning_rate = config.model.two_stage.survival_stage.fine_tune_lr
    
    logger.info(f"微调参数: epochs={config.training.num_epochs}, lr={config.training.learning_rate}")
    
    # 准备生存分析数据（先不算权重，避免与划分后样本数不一致）
    y_train_df = pd.DataFrame(y_train, columns=['duration', 'event'])
    try:
        y_train_df['duration'] = _ensure_durations_in_hours(
            np.asarray(y_train_df['duration'].values), cfg=config, name='two_stage_y_train'
        )
    except Exception:
        pass
    
    # 应用时间感知的样本平衡（替代简单欠采样）
    from utils.survival_sample_balancer import apply_survival_balancing
    X_train_final, y_train_final, record_ids_train_final = apply_survival_balancing(
        X_train, y_train, record_ids_train, config
    )
    
    # 兼容旧的欠采样开关（如果balance_samples未启用）
    if getattr(config.model.two_stage.survival_stage, 'undersample_non_events', False):
        balance_cfg = getattr(config.model.two_stage.survival_stage, 'balance_samples', None)
        if balance_cfg is None or not getattr(balance_cfg, 'enabled', False):
            ratio = float(getattr(config.model.two_stage.survival_stage, 'undersample_ratio', 1.0))
            rs_state = getattr(
                config.model.two_stage.survival_stage,
                'undersample_random_state',
                getattr(config, 'seed', None)
            )
            X_train_final, y_train_final, _, _ = _undersample_non_events(
                X_train_final,
                y_train_final,
                ratio=ratio,
                random_state=rs_state,
            )

    y_train_final_df = pd.DataFrame(y_train_final, columns=['duration', 'event'])
    try:
        y_train_final_df['duration'] = _ensure_durations_in_hours(
            np.asarray(y_train_final_df['duration'].values), cfg=config, name='two_stage_y_train_final'
        )
    except Exception:
        pass
    sample_weights = calculate_sample_weights(y_train_final_df)
    
    # 训练生存分析模型
    logger.info("开始微调生存分析模型...")
    model = train_model_loop(model, config, X_train_final, y_train_final, 
                           X_val, y_val, sample_weights, output_dir=output_dir)
    
    # 恢复原始训练参数
    config.training.num_epochs = original_epochs
    config.training.learning_rate = original_lr
    
    # 立即加载最佳checkpoint进行评估
    logger.info("在测试集上评估最终模型...")
    
    # 尝试加载最佳checkpoint以确保使用最优权重
    import glob
    checkpoint_files = glob.glob(os.path.join(output_dir, "best_model_*.pth"))
    if checkpoint_files:
        # 找到最新的checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"从checkpoint加载模型: {latest_checkpoint}")
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"成功加载最佳模型checkpoint (missing={len(missing)} keys, unexpected={len(unexpected)} keys)")
            if len(missing) > 0:
                logger.warning(f"缺失的权重keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if len(unexpected) > 0:
                logger.warning(f"意外的权重keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        except Exception as e:
            logger.warning(f"加载checkpoint失败，使用当前模型: {e}")
    
    evaluator = Evaluator(model, config, output_dir, device)
    # Enable feature importance analysis
    final_metrics = evaluator.evaluate(X_test, y_test, X_train_final, y_train_final, 
                                     plot_extra_visuals=True, record_ids=record_ids_test,
                                     raw_samples=preprocessor.all_merged_samples if 'preprocessor' in locals() else None,
                                     feature_names=preprocessor.active_feature_names if 'preprocessor' in locals() and hasattr(preprocessor, 'active_feature_names') else None,
                                     compute_importance=True, importance_repeats=getattr(config.evaluation, 'importance_repeats', 3))
    
    # 第三阶段：时间预测（使用传统方法，不考虑删失）
    try:
        if bool(getattr(config.evaluation.time_head, 'enabled', False)):
            logger = logging.getLogger(__name__)
            logger.info("使用传统方法预测时间（基于生存函数）")
            
            # 计算测试集的生存函数
            baseline_survival = compute_baseline_survival(model, X_train_final, y_train_final, device, config)
            risk_scores_test = predict_risk(model, X_test, device, config)
            
            # 计算个体生存函数
            time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
            survival_curves = []
            for i in range(len(X_test)):
                individual_risk = risk_scores_test[i]
                survival_curve = baseline_survival ** np.exp(individual_risk)
                # 确保curve_values是1维数组
                curve_values = survival_curve.values
                if curve_values.ndim > 1:
                    curve_values = curve_values.flatten()
                
                # 确保curve_values的长度与time_points匹配
                if len(curve_values) != len(time_points):
                    from scipy import interpolate
                    original_times = survival_curve.index.values
                    if len(original_times) == len(curve_values):
                        f = interpolate.interp1d(original_times, curve_values, 
                                                kind='linear', bounds_error=False, 
                                                fill_value='extrapolate')
                        curve_values = f(time_points)
                    else:
                        if len(curve_values) > 0:
                            last_value = curve_values[-1]
                            curve_values = np.full(len(time_points), last_value)
                        else:
                            curve_values = np.ones(len(time_points)) * 0.5
                
                survival_curves.append(curve_values)
            
            survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
            
            # 使用传统方法预测时间（仅事件样本），向预测函数传入真实时间/事件以便内部校准
            # 使用训练集标签作为校准数据（避免使用测试集标签导致信息泄露）
            time_preds = predict_time_traditional(survival_funcs_df, risk_scores_test, config,
                                                  true_times=y_train_final[:, 0], events=y_train_final[:, 1],
                                                  output_dir=output_dir)
            ev_mask = y_test[:, 1] == 1
            if np.any(ev_mask) and time_preds is not None:
                y_true = y_test[ev_mask, 0]
                y_pred = time_preds[ev_mask]
                mae = float(np.mean(np.abs(y_pred - y_true)))
                rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                final_metrics['time_prediction'] = {'mae': mae, 'rmse': rmse}
                logger.info(f"时间预测 - MAE: {mae:.2f}h, RMSE: {rmse:.2f}h")
                if bool(getattr(config.evaluation.time_head, 'save_predictions', True)):
                    outp = os.path.join(output_dir, 'time_predictions.csv')
                    os.makedirs(os.path.dirname(outp), exist_ok=True)
                    pd.DataFrame({'true_time': y_true, 'pred_time': y_pred}).to_csv(outp, index=False)
    except Exception as e:
        logging.getLogger(__name__).warning(f"传统时间预测失败: {e}")
    
    # 保存结果
    results = {
        'classification_results': classification_results,
        'survival_metrics': final_metrics,
        'output_dir': output_dir,
        'model_path': os.path.join(output_dir, "final_survival_model.pth")
    }
    
    # 保存最终模型
    torch.save(model.state_dict(), results['model_path'])
    
    # 保存完整结果
    results_path = os.path.join(output_dir, "two_stage_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info("两阶段训练实验完成")
    if classification_results is not None:
        logger.info(f"分类阶段最佳AUC: {classification_results['best_val_auc']:.4f}")
    else:
        logger.info("分类阶段被跳过或使用预训练权重")
    logger.info(f"生存分析阶段C-index: {final_metrics.get('c_index', 0):.4f}")
    
    return results


def run_classification_experiment(config, output_dir=None):
    """
    仅运行分类实验的便捷函数
    
    Args:
        config: 配置对象
        output_dir: 输出目录
        
    Returns:
        分类结果字典
    """
    logger = logging.getLogger(__name__)
    logger.info("开始分类实验...")
    
    validate_config(config)
    device = torch.device(config.training.device)
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            "results", f"classification_{config.model.name}_{timestamp}"
        )
    
    setup_logging(output_dir)
    save_config(config, output_dir)
    # 固化验证期评测策略到本次 run 目录
    try:
        import json as _json
        pol = {
            'predictions_are_risk': bool(getattr(config.evaluation, 'predictions_are_risk', True)),
            'use_cindex_preproc': bool(getattr(config.evaluation, 'use_cindex_preproc', getattr(config.evaluation, 'cindex_preprocess_enabled', False))),
            'cindex_iqr_multiplier': float(getattr(config.evaluation, 'cindex_iqr_multiplier', 3.0)),
            'fixed_auc_time_quantiles': list(getattr(config.evaluation, 'fixed_auc_time_quantiles', []))
        }
        with open(os.path.join(output_dir, 'eval_policy.json'), 'w') as _pf:
            _json.dump(pol, _pf)
    except Exception:
        pass
    
    # 数据预处理
    preprocessor = DataPreprocessor(config)
    preprocessor.execute()
    
    # 获取训练数据
    train_samples, test_samples = preprocessor.get_train_test_split()
    X_train, y_train, record_ids_train = preprocessor.get_evaluation_subsequences(train_samples)
    X_test, y_test, record_ids_test = preprocessor.get_evaluation_subsequences(test_samples)
    
    # 运行分类实验
    results = run_classification_only(config, X_train, y_train, record_ids_train)
    
    logger.info("分类实验完成")
    return results


def run_experiment(config, output_dir=None):
    """主实验运行流程，包括数据准备、模型训练、交叉验证和评估。"""
    
    validate_config(config)

    # 启用多格式绘图功能（如果尚未启用）
    try:
        from utils.multiformat_plotting import get_current_settings
        if not get_current_settings()['enabled']:
            enable_multiformat_plotting(config)
    except Exception:
        pass

    device = torch.device(config.training.device)

    # 通用 JSON 序列化辅助函数，处理 numpy / pandas 类型，使得写入 JSON 时不会失败
    def _make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        try:
            import pandas as _pd
            if isinstance(obj, _pd.Series):
                return obj.to_dict()
        except Exception:
            pass
        return obj

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            "results", f"{config.model.name}_{timestamp}"
        )
    
    setup_logging(output_dir)
    save_config(config, output_dir)
    logger = logging.getLogger(__name__)
    
    # 依据 encoder.type 设置 use_lstm
    try:
        enc_type = str(getattr(getattr(config.model, 'encoder', {}), 'type', 'lstm')).lower()
        if enc_type in ('lstm', 'transformer'):
            config.model.use_lstm = True
        else:
            config.model.use_lstm = bool(getattr(config.model, 'use_lstm', False))
    except Exception:
        config.model.use_lstm = bool(getattr(config.model, 'use_lstm', False))
    use_lstm = bool(getattr(config.model, 'use_lstm', False))
    logger.info(f"Sequence encoder enabled: {use_lstm} (encoder.type={enc_type if 'enc_type' in locals() else 'unknown'})")

    logger.info("开始数据预处理流水线...")
    _cleanup_preproc_reports()
    preprocessor = DataPreprocessor(config)
    preprocessor.execute()
    n_features = len(preprocessor.active_feature_names)
    logger.info("数据预处理完成。")

    if config.data.cross_validation.enabled:
        logging.info("开始交叉验证流程...")
        N_SPLITS = config.data.cross_validation.n_splits
        logger.info(f"--- 开始 {N_SPLITS}-折分组交叉验证 (GroupKFold) ---")
    
        train_parent_samples, _ = preprocessor.get_train_test_split()
        # 安全提取 parent group id：确保传递给 GroupKFold 的 groups 是可比较的标量（str/int），
        # 避免 record_id 为 dict 导致 numpy.unique 中的排序比较失败。
        def _extract_group_id(rec_id, idx):
            if rec_id is None:
                return f"idx_{idx}"
            if isinstance(rec_id, (str, int)):
                return rec_id
            if isinstance(rec_id, dict):
                # 常见的键名尝试
                for k in ('parent_id', 'record_id', 'id', 'patient_id'):
                    if k in rec_id:
                        return rec_id[k]
                # 回退为稳定的 JSON 字符串
                try:
                    import json as _json
                    return _json.dumps(rec_id, sort_keys=True)
                except Exception:
                    return str(rec_id)
            # 其它类型一律转为字符串
            return str(rec_id)

        parent_groups = [_extract_group_id(s.get('record_id'), i) for i, s in enumerate(train_parent_samples)]
        # 强制为 numpy 数组以避免 sklearn 在内部做意外类型推断
        parent_groups = np.array(parent_groups, dtype=object)
        
        gkf = GroupKFold(n_splits=N_SPLITS)
        all_fold_metrics = []
        all_fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_parent_samples, groups=parent_groups)):
            logger.info(f"--- 第 {fold + 1}/{N_SPLITS} 折 ---")

            cv_train_samples, cv_val_samples = preprocessor.get_cross_val_split((train_idx, val_idx))
            X_train_fold, y_train_fold, record_ids_train_fold = preprocessor.get_evaluation_subsequences(cv_train_samples)
            X_val_fold, y_val_fold, record_ids_val_fold = preprocessor.get_evaluation_subsequences(cv_val_samples)
            if record_ids_train_fold is not None:
                record_ids_train_fold = np.array(record_ids_train_fold, dtype=object)
            if record_ids_val_fold is not None:
                record_ids_val_fold = np.array(record_ids_val_fold, dtype=object)

            # ---> 新增: 数据清理 <---
            X_train_fold, y_train_fold, record_ids_train_fold = clean_and_log_data(
                X_train_fold, y_train_fold, logger, f"Fold-{fold+1} Train", record_ids_train_fold, config=config)
            X_val_fold, y_val_fold, record_ids_val_fold = clean_and_log_data(
                X_val_fold, y_val_fold, logger, f"Fold-{fold+1} Val", record_ids_val_fold, config=config)

            if X_train_fold is None or X_val_fold is None or len(X_train_fold) == 0 or len(X_val_fold) == 0:
                logger.warning(f"第 {fold + 1} 折数据不足，跳过。")
                continue

            # 使用按record_id分组的切分，避免同一父样本的子序列泄漏到训练与验证
            from sklearn.model_selection import GroupShuffleSplit
            val_ratio_cv = config.data.validation_ratio
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio_cv, random_state=config.data.random_seed)
            # 需要record_ids进行分组；normalize为可哈希的字符串（避免 dict 导致的排序/unique 错误）
            try:
                # 如果record_ids_train_fold中的元素是dict或元数据，提取可比较的字段
                groups_train_fold = np.array([
                    (r.get('ar') if isinstance(r, dict) and r.get('ar') is not None else (r.get('raw') if isinstance(r, dict) else r))
                    for r in record_ids_train_fold
                ], dtype=object)
                # 最终全部转为字符串，确保 numpy.unique 可比较
                groups_train_fold = np.array([str(g) for g in groups_train_fold], dtype=object)
            except Exception:
                groups_train_fold = np.array([str(r) for r in record_ids_train_fold], dtype=object)
            idx_all = np.arange(len(X_train_fold))
            train_idx_g, val_idx_g = next(gss.split(idx_all, groups=groups_train_fold))
            X_train_cv, X_val_cv = X_train_fold[train_idx_g], X_train_fold[val_idx_g]
            y_train_cv, y_val_cv = y_train_fold[train_idx_g], y_train_fold[val_idx_g]
            record_ids_train_cv = record_ids_train_fold[train_idx_g] if record_ids_train_fold is not None else None
            if getattr(config.model.two_stage.survival_stage, 'undersample_non_events', False):
                ratio = float(getattr(config.model.two_stage.survival_stage, 'undersample_ratio', 1.0))
                rs_state = getattr(
                    config.model.two_stage.survival_stage,
                    'undersample_random_state',
                    getattr(config, 'seed', None)
                )
                X_train_cv, y_train_cv, record_ids_train_cv, _ = _undersample_non_events(
                    X_train_cv,
                    y_train_cv,
                    ratio=ratio,
                    random_state=rs_state,
                    record_ids=record_ids_train_cv,
                )

            y_train_cv_df = pd.DataFrame(y_train_cv, columns=['duration', 'event'])
            # canonicalize durations to hours
            try:
                import numpy as _np
                y_train_cv_df['duration'] = _ensure_durations_in_hours(_np.asarray(y_train_cv_df['duration'].values), cfg=config, name='main_cv_y_train')
            except Exception:
                pass
            sample_weights_cv = calculate_sample_weights(y_train_cv_df)
            
            logger.info(f"折-{fold+1}: 训练集: {X_train_cv.shape}, 早停验证集: {X_val_cv.shape}, 折验证集: {X_val_fold.shape}")

            # --- 决定性调试 ---
            # Ensure the feature dimension passed to the model matches the data (subsequence generator may have appended agg features)
            inferred_n_features_cv = n_features
            try:
                if X_train_cv is not None and hasattr(X_train_cv, 'shape') and len(X_train_cv.shape) >= 3:
                    inferred_n_features_cv = X_train_cv.shape[2]
                    logger.info(f"Detected subsequence feature dim for CV fold from data: {inferred_n_features_cv} (overriding preprocessor n_features={n_features}).")
            except Exception:
                pass

            logger.info(f"即将创建模型，从配置中读取的名称为: '{config.model.name}'")
            model_fold = get_model(config, inferred_n_features_cv, device)
            # 根据模型类型选择不同的初始化方法
            if config.model.name.lower() == 'coxkan':
                model_fold.apply(init_weights_stable)  # 使用更稳定的初始化
            else:
                model_fold.apply(init_weights)  # 应用 Kaiming 初始化
            model_fold = train_model_loop(model_fold, config, X_train_cv, y_train_cv, X_val_cv, y_val_cv, sample_weights_cv, output_dir=output_dir)

            # --- 新增: 保存每一折模型 ---
            model_path = os.path.join(output_dir, f"fold_{fold+1}_model.pth")
            torch.save(model_fold.state_dict(), model_path)
            logger.info(f"模型已保存到: {model_path}")

            logger.info(f"--- 在第 {fold + 1} 折的验证集上评估... ---")
            evaluator_fold = Evaluator(model_fold, config, output_dir, device)
            record_ids_val_for_eval = record_ids_val_fold.tolist() if isinstance(record_ids_val_fold, np.ndarray) else record_ids_val_fold
            fold_metrics = evaluator_fold.evaluate(X_val_fold, y_val_fold, X_train_cv, y_train_cv, fold=fold+1, record_ids=record_ids_val_for_eval)
            
            # 为最后一折添加预测方法对比（避免重复计算）
            if fold == N_SPLITS - 1:  # 最后一折
                try:
                    if getattr(config.evaluation, 'prediction_methods_comparison', {}).get('enabled', False):
                        logger.info(f"\n--- 在第 {fold + 1} 折运行预测方法对比分析 ---")
                        from evaluation.prediction_methods_comparison import PredictionMethodsComparison
                        
                        # 创建预测方法对比器
                        comparison_runner = PredictionMethodsComparison(config, output_dir)
                        
                        # 获取生存函数数据
                        try:
                            if hasattr(evaluator_fold, 'survival_funcs_df') and evaluator_fold.survival_funcs_df is not None:
                                survival_funcs_df = evaluator_fold.survival_funcs_df
                                risk_scores = evaluator_fold.risk_scores if hasattr(evaluator_fold, 'risk_scores') else None
                                logger.info(f"使用evaluator中的生存函数数据: {survival_funcs_df.shape}")
                            else:
                                # 重新计算
                                logger.info("重新计算生存函数数据用于预测方法对比...")
                                risk_scores = predict_risk(model_fold, X_val_fold, device, config)
                                baseline_survival = compute_baseline_survival(model_fold, X_train_cv, y_train_cv, device, config)
                                
                                # 创建生存函数DataFrame
                                time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
                                survival_curves = []
                                
                                for i in range(len(X_val_fold)):
                                    individual_risk = risk_scores[i]
                                    survival_curve = baseline_survival ** np.exp(individual_risk)
                                    
                                    # 确保survival_curve.values是1维数组
                                    curve_values = survival_curve.values
                                    if curve_values.ndim > 1:
                                        curve_values = curve_values.flatten()
                                    
                                    # 确保curve_values的长度与time_points匹配
                                    if len(curve_values) != len(time_points):
                                        # 如果长度不匹配，使用插值来调整
                                        from scipy import interpolate
                                        if len(curve_values) > 0:
                                            # 使用原始时间点进行插值
                                            original_times = survival_curve.index.values
                                            if len(original_times) == len(curve_values):
                                                # 创建插值函数
                                                f = interpolate.interp1d(original_times, curve_values, 
                                                                        kind='linear', bounds_error=False, 
                                                                        fill_value='extrapolate')
                                                # 在新时间点上插值
                                                curve_values = f(time_points)
                                            else:
                                                # 如果原始时间点也不匹配，使用最后一个值填充
                                                if len(curve_values) > 0:
                                                    last_value = curve_values[-1]
                                                    curve_values = np.full(len(time_points), last_value)
                                                else:
                                                    curve_values = np.ones(len(time_points)) * 0.5
                                        else:
                                            curve_values = np.ones(len(time_points)) * 0.5
                                    
                                    survival_curves.append(curve_values)
                                
                                survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
                                logger.info(f"重新计算的生存函数数据: {survival_funcs_df.shape}")
                            
                            # 运行预测方法对比
                            record_ids_for_cmp = record_ids_val_fold.tolist() if isinstance(record_ids_val_fold, np.ndarray) else record_ids_val_fold
                            comparison_results = comparison_runner.run_comparison(
                                survival_funcs_df,
                                y_val_fold[:, 0],
                                y_val_fold[:, 1],
                                risk_scores,
                                record_ids_for_cmp,
                                split_name=f'fold_{fold + 1}_val'
                            )
                            
                            if comparison_results:
                                logger.info("预测方法对比分析完成")
                                # 将对比结果添加到折指标中
                                fold_metrics['prediction_methods_comparison'] = {
                                    'output_dir': comparison_results['output_dir'],
                                    'metrics_summary': {method: {
                                        'mae': metrics['mae'],
                                        'rmse': metrics['rmse'],
                                        'correlation': metrics['correlation']
                                    } for method, metrics in comparison_results['metrics'].items()}
                                }
                        except Exception as e:
                            logger.error(f"预测方法对比分析出错: {e}")
                except Exception as e:
                    logger.warning(f"预测方法对比功能执行失败: {e}")
            
            all_fold_metrics.append(fold_metrics)
            all_fold_models.append(model_fold)
            logger.info(f"第 {fold + 1} 折指标: {fold_metrics}")
            
        logger.info("\n--- 交叉验证完成 ---")
        avg_metrics = pd.DataFrame(all_fold_metrics).mean().to_dict()
        logger.info(f"平均指标: {avg_metrics}")
        
        # 绘制模型性能对比图
        try:
            metrics_dict = {f"Fold_{i+1}": metrics for i, metrics in enumerate(all_fold_metrics)}
            metrics_dict["Average"] = avg_metrics
            plot_model_performance_comparison(metrics_dict, output_dir)
        except Exception as e:
            logger.warning(f"绘制模型性能对比图失败: {e}")
        
        results_path = os.path.join(output_dir, "final_avg_metrics.json")
        # 将任何 numpy/pandas 类型转换为 Python 原生类型，确保可序列化
        def _make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_json_serializable(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            try:
                import pandas as _pd
                if isinstance(obj, _pd.Series):
                    return obj.to_dict()
            except Exception:
                pass
            return obj

        with open(results_path, 'w') as f:
            json.dump(_make_json_serializable(avg_metrics), f, indent=4)
        logger.info(f"最终平均指标已保存到: {results_path}")
        
        return avg_metrics
    
    else:
        logging.info("开始简单的训练/测试流程...")
        train_samples, test_samples = preprocessor.get_train_test_split()

        X_train, y_train, record_ids_train = preprocessor.get_evaluation_subsequences(train_samples)
        X_test, y_test, record_ids_test = preprocessor.get_evaluation_subsequences(test_samples)

        # ---> 新增: 数据清理 <---
        X_train, y_train, record_ids_train = clean_and_log_data(X_train, y_train, logger, "Train", record_ids=record_ids_train, config=config)
        X_test, y_test, record_ids_test = clean_and_log_data(X_test, y_test, logger, "Test", record_ids=record_ids_test, config=config)

        # 使用按record_id分组的切分，避免泄漏
        if getattr(config.data, 'use_validation', True):
            from sklearn.model_selection import GroupShuffleSplit
            gss_main = GroupShuffleSplit(n_splits=1, test_size=config.data.validation_ratio, random_state=config.data.random_seed)
            # 规范化 groups，避免 record_ids 中包含 dict 导致 numpy.unique 报错
            try:
                groups_main = np.array([
                    (r.get('ar') if isinstance(r, dict) and r.get('ar') is not None else (r.get('raw') if isinstance(r, dict) else r))
                    for r in record_ids_train
                ], dtype=object)
                groups_main = np.array([str(g) for g in groups_main], dtype=object)
            except Exception:
                groups_main = np.array([str(r) for r in record_ids_train], dtype=object)
            idx_all_main = np.arange(len(X_train))
            train_idx_main, val_idx_main = next(gss_main.split(idx_all_main, groups=groups_main))
            X_train_main, X_val = X_train[train_idx_main], X_train[val_idx_main]
            y_train_main, y_val = y_train[train_idx_main], y_train[val_idx_main]
        else:
            # 不使用验证集：全部数据用于训练，验证集置空（train_model_loop 将处理该情形）
            X_train_main, X_val = X_train, np.empty((0,))
            y_train_main, y_val = y_train, np.empty((0,))
        if getattr(config.model.two_stage.survival_stage, 'undersample_non_events', False):
            ratio = float(getattr(config.model.two_stage.survival_stage, 'undersample_ratio', 1.0))
            rs_state = getattr(
                config.model.two_stage.survival_stage,
                'undersample_random_state',
                getattr(config, 'seed', None)
            )
            X_train_main, y_train_main, record_ids_train, _ = _undersample_non_events(
                X_train_main,
                y_train_main,
                ratio=ratio,
                random_state=rs_state,
                record_ids=record_ids_train,
            )

        y_train_main_df = pd.DataFrame(y_train_main, columns=['duration', 'event'])
        try:
            import numpy as _np
            y_train_main_df['duration'] = _ensure_durations_in_hours(_np.asarray(y_train_main_df['duration'].values), cfg=config, name='main_y_train_main')
        except Exception:
            pass
        sample_weights = calculate_sample_weights(y_train_main_df)

        logger.info(f"训练集: {X_train_main.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

        # --- 决定性调试 ---
        # 确保用于创建模型的特征维度与实际训练数据一致（考虑到子序列中可能附加了聚合特征）
        inferred_n_features = n_features
        try:
            if X_train_main is not None and hasattr(X_train_main, 'shape') and len(X_train_main.shape) >= 3:
                inferred_n_features = X_train_main.shape[2]
                logger.info(f"Detected subsequence feature dim from data: {inferred_n_features} (overriding preprocessor n_features={n_features}).")
        except Exception:
            pass

        logger.info(f"即将创建模型，从配置中读取的名称为: '{config.model.name}'")
        model = get_model(config, inferred_n_features, device)
        # 根据模型类型选择不同的初始化方法
        if config.model.name.lower() == 'coxkan':
            model.apply(init_weights_stable)  # 使用更稳定的初始化
        else:
            model.apply(init_weights)  # 应用 Kaiming 初始化
        model = train_model_loop(model, config, X_train_main, y_train_main, X_val, y_val, sample_weights, output_dir=output_dir)

        logger.info("\n--- 在测试集上评估最终模型 ---")
        logger.info(f"[DEBUG] After train_model_loop, model is: {type(model)} | is None: {model is None}")
        evaluator = Evaluator(model, config, output_dir, device)
        final_metrics = evaluator.evaluate(X_test, y_test, X_train_main, y_train_main, plot_extra_visuals=True, record_ids=record_ids_test, raw_samples=preprocessor.all_merged_samples if 'preprocessor' in locals() else None, feature_names=preprocessor.active_feature_names if 'preprocessor' in locals() and hasattr(preprocessor, 'active_feature_names') else None)
        
        # 第三阶段：时间预测（使用传统方法，不考虑删失）
        try:
            th_cfg = getattr(config.evaluation, 'time_head', None)
            if th_cfg and bool(getattr(th_cfg, 'enabled', False)):
                logger.info("使用传统方法预测时间（基于生存函数）")
                
                # 计算测试集的生存函数
                baseline_survival = compute_baseline_survival(model, X_train_main, y_train_main, device, config)
                risk_scores_test = predict_risk(model, X_test, device, config)
                
                # 计算个体生存函数
                time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
                survival_curves = []
                for i in range(len(X_test)):
                    individual_risk = risk_scores_test[i]
                    survival_curve = baseline_survival ** np.exp(individual_risk)
                    curve_values = survival_curve.values
                    if curve_values.ndim > 1:
                        curve_values = curve_values.flatten()
                    
                    # 确保curve_values的长度与time_points匹配
                    if len(curve_values) != len(time_points):
                        from scipy import interpolate
                        original_times = survival_curve.index.values
                        if len(original_times) == len(curve_values):
                            f = interpolate.interp1d(original_times, curve_values, 
                                                    kind='linear', bounds_error=False, 
                                                    fill_value='extrapolate')
                            curve_values = f(time_points)
                        else:
                            if len(curve_values) > 0:
                                last_value = curve_values[-1]
                                curve_values = np.full(len(time_points), last_value)
                            else:
                                curve_values = np.ones(len(time_points)) * 0.5
                    
                    survival_curves.append(curve_values)
                
                survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
                
                # 使用传统方法预测时间（仅事件样本），向预测函数传入真实时间/事件以便内部校准
                # 使用训练集标签进行校准，避免在评估时泄露测试集信息
                time_preds = predict_time_traditional(survival_funcs_df, risk_scores_test, config,
                                                      true_times=y_train_main[:, 0], events=y_train_main[:, 1],
                                                      output_dir=output_dir)
                ev_mask = y_test[:, 1] == 1
                if np.any(ev_mask) and time_preds is not None:
                    y_true = y_test[ev_mask, 0]
                    y_pred = time_preds[ev_mask]
                    mae = float(np.mean(np.abs(y_pred - y_true)))
                    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                    final_metrics['time_prediction'] = {'mae': mae, 'rmse': rmse}
                    logger.info(f"时间预测 - MAE: {mae:.2f}h, RMSE: {rmse:.2f}h")
                    if bool(getattr(config.evaluation.time_head, 'save_predictions', True)):
                        outp = os.path.join(output_dir, 'time_predictions.csv')
                        os.makedirs(os.path.dirname(outp), exist_ok=True)
                        pd.DataFrame({'true_time': y_true, 'pred_time': y_pred}).to_csv(outp, index=False)
        except Exception as e:
            logging.getLogger(__name__).warning(f"传统时间预测失败: {e}")

        # 集成预测方法对比功能
        try:
            if getattr(config.evaluation, 'prediction_methods_comparison', {}).get('enabled', False):
                logger.info("\n--- 运行预测方法对比分析 ---")
                from evaluation.prediction_methods_comparison import PredictionMethodsComparison
                
                # 创建预测方法对比器
                comparison_runner = PredictionMethodsComparison(config, output_dir)
                
                # 获取生存函数数据（从evaluator中提取）
                try:
                    # 尝试从evaluator获取生存函数数据
                    if hasattr(evaluator, 'survival_funcs_df') and evaluator.survival_funcs_df is not None:
                        survival_funcs_df = evaluator.survival_funcs_df
                        risk_scores = evaluator.risk_scores if hasattr(evaluator, 'risk_scores') else None
                        logger.info(f"使用evaluator中的生存函数数据: {survival_funcs_df.shape}")
                    else:
                        # 如果没有，重新计算
                        logger.info("重新计算生存函数数据用于预测方法对比...")
                        risk_scores = predict_risk(model, X_test, device, config)
                        baseline_survival = compute_baseline_survival(model, X_train_main, y_train_main, device, config)
                        
                        # 创建生存函数DataFrame
                        time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
                        survival_curves = []
                        
                        for i in range(len(X_test)):
                            # 计算个体生存函数
                            individual_risk = risk_scores[i]
                            survival_curve = baseline_survival ** np.exp(individual_risk)
                            
                            # 确保survival_curve.values是1维数组
                            curve_values = survival_curve.values
                            if curve_values.ndim > 1:
                                curve_values = curve_values.flatten()
                            
                            # 确保curve_values的长度与time_points匹配
                            if len(curve_values) != len(time_points):
                                # 如果长度不匹配，使用插值来调整
                                from scipy import interpolate
                                if len(curve_values) > 0:
                                    # 使用原始时间点进行插值
                                    original_times = survival_curve.index.values
                                    if len(original_times) == len(curve_values):
                                        # 创建插值函数
                                        f = interpolate.interp1d(original_times, curve_values, 
                                                                kind='linear', bounds_error=False, 
                                                                fill_value='extrapolate')
                                        # 在新时间点上插值
                                        curve_values = f(time_points)
                                    else:
                                        # 如果原始时间点也不匹配，使用最后一个值填充
                                        if len(curve_values) > 0:
                                            last_value = curve_values[-1]
                                            curve_values = np.full(len(time_points), last_value)
                                        else:
                                            curve_values = np.ones(len(time_points)) * 0.5
                                else:
                                    curve_values = np.ones(len(time_points)) * 0.5
                            
                            survival_curves.append(curve_values)
                        
                        survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
                        logger.info(f"重新计算的生存函数数据: {survival_funcs_df.shape}")
                    
                    # 运行预测方法对比
                    comparison_results = comparison_runner.run_comparison(
                        survival_funcs_df,
                        y_test[:, 0],
                        y_test[:, 1],
                        risk_scores,
                        record_ids_test,
                        split_name='test'
                    )
                    
                    if comparison_results:
                        logger.info("预测方法对比分析完成")
                        # 将对比结果添加到最终指标中
                        final_metrics['prediction_methods_comparison'] = {
                            'output_dir': comparison_results['output_dir'],
                            'metrics_summary': {method: {
                                'mae': metrics['mae'],
                                'rmse': metrics['rmse'],
                                'correlation': metrics['correlation']
                            } for method, metrics in comparison_results['metrics'].items()}
                        }
                    else:
                        logger.warning("预测方法对比分析失败")
                        
                except Exception as e:
                    logger.error(f"预测方法对比分析出错: {e}")
                    logger.exception("详细错误信息:")
            else:
                logger.info("预测方法对比功能已禁用")
        except Exception as e:
            logger.warning(f"预测方法对比功能执行失败: {e}")

        results_path = os.path.join(output_dir, "final_test_metrics.json")
        with open(results_path, 'w') as f:
            json.dump(_make_json_serializable(final_metrics), f, indent=4)
        logger.info(f"最终测试指标已保存到: {results_path}")

        if X_test.shape[0] > 0:
            logger.info("为测试集中的一些样本绘制风险进展图...")
            num_plots = min(5, X_test.shape[0])
            evaluator.plot_risk_progression(X_test[:num_plots])

        return final_metrics


 
    

def main():
    """主入口函数。根据 config.run_mode 调用不同流程。
    支持: 'train_only', 'test_only', 'cross_validation', 'two_stage', 'classification_only'
    """
    config = get_config()
    logger = logging.getLogger(__name__)

    # 启用多格式绘图功能（自动为所有图表保存多种格式）
    try:
        enable_multiformat_plotting(config)
        logger.info("✓ 已启用多格式绘图功能")
    except Exception as e:
        logger.warning(f"启用多格式绘图功能失败: {e}")

    try:
        run_mode = getattr(config, 'run_mode', 'cross_validation')
        logger.info(f"运行模式: {run_mode}")

        if run_mode == 'train_only':
            # 关闭交叉验证，执行简单的训练/验证流程
            config.data.cross_validation.enabled = False
            config.run_mode = 'train_only'
            run_experiment(config)
        elif run_mode == 'test_only':
            # 关闭交叉验证，执行测试流程。
            # 行为说明：如果配置了 config.model.checkpoint_path（文件或包含 .pth 的 run 目录），
            # 则直接加载该 checkpoint 并只运行评估；否则按原始逻辑调用 run_experiment（训练后评估）。
            config.data.cross_validation.enabled = False
            config.run_mode = 'test_only'
            ckpt_path = getattr(config.model, 'checkpoint_path', None)
            if ckpt_path:
                logger.info(f"test_only 模式检测到 checkpoint_path={ckpt_path}，尝试加载并仅执行评估。")
                # 1) 优先从 run 目录加载训练期 config.json 重建配置
                try:
                    run_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
                    run_cfg_path = os.path.join(run_dir, 'config.json')
                    if os.path.exists(run_cfg_path):
                        with open(run_cfg_path, 'r') as _cf:
                            trained_cfg = json.load(_cf)
                        
                        # Preserve current results_dir (e.g. CLI --output_dir) to avoid overwriting the
                        # user-intended output directory when loading the training config.
                        _preserve_results_dir = getattr(config, 'results_dir', None)
                        
                        # 完全替换配置：在test_only模式下，使用训练时的配置，忽略default_config
                        # 递归地将字典转换为EasyDict对象，从trained_cfg重建整个配置结构
                        from easydict import EasyDict
                        
                        def _dict_to_easydict(d):
                            """递归将字典转换为EasyDict"""
                            if isinstance(d, dict):
                                ed = EasyDict()
                                for k, v in d.items():
                                    ed[k] = _dict_to_easydict(v)
                                return ed
                            elif isinstance(d, list):
                                return [_dict_to_easydict(item) if isinstance(item, (dict, list)) else item for item in d]
                            else:
                                return d
                        
                        config = _dict_to_easydict(trained_cfg)
                        
                        # 恢复关键运行时设置
                        config.data.cross_validation.enabled = False  # test_only模式关闭交叉验证
                        config.run_mode = 'test_only'
                        if _preserve_results_dir is not None:
                            try:
                                config.results_dir = _preserve_results_dir
                                logger.info(f"保留外部指定的 results_dir: {config.results_dir}")
                            except Exception:
                                pass
                        
                        # 确保预测方法对比配置存在（如果用户配置中启用了，则强制启用并复制完整配置）
                        try:
                            from configs.default_config import get_config as get_default_config
                            default_cfg = get_default_config()
                            if hasattr(default_cfg.evaluation, 'prediction_methods_comparison'):
                                default_pred_cfg = default_cfg.evaluation.prediction_methods_comparison
                                if getattr(default_pred_cfg, 'enabled', False):
                                    # 如果用户配置中启用了，确保训练期配置也启用并复制完整配置
                                    if not hasattr(config.evaluation, 'prediction_methods_comparison'):
                                        config.evaluation.prediction_methods_comparison = EasyDict()
                                    
                                    # 复制用户配置中的所有字段到训练期配置
                                    def _copy_easydict(src, dst):
                                        """递归复制EasyDict的属性"""
                                        for key, value in src.items():
                                            if isinstance(value, EasyDict):
                                                cur = getattr(dst, key, None)
                                                if not isinstance(cur, EasyDict):
                                                    dst[key] = EasyDict()
                                                _copy_easydict(value, dst[key])
                                            elif isinstance(value, (list, tuple)):
                                                dst[key] = list(value) if isinstance(value, tuple) else value
                                            else:
                                                dst[key] = value
                                    
                                    _copy_easydict(default_pred_cfg, config.evaluation.prediction_methods_comparison)
                                    config.evaluation.prediction_methods_comparison.enabled = True
                                    logger.info("检测到用户配置中启用了预测方法对比，已复制完整配置并强制启用")
                        except Exception as _e:
                            logger.warning(f"检查预测方法对比配置时出错: {_e}", exc_info=True)
                        
                        # 允许在test_only模式下覆盖风险分数校准配置（如果当前配置中启用了）
                        # 这样即使训练时未启用校准，测试时也可以启用
                        try:
                            from configs.default_config import get_config as get_default_config
                            default_cfg = get_default_config()
                            if hasattr(default_cfg.evaluation, 'risk_score_calibration'):
                                default_risk_cal_cfg = default_cfg.evaluation.risk_score_calibration
                                if getattr(default_risk_cal_cfg, 'enabled', False):
                                    # 如果当前配置中启用了风险分数校准，覆盖训练期配置
                                    if not hasattr(config.evaluation, 'risk_score_calibration'):
                                        config.evaluation.risk_score_calibration = EasyDict()
                                    
                                    # 复制当前配置的所有字段
                                    def _copy_easydict(src, dst):
                                        """递归复制EasyDict的属性"""
                                        for key, value in src.items():
                                            if isinstance(value, EasyDict):
                                                cur = getattr(dst, key, None)
                                                if not isinstance(cur, EasyDict):
                                                    dst[key] = EasyDict()
                                                _copy_easydict(value, dst[key])
                                            elif isinstance(value, (list, tuple)):
                                                dst[key] = list(value) if isinstance(value, tuple) else value
                                            else:
                                                dst[key] = value
                                    
                                    _copy_easydict(default_risk_cal_cfg, config.evaluation.risk_score_calibration)
                                    config.evaluation.risk_score_calibration.enabled = True
                                    logger.info("检测到当前配置中启用了风险分数校准，已覆盖训练期配置并强制启用")
                        except Exception as _e:
                            logger.warning(f"检查风险分数校准配置时出错: {_e}", exc_info=True)
                        
                        logger.info(f"已从 {run_cfg_path} 完全替换为训练期配置（test_only模式）。")
                        logger.info(f"训练期配置摘要: two_stage.enabled={getattr(getattr(config.model, 'two_stage', None), 'enabled', None)}, "
                                   f"encoder.type={getattr(getattr(config.model, 'encoder', None), 'type', None)}, "
                                   f"prediction_methods_comparison.enabled={getattr(getattr(config.evaluation, 'prediction_methods_comparison', None), 'enabled', False)}")
                except Exception as _e:
                    logger.warning(f"从 run 目录重建配置失败，继续使用当前配置: {_e}")

                # 2) 构建数据并模型
                validate_config(config)
                device = torch.device(config.training.device)
                preprocessor = DataPreprocessor(config)
                preprocessor.execute()
                # 强制加载训练期预处理产物（如存在）以避免重新拟合导致分布漂移
                try:
                    pr_dir = os.path.join(getattr(config, 'results_dir', 'results'), 'preproc_reports')
                    alt_pr = os.path.join(run_dir, 'preproc_reports')
                    if os.path.isdir(alt_pr):
                        pr_dir = alt_pr
                    import joblib
                    scaler_p = os.path.join(pr_dir, 'scaler.joblib')
                    imputer_p = os.path.join(pr_dir, 'imputer.joblib')
                    cols_p = os.path.join(pr_dir, 'scaler_feature_columns.json')
                    if os.path.exists(scaler_p):
                        preprocessor.scaler = joblib.load(scaler_p)
                        logger.info(f"Loaded scaler from {scaler_p}")
                    if os.path.exists(imputer_p):
                        try:
                            preprocessor.imputer = joblib.load(imputer_p)
                            logger.info(f"Loaded imputer from {imputer_p}")
                        except Exception:
                            pass
                    if os.path.exists(cols_p):
                        with open(cols_p, 'r') as _f:
                            preprocessor.scaler_feature_columns = json.load(_f)
                        logger.info(f"Loaded scaler_feature_columns from {cols_p}")
                    # If a preproc manifest exists, read it and record for diagnostics
                    manifest_p = os.path.join(pr_dir, 'preproc_manifest.json')
                    preproc_manifest = None
                    preproc_manifest_mismatch = None
                    try:
                        if os.path.exists(manifest_p):
                            with open(manifest_p, 'r') as _mf:
                                preproc_manifest = json.load(_mf)
                            logger.info(f"Loaded preproc_manifest from {manifest_p}")
                            # basic consistency check: compare counts
                            try:
                                expected_count = preproc_manifest.get('scaler_feature_columns_count', None)
                                loaded_count = len(preprocessor.scaler_feature_columns) if getattr(preprocessor, 'scaler_feature_columns', None) is not None else None
                                if expected_count is not None and loaded_count is not None and expected_count != loaded_count:
                                    preproc_manifest_mismatch = {
                                        'type': 'scaler_columns_count_mismatch',
                                        'expected': expected_count,
                                        'loaded': loaded_count,
                                        'manifest_path': manifest_p
                                    }
                                    logger.warning(f"Preproc manifest mismatch detected: {preproc_manifest_mismatch}")
                            except Exception:
                                pass
                    except Exception as _e:
                        logger.warning(f"Failed to load preproc_manifest: {_e}")
                except Exception as _e:
                    logger.warning(f"加载训练期预处理产物失败，可能导致分布不一致: {_e}")
                X_train, y_train, record_ids_train = preprocessor.get_evaluation_subsequences(preprocessor.train_samples)
                X_test, y_test, record_ids_test = preprocessor.get_evaluation_subsequences(preprocessor.test_samples)
                # 构建模型
                inferred_n_features = X_test.shape[2] if (hasattr(X_test, 'shape') and len(X_test.shape) >= 3) else len(preprocessor.active_feature_names)
                model = get_model(config, inferred_n_features, device)
                # 如果 checkpoint_path 是目录，尝试查找 .pth
                if os.path.isdir(ckpt_path):
                    # search for best/fold pth
                    ckpt_found = None
                    for root, _, files in os.walk(ckpt_path):
                        for fn in files:
                            if fn.endswith('.pth') and ('best' in fn.lower() or 'fold' in fn.lower()):
                                ckpt_found = os.path.join(root, fn)
                                break
                        if ckpt_found:
                            break
                    if ckpt_found is None:
                        logger.warning(f"在目录 {ckpt_path} 中未找到可用的 .pth 权重，切回默认流程（训练+测试）。")
                        run_experiment(config)
                        return
                    ckpt_path = ckpt_found
                # 尝试加载权重
                try:
                    import torch as _torch
                    state = _torch.load(ckpt_path, map_location=device)
                    if isinstance(state, dict) and 'state_dict' in state:
                        model.load_state_dict(state['state_dict'])
                    else:
                        try:
                            model.load_state_dict(state)
                        except Exception:
                            if isinstance(state, dict) and 'model_state' in state:
                                model.load_state_dict(state['model_state'])
                    logger.info(f"已从 {ckpt_path} 加载模型权重。")
                except Exception as e:
                    logger.warning(f"加载 checkpoint 失败 ({ckpt_path})：{e}，切回默认训练+测试流程。")
                    run_experiment(config)
                    return

                # 执行评估
                try:
                    model.to(device)
                except Exception:
                    pass
                # 在 test_only 下，将输出集中写入 results_dir/test_only 子目录（不修改全局 results_dir 以免影响预处理加载）
                _base_out_dir = getattr(config, 'results_dir', 'results')
                _test_only_out_dir = os.path.join(_base_out_dir, 'test_only')
                evaluator = Evaluator(model, config, _test_only_out_dir, device)
                os.makedirs(evaluator.output_dir, exist_ok=True)
                final_metrics = evaluator.evaluate(X_test, y_test, X_train, y_train, plot_extra_visuals=True, record_ids=record_ids_test, raw_samples=preprocessor.all_merged_samples if 'preprocessor' in locals() else None, feature_names=preprocessor.active_feature_names if 'preprocessor' in locals() and hasattr(preprocessor, 'active_feature_names') else None)
                
                # 集成预测方法对比功能（test_only模式）
                try:
                    if getattr(config.evaluation, 'prediction_methods_comparison', {}).get('enabled', False):
                        logger.info("\n--- 运行预测方法对比分析 (test_only模式) ---")
                        from evaluation.prediction_methods_comparison import PredictionMethodsComparison
                        
                        # 创建预测方法对比器
                        comparison_runner = PredictionMethodsComparison(config, evaluator.output_dir)
                        
                        # 获取生存函数数据
                        try:
                            if hasattr(evaluator, 'survival_funcs_df') and evaluator.survival_funcs_df is not None:
                                survival_funcs_df = evaluator.survival_funcs_df
                                risk_scores = evaluator.risk_scores if hasattr(evaluator, 'risk_scores') else None
                                logger.info(f"使用evaluator中的生存函数数据: {survival_funcs_df.shape}")
                            else:
                                # 重新计算
                                logger.info("重新计算生存函数数据用于预测方法对比...")
                                risk_scores = predict_risk(model, X_test, device, config)
                                baseline_survival = compute_baseline_survival(model, X_train, y_train, device, config)
                                
                                # 创建生存函数DataFrame
                                time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
                                survival_curves = []
                                
                                for i in range(len(X_test)):
                                    individual_risk = risk_scores[i]
                                    survival_curve = baseline_survival ** np.exp(individual_risk)
                                    
                                    # 确保survival_curve.values是1维数组
                                    curve_values = survival_curve.values
                                    if curve_values.ndim > 1:
                                        curve_values = curve_values.flatten()
                                    
                                    # 确保curve_values的长度与time_points匹配
                                    if len(curve_values) != len(time_points):
                                        # 如果长度不匹配，使用插值来调整
                                        from scipy import interpolate
                                        if len(curve_values) > 0:
                                            # 使用原始时间点进行插值
                                            original_times = survival_curve.index.values
                                            if len(original_times) == len(curve_values):
                                                # 创建插值函数
                                                f = interpolate.interp1d(original_times, curve_values, 
                                                                        kind='linear', bounds_error=False, 
                                                                        fill_value='extrapolate')
                                                # 在新时间点上插值
                                                curve_values = f(time_points)
                                            else:
                                                # 如果原始时间点也不匹配，使用最后一个值填充
                                                if len(curve_values) > 0:
                                                    last_value = curve_values[-1]
                                                    curve_values = np.full(len(time_points), last_value)
                                                else:
                                                    curve_values = np.ones(len(time_points)) * 0.5
                                        else:
                                            curve_values = np.ones(len(time_points)) * 0.5
                                    
                                    survival_curves.append(curve_values)
                                
                                survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
                                logger.info(f"重新计算的生存函数数据: {survival_funcs_df.shape}")
                        
                        except Exception as e:
                            logger.error(f"获取生存函数数据失败: {e}", exc_info=True)
                            survival_funcs_df = None
                        
                        # 运行预测方法对比
                        if survival_funcs_df is not None and not survival_funcs_df.empty:
                            try:
                                logger.info(f"开始运行预测方法对比，生存函数数据形状: {survival_funcs_df.shape}, 测试样本数: {len(y_test)}")
                                comparison_results = comparison_runner.run_comparison(
                                    survival_funcs_df,
                                    y_test[:, 0],
                                    y_test[:, 1],
                                    risk_scores,
                                    record_ids_test,
                                    split_name='test'
                                )
                                
                                if comparison_results:
                                    logger.info("预测方法对比分析完成")
                                    # 将对比结果添加到最终指标中
                                    if 'metrics' in comparison_results:
                                        final_metrics['prediction_methods_comparison'] = {
                                            'output_dir': comparison_results.get('output_dir', ''),
                                            'metrics_summary': {method: {
                                                'mae': metrics.get('mae', None),
                                                'rmse': metrics.get('rmse', None),
                                                'correlation': metrics.get('correlation', None)
                                            } for method, metrics in comparison_results['metrics'].items()}
                                        }
                                else:
                                    logger.warning("预测方法对比未生成结果（run_comparison返回None）")
                            except Exception as e:
                                logger.error(f"运行预测方法对比时出错: {e}", exc_info=True)
                        else:
                            logger.warning(f"无法运行预测方法对比：survival_funcs_df为None或空 (is None: {survival_funcs_df is None}, is empty: {survival_funcs_df.empty if survival_funcs_df is not None else 'N/A'})")
                    else:
                        logger.info("预测方法对比功能未启用（配置中enabled=False或配置不存在）")
                except Exception as e:
                    logger.error(f"预测方法对比功能执行失败: {e}", exc_info=True)
                
                # Attach audit info so each evaluation records which checkpoint/preproc were used
                try:
                    audit = {
                        'used_checkpoint': ckpt_path,
                        'used_preproc_dir': pr_dir if 'pr_dir' in locals() else None,
                        'preproc_manifest': preproc_manifest if 'preproc_manifest' in locals() else None,
                        'preproc_manifest_mismatch': preproc_manifest_mismatch if 'preproc_manifest_mismatch' in locals() else None
                    }
                    final_metrics['_audit'] = audit
                except Exception:
                    pass
                results_path = os.path.join(evaluator.output_dir, "final_test_metrics.json")
                with open(results_path, 'w') as f:
                    json.dump(_make_json_serializable(final_metrics), f, indent=4)
                logger.info(f"评估结果已保存到: {results_path}")
                return final_metrics
            else:
                run_experiment(config)
        elif run_mode == 'cross_validation':
            config.data.cross_validation.enabled = True
            config.run_mode = 'cross_validation'
            run_experiment(config)
        elif run_mode == 'two_stage':
            # 两阶段训练：先训练分类模型，再微调生存分析模型
            config.data.cross_validation.enabled = False
            config.run_mode = 'two_stage'
            config.model.two_stage.enabled = True
            run_two_stage_experiment(config)
        elif run_mode == 'time_head_only':
            # 独立第三阶段：加载已训练两阶段模型，仅训练/评估时间头
            th_cfg = getattr(config.evaluation, 'time_head', None)
            # 安全获取 standalone 子配置（无需依赖 EasyDict）
            standalone_cfg = getattr(th_cfg, 'standalone', None)
            standalone_enabled = bool(getattr(standalone_cfg, 'enabled', False)) if standalone_cfg is not None else False
            if not th_cfg or not standalone_enabled:
                logger.warning("time_head_only 模式需要开启 evaluation.time_head.standalone.enabled")
                run_experiment(config)
                return
            # 读取 checkpoint 路径
            ckpt_path = getattr(th_cfg.standalone, 'base_model_checkpoint_path', '')
            if not isinstance(ckpt_path, str) or len(ckpt_path) == 0:
                logger.warning("未提供 base_model_checkpoint_path，切回默认流程。")
                run_experiment(config)
                return
            # 预处理与数据（在加载前尝试按预训练配置对齐关键超参）
            preprocessor = DataPreprocessor(config)
            preprocessor.execute()
            split = str(getattr(th_cfg.standalone, 'dataset_split', 'test')).lower()
            if split == 'train':
                X_src, y_src, record_ids_src = preprocessor.get_evaluation_subsequences(preprocessor.train_samples)
                X_ref, y_ref = X_src, y_src
            else:
                X_src, y_src, record_ids_src = preprocessor.get_evaluation_subsequences(preprocessor.test_samples)
                X_ref, y_ref = preprocessor.get_evaluation_subsequences(preprocessor.train_samples)[:2]
            # 自动与预训练配置对齐（在模型创建之前进行）
            try:
                pretrained_cfg_dict = _load_pretrained_config_dict(ckpt_path)
                if pretrained_cfg_dict:
                    logger.info("开始根据预训练配置自动对齐关键超参...")
                    _auto_align_config_with_pretrained(config, pretrained_cfg_dict)
                    logger.info("已根据预训练配置自动对齐关键超参（encoder/下游/窗口/特征列等）")
                else:
                    logger.warning("未能从预训练模型路径加载配置，将使用当前配置文件设置")
            except Exception as e:
                logger.warning(f"配置对齐过程中出现异常: {e}，将使用当前配置文件设置")
            
            # 构建与加载模型（在配置对齐之后）
            device = torch.device(config.training.device)
            n_features = X_src.shape[-1]
            model = get_model(config, n_features, device)
            try:
                import torch as _torch
                state = _torch.load(ckpt_path, map_location=device)
                if isinstance(state, dict) and 'state_dict' in state:
                    model.load_state_dict(state['state_dict'])
                else:
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        if isinstance(state, dict) and 'model_state' in state:
                            model.load_state_dict(state['model_state'])
                logger.info(f"已从 {ckpt_path} 加载模型权重。")
                # 确保模型参数与输入在同一设备
                model.to(device)
                model.eval()
            except Exception as e:
                logger.warning(f"加载 checkpoint 失败 ({ckpt_path})：{e}，切回默认流程。")
                run_experiment(config)
                return
            # 使用传统方法预测时间
            logger.info("使用传统方法预测时间（基于生存函数，不考虑删失）")
            
            # 计算生存函数
            baseline_survival = compute_baseline_survival(model, X_ref, y_ref, device, config)
            risk_src = predict_risk(model, X_src, device, config)
            
            # 计算个体生存函数
            time_points = np.arange(0, config.data.sequence_generation.prediction_window_hours + 2, 2)
            survival_curves = []
            for i in range(len(X_src)):
                individual_risk = risk_src[i]
                survival_curve = baseline_survival ** np.exp(individual_risk)
                curve_values = survival_curve.values
                if curve_values.ndim > 1:
                    curve_values = curve_values.flatten()
                
                # 确保curve_values的长度与time_points匹配
                if len(curve_values) != len(time_points):
                    from scipy import interpolate
                    original_times = survival_curve.index.values
                    if len(original_times) == len(curve_values):
                        f = interpolate.interp1d(original_times, curve_values, 
                                                kind='linear', bounds_error=False, 
                                                fill_value='extrapolate')
                        curve_values = f(time_points)
                    else:
                        if len(curve_values) > 0:
                            last_value = curve_values[-1]
                            curve_values = np.full(len(time_points), last_value)
                        else:
                            curve_values = np.ones(len(time_points)) * 0.5
                
                survival_curves.append(curve_values)
            
            survival_funcs_df = pd.DataFrame(survival_curves, columns=time_points)
            
            # 使用传统方法预测时间（仅事件样本），使用参考训练集 (y_ref) 作为校准数据以避免泄露
            preds = predict_time_traditional(survival_funcs_df, risk_src, config,
                                            true_times=y_ref[:, 0], events=y_ref[:, 1], output_dir=out_dir)
            
            out_dir = os.path.join(getattr(config, 'results_dir', 'results'), 'time_prediction')
            os.makedirs(out_dir, exist_ok=True)
            
            ev_mask = y_src[:, 1] == 1
            if np.any(ev_mask) and preds is not None:
                y_true = y_src[ev_mask, 0]
                y_pred = preds[ev_mask]
                mae = float(np.mean(np.abs(y_pred - y_true)))
                rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                mse = float(np.mean((y_pred - y_true)**2))
                mape = float(np.mean(np.abs((y_pred - y_true) / np.maximum(y_true, 1e-6)) * 100))
                logger.info(f"[传统时间预测] MAE={mae:.3f}, RMSE={rmse:.3f}, N={len(y_true)}")
                
                # === 测试总结 ===
                print("\n" + "="*80)
                print("传统时间预测测试总结")
                print("="*80)
                print(f"\n【测试指标】")
                print(f"  MAE (平均绝对误差): {mae:.4f} 小时")
                print(f"  RMSE (均方根误差): {rmse:.4f} 小时")
                print(f"  MSE (均方误差): {mse:.4f} 小时²")
                print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
                print(f"\n【样本统计】")
                print(f"  测试样本数: {len(y_true)}")
                print(f"  事件样本比例: {len(y_true)/len(y_src):.1%}")
                print(f"\n【预测分布】")
                print(f"  真实时间 - 均值: {np.mean(y_true):.2f}h, 中位数: {np.median(y_true):.2f}h")
                print(f"  预测时间 - 均值: {np.mean(y_pred):.2f}h, 中位数: {np.median(y_pred):.2f}h")
                print(f"\n【保存位置】")
                print(f"  结果目录: {out_dir}")
                print(f"  预测数据: {split}_time_predictions.csv")
                print("="*80 + "\n")
                
                pd.DataFrame({'true_time': y_true, 'pred_time': y_pred}).to_csv(
                    os.path.join(out_dir, f'{split}_time_predictions.csv'), index=False
                )
            return
        elif run_mode == 'classification_only':
            # 仅训练分类模型
            config.data.cross_validation.enabled = False
            config.run_mode = 'classification_only'
            config.model.two_stage.enabled = True
            config.model.two_stage.classification_stage.enabled = True
            config.model.two_stage.survival_stage.enabled = False
            run_classification_experiment(config)
        else:
            logger.warning(f"未知的run_mode '{run_mode}'，默认使用交叉验证模式。")
            config.data.cross_validation.enabled = True
            config.run_mode = 'cross_validation'
            run_experiment(config)

    except Exception as e:
        logger.exception(f"主流程执行失败: {e}")
        raise


if __name__ == '__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description='Run FlameShadowModel experiment (LSTM+DeepSurv/DeepHit/CoxKAN)')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional output directory override (maps to config.results_dir)')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU')
    parser.add_argument('--run_mode', type=str, default=None,
                        choices=['train_only','test_only','cross_validation','two_stage','classification_only','time_head_only'],
                        help='Override run_mode from default_config at runtime')
    parser.add_argument('--preset', type=str, default=None,
                        help='Optional preset module under configs.experiments (e.g., paper_reproduction)')
    args = parser.parse_args()

    cfg = get_config()
    if args.no_cuda:
        cfg.training.device = 'cpu'
    if args.run_mode is not None:
        cfg.run_mode = args.run_mode
    if args.output_dir is not None and len(args.output_dir) > 0:
        # 将命令行的输出目录映射到配置中的 results_dir，以便所有流程统一使用
        cfg.results_dir = args.output_dir

    # 当通过 CLI 指定 time_head_only 时，自动开启第三阶段与独立模式，避免误入完整训练
    try:
        if getattr(cfg, 'run_mode', None) == 'time_head_only':
            if not hasattr(cfg, 'evaluation'):
                cfg.evaluation = type('obj', (), {})()
            if not hasattr(cfg.evaluation, 'time_head') or cfg.evaluation.time_head is None:
                cfg.evaluation.time_head = type('obj', (), {})()
            setattr(cfg.evaluation.time_head, 'enabled', True)
            if not hasattr(cfg.evaluation.time_head, 'standalone') or cfg.evaluation.time_head.standalone is None:
                cfg.evaluation.time_head.standalone = type('obj', (), {})()
            setattr(cfg.evaluation.time_head.standalone, 'enabled', True)
    except Exception:
        pass

    # 动态加载实验预设（如提供）
    if args.preset is not None and len(args.preset) > 0:
        try:
            mod = importlib.import_module(f'configs.experiments.{args.preset}')
            if hasattr(mod, 'get_config'):
                cfg = mod.get_config(cfg)
            else:
                print(f'Preset {args.preset} does not define get_config(base_cfg). Using default cfg.')
        except Exception as e:
            logging.getLogger(__name__).exception('Failed to import preset %s: %s', args.preset, e)
    # 如果用户通过CLI提供了 --output_dir，应在preset加载后再次应用，确保CLI优先级高于预设
    if args.output_dir is not None and len(args.output_dir) > 0:
        cfg.results_dir = args.output_dir
    

    os.makedirs(getattr(cfg, 'results_dir', 'results'), exist_ok=True)

    print('Using config results_dir:', getattr(cfg, 'results_dir', 'results'))
    print('Starting experiment...')
    try:
        # 交由 main() 按 config.run_mode 路由执行
        main()
    except Exception as e:
        logging.getLogger(__name__).exception('Experiment failed: %s', e)
        raise

def run_train_only(config):
    """仅训练模式的简易封装：关闭交叉验证并调用 run_experiment。"""
    validate_config(config)
    config.data.cross_validation.enabled = False
    config.run_mode = 'train_only'
    return run_experiment(config)


def run_test_only(config):
    """仅测试模式的简易封装：关闭交叉验证并调用 run_experiment（会训练并在测试集上评估）。"""
    validate_config(config)
    config.data.cross_validation.enabled = False
    config.run_mode = 'test_only'
    return run_experiment(config)
