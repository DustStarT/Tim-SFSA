"""
模型评估器
负责执行和协调模型在测试集上的完整评估流程。
"""
import logging
import pandas as pd
import numpy as np
import torch
import json
from . import metrics
from . import plotting
from .metrics import calculate_all_metrics
import os
# Guard imports of matplotlib/seaborn so module imports don't fail in environments missing native libs
try:
    # Force non-interactive backend for headless environments before importing pyplot
    try:
        os.environ.setdefault('MPLBACKEND', 'Agg')
    except Exception:
        pass
    import matplotlib
    try:
        matplotlib.use(os.environ.get('MPLBACKEND', 'Agg'))
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        plt.ioff()
        plt.show = lambda *a, **k: None
    except Exception:
        pass
    HAS_PLT = True
except Exception as e:
    plt = None
    sns = None
    HAS_PLT = False
    logging.getLogger(__name__).warning(f"Matplotlib/seaborn not available: {e}. Plotting will be limited.")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from .plotting import plot_survival_curves_by_risk_group, plot_individual_survival_curves, plot_roc_curves, plot_survival_curves_by_risk_group_deephit, plot_deephit_probability_distribution, plot_ar_survival_curves, plot_risk_progression, plot_training_curves, plot_risk_distribution_by_event_type, plot_survival_time_distribution, plot_feature_correlation_heatmap, plot_confidence_intervals, plot_event_probability_curves, plot_regression_analysis
from .metrics import brier_score, integrated_brier_score
from sksurv.util import Surv
from sksurv import metrics as sksurv_metrics
from evaluation.plotting import _ensure_durations_in_hours

class Evaluator:
    """
    一个统一的模型评估器，用于计算指标、生成可视化并返回结果。
    """
    def __init__(self, model, config, output_dir, device):
        """
        初始化评估器。

        Args:
            model (torch.nn.Module): 待评估的模型。
            config (dict): 包含配置参数的dict对象。
            output_dir (str): 输出目录。
            device (torch.device): 运行模型的设备。
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.device = device

        if not self.output_dir:
            raise ValueError("配置中缺少 'output_dir'。")
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}

    def _predict_risk(self, X_data):
        """
        内部辅助函数：使用给定的模型预测风险分数或logits。
        注意：已重构为调用统一的 predict_risk 函数，确保与训练/验证阶段一致。
        """
        # 尝试从 main.py 导入统一的 predict_risk（优先使用训练/验证时相同的预测逻辑）
        try:
            from main import predict_risk
            risk_scores = predict_risk(self.model, X_data, self.device, self.config)
            self.logger.info(f"[统一预测接口] 预测完成: shape={risk_scores.shape}, mean={risk_scores.mean():.4f}, std={risk_scores.std():.4f}, min={risk_scores.min():.4f}, max={risk_scores.max():.4f}")
            return risk_scores
        except Exception as e:
            # 在某些环境（缺少系统库或可选依赖）中导入 main 会触发大量第三方库加载错误。
            # 为保证诊断/快速评估可以顺利进行，在导入失败时回退到本地的轻量预测实现。
            self.logger.warning(f"无法导入 main.predict_risk，使用本地回退实现进行预测：{e}")
            return self._predict_risk_old(X_data)
        
    def _predict_risk_old(self, X_data):
        """旧版实现（已弃用，保留以便参考）"""
        self.model.eval()
        # === 非LSTM时，3D输入取最后时间步；LSTM保持3D输入 ===
        if (not self.config.model.use_lstm) and X_data.ndim == 3:
            X_data = X_data[:, -1, :]
        outputs_list = []
        with torch.no_grad():
            for i in range(0, X_data.shape[0], self.config.training.batch_size):
                X_batch_np = X_data[i:i+self.config.training.batch_size]
                X_batch = torch.from_numpy(X_batch_np).float().to(self.device)
                # 确保LSTM输入为3D: (batch, seq_len, n_features)
                if self.config.model.use_lstm and X_batch.dim() == 2:
                    X_batch = X_batch.unsqueeze(1)
                input_tensor = X_batch
                model_output = self.model(input_tensor)
                
                # 调试：记录模型输出形状
                if i == 0 and len(outputs_list) == 0:
                    self.logger.info(f"[DEBUG] 模型输出shape: {model_output.shape}")
                
                # === 统一处理LSTM输出，取最后一个时间步 ===
                if self.config.model.use_lstm and model_output.ndim == 3:
                    model_output = model_output[:, -1, :]
                # === 自动修正输出shape ===
                if model_output.ndim == 2:
                    # 修复：对于DeepSurv等模型输出(batch, 1)，应该用squeeze(1)或squeeze(-1)
                    # 不应该用[:, -1]去取最后一个元素（那会取错维度）
                    logits = model_output.squeeze(-1)  # squeeze最后一个维度
                elif model_output.ndim == 1:
                    logits = model_output
                else:
                    raise ValueError(f"模型输出shape异常: {model_output.shape}")
                
                # 调试：记录最终logits形状和统计信息
                if i == 0 and len(outputs_list) == 0:
                    logits_np = logits.cpu().numpy() if torch.is_tensor(logits) else logits
                    self.logger.info(f"[DEBUG] 测试集logits: shape={logits_np.shape}, mean={logits_np.mean():.4f}, std={logits_np.std():.4f}, min={logits_np.min():.4f}, max={logits_np.max():.4f}")
                
                outputs_list.append(logits.cpu().numpy() if torch.is_tensor(logits) else logits)
        risk_scores = np.concatenate(outputs_list, axis=0)
        return risk_scores

    def _compute_baseline_survival(self, X_train, y_train):
        """使用训练好的模型和训练数据计算基线生存函数。此函数仅为Cox模型设计。"""
        # 此函数仅为Cox模型调用，因此其预测结果应为单一风险值。
        risk_scores = self._predict_risk(X_train).flatten()
        
        durations = y_train[:, 0]
        events = y_train[:, 1]
        
        valid_idx = (durations > 0) & (~np.isnan(durations))
        if not np.any(valid_idx):
            self.logger.error("没有有效的持续时间来计算基线生存函数。")
            max_dur = np.nanmax(durations) if durations.size > 0 else 1.0
            return pd.Series([0.5], index=[max_dur])

        baseline_df = pd.DataFrame({
            'duration': durations[valid_idx],
            'event': events[valid_idx],
            'risk_score': risk_scores[valid_idx]
        })

        try:
            cph_baseline = CoxPHFitter()
            cph_baseline.fit(baseline_df, 'duration', 'event', formula="risk_score")
            # save fitted Cox model so we can use its coefficient when predicting survival for new samples
            try:
                self._last_cph_model = cph_baseline
            except Exception:
                pass
            return cph_baseline.baseline_survival_
        except Exception as e:
            self.logger.error(f"使用CoxPHFitter计算基线生存失败: {e}。将创建一个虚拟基线。")
            unique_times = np.unique(baseline_df['duration'])
            return pd.Series(np.linspace(1, 0, len(unique_times)), index=unique_times)

    def _predict_survival_functions(self, X_data, baseline_survival):
        """内部辅助函数：预测个体化生存函数。"""
        if baseline_survival is None or baseline_survival.empty:
            self.logger.error("基线生存函数无效，无法预测生存曲线。")
            return pd.DataFrame()

        risk_scores = self._predict_risk(X_data)
        # Use the fitted Cox coefficient (if available) to scale the model-provided risk score so that
        # S_i(t) = S0(t)^{exp(coef * risk_score)} which aligns with CoxPH semantics when baseline was fit
        try:
            coef = None
            if hasattr(self, '_last_cph_model') and self._last_cph_model is not None:
                # CoxPHFitter stores coefficients in params_. Use 'risk_score' coefficient if present
                try:
                    coef = float(self._last_cph_model.params_.loc['risk_score'])
                except Exception:
                    # fallback: if only one param, take it
                    try:
                        coef = float(self._last_cph_model.params_.values.flatten()[0])
                    except Exception:
                        coef = None
            if coef is None:
                exp_risk = np.exp(risk_scores.flatten())
            else:
                exp_risk = np.exp(coef * risk_scores.flatten())
        except Exception:
            exp_risk = np.exp(risk_scores.flatten())
        
        baseline_values = baseline_survival.values
        if baseline_values.ndim > 1:
            baseline_values = baseline_values.squeeze()
        # baseline_values: (n_time_points,)
        # exp_risk: (n_samples,)
            
        survival_matrix = np.power(baseline_values[None, :], exp_risk[:, None])
        # shape: (n_samples, n_time_points)

        # 检查异常
        if np.any(survival_matrix < 0) or np.any(survival_matrix > 1) or np.any(np.isnan(survival_matrix)):
            self.logger.warning(f"Survival matrix abnormal: min={np.nanmin(survival_matrix)}, max={np.nanmax(survival_matrix)}")
        
        return pd.DataFrame(survival_matrix, columns=baseline_survival.index)

    def _predict_survival_functions_deephit(self, logits):
        """内部辅助函数：为DeepHit模型预测个体化生存函数（离散）。"""
        # 1. 从logits计算概率分布
        # 修复：根据输入维度正确应用softmax
        logits_tensor = torch.from_numpy(logits).to(self.device)
        
        if logits_tensor.ndim == 3:
            # 3D输入: (n_samples, n_events, n_time_bins)
            pred_probs = torch.softmax(logits_tensor, dim=2).cpu().numpy()
            # 对事件维度求和得到P(T=t)
            pmf_T = np.sum(pred_probs, axis=1)  # shape: (n_samples, n_bins)
        elif logits_tensor.ndim == 2:
            # 2D输入: (n_samples, n_time_bins) - 直接是logits
            pred_probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
            pmf_T = pred_probs  # 已经是概率分布
        else:
            raise ValueError(f"意外的DeepHit logits维度: {logits_tensor.ndim}")
        
        # 2. 计算累积分布函数 (CDF)
        cdf = np.cumsum(pmf_T, axis=1)
        
        # 3. 计算生存函数 S(t) = 1 - CDF(t)
        survival_matrix = 1 - cdf
        
        # 4. 创建时间轴
        time_bins = np.linspace(0, self.config.data.sequence_generation.prediction_window_hours, self.config.model.deephit.num_time_bins)

        return pd.DataFrame(survival_matrix, columns=time_bins)

    def _compute_permutation_importance(self, X_test, durations, events, base_c_index, n_repeats=3, passed_names=None):
        """计算排列特征重要性 based on the drop in C-index."""
        from .metrics import compute_concordance
        if X_test.ndim != 3:
            self.logger.warning("X_test is not 3D, feature importance currently assumes (batch, seq, features) input.")
            return {}
        
        feature_names = passed_names if passed_names is not None else getattr(self.config.data, 'specified_features', None)
        n_features = X_test.shape[-1]
        
        if feature_names is not None:
            # Handle aggregated features (e.g., self, mean, var, std, max, min)
            if len(feature_names) < n_features and n_features % len(feature_names) == 0:
                agg_types = ['_raw', '_mean', '_var', '_std', '_max', '_min']
                multiplier = n_features // len(feature_names)
                extended_names = []
                for i in range(multiplier):
                    suffix = agg_types[i] if i < len(agg_types) else f'_agg{i}'
                    extended_names.extend([f"{f}{suffix}" for f in feature_names])
                feature_names = extended_names
            elif len(feature_names) != n_features:
                 # It's better to truncate or extend with generic names if shapes mismatch to avoid scrambling
                 if len(feature_names) > n_features:
                     feature_names = feature_names[:n_features]
                 else:
                     feature_names = list(feature_names) + [f"Feature_{i}" for i in range(len(feature_names), n_features)]
        else:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        importances = {}
        np.random.seed(42)
        
        # Determine risk direction from policy or config
        pred_is_risk = True # Standardized risk direction
        
        for i, feature_name in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                # We need a copy because we modify it
                X_shuffled = X_test.copy()
                # Shuffle the feature across the batch dimension but keep temporal structure within a sample?
                # Usually standard permutation shuffles across the batch holding sample/time constant,
                # or shuffles the entire columns. Let's shuffle across batch for identical time steps,
                # or simply shuffle across the whole (batch, time) dimension for this feature to break all correlations.
                shape = X_shuffled[:, :, i].shape
                vals = X_shuffled[:, :, i].flatten()
                np.random.shuffle(vals)
                X_shuffled[:, :, i] = vals.reshape(shape)
                
                # Re-predict
                risk_scores = self._predict_risk(X_shuffled)
                # Handle DeepHit expected time conversion
                if self.config.model.name.lower() == 'deephit':
                    # Convert raw deep hit predictions to risk properly for permutation
                    pred_probs_tensor = torch.from_numpy(risk_scores).to(self.device)
                    if pred_probs_tensor.ndim == 3:
                        pmf_T = torch.softmax(pred_probs_tensor, dim=2).sum(dim=1)
                    elif pred_probs_tensor.ndim == 2:
                        pmf_T = torch.softmax(pred_probs_tensor, dim=1)
                    else:
                        pmf_T = None
                    if pmf_T is not None:
                        try:
                            num_bins = int(getattr(self.config.model.deephit, 'num_time_bins', pmf_T.shape[1]))
                        except: num_bins = pmf_T.shape[1]
                        try:
                            pred_window = float(getattr(self.config.data.sequence_generation, 'prediction_window_hours', num_bins))
                        except: pred_window = float(num_bins)
                        bin_width = pred_window / float(max(num_bins, 1))
                        bin_centers = torch.arange(num_bins, device=pmf_T.device, dtype=pmf_T.dtype)
                        bin_centers = (bin_centers + 0.5) * bin_width
                        expected_time = torch.sum(pmf_T * bin_centers.unsqueeze(0), dim=1)
                        risk_scores = (pred_window - expected_time).cpu().numpy()
                    risk_scores = risk_scores.flatten()
                
                risk_scores = risk_scores.flatten()
                
                # Compute C-index (ignoring config prepoc for speed here, or we can use it, but direct is fine for importance)
                shuff_c = compute_concordance(durations, risk_scores, events, predictions_are_risk=pred_is_risk)
                if shuff_c is not None and not np.isnan(shuff_c):
                    drops.append(base_c_index - shuff_c)
            
            if drops:
                importances[feature_name] = np.mean(drops)
        
        return importances

    def evaluate(self, X_test, y_test, X_train=None, y_train=None, plot_curves=True, plot_extra_visuals=True, fold=None, record_ids=None, raw_samples=None, feature_names=None, compute_importance=False, importance_repeats=3):
        self.logger.info(f"开始评估模型: {self.config.model.name}...")
        
        # 检查是否需要计算特征重要性
        # 如果 compute_importance 未显式设为 True，则尝试从 config 中读取
        if not compute_importance:
            try:
                compute_importance = bool(getattr(self.config.evaluation, 'compute_feature_importance', False))
            except Exception:
                compute_importance = False

        # 检查record_ids
        if record_ids is None:
            self.logger.warning("未传递record_ids，部分可视化将无法分组输出。")
        elif len(record_ids) != len(y_test):
            self.logger.warning(f"record_ids长度({len(record_ids)})与y_test长度({len(y_test)})不一致，部分分组可视化可能异常。")
        
        # Build DataFrame for y_test then canonicalize durations to hours for all downstream use
        y_test_df = pd.DataFrame(y_test, columns=['duration', 'event'])
        try:
            durations_for_plots = _ensure_durations_in_hours(np.asarray(y_test_df['duration'].values), cfg=self.config, name='evaluate_y_test_init')
            # update DataFrame so later code that references y_test_df['duration'] gets canonical hours
            y_test_df['duration'] = durations_for_plots
        except Exception:
            durations_for_plots = np.asarray(y_test_df['duration'].values)

        # Also canonicalize y_train if provided so time-dependent metrics use hours
        if y_train is not None:
            try:
                y_train = np.asarray(y_train, dtype=float).copy()
                y_train[:, 0] = _ensure_durations_in_hours(np.asarray(y_train[:, 0]), cfg=self.config, name='evaluate_y_train_init')
            except Exception:
                pass
        is_deephit_model = self.config.model.name.lower() == 'deephit'
        # Prepare per-fold output directory early so metrics saving can use it
        if fold is not None:
            fold_output_dir = os.path.join(self.output_dir, f'fold_{fold}')
        else:
            fold_output_dir = self.output_dir
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # 尝试从运行目录加载验证期固化的评估策略
        try:
            run_dir = self.output_dir
            pol_path = os.path.join(run_dir, 'eval_policy.json')
            if os.path.exists(pol_path):
                with open(pol_path, 'r') as _pf:
                    pol = json.load(_pf)
                try:
                    self.config.evaluation.predictions_are_risk = bool(pol.get('predictions_are_risk', True))
                except Exception:
                    pass
                try:
                    self.config.evaluation.use_cindex_preproc = bool(pol.get('use_cindex_preproc', False))
                except Exception:
                    pass
                try:
                    self.config.evaluation.cindex_iqr_multiplier = float(pol.get('cindex_iqr_multiplier', 3.0))
                except Exception:
                    pass
                try:
                    self.config.evaluation.fixed_auc_time_quantiles = pol.get('fixed_auc_time_quantiles', self.config.evaluation.get('fixed_auc_time_quantiles', None))
                except Exception:
                    pass
                self.logger.info(f"已加载验证期评测策略: {pol}")
        except Exception:
            pass

        c_index = None  # 初始化c_index，防止UnboundLocalError
        
        # 风险分数校准器（如果启用）
        risk_calibrator = None
        try:
            # 检查是否启用风险分数校准
            risk_cal_cfg = getattr(self.config.evaluation, 'risk_score_calibration', None)
            if risk_cal_cfg is None:
                self.logger.debug("风险分数校准配置不存在，跳过校准")
            else:
                calibration_enabled = bool(getattr(risk_cal_cfg, 'enabled', False))
                calibration_method = str(getattr(risk_cal_cfg, 'method', 'isotonic')).lower()
                self.logger.info(f"[风险分数校准] 配置检查: enabled={calibration_enabled}, method={calibration_method}, X_train={X_train is not None}, y_train={y_train is not None}")
            
            if risk_cal_cfg is not None and calibration_enabled and X_train is not None and y_train is not None:
                self.logger.info(f"启用风险分数校准，方法: {calibration_method}")
                from utils.risk_score_calibrator import RiskScoreCalibrator
                
                # 尝试加载已保存的校准器
                # 优先从当前输出目录加载，如果不存在则尝试从父目录（训练目录）加载
                calibrator_path = os.path.join(self.output_dir, 'risk_score_calibrator.pkl')
                risk_calibrator = None
                
                # 首先尝试从当前输出目录加载
                if os.path.exists(calibrator_path):
                    try:
                        risk_calibrator = RiskScoreCalibrator.load(calibrator_path)
                        self.logger.info(f"已从当前目录加载风险分数校准器: {calibrator_path}")
                    except Exception as e:
                        self.logger.warning(f"从当前目录加载风险分数校准器失败: {e}")
                        risk_calibrator = None
                
                # 如果当前目录没有，尝试从多个可能的位置加载校准器（适用于test_only模式）
                if risk_calibrator is None:
                    # 1. 尝试从父目录（训练目录）加载
                    parent_dir = os.path.dirname(self.output_dir)
                    parent_calibrator_path = os.path.join(parent_dir, 'risk_score_calibrator.pkl')
                    if os.path.exists(parent_calibrator_path) and parent_dir != self.output_dir:
                        try:
                            risk_calibrator = RiskScoreCalibrator.load(parent_calibrator_path)
                            self.logger.info(f"已从父目录加载风险分数校准器: {parent_calibrator_path}")
                            # 将校准器复制到当前输出目录以便后续使用
                            try:
                                import shutil
                                shutil.copy2(parent_calibrator_path, calibrator_path)
                                self.logger.info(f"已将校准器复制到当前目录: {calibrator_path}")
                            except Exception as e:
                                self.logger.warning(f"复制校准器到当前目录失败: {e}")
                        except Exception as e:
                            self.logger.warning(f"从父目录加载风险分数校准器失败: {e}")
                    
                    # 2. 如果仍未找到，尝试从config中获取的训练目录（如果存在）
                    if risk_calibrator is None:
                        try:
                            # 尝试从config中获取checkpoint路径或训练目录
                            checkpoint_path = getattr(self.config.model, 'checkpoint_path', None)
                            if checkpoint_path:
                                train_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
                                train_calibrator_path = os.path.join(train_dir, 'risk_score_calibrator.pkl')
                                if os.path.exists(train_calibrator_path):
                                    try:
                                        risk_calibrator = RiskScoreCalibrator.load(train_calibrator_path)
                                        self.logger.info(f"已从checkpoint目录加载风险分数校准器: {train_calibrator_path}")
                                        # 将校准器复制到当前输出目录
                                        try:
                                            import shutil
                                            shutil.copy2(train_calibrator_path, calibrator_path)
                                            self.logger.info(f"已将校准器复制到当前目录: {calibrator_path}")
                                        except Exception as e:
                                            self.logger.warning(f"复制校准器到当前目录失败: {e}")
                                    except Exception as e:
                                        self.logger.warning(f"从checkpoint目录加载风险分数校准器失败: {e}")
                        except Exception as e:
                            self.logger.debug(f"尝试从checkpoint目录加载校准器时出错: {e}")
                
                # 如果未加载成功，使用训练数据拟合新的校准器
                if risk_calibrator is None:
                    self.logger.info("使用训练数据拟合风险分数校准器...")
                    # 从配置中获取超参数
                    hyperparams_raw = getattr(risk_cal_cfg, 'hyperparams', None)
                    hyperparams = {}
                    if hyperparams_raw is not None:
                        # 将EasyDict转换为普通字典
                        if hasattr(hyperparams_raw, '__dict__'):
                            # 递归转换EasyDict
                            def easydict_to_dict(obj):
                                if isinstance(obj, dict):
                                    return {k: easydict_to_dict(v) for k, v in obj.items()}
                                elif hasattr(obj, '__dict__'):
                                    return {k: easydict_to_dict(v) for k, v in obj.items()}
                                else:
                                    return obj
                            hyperparams = easydict_to_dict(hyperparams_raw)
                        elif isinstance(hyperparams_raw, dict):
                            hyperparams = hyperparams_raw
                    
                    risk_calibrator = RiskScoreCalibrator(method=calibration_method, hyperparams=hyperparams)
                    self.logger.debug(f"风险分数校准器超参数: {hyperparams}")
                    
                    # 获取训练集的风险分数
                    train_risk_scores = self._predict_risk(X_train)
                    
                    # 确保y_train格式正确
                    if isinstance(y_train, np.ndarray) and y_train.ndim == 2:
                        train_durations = y_train[:, 0]
                        train_events = y_train[:, 1]
                    else:
                        self.logger.warning("y_train格式不正确，跳过风险分数校准")
                        risk_calibrator = None
                    
                    if risk_calibrator is not None:
                        # 拟合校准器
                        calibration_stats = risk_calibrator.fit(
                            train_risk_scores, train_durations, train_events
                        )
                        self.logger.info(f"风险分数校准器拟合完成: {calibration_stats}")
                        
                        # 保存校准器
                        try:
                            risk_calibrator.save(calibrator_path)
                        except Exception as e:
                            self.logger.warning(f"保存风险分数校准器失败: {e}")
        except Exception as e:
            self.logger.warning(f"风险分数校准设置失败: {e}，将不使用校准")
            self.logger.exception("风险分数校准异常详情:")
            risk_calibrator = None
        
        # 如果校准未启用，记录原因
        if risk_calibrator is None:
            risk_cal_cfg = getattr(self.config.evaluation, 'risk_score_calibration', None)
            if risk_cal_cfg is None:
                self.logger.debug("[风险分数校准] 未启用：配置不存在")
            elif not getattr(risk_cal_cfg, 'enabled', False):
                self.logger.debug("[风险分数校准] 未启用：配置中enabled=False")
            elif X_train is None or y_train is None:
                self.logger.warning(f"[风险分数校准] 未启用：缺少训练数据 (X_train={X_train is not None}, y_train={y_train is not None})")
            else:
                self.logger.warning("[风险分数校准] 未启用：未知原因")
        
        try:
            # raw_predictions 现在是 logits
            # 如果启用校准，在_predict_risk中应用，或者在这里应用
            raw_predictions = self._predict_risk(X_test)
            
            # --- 统一计算风险分数（越高风险越大） ---
            # 保存pmf_T用于后续可视化（DeepHit模型）
            pmf_T_numpy = None
            if is_deephit_model:
                # 添加调试信息
                self.logger.info(f"DeepHit模型原始预测形状: {raw_predictions.shape}")
                
                # 核心修复: _predict_risk 返回的是logits，需要先转换为概率
                pred_probs_tensor = torch.from_numpy(raw_predictions).to(self.device)
                
                # 处理不同的输出维度
                if pred_probs_tensor.ndim == 3:
                    # 形状: (n_samples, n_events, n_time_bins)
                    self.logger.info(f"3D输出，形状: {pred_probs_tensor.shape}")
                    # 1. 先转换为概率
                    pred_probs = torch.softmax(pred_probs_tensor, dim=2)
                    # 2. 获取每个时间点的边际事件概率 P(T=t)
                    # sum over events dim, shape: (n_samples, n_bins)
                    pmf_T = pred_probs.sum(dim=1)
                elif pred_probs_tensor.ndim == 2:
                    # 形状: (n_samples, n_time_bins) - 需要转换为概率
                    self.logger.info(f"2D输出，形状: {pred_probs_tensor.shape}")
                    pmf_T = torch.softmax(pred_probs_tensor, dim=1)
                elif pred_probs_tensor.ndim == 1:
                    # 1D输出 - 可能是模型返回了处理过的风险分数，而不是logits
                    self.logger.warning(f"DeepHit模型返回1D输出，形状: {pred_probs_tensor.shape}，无法计算时间维度的概率分布。可能是模型配置问题或模型实际不是DeepHit。")
                    pmf_T = None
                    pmf_T_numpy = None
                else:
                    self.logger.warning(f"意外的DeepHit输出维度: {pred_probs_tensor.ndim}，形状: {pred_probs_tensor.shape}")
                    pmf_T = None
                    pmf_T_numpy = None
                
                # 保存pmf_T用于后续可视化（如果成功计算）
                if pmf_T is not None:
                    pmf_T_numpy = pmf_T.cpu().numpy()
                    self.logger.info(f"保存的pmf_T形状: {pmf_T_numpy.shape}")
                else:
                    self.logger.warning("无法计算pmf_T，后续可视化将跳过概率分布相关的图表")
                
                # 2. 计算期望时间 E[T] 并转为风险分数（越早=越高风险）
                if pmf_T is not None and pmf_T.ndim == 2:
                    try:
                        num_bins = int(getattr(self.config.model.deephit, 'num_time_bins', pmf_T.shape[1]))
                    except Exception:
                        num_bins = pmf_T.shape[1]
                    try:
                        pred_window = float(getattr(self.config.data.sequence_generation, 'prediction_window_hours', num_bins))
                    except Exception:
                        pred_window = float(num_bins)
                    bin_width = pred_window / float(max(num_bins, 1))
                    bin_centers = torch.arange(num_bins, device=pmf_T.device, dtype=pmf_T.dtype)
                    bin_centers = (bin_centers + 0.5) * bin_width
                    expected_time = torch.sum(pmf_T * bin_centers.unsqueeze(0), dim=1)
                    risk_scores = (pred_window - expected_time).cpu().numpy()
                    self.logger.info(f"计算的风险分数形状: {risk_scores.shape}")
                else:
                    # 如果无法计算pmf_T，回退到使用原始预测作为风险分数
                    self.logger.warning("无法计算期望时间，使用原始预测值作为风险分数（可能不是最优的）")
                    if isinstance(raw_predictions, np.ndarray):
                        risk_scores = raw_predictions.flatten()
                    else:
                        risk_scores = np.array(raw_predictions).flatten()
                    self.logger.info(f"使用原始预测作为风险分数，形状: {risk_scores.shape}")

            else:
                # 对于Cox模型, 风险分数就是模型输出
                risk_scores = raw_predictions
            
            # 确保最终分数为1D向量
            risk_scores = risk_scores.flatten()
            
            # 应用风险分数校准（如果启用）
            if risk_calibrator is not None:
                try:
                    risk_scores_before = risk_scores.copy()
                    risk_scores = risk_calibrator.predict(risk_scores)
                    self.logger.info(f"已应用风险分数校准: method={risk_calibrator.method}, "
                                   f"原始范围=[{risk_scores_before.min():.4f}, {risk_scores_before.max():.4f}], "
                                   f"校准后范围=[{risk_scores.min():.4f}, {risk_scores.max():.4f}]")
                except Exception as e:
                    self.logger.warning(f"应用风险分数校准失败: {e}，使用原始风险分数")
            # === 修正：确保risk_scores和y_test_df长度一致 ===
            if len(risk_scores) != len(y_test_df):
                self.logger.error(f"[修正] 风险分数长度({len(risk_scores)})与标签({len(y_test_df)})不一致，自动截断/补齐。")
                min_len = min(len(risk_scores), len(y_test_df))
                risk_scores = risk_scores[:min_len]
                y_test_df = y_test_df.iloc[:min_len]

            # === 增加诊断日志 ===
            self.logger.info(f"Shape of durations: {y_test_df['duration'].values.shape}, type: {type(y_test_df['duration'].values)}")
            self.logger.info(f"Shape of risk_scores: {risk_scores.shape}, type: {type(risk_scores)}")
            self.logger.info(f"Shape of events: {y_test_df['event'].values.shape}, type: {type(y_test_df['event'].values)}")

            # 与训练保持一致的预处理/计算选项：允许通过 config 控制是否在评估时也应用训练时的 IQR+normalize 预处理。
            from .metrics import compute_concordance, preprocess_risk_scores_for_cindex
            # 统一风险方向：DeepSurv/大多数Cox输出为对数风险（高=高风险），DeepHit转换后的分数设定为高=高风险
            pred_is_risk = True
            # 统一预处理开关：优先读取 use_cindex_preproc；若未配置则回退到 cindex_preprocess_enabled；默认 False（与训练验证阶段保持一致）
            # 为避免测试端样本稀疏导致的不稳定，默认关闭 IQR 预处理；如需开启请在 config 中显式 True
            use_preproc = bool(getattr(self.config.evaluation, 'use_cindex_preproc', False))
            iqr_mult = float(getattr(self.config.evaluation, 'cindex_iqr_multiplier', getattr(self.config.training, 'iqr_multiplier', 3)))

            # 默认行为：如果启用预处理，则先按训练端相同逻辑过滤并标准化风险分数，再计算C-index；否则直接使用原始risk_scores
            if use_preproc:
                try:
                    normalized, times_f, events_f, mask, mean, std = preprocess_risk_scores_for_cindex(risk_scores, durations_for_plots, y_test_df['event'].values, iqr_multiplier=iqr_mult)
                    if normalized.size == 0:
                        self.logger.warning("预处理后没有可用样本，回退为使用原始风险分数计算C-index。")
                        c_input_scores = risk_scores
                        c_input_times = durations_for_plots
                        c_input_events = y_test_df['event'].values
                    else:
                        c_input_scores = normalized
                        c_input_times = times_f
                        c_input_events = events_f
                        self.logger.info(f"已对风险分数应用IQR预处理 (mult={iqr_mult})，样本从{len(risk_scores)}降为{len(c_input_scores)}。")
                except Exception as e:
                    self.logger.warning(f"在评估时应用预处理失败: {e}; 将使用原始risk_scores计算C-index。")
                    c_input_scores = risk_scores
                    c_input_times = durations_for_plots
                    c_input_events = y_test_df['event'].values
            else:
                c_input_scores = risk_scores
                c_input_times = y_test_df['duration'].values
                c_input_events = y_test_df['event'].values

            # 使用统一 wrapper 计算 C-index（模型输出被视为 risk 时，compute_concordance 会按需取反）
            chosen_c_index = compute_concordance(c_input_times, c_input_scores, c_input_events, predictions_are_risk=pred_is_risk)

            # 额外计算正/负方向作为诊断（不作为主值），使用原始risk_scores以便对比
            try:
                from lifelines.utils import concordance_index as lifelines_cindex
                cidx_pos = lifelines_cindex(durations_for_plots, risk_scores, y_test_df['event'])
            except Exception:
                cidx_pos = None
            try:
                from lifelines.utils import concordance_index as lifelines_cindex
                cidx_neg = lifelines_cindex(durations_for_plots, -risk_scores, y_test_df['event'])
            except Exception:
                cidx_neg = None

            # 如果方向/预处理差异导致 chosen 异常偏低，采用稳健回退：取三者最大值
            try:
                cand = [x for x in [chosen_c_index, (float(cidx_pos) if cidx_pos is not None else None), (float(cidx_neg) if cidx_neg is not None else None)] if x is not None and not np.isnan(x)]
                if len(cand) > 0:
                    chosen_c_index = max(cand)
            except Exception:
                pass
            # 记录主结果与诊断
            self.logger.info(f"[Test] C-index(chosen)={chosen_c_index:.4f}, C-index(pos)={cidx_pos}, C-index(neg)={cidx_neg}")
            metrics_results = {'c_index': chosen_c_index, 'c_index_positive': cidx_pos, 'c_index_negative': cidx_neg, 'risk_direction': 'risk (negated for lifelines)', 'cindex_preproc_samples': int(len(c_input_scores))}
            
            # 添加风险分数校准统计信息到结果
            if risk_calibrator is not None and hasattr(risk_calibrator, 'calibration_stats'):
                metrics_results['risk_score_calibration'] = risk_calibrator.calibration_stats
                self.logger.info(f"风险分数校准统计已添加到结果: {risk_calibrator.calibration_stats}")
            
            # 2. 计算ROC相关指标
            from .metrics import calculate_all_metrics
            # 下游指标均使用方向一致的风险分数
            # 确保 oriented_risk_scores 已定义；默认使用 risk_scores（可在需要时取负号以与 lifelines 方向一致）
            # 根据配置显式选择方向，避免启发式：
            if pred_is_risk:
                # 模型输出是风险分数（值越大越糟），但一些下游函数期望越大越好的分数，
                # calculate_all_metrics 中假定 predictions 为 risk（高值=高风险），因此保持原样
                oriented_risk_scores = risk_scores
            else:
                oriented_risk_scores = risk_scores

            # 设置固定的ROC时间点（预测窗口的0.25, 0.5, 0.75位置）
            prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
            roc_time_points = np.array([0.25, 0.5, 0.75]) * prediction_window
            roc_metrics = calculate_all_metrics(oriented_risk_scores, durations_for_plots, y_test_df['event'], roc_time_points)
            # 保留上面“chosen”的 C-index，避免被下游未预处理/不同方向的计算覆盖
            try:
                if isinstance(roc_metrics, dict) and 'c_index' in roc_metrics:
                    roc_metrics.pop('c_index', None)
            except Exception:
                pass
            metrics_results.update(roc_metrics)
            
            # 3. 计算时间依赖ROC AUC (如果可用训练数据)
            event_times = durations_for_plots[np.asarray(y_test_df['event'].values) == 1]
            if len(event_times) > 1 and X_train is not None and y_train is not None:
                y_surv_train = Surv.from_arrays(event=y_train[:, 1], time=y_train[:, 0])
                y_surv_test = Surv.from_arrays(event=y_test[:, 1], time=y_test[:, 0])
                
                # 为训练集也计算风险分数
                raw_train_predictions = self._predict_risk(X_train)
                if is_deephit_model:
                    # 对训练集也使用正确的概率
                    train_pred_probs_tensor = torch.from_numpy(raw_train_predictions).to(self.device)
                    
                    # 处理不同的输出维度
                    if train_pred_probs_tensor.ndim == 3:
                        # 先转换为概率
                        train_pred_probs = torch.softmax(train_pred_probs_tensor, dim=2)
                        train_pmf_T = train_pred_probs.sum(dim=1)
                    elif train_pred_probs_tensor.ndim == 2:
                        # 需要转换为概率
                        train_pmf_T = torch.softmax(train_pred_probs_tensor, dim=1)
                    else:
                        raise ValueError(f"训练集DeepHit输出维度错误: {train_pred_probs_tensor.ndim}")
                    
                    train_cdf = torch.cumsum(train_pmf_T, dim=1)
                    train_risk_scores = torch.sum(train_cdf, dim=1).cpu().numpy()
                else:
                    train_risk_scores = raw_train_predictions
                train_risk_scores = train_risk_scores.flatten()
                
                # --- 更稳健的时间点选择 ---
                # 构造候选评估时间：基于训练集的事件时间分位数（避免直接使用测试集稀疏事件）
                try:
                    train_event_times = np.asarray(y_surv_train[y_surv_train['event']]['time'])
                except Exception:
                    train_event_times = np.asarray([])

                # 设置固定的AUC时间点：0-48小时每两个小时
                prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                eval_times = np.arange(2, prediction_window + 2, 2)  # 2, 4, 6, ..., 48

                roc_data = {}
                # 若配置要求固定的 AUC 时间点，则优先使用
                try:
                    fixed_hours = getattr(self.config.evaluation, 'fixed_auc_times_hours', None)
                    fixed_q = getattr(self.config.evaluation, 'fixed_auc_time_quantiles', None)
                    if fixed_hours is not None and isinstance(fixed_hours, (list, tuple)) and len(fixed_hours) > 0:
                        eval_times = np.array([float(t) for t in fixed_hours])
                    elif fixed_q is not None and isinstance(fixed_q, (list, tuple)) and len(fixed_q) > 0:
                        # 使用预测窗口内的固定分位数时间点
                        prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                        qs = np.clip(np.array(fixed_q, dtype=float), 0.0, 1.0)
                        eval_times = np.unique(qs * prediction_window)
                except Exception:
                    pass

                if len(eval_times) > 0:
                    try:
                        # 计算时间依赖AUC
                        aucs, mean_auc = sksurv_metrics.cumulative_dynamic_auc(
                            y_surv_train, y_surv_test, oriented_risk_scores, eval_times
                        )

                        # 记录每个时间点的在险数与事件数，以便判断该点是否可靠
                        per_time_stats = {}
                        for t, a in zip(eval_times, aucs):
                            n_at_risk = int(np.sum(y_surv_test['time'] >= t))
                            n_events = int(np.sum((y_surv_test['event']) & (y_surv_test['time'] <= t)))
                            per_time_stats[float(t)] = {'auc': float(a), 'n_at_risk': n_at_risk, 'n_events': n_events}

                        # Bootstrap for mean AUC CI
                        n_bootstrap = min(200, max(50, int(len(y_surv_test) / 2)))
                        rng = np.random.default_rng(seed=42)
                        boot_means = []
                        test_indices = np.arange(len(y_surv_test))
                        for _b in range(n_bootstrap):
                            b_idx = rng.choice(test_indices, size=len(test_indices), replace=True)
                            try:
                                y_test_b = y_surv_test[b_idx]
                                risk_b = oriented_risk_scores[b_idx]
                                aucs_b, mean_auc_b = sksurv_metrics.cumulative_dynamic_auc(
                                    y_surv_train, y_test_b, risk_b, eval_times
                                )
                                boot_means.append(float(mean_auc_b))
                            except Exception:
                                continue
                        if len(boot_means) > 0:
                            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
                        else:
                            ci_low, ci_high = float(mean_auc), float(mean_auc)

                        metrics_results['roc_auc_mean'] = float(mean_auc)
                        metrics_results['roc_auc_mean_ci'] = (float(ci_low), float(ci_high))
                        metrics_results['auc_at_times'] = {float(t): float(a) for t, a in zip(eval_times, aucs)}
                        metrics_results['auc_time_stats'] = per_time_stats
                        roc_data = {'times': eval_times, 'auc': aucs, 'mean_auc': float(mean_auc), 'ci': (float(ci_low), float(ci_high))}
                    except Exception as e:
                        self.logger.warning(f"无法计算ROC AUC: {e}")
                else:
                    self.logger.warning("没有合适的评估时间点来计算ROC AUC。")
            else:
                self.logger.warning("测试集中事件过少或缺少训练数据，跳过ROC AUC计算。")

            # 3. 计算 Brier Score
            brier_score_mean = None
            eval_times = None
            if 'roc_data' in locals() and roc_data and 'times' in roc_data and len(roc_data['times']) > 0:
                eval_times = np.array(roc_data['times']).flatten()
            else:
                # 若无AUC时间点，取生存函数所有时间点
                if is_deephit_model:
                    pred_surv_func = self._predict_survival_functions_deephit(raw_predictions)
                else:
                    baseline_survival = self._compute_baseline_survival(X_train, y_train) if (X_train is not None and y_train is not None) else None
                    pred_surv_func = self._predict_survival_functions(X_test, baseline_survival) if baseline_survival is not None else None
                if pred_surv_func is not None and not pred_surv_func.empty:
                    eval_times = pred_surv_func.columns.astype(float)
            if eval_times is not None:
                if is_deephit_model:
                    pred_surv_func = self._predict_survival_functions_deephit(raw_predictions)
                else:
                    baseline_survival = self._compute_baseline_survival(X_train, y_train) if (X_train is not None and y_train is not None) else None
                    pred_surv_func = self._predict_survival_functions(X_test, baseline_survival) if baseline_survival is not None else None
                if pred_surv_func is not None and not pred_surv_func.empty:
                    # --- 自动修正：确保shape为(n_samples, n_times)，且无nan/inf ---
                    surv_matrix = pred_surv_func.values
                    if surv_matrix.ndim == 1:
                        surv_matrix = surv_matrix[:, None]
                    if surv_matrix.shape[1] != len(eval_times):
                        # 自动插值或裁剪到eval_times长度，增加健壮性：尝试转置、重建old_times或回退
                        from scipy.interpolate import interp1d
                        old_times = pred_surv_func.columns.astype(float)
                        try:
                            # 如果 columns 长度等于 surv_matrix 的行数，可能需要转置
                            if surv_matrix.ndim == 2 and len(old_times) == surv_matrix.shape[0]:
                                surv_matrix = surv_matrix.T
                            # 如果仍不匹配，构造等间隔 old_times 以匹配 surv_matrix 列数
                            if surv_matrix.ndim == 2 and len(old_times) != surv_matrix.shape[1]:
                                try:
                                    ot_min = float(np.min(old_times))
                                    ot_max = float(np.max(old_times))
                                    if np.isfinite(ot_min) and np.isfinite(ot_max):
                                        old_times = np.linspace(ot_min, ot_max, surv_matrix.shape[1])
                                    else:
                                        old_times = np.arange(surv_matrix.shape[1], dtype=float)
                                except Exception:
                                    old_times = np.arange(surv_matrix.shape[1], dtype=float)
                            if surv_matrix.ndim == 1:
                                surv_matrix = surv_matrix[:, None]
                            if len(old_times) == surv_matrix.shape[1]:
                                f_interp = interp1d(old_times, surv_matrix, kind='linear', axis=1, fill_value='extrapolate')
                                surv_matrix = f_interp(eval_times)
                            else:
                                self.logger.warning(f"[EVAL] 无法对齐时间轴进行插值: old_times_len={len(old_times)}, surv_cols={surv_matrix.shape[1]}, eval_times_len={len(eval_times)}")
                        except Exception as e:
                            self.logger.error(f"[EVAL] 插值过程中失败: {e}")
                    # 过滤nan/inf
                    mask = np.isfinite(surv_matrix).all(axis=1)
                    surv_matrix = surv_matrix[mask]
                    # ensure times for brier are in hours
                    y_test_brier = pd.DataFrame({'time': durations_for_plots[mask], 'event': y_test_df['event'].values[mask]})
                    y_train_brier = pd.DataFrame({'time': y_train[:,0], 'event': y_train[:,1]}) if y_train is not None else pd.DataFrame({'time': [], 'event': []})

                    # --- 新增健壮性: 确保 eval_times 在测试集随访区间内 ---
                    try:
                        test_min_time = float(np.min(y_test_brier['time'])) if len(y_test_brier) > 0 else 0.0
                        test_max_time = float(np.max(y_test_brier['time'])) if len(y_test_brier) > 0 else 0.0
                    except Exception:
                        test_min_time, test_max_time = 0.0, 0.0

                    # 保证 eval_times 为 numpy 数组并排序唯一化
                    eval_times = np.array(eval_times).astype(float).flatten()
                    eval_times = np.unique(np.sort(eval_times))

                    # 只保留落入测试随访区间 [test_min_time, test_max_time] 内的时间点
                    if len(y_test_brier) == 0:
                        self.logger.warning("测试集中没有有效样本，跳过 Brier score 计算。")
                        eval_times_filtered = np.array([])
                    else:
                        # 使用开区间过滤（排除边界，避免Brier Score计算错误）
                        eval_times_filtered = eval_times[(eval_times > test_min_time) & (eval_times < test_max_time)]

                    if eval_times_filtered.size == 0:
                        self.logger.warning(f"没有适合的评估时间点用于 Brier（原始 eval_times={eval_times.tolist()}, 测试集时间范围=({test_min_time},{test_max_time})），将跳过 Brier 计算。")
                    else:
                        # 如果 surv_matrix 的时间轴与 eval_times_filtered 不一致，进行插值到新的时间点
                        if surv_matrix.shape[1] != len(eval_times_filtered):
                            from scipy.interpolate import interp1d
                            old_times = pred_surv_func.columns.astype(float)
                            try:
                                if surv_matrix.ndim == 2 and len(old_times) == surv_matrix.shape[0]:
                                    surv_matrix = surv_matrix.T
                                if surv_matrix.ndim == 2 and len(old_times) != surv_matrix.shape[1]:
                                    try:
                                        ot_min = float(np.min(old_times))
                                        ot_max = float(np.max(old_times))
                                        if np.isfinite(ot_min) and np.isfinite(ot_max):
                                            old_times = np.linspace(ot_min, ot_max, surv_matrix.shape[1])
                                        else:
                                            old_times = np.arange(surv_matrix.shape[1], dtype=float)
                                    except Exception:
                                        old_times = np.arange(surv_matrix.shape[1], dtype=float)
                                if surv_matrix.ndim == 1:
                                    surv_matrix = surv_matrix[:, None]
                                if len(old_times) == surv_matrix.shape[1]:
                                    f_interp = interp1d(old_times, surv_matrix, kind='linear', axis=1, fill_value='extrapolate')
                                    surv_matrix = f_interp(eval_times_filtered)
                                else:
                                    self.logger.warning(f"[EVAL] 无法对齐时间轴进行插值(filtered): old_times_len={len(old_times)}, surv_cols={surv_matrix.shape[1]}, eval_times_filtered_len={len(eval_times_filtered)}")
                            except Exception as e:
                                self.logger.error(f"[EVAL] filtered 插值过程中失败: {e}")

                        try:
                            brier_scores_over_time = brier_score(
                                y_train_brier,
                                y_test_brier,
                                surv_matrix,
                                eval_times_filtered
                            )
                            # Normalize result to numpy array (brier_score wrapper may return scalar)
                            try:
                                brier_scores_over_time = np.asarray(brier_scores_over_time)
                            except Exception:
                                brier_scores_over_time = np.array([])
                            # Ensure at least 1-D
                            if brier_scores_over_time.ndim == 0:
                                brier_scores_over_time = brier_scores_over_time.reshape(1)
                        except Exception as e:
                            # 如果 sksurv 的 brier 计算失败（例如训练样本不足或内部错误），
                            # 回退到一个更鲁棒但不完美的实现：对每个时间点使用可用的测试样本
                            # 计算不加权（naive）Brier，排除在时间点之前被删失的样本。
                            self.logger.warning(f"Brier Score计算失败(sksurv): {e}; 尝试不加权回退计算（排除在t之前被删失样本）")
                            try:
                                durations_arr = np.asarray(y_test_brier['time'].values).flatten()
                                events_arr = np.asarray(y_test_brier['event'].values).flatten()
                                brier_list = []
                                # surv_matrix shape: (n_samples, n_times)
                                for ti, t in enumerate(eval_times_filtered):
                                    # 有效样本：在 t 时刻依然在险（duration >= t）或已经发生事件（duration < t and event==1)
                                    mask_valid = (durations_arr >= t) | ((durations_arr < t) & (events_arr == 1))
                                    if np.sum(mask_valid) < max(5, int(0.01 * len(durations_arr))):
                                        # 样本太少，无法可靠估计该时间点
                                        brier_list.append(np.nan)
                                        continue
                                    event_at_t = ((durations_arr <= t) & (events_arr == 1)).astype(float)
                                    pred_event_prob = 1.0 - np.asarray(surv_matrix)[mask_valid, ti]
                                    # 计算 naive Brier（对可用样本的简单均值）
                                    bs_t = np.mean((event_at_t[mask_valid] - pred_event_prob) ** 2)
                                    brier_list.append(float(bs_t))
                                if len(brier_list) == 0:
                                    brier_scores_over_time = np.array([])
                                else:
                                    brier_scores_over_time = np.array(brier_list)
                            except Exception as e2:
                                self.logger.warning(f"回退Brier计算也失败: {e2}")
                                brier_scores_over_time = np.array([])

                        # 计算最终的 mean（排除 NaN）
                        # Ensure brier_scores_over_time is a numpy array before checks
                        if not hasattr(brier_scores_over_time, 'size'):
                            try:
                                brier_scores_over_time = np.asarray(brier_scores_over_time)
                            except Exception:
                                brier_scores_over_time = np.array([])

                        if brier_scores_over_time.size == 0 or np.all(np.isnan(brier_scores_over_time)):
                            self.logger.warning("无法获得有效的 Brier scores over time，跳过 Brier 平均值计算。")
                        else:
                            valid_bs = brier_scores_over_time[~np.isnan(brier_scores_over_time)]
                            if valid_bs.size > 0:
                                brier_score_mean = float(np.mean(valid_bs))
                                metrics_results['brier_score_mean'] = brier_score_mean
                            else:
                                self.logger.warning("所有时间点的 Brier score 都为 NaN，跳过 Brier 平均值计算。")

                    # 为避免在控制台打印超大数组，将完整的 metrics_results 保存为文件（pickle），并记录简短摘要
                    try:
                        import pickle
                        metrics_path = os.path.join(fold_output_dir, f"metrics_results{f'_fold{fold}' if fold is not None else ''}.pkl")
                        with open(metrics_path, 'wb') as mf:
                            pickle.dump(metrics_results, mf)
                        # 简短摘要日志
                        summary = {
                            'c_index': metrics_results.get('c_index', None),
                            'integrated_auc': metrics_results.get('integrated_auc', None),
                            'roc_auc_mean': metrics_results.get('roc_auc_mean', None),
                            'brier_score_mean': metrics_results.get('brier_score_mean', None)
                        }
                        self.logger.info(f"指标已保存: {metrics_path}; 摘要: {summary}")
                    except Exception as e:
                        # 如果无法保存，则退回到安全的简短日志（避免打印大数组）
                        self.logger.warning(f"无法将完整指标保存到文件: {e}. 记录简短摘要。")
                        self.logger.info("指标摘要: c_index=%s, integrated_auc=%s, roc_auc_mean=%s, brier_score_mean=%s", 
                                         metrics_results.get('c_index', None),
                                         metrics_results.get('integrated_auc', None),
                                         metrics_results.get('roc_auc_mean', None),
                                         metrics_results.get('brier_score_mean', None))

            # Ensure we have a unified durations array (hours) for plotting calls regardless of diagnostic write
            try:
                durations_for_plots = _ensure_durations_in_hours(np.asarray(y_test_df['duration'].values), cfg=self.config, name='plotting_durations_init')
            except Exception:
                try:
                    durations_for_plots = np.asarray(y_test_df['duration'].values)
                except Exception:
                    durations_for_plots = np.array([])

            # --- 可选：为诊断保存 predictions.npz（如果尚未存在），并确保 durations 写为小时 ---
            try:
                diag_npz = os.path.join(fold_output_dir, 'diag_out_predictions.npz')
                if not os.path.exists(diag_npz):
                    preds_to_save = risk_scores if (not is_deephit_model) else raw_predictions
                    raw_durations = np.asarray(y_test_df['duration'].values)
                    # converted durations in hours (observed or fraction->hours)
                    converted_hours = _ensure_durations_in_hours(raw_durations, cfg=self.config, name='evaluator_saved_durations')
                    save_events = y_test_df['event'].values
                    # record_ids may not be sliceable if None
                    try:
                        save_rids = np.array(record_ids, dtype=object) if record_ids is not None else np.arange(len(converted_hours))
                    except Exception:
                        save_rids = np.arange(len(converted_hours))

                    # Prepare optional per-time outputs for plotting consumers
                    per_time_array = None
                    try:
                        if is_deephit_model:
                            # 优先使用已计算的pmf_T_numpy（如果可用）
                            if pmf_T_numpy is not None:
                                per_time_array = pmf_T_numpy
                            else:
                                # 回退：从raw_predictions重新计算
                                try:
                                    logits = np.array(raw_predictions)
                                    # If logits have last-dim >1, assume binary/time logits and softmax across last dim to get pmf
                                    if logits.ndim == 3 and logits.shape[-1] > 1:
                                        # softmax along last axis then take probability of event occurrence index (commonly index 1)
                                        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                                        sm = exp / np.sum(exp, axis=-1, keepdims=True)
                                        # If shape (n, bins, 2) take probability of positive event per bin
                                        per_time_array = sm[..., 1]
                                    elif logits.ndim == 2:
                                        # Already per-bin pmf
                                        per_time_array = logits
                                    else:
                                        # fallback: attempt softmax over axis=1
                                        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                                        per_time_array = exp / np.sum(exp, axis=1, keepdims=True)
                                except Exception:
                                    per_time_array = None
                        else:
                            # For Cox-like models, if we computed survival_funcs_df above, save the per-sample survival curves
                            try:
                                if 'survival_funcs_df' in locals() and survival_funcs_df is not None and not survival_funcs_df.empty:
                                    # survival_funcs_df columns are time bins; convert to numpy array (n_samples, n_bins)
                                    per_time_array = np.asarray(survival_funcs_df.values)
                            except Exception:
                                per_time_array = None
                    except Exception:
                        per_time_array = None

                    # Save both raw window/fraction semantics and converted hours. Keep 'durations' key for backward compatibility.
                    save_dict = dict(
                        preds=np.array(preds_to_save),
                        durations=np.array(converted_hours),
                        durations_window_raw=np.array(raw_durations),
                        durations_observed_hours=np.array(converted_hours),
                        events=np.array(save_events),
                        record_ids=save_rids
                    )
                    if per_time_array is not None:
                        # Name it pred_probs if it is pmf-like (prob mass per bin), otherwise survival_funcs
                        try:
                            # Heuristic: values in [0,1] and rows sum <=~1 -> treat as pred_probs (pmf)
                            pt = np.asarray(per_time_array)
                            row_sums = np.nansum(np.nan_to_num(pt), axis=1)
                            if np.all((pt >= -1e-8) & (pt <= 1.0 + 1e-8)) and np.nanmax(row_sums) <= 1.5:
                                save_dict['pred_probs'] = pt
                            else:
                                save_dict['survival_funcs'] = pt
                        except Exception:
                            save_dict['survival_funcs'] = np.asarray(per_time_array)

                    np.savez_compressed(diag_npz, **save_dict)
                    self.logger.info(f"Saved diagnostic predictions to {diag_npz} (including per-time arrays: {'pred_probs' in save_dict or 'survival_funcs' in save_dict})")
                # Prepare a unified durations array (hours) for plotting calls.
                try:
                    durations_for_plots = np.asarray(converted_hours)
                except Exception:
                    try:
                        durations_for_plots = _ensure_durations_in_hours(np.asarray(y_test_df['duration'].values), cfg=self.config, name='plotting_durations')
                    except Exception:
                        durations_for_plots = np.asarray(y_test_df['duration'].values)
            except Exception:
                # Do not fail evaluation if saving diagnostics fails
                pass

            # 4. 绘图
            if plot_curves:
                self.logger.info("生成评估图表...")

                # 如果提供了 fold，则为本折创建单独的输出目录，便于每折图像和文件管理
                if fold is not None:
                    fold_output_dir = os.path.join(self.output_dir, f'fold_{fold}')
                else:
                    fold_output_dir = self.output_dir
                os.makedirs(fold_output_dir, exist_ok=True)
                # 为输出文件使用带折数的模型名，方便每折文件命名
                model_name_with_fold = self.config.model.name + (f"_fold{fold}" if fold is not None else "")

                if is_deephit_model:
                    # 尝试从DeepHit logits生成生存函数
                    try:
                        # 检查raw_predictions的维度，只有2D或3D才能处理
                        if hasattr(raw_predictions, 'ndim') and raw_predictions.ndim >= 2:
                            survival_funcs_df = self._predict_survival_functions_deephit(raw_predictions) # raw_predictions是logits
                            # 存储生存函数数据和风险分数供预测方法对比使用
                            self.survival_funcs_df = survival_funcs_df
                            self.risk_scores = risk_scores
                            risk_scores_for_grouping = risk_scores
                        else:
                            # 如果raw_predictions是1D，说明模型返回的不是DeepHit logits
                            self.logger.warning(f"DeepHit模型返回1D输出，形状: {raw_predictions.shape if hasattr(raw_predictions, 'shape') else 'unknown'}，尝试使用Cox模型方法生成生存函数")
                            survival_funcs_df = None
                            risk_scores_for_grouping = risk_scores
                    except Exception as e:
                        self.logger.warning(f"无法从DeepHit logits生成生存函数: {e}，尝试使用Cox模型方法")
                        survival_funcs_df = None
                        risk_scores_for_grouping = risk_scores
                else:
                    # 为非DeepHit模型也计算和存储生存函数数据
                    baseline_survival = self._compute_baseline_survival(X_train, y_train) if (X_train is not None and y_train is not None) else None
                    if baseline_survival is not None:
                        survival_funcs_df = self._predict_survival_functions(X_test, baseline_survival)
                        # 存储生存函数数据和风险分数供预测方法对比使用
                        self.survival_funcs_df = survival_funcs_df
                        self.risk_scores = risk_scores
                        risk_scores_for_grouping = risk_scores
                    else:
                        survival_funcs_df = None
                        risk_scores_for_grouping = risk_scores
                
                # 如果生存函数生成失败，尝试从风险分数重建（用于Cox模型）
                if survival_funcs_df is None and X_train is not None and y_train is not None:
                    try:
                        self.logger.info("尝试从风险分数重建生存函数...")
                        baseline_survival = self._compute_baseline_survival(X_train, y_train)
                        if baseline_survival is not None:
                            survival_funcs_df = self._predict_survival_functions(X_test, baseline_survival)
                            self.survival_funcs_df = survival_funcs_df
                            self.risk_scores = risk_scores
                            risk_scores_for_grouping = risk_scores
                            self.logger.info(f"成功重建生存函数: {survival_funcs_df.shape}")
                    except Exception as e:
                        self.logger.warning(f"无法重建生存函数: {e}")
                
                # 为风险分组计算中位数（用于绘图）
                risk_median = np.median(risk_scores_for_grouping)
                risk_groups = pd.Series('Low Risk', index=y_test_df.index)
                risk_groups[risk_scores_for_grouping >= risk_median] = 'High Risk'

                # 1. 绘制离散的风险组生存曲线并获取log-rank检验结果
                logrank_p_value, logrank_statistic = plotting.plot_survival_curves_by_risk_group_deephit(
                    y_test_df=y_test_df,
                    risk_groups=risk_groups,
                    survival_funcs_df=survival_funcs_df,
                    model_name=model_name_with_fold,
                    output_dir=fold_output_dir
                )
                
                # 将log-rank检验结果添加到评估指标中
                if logrank_p_value is not None:
                    metrics_results['logrank_p_value'] = float(logrank_p_value)
                    metrics_results['logrank_statistic'] = float(logrank_statistic)
                    significance = "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
                    metrics_results['logrank_significance'] = significance
                    
                    # 计算生存时间预测误差指标
                    try:
                        from .metrics import calculate_survival_time_metrics
                        prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                        survival_time_metrics = calculate_survival_time_metrics(
                            survival_funcs_df, durations_for_plots, y_test_df['event'].values, prediction_window, risk_scores
                        )
                        metrics_results.update(survival_time_metrics)
                    except Exception as e:
                        self.logger.warning(f"计算生存时间预测误差指标失败: {e}")
                    
                    # 1.5 Permutation Feature Importance
                    feature_importance = {}
                    if plot_extra_visuals:
                        try:
                            self.logger.info("Computing Permutation Feature Importance on test set...")
                            feature_importance = self._compute_permutation_importance(X_test, durations_for_plots, y_test_df['event'].values, base_c_index=chosen_c_index, passed_names=feature_names)
                            if feature_importance is not None and len(feature_importance) > 0:
                                fi_output_dir = os.path.join(fold_output_dir, "feature_importance")
                                os.makedirs(fi_output_dir, exist_ok=True)
                                plotting.plot_feature_importance_barh(feature_importance, model_name=model_name_with_fold, output_dir=fi_output_dir)
                                # Extended visualizations: annotated bar, radar chart, heatmap
                                try:
                                    plotting.plot_feature_importance_extended(feature_importance, model_name=model_name_with_fold, output_dir=fi_output_dir)
                                except Exception as _ext_e:
                                    self.logger.warning(f"plot_feature_importance_extended failed: {_ext_e}")
                                # Save numerical values
                                pd.Series(feature_importance).to_csv(os.path.join(fi_output_dir, f'{model_name_with_fold}_feature_importance.csv'))
                        except Exception as e:
                            self.logger.warning(f"Permutation Feature Importance failed: {e}")

                    # 1.6 Plot AR dynamic features overlay for good/bad AR cases
                    if plot_extra_visuals and X_test.ndim == 3 and survival_funcs_df is not None:
                        try:
                            self.logger.info("Generating AR Dynamic Survival/Feature Overlay Plots...")
                            # we need the original features, which are normalized here, but we can just plot the normalized ones directly.
                            f_names_to_use = feature_names if feature_names is not None else getattr(self.config.data, 'specified_features', None)
                            dyn_output_dir = os.path.join(fold_output_dir, "dynamic_trajectories")
                            os.makedirs(dyn_output_dir, exist_ok=True)
                            plotting.plot_ar_dynamic_prediction_with_features(
                                X_test=X_test, 
                                durations_for_plots=durations_for_plots, 
                                events=y_test_df['event'].values,
                                survival_funcs_df=survival_funcs_df,
                                pmf_T_numpy=pmf_T_numpy,
                                risk_scores=risk_scores,
                                record_ids=record_ids,
                                feature_names=f_names_to_use,
                                model_name=model_name_with_fold,
                                output_dir=dyn_output_dir,
                                raw_samples=raw_samples,
                                feature_importances=feature_importance
                            )
                        except Exception as e:
                            self.logger.warning(f"AR Dynamic Overlay Plot failed: {e}")

                    # 2. 个体生存曲线，带活动区
                    if plot_extra_visuals:
                        # Choose representative samples: top N highest risk and N lowest risk (if possible)
                        try:
                            num_to_plot = min(10, survival_funcs_df.shape[0])
                            # rank by risk_scores (higher -> earlier event for deephit)
                            risk_order = np.argsort(risk_scores)  # ascending
                            low_idx = risk_order[: num_to_plot // 2]
                            high_idx = risk_order[-(num_to_plot - len(low_idx)):]
                            sel_idx = np.concatenate([high_idx[::-1], low_idx])
                            sel_idx = sel_idx[:num_to_plot]
                        except Exception:
                            sel_idx = np.arange(min(10, survival_funcs_df.shape[0]))

                        sel_surv = survival_funcs_df.iloc[sel_idx]
                        sel_y = y_test_df.iloc[sel_idx]
                        sel_record_ids = (record_ids[sel_idx] if (record_ids is not None and len(record_ids) >= sel_idx.max()+1) else None)
                        plotting.plot_individual_survival_curves(
                            survival_funcs_df=sel_surv,
                            y_test_df=sel_y,
                            model_name=model_name_with_fold,
                            output_dir=fold_output_dir,
                            fold=fold,
                            n_samples=len(sel_surv),
                            is_discrete=True,
                            record_ids=sel_record_ids
                        )
                    # 3. 概率分布图
                    # 使用之前计算的pmf_T（如果可用），否则重新计算
                    if pmf_T_numpy is not None:
                        pred_probs = pmf_T_numpy
                        self.logger.info(f"使用已计算的pmf_T，形状: {pred_probs.shape}")
                    else:
                        # 回退：重新计算（应该不会到这里，但保留以防万一）
                        self.logger.warning(f"pmf_T_numpy为None，尝试从raw_predictions重新计算。raw_predictions形状: {raw_predictions.shape if hasattr(raw_predictions, 'shape') else type(raw_predictions)}")
                        try:
                            # 确保raw_predictions是numpy数组
                            raw_predictions_arr = np.asarray(raw_predictions)
                            if raw_predictions_arr.ndim == 0:
                                # 标量，无法处理
                                self.logger.warning("raw_predictions是标量，无法计算概率分布")
                                pred_probs = None
                            elif raw_predictions_arr.ndim == 1:
                                # 1D数组，可能是(n_samples,)或(n_time_bins,)
                                # 检查是否是所有样本的单个值，还是单个样本的多时间点
                                if len(raw_predictions_arr) == len(y_test_df):
                                    # 可能是(n_samples,) - 每个样本一个风险分数，无法得到时间分布
                                    self.logger.warning(f"raw_predictions是1D数组，形状{raw_predictions_arr.shape}，无法计算时间维度的概率分布")
                                    pred_probs = None
                                else:
                                    # 可能是单个样本的(n_time_bins,)，尝试reshape
                                    self.logger.warning(f"raw_predictions是1D数组，形状{raw_predictions_arr.shape}，尝试reshape为2D")
                                    # 如果长度是num_time_bins的倍数，尝试reshape
                                    try:
                                        num_bins = int(getattr(self.config.model.deephit, 'num_time_bins', 24))
                                        if len(raw_predictions_arr) % num_bins == 0:
                                            n_samples = len(raw_predictions_arr) // num_bins
                                            raw_predictions_arr = raw_predictions_arr.reshape(n_samples, num_bins)
                                            logits_tensor = torch.from_numpy(raw_predictions_arr).to(self.device)
                                            pred_probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
                                            self.logger.info(f"成功reshape并计算概率分布，形状: {pred_probs.shape}")
                                        else:
                                            pred_probs = None
                                    except Exception as e:
                                        self.logger.warning(f"尝试reshape失败: {e}")
                                        pred_probs = None
                            else:
                                # 2D或3D，正常处理
                                logits_tensor = torch.from_numpy(raw_predictions_arr).to(self.device)
                                if logits_tensor.ndim == 3:
                                    # 形状: (n_samples, n_events, n_time_bins)
                                    pred_probs_3d = torch.softmax(logits_tensor, dim=2)
                                    pred_probs = pred_probs_3d.sum(dim=1).cpu().numpy()
                                elif logits_tensor.ndim == 2:
                                    pred_probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
                                else:
                                    self.logger.warning(f"无法处理raw_predictions的维度: {logits_tensor.ndim}，跳过概率分布图")
                                    pred_probs = None
                        except Exception as e:
                            self.logger.warning(f"从raw_predictions重新计算概率分布失败: {e}")
                            pred_probs = None
                    if pred_probs is not None:
                        plotting.plot_deephit_probability_distribution(
                            pred_probs=pred_probs,
                            model_name=model_name_with_fold,
                            output_dir=fold_output_dir,
                            fold=fold
                        )
                        # 4. 事件概率分布曲线（带活动区/事件点）
                        if plot_extra_visuals:
                            plotting.plot_event_probability_curves(
                                pred_probs=pred_probs,
                                durations=durations_for_plots,
                                events=y_test_df['event'].values,
                                record_ids=record_ids if record_ids is not None else np.arange(len(y_test_df)),
                                model_name=model_name_with_fold,
                                output_dir=fold_output_dir,
                                fold=fold,
                                n_samples=5
                            )
                            # New: time-wise separation visualization for DeepHit (use per-bin probabilities)
                            try:
                                plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                            except Exception:
                                plot_modes = {}
                            if getattr(plot_modes, 'enable_timewise_separation', True):
                                try:
                                    plotting.plot_timewise_separation(
                                        survival_funcs_df=None,
                                        pred_probs=pred_probs,
                                        events=y_test_df['event'].values,
                                        model_name=model_name_with_fold,
                                        output_dir=fold_output_dir,
                                        fold=fold,
                                        smooth=bool(getattr(plot_modes, 'timewise_separation_smooth', False))
                                    )
                                except Exception as _e:
                                    self.logger.warning(f"plot_timewise_separation (DeepHit) failed: {_e}")
                    else:
                        self.logger.warning("pred_probs为None，跳过概率分布相关的可视化")
                    # 4b. 累计风险示例（配置控制）
                    try:
                        plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                    except Exception:
                        plot_modes = {}
                    # 禁用cumulative_risk_examples，使用individual_survival_curves代替
                    # if plot_extra_visuals and getattr(plot_modes, 'enable_cumulative_risk_examples', True):
                    #     plotting.plot_cumulative_risk_examples(
                    #         survival_funcs_df=survival_funcs_df,
                    #         durations=durations_for_plots,
                    #         events=y_test_df['event'].values,
                    #         record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                    #         model_name=model_name_with_fold,
                    #         output_dir=fold_output_dir,
                    #         n_examples=6,
                    #         fold=fold
                    #     )
                    # 5. 活动区生存曲线
                    if plot_extra_visuals:
                        plotting.plot_ar_survival_curves(
                            survival_probs=survival_funcs_df,
                            durations=durations_for_plots,
                            events=y_test_df['event'].values,
                            record_ids=record_ids if record_ids is not None else np.arange(len(y_test_df)),
                            output_dir=fold_output_dir,
                            fold=fold,
                            n_ar=6,
                            model_name=model_name_with_fold,
                            output_all_ars=True,
                            subdir='ar_survival_curves'
                        )
                    # 6. 风险进展图
                    num_plots = min(10, X_test.shape[0])
                    plotting.plot_risk_progression(
                        risk_scores[:num_plots],
                        y_test[:num_plots],
                        os.path.join(fold_output_dir, f"risk_progression.png"),
                        fold=fold
                    )
                    # 6b. 风险时间序列示例（半小时分辨率或连续插值）
                    if plot_extra_visuals and getattr(plot_modes, 'enable_risk_timecourse_examples', True):
                        try:
                            if pred_probs is not None:
                                plotting.plot_risk_timecourse_examples(
                                    pred_probs=pred_probs,
                                    durations=durations_for_plots,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    n_examples=4,
                                    fold=fold,
                                    continuous_interpolation=True
                                )
                            else:
                                self.logger.warning("pred_probs为None，跳过风险时间序列示例可视化")
                        except Exception as e:
                            self.logger.warning(f"plot_risk_timecourse_examples failed: {e}")
                    # 6c. 样本危险函数 h(t) 示例曲线（平滑处理，避免末端跳变）
                    try:
                        plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                    except Exception:
                        plot_modes = {}
                    if plot_extra_visuals and getattr(plot_modes, 'enable_hazard_examples', True):
                        try:
                            # 读取可选参数
                            hz_scale = getattr(plot_modes, 'hazard_scale', 1.0)
                            hz_clip_q = getattr(plot_modes, 'hazard_clip_q', 99.0)
                            hz_smooth = getattr(plot_modes, 'hazard_smooth_window', None)
                            hz_show_cum = getattr(plot_modes, 'hazard_show_cumhaz', False)
                            plotting.plot_hazard_curves_examples(
                                survival_funcs=survival_funcs_df,
                                durations=durations_for_plots,
                                events=y_test_df['event'].values,
                                record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                model_name=model_name_with_fold,
                                output_dir=fold_output_dir,
                                n_examples=4,
                                fold=fold,
                                hazard_scale=hz_scale,
                                clip_quantile=hz_clip_q,
                                smooth_window=hz_smooth,
                                show_cumhaz=hz_show_cum
                            )
                        except Exception as e:
                            self.logger.warning(f"plot_hazard_curves_examples (DeepHit) failed: {e}")

                    # 6d. 每分钟非累计风险（事件概率/分钟）示例
                    try:
                        plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                    except Exception:
                        plot_modes = {}
                    try:
                        if plot_extra_visuals and getattr(plot_modes, 'enable_per_minute_risk', True):
                            log_scale = getattr(plot_modes, 'per_minute_risk_log_scale', True)
                            smooth_minutes = getattr(plot_modes, 'per_minute_risk_smooth_minutes', None)
                            try:
                                from .plotting import plot_non_cumulative_risk_per_minute_examples
                                agg_mins = getattr(plot_modes, 'per_minute_risk_aggregate_minutes', None)
                                plot_non_cumulative_risk_per_minute_examples(
                                    survival_funcs_df=survival_funcs_df,
                                    durations=durations_for_plots,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    n_examples=6,
                                    fold=fold,
                                    log_scale=log_scale,
                                    smooth_minutes=smooth_minutes,
                                    aggregate_minutes=agg_mins
                                )
                            except Exception as e:
                                self.logger.warning(f"plot_non_cumulative_risk_per_minute_examples (DeepHit) failed: {e}")
                    except Exception:
                        pass

                    # 7. 风险分布、生存时间分布、置信区间
                    plotting.plot_risk_distribution_by_event_type(
                        risk_scores=risk_scores,
                        events=y_test_df['event'].values,
                        model_name=model_name_with_fold,
                        output_dir=fold_output_dir
                    )
                    plotting.plot_survival_time_distribution(
                        durations=durations_for_plots,
                        events=y_test_df['event'].values,
                        model_name=model_name_with_fold,
                        output_dir=fold_output_dir
                    )
                    plotting.plot_confidence_intervals(
                        survival_curves=survival_funcs_df,
                        model_name=model_name_with_fold,
                        output_dir=fold_output_dir
                    )
                    # 8. 回归性分析（事件样本）
                    event_mask = y_test_df['event'].values == 1
                    if np.sum(event_mask) > 0:
                        # 检查pred_probs的维度
                        if pred_probs is not None and pred_probs.ndim == 2 and pred_probs.shape[1] > 1:
                            # 预测事件时间=概率分布的期望
                            pred_event_times = np.sum(pred_probs[event_mask] * np.arange(pred_probs.shape[1]), axis=1)
                            true_event_times = durations_for_plots[event_mask]
                            ar_ids = record_ids[event_mask] if record_ids is not None else None
                            reg_stats = plotting.plot_regression_analysis(
                                pred_event_times=pred_event_times,
                                true_event_times=true_event_times,
                                model_name=model_name_with_fold,
                                output_dir=fold_output_dir,
                                record_ids=ar_ids
                            )
                            metrics_results.update({f'regression_{k}': v for k, v in reg_stats.items()})
                        else:
                            self.logger.warning("pred_probs维度不正确，跳过回归性分析")
                    # 9. ROC分析
                    if plot_curves:
                        from .plotting import plot_roc_analysis, plot_roc_curves, plot_auc_over_time
                        
                        # 综合ROC分析
                        roc_analysis_results = plotting.plot_roc_analysis(
                            predictions=risk_scores,
                            durations=durations_for_plots,
                            events=y_test_df['event'].values,
                            output_dir=fold_output_dir,
                            model_name=model_name_with_fold
                        )
                        
                        # 单独的ROC曲线
                        if 'time_dependent_roc' in roc_metrics:
                            plotting.plot_roc_curves(
                                roc_metrics['time_dependent_roc'],
                                fold_output_dir,
                                model_name_with_fold
                            )
                        
                        # AUC时间曲线 - 使用更密集的时间点（每2小时）
                        try:
                            from .metrics import calculate_auc_at_multiple_times
                            prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                            auc_time_points = np.arange(1, prediction_window + 2, 2)  # 0, 2, 4, ..., 48
                            dense_auc_scores = calculate_auc_at_multiple_times(
                                oriented_risk_scores, durations_for_plots, y_test_df['event'], auc_time_points
                            )
                            plotting.plot_auc_over_time(
                                dense_auc_scores,
                                fold_output_dir,
                                model_name_with_fold,
                                mean_ci=metrics_results.get('roc_auc_mean_ci'),
                                time_stats=metrics_results.get('auc_time_stats')
                            )
                        except Exception as e:
                            logger.warning(f"绘制密集AUC时间曲线失败: {e}")
                            # 回退到使用roc_metrics中的auc_at_times
                            if 'auc_at_times' in roc_metrics:
                                plotting.plot_auc_over_time(
                                    roc_metrics['auc_at_times'],
                                    fold_output_dir,
                                    model_name_with_fold,
                                    mean_ci=metrics_results.get('roc_auc_mean_ci'),
                                    time_stats=metrics_results.get('auc_time_stats')
                                )

                if X_train is not None and y_train is not None:
                    # Cox模型的绘图流程
                    baseline_survival = self._compute_baseline_survival(X_train, y_train)
                    survival_funcs_df = self._predict_survival_functions(X_test, baseline_survival)
                    
                    if not survival_funcs_df.empty:
                        # 风险分组 (使用风险分数，DeepSurv中风险分数越高表示风险越高)
                        risk_scores_for_grouping = risk_scores
                        risk_median = np.median(risk_scores_for_grouping)
                        risk_groups = pd.Series('Low Risk', index=y_test_df.index)
                        risk_groups[risk_scores_for_grouping >= risk_median] = 'High Risk'

                        # 绘制生存曲线并获取log-rank检验结果
                        logrank_p_value, logrank_statistic = plotting.plot_survival_curves_by_risk_group(
                            y_test_df=y_test_df,
                            risk_groups=risk_groups,
                            survival_funcs_df=survival_funcs_df,
                            model_name=model_name_with_fold,
                            output_dir=fold_output_dir
                        )
                        
                        # 将log-rank检验结果添加到评估指标中
                        if logrank_p_value is not None:
                            metrics_results['logrank_p_value'] = float(logrank_p_value)
                            metrics_results['logrank_statistic'] = float(logrank_statistic)
                            significance = "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
                            metrics_results['logrank_significance'] = significance
                        
                        # 计算生存时间预测误差指标
                        try:
                            from .metrics import calculate_survival_time_metrics
                            prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                            survival_time_metrics = calculate_survival_time_metrics(
                                survival_funcs_df, durations_for_plots, y_test_df['event'].values, prediction_window, risk_scores
                            )
                            metrics_results.update(survival_time_metrics)
                        except Exception as e:
                            self.logger.warning(f"计算生存时间预测误差指标失败: {e}")
                        
                        if len(y_test_df[y_test_df['event']==1]) + len(y_test_df[y_test_df['event']==0]) > 0:
                            try:
                                num_to_plot = min(12, survival_funcs_df.shape[0])
                                # select a spread: quantiles of risk
                                idxs = np.linspace(0, survival_funcs_df.shape[0]-1, num_to_plot).astype(int)
                                sel_surv = survival_funcs_df.iloc[idxs]
                                sel_y = y_test_df.iloc[idxs]
                                sel_rids = (record_ids[idxs] if (record_ids is not None and len(record_ids) >= idxs.max()+1) else None)
                                plotting.plot_individual_survival_curves(
                                    survival_funcs_df=sel_surv,
                                    y_test_df=sel_y,
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    fold=fold,
                                    n_samples=len(sel_surv),
                                    is_discrete=False,
                                    record_ids=sel_rids
                                )
                            except Exception as e:
                                self.logger.warning(f"个体生存曲线选择或绘制失败: {e}")
                        # 额外：为基于生存函数的模型生成累计风险与概率时间序列等可视化
                        try:
                            plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                        except Exception:
                            plot_modes = {}

                        # 4a) 个体累计风险示例（已禁用，使用individual_survival_curves代替）
                        # try:
                        #     if getattr(self.config.evaluation, 'plot_extra_visuals', True) and getattr(plot_modes, 'enable_cumulative_risk_examples', True):
                        #         plotting.plot_cumulative_risk_examples(
                        #             survival_funcs_df=survival_funcs_df,
                        #             durations=durations_for_plots,
                        #             events=y_test_df['event'].values,
                        #             record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                        #             model_name=model_name_with_fold,
                        #             output_dir=fold_output_dir,
                        #             n_examples=4,
                        #             fold=fold
                        #         )
                        # except Exception as e:
                        #     self.logger.warning(f"plot_cumulative_risk_examples (Cox) failed: {e}")

                        # 4b) 平均累计风险曲线（全体，必要时可按高低风险分组）
                        try:
                            plotting.plot_mean_cumulative_risk(
                                survival_probs=survival_funcs_df,
                                model_name=model_name_with_fold,
                                output_dir=fold_output_dir,
                                by_risk_group=None,
                                fold=fold
                            )
                        except Exception as e:
                            self.logger.warning(f"plot_mean_cumulative_risk (Cox) failed: {e}")

                        # New: time-wise separation visualization for Cox-like models (derive per-time risk from S)
                        try:
                            plot_modes = getattr(self.config.evaluation, 'plot_modes', None) or {}
                        except Exception:
                            plot_modes = {}
                        if getattr(plot_modes, 'enable_timewise_separation', True):
                            try:
                                plotting.plot_timewise_separation(
                                    survival_funcs_df=survival_funcs_df,
                                    pred_probs=None,
                                    events=y_test_df['event'].values,
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    fold=fold,
                                    smooth=bool(getattr(plot_modes, 'timewise_separation_smooth', False))
                                )
                            except Exception as _e:
                                self.logger.warning(f"plot_timewise_separation (Cox) failed: {_e}")

                        # 4c) 活动区累计风险曲线（事件样本颜色更深序）
                        try:
                            if plot_extra_visuals and hasattr(plotting, 'plot_ar_cumulative_risk_curves'):
                                plotting.plot_ar_cumulative_risk_curves(
                                    survival_probs=survival_funcs_df,
                                    durations=durations_for_plots,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    output_dir=fold_output_dir,
                                    fold=fold,
                                    n_ar=6,
                                    model_name=model_name_with_fold
                                )
                        except Exception as e:
                            self.logger.warning(f"plot_ar_cumulative_risk_curves (Cox) failed: {e}")

                        # 4c-bis) 活动区生存曲线：输出全部活动区到子目录
                        try:
                            if plot_extra_visuals:
                                plotting.plot_ar_survival_curves(
                                    survival_probs=survival_funcs_df,
                                    durations=durations_for_plots,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    output_dir=fold_output_dir,
                                    fold=fold,
                                    n_ar=6,
                                    model_name=model_name_with_fold,
                                    output_all_ars=True,
                                    subdir='ar_survival_curves'
                                )
                        except Exception as e:
                            self.logger.warning(f"plot_ar_survival_curves (Cox) failed: {e}")

                        # 4d) 风险概率/事件概率时间序列（pmf，分开画4个样本，优先2事件+2删失）
                        try:
                            if getattr(self.config.evaluation, 'plot_extra_visuals', True) and getattr(plot_modes, 'enable_risk_timecourse_examples', True):
                                surv_vals = survival_funcs_df.values
                                # pmf: p_k = S(t_{k-1}) - S(t_k)
                                S_prev = np.concatenate([np.ones((surv_vals.shape[0], 1)), surv_vals[:, :-1]], axis=1)
                                pmf = np.maximum(S_prev - surv_vals, 0.0)
                                # 归一化（仅对和>0的行）
                                row_sums = pmf.sum(axis=1, keepdims=True)
                                nz = (row_sums.squeeze() > 0)
                                if nz.any():
                                    pmf[nz] = pmf[nz] / row_sums[nz]

                                plotting.plot_risk_timecourse_examples(
                                    pred_probs=pmf,
                                    durations=y_test_df['duration'].values,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    n_examples=4,
                                    fold=fold,
                                    continuous_interpolation=True,
                                    display_mode='mass'
                                )
                        except Exception as e:
                            self.logger.warning(f"plot_risk_timecourse_examples (Cox) failed: {e}")

                        # 4e) 样本危险函数 h(t) 示例曲线（平滑处理）
                        try:
                            if plot_extra_visuals and getattr(plot_modes, 'enable_hazard_examples', True):
                                hz_scale = getattr(plot_modes, 'hazard_scale', 1.0)
                                hz_clip_q = getattr(plot_modes, 'hazard_clip_q', 99.0)
                                hz_smooth = getattr(plot_modes, 'hazard_smooth_window', None)
                                hz_show_cum = getattr(plot_modes, 'hazard_show_cumhaz', False)
                                plotting.plot_hazard_curves_examples(
                                    survival_funcs=survival_funcs_df,
                                    durations=durations_for_plots,
                                    events=y_test_df['event'].values,
                                    record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                    model_name=model_name_with_fold,
                                    output_dir=fold_output_dir,
                                    n_examples=4,
                                    fold=fold,
                                    hazard_scale=hz_scale,
                                    clip_quantile=hz_clip_q,
                                    smooth_window=hz_smooth,
                                    show_cumhaz=hz_show_cum
                                )
                        except Exception as e:
                            self.logger.warning(f"plot_hazard_curves_examples (Cox) failed: {e}")

                        # 4f) 每分钟非累计风险示例
                        try:
                            if plot_extra_visuals and getattr(plot_modes, 'enable_per_minute_risk', True):
                                log_scale = getattr(plot_modes, 'per_minute_risk_log_scale', True)
                                smooth_minutes = getattr(plot_modes, 'per_minute_risk_smooth_minutes', None)
                                try:
                                    from .plotting import plot_non_cumulative_risk_per_minute_examples
                                    agg_mins = getattr(plot_modes, 'per_minute_risk_aggregate_minutes', None)
                                    plot_non_cumulative_risk_per_minute_examples(
                                        survival_funcs_df=survival_funcs_df,
                                        durations=durations_for_plots,
                                        events=y_test_df['event'].values,
                                        record_ids=(record_ids if record_ids is not None else np.arange(len(y_test_df))),
                                        model_name=model_name_with_fold,
                                        output_dir=fold_output_dir,
                                        n_examples=6,
                                        fold=fold,
                                        log_scale=log_scale,
                                        smooth_minutes=smooth_minutes,
                                        aggregate_minutes=agg_mins
                                    )
                                except Exception as e:
                                    self.logger.warning(f"plot_non_cumulative_risk_per_minute_examples (Cox) failed: {e}")
                        except Exception:
                            pass
                
                # 为Cox模型添加额外的可视化
                if plot_extra_visuals and X_test.shape[0] > 0:
                    self.logger.info("为Cox模型添加额外可视化...")
                    
                    # 风险分布图
                    plotting.plot_risk_distribution_by_event_type(
                        risk_scores=risk_scores,
                        events=y_test_df['event'].values,
                        model_name=model_name_with_fold,
                        output_dir=fold_output_dir
                    )
                    
                    # 生存时间分布图
                    plotting.plot_survival_time_distribution(
                        durations=y_test_df['duration'].values,
                        events=y_test_df['event'].values,
                        model_name=model_name_with_fold,
                        output_dir=fold_output_dir
                    )
                    
                    # ROC分析
                    if plot_curves:
                        from .plotting import plot_roc_analysis, plot_roc_curves, plot_auc_over_time
                        
                        # 综合ROC分析
                        roc_analysis_results = plotting.plot_roc_analysis(
                            predictions=risk_scores,
                            durations=y_test_df['duration'].values,
                            events=y_test_df['event'].values,
                            output_dir=fold_output_dir,
                            model_name=model_name_with_fold
                        )
                        
                        # 单独的ROC曲线
                        if 'time_dependent_roc' in roc_metrics:
                            plotting.plot_roc_curves(
                                roc_metrics['time_dependent_roc'],
                                fold_output_dir,
                                model_name_with_fold
                            )

                        # AUC时间曲线 - 使用更密集的时间点（每2小时）
                        try:
                            from .metrics import calculate_auc_at_multiple_times
                            prediction_window = float(self.config.data.sequence_generation.prediction_window_hours)
                            auc_time_points = np.arange(2, prediction_window + 2, 2)  # 2, 4, 6, ..., 48
                            dense_auc_scores = calculate_auc_at_multiple_times(
                                risk_scores, y_test_df['duration'].values, y_test_df['event'].values, auc_time_points
                            )
                            plotting.plot_auc_over_time(
                                dense_auc_scores,
                                fold_output_dir,
                                model_name_with_fold,
                                mean_ci=metrics_results.get('roc_auc_mean_ci'),
                                time_stats=metrics_results.get('auc_time_stats')
                            )
                        except Exception as e:
                            logger.warning(f"绘制密集AUC时间曲线失败: {e}")
                            # 回退到使用roc_metrics中的auc_at_times
                            if 'auc_at_times' in roc_metrics:
                                plotting.plot_auc_over_time(
                                    roc_metrics['auc_at_times'],
                                    fold_output_dir,
                                    model_name_with_fold,
                                    mean_ci=metrics_results.get('roc_auc_mean_ci'),
                                    time_stats=metrics_results.get('auc_time_stats')
                                )
                    
                    # 置信区间图
                    if not survival_funcs_df.empty:
                        plotting.plot_confidence_intervals(
                            survival_curves=survival_funcs_df,
                            model_name=model_name_with_fold,
                            output_dir=fold_output_dir
                        )

            # 保存简表：平均AUC与C-index
            try:
                import csv
                summary_rows = []
                summary_rows.append({'metric': 'c_index', 'value': metrics_results.get('c_index', np.nan)})
                if 'roc_auc_mean' in metrics_results:
                    summary_rows.append({'metric': 'roc_auc_mean', 'value': metrics_results.get('roc_auc_mean', np.nan)})
                if 'roc_auc_mean_ci' in metrics_results and metrics_results.get('roc_auc_mean_ci'):
                    ci = metrics_results['roc_auc_mean_ci']
                    summary_rows.append({'metric': 'roc_auc_mean_ci_low', 'value': ci[0]})
                    summary_rows.append({'metric': 'roc_auc_mean_ci_high', 'value': ci[1]})
                if 'integrated_auc' in metrics_results:
                    summary_rows.append({'metric': 'integrated_auc', 'value': metrics_results.get('integrated_auc', np.nan)})
                csv_path = os.path.join(fold_output_dir, f'metrics_summary{f"_fold{fold}" if fold is not None else ""}.csv')
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=['metric', 'value'])
                    writer.writeheader()
                    for r in summary_rows:
                        writer.writerow(r)
                self.logger.info(f"简表已保存: {csv_path}")
            except Exception as e:
                self.logger.warning(f"保存简表CSV失败: {e}")

            # --- 特征重要性分析 (Permutation Importance) ---
            if compute_importance:
                self.logger.info(f"开始计算特征重要性 (repeats={importance_repeats})...")
                try:
                    # 使用当前 fold 的 C-index 作为基线
                    base_c = metrics_results.get('c_index', 0.5)
                    importances = self._compute_permutation_importance(
                        X_test, 
                        durations_for_plots, 
                        y_test_df['event'].values, 
                        base_c, 
                        n_repeats=importance_repeats,
                        passed_names=feature_names
                    )
                    metrics_results['feature_importance'] = importances
                    
                    # 保存结果到 JSON
                    fi_json_path = os.path.join(fold_output_dir, f'feature_importance{f"_fold{fold}" if fold is not None else ""}.json')
                    with open(fi_json_path, 'w', encoding='utf-8') as f:
                        json.dump(importances, f, indent=4, ensure_ascii=False)
                    self.logger.info(f"特征重要性数据已保存: {fi_json_path}")
                    
                    # 绘制重要性图表
                    if plot_curves:
                        try:
                            from .plotting import plot_feature_importance_extended
                            plot_feature_importance_extended(
                                importances, 
                                model_name_with_fold, 
                                fold_output_dir,
                                top_k=int(getattr(self.config.evaluation, 'importance_top_k', 0)) # 0 means all
                            )
                        except Exception as e_plot:
                            self.logger.warning(f"绘制特征重要性图表失败: {e_plot}")
                except Exception as e:
                    self.logger.warning(f"特征重要性计算失败: {e}")

            return metrics_results

        except Exception as e:
            self.logger.error(f"模型评估过程中出错: {e}", exc_info=True)
            return {'error': str(e)}

        # 在记录日志前检查c_index是否已成功计算
        if c_index is not None:
            self.logger.info(f"C-Index on test set: {c_index:.4f}")
        else:
            self.logger.error("C-Index calculation failed.")

    def _plot_risk_distribution(self, risk_scores, events, model_name):
        """绘制风险分布图"""
        try:
            # delegate to centralized plotting utility if available
            from .plotting import plot_risk_distribution_by_event_type
            plot_risk_distribution_by_event_type(risk_scores, events, model_name, self.output_dir)
            return
        except Exception:
            pass

        # Fallback: try matplotlib directly, otherwise save CSV
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            for event_type, event_name in [(0, 'Censored'), (1, 'Event')]:
                mask = events == event_type
                if np.any(mask):
                    plt.hist(risk_scores[mask], bins=50, alpha=0.6, label=event_name, density=True)
            plt.title('Risk Score Distribution')
            plt.xlabel('Risk Score')
            plt.ylabel('Density')
            plt.legend()
            save_path = os.path.join(self.output_dir, f'{model_name}_risk_distribution.png')
            plt.savefig(save_path)
            plt.close()
            return
        except Exception as e:
            # Save CSV fallback
            try:
                import csv
                out_csv = os.path.join(self.output_dir, f'{model_name}_risk_distribution.csv')
                with open(out_csv, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(['risk_score', 'event'])
                    for r, ev in zip(risk_scores, events):
                        writer.writerow([float(r), int(ev)])
                self.logger.info(f'风险分布CSV已保存: {out_csv}')
            except Exception:
                self.logger.warning(f'无法绘制或保存风险分布图: {e}')
            return

    def _plot_feature_importance(self, feature_importance, model_name):
        """绘制特征重要性图"""
        try:
            from .plotting import plot_feature_importance
            plot_feature_importance(feature_importance, model_name, self.output_dir)
            return
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            feature_importance.sort_values(ascending=True).plot(kind='barh')
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
            plt.savefig(save_path)
            plt.close()
            return
        except Exception as e:
            self.logger.warning(f'无法绘制特征重要性图: {e}')
            return

    def plot_risk_progression(self, X_subset, y_subset=None, record_ids=None, out_path=None, fold=None):
        """
        为给定的样本子集绘制或保存风险进展图。如果 plotting 模块提供实现则委托给它；否则保存一个CSV并尝试绘制一个简单的图。

        参数:
            X_subset (np.ndarray): 输入特征子集，形状可为 (n, seq_len, n_features) 或 (n, n_features)
            y_subset (np.ndarray or None): 对应的 (duration,event) 矩阵，可选
            record_ids (array-like or None): 对应的 record ids，可选
            out_path (str or None): 输出文件路径，如果为None则使用 output_dir/risk_progression_fold{fold}.png
            fold (int or None): 可选fold信息
        返回:
            如果成功返回输出路径，否则返回 None。
        """
        # 优先 delegating 给 plotting 模块（如果存在同名函数）
        try:
            if hasattr(plotting, 'plot_risk_progression'):
                # plotting.plot_risk_progression 的签名在不同实现间可能不同，所以采用常见参数名
                try:
                    return plotting.plot_risk_progression(
                        risk_scores=None,  # plotting 实现可能会忽略此 None 并接收 X/y
                        X=X_subset,
                        y=y_subset,
                        record_ids=record_ids,
                        output_path=out_path,
                        fold=fold,
                        evaluator=self
                    )
                except TypeError:
                    # 回退到另一种常见签名
                    return plotting.plot_risk_progression(X_subset, y_subset, record_ids, out_path)
        except Exception as e:
            self.logger.debug(f"delegating to plotting.plot_risk_progression 失败，回退到内部实现: {e}")

        # 如果 plotting 模块无实现或委托失败，执行内部回退实现
        try:
            risk_scores = self._predict_risk(X_subset).flatten()
        except Exception as e:
            self.logger.error(f"无法对给定子集预测风险: {e}")
            return None

        # 尝试恢复 y_subset / record_ids
        if y_subset is None:
            y_subset = getattr(self, '_last_y_test', None)
            if y_subset is not None:
                try:
                    y_subset = y_subset[: X_subset.shape[0]]
                except Exception:
                    pass
        if record_ids is None:
            record_ids = getattr(self, '_last_record_ids', None)
            if record_ids is not None:
                try:
                    record_ids = record_ids[: X_subset.shape[0]]
                except Exception:
                    pass

        # 默认输出路径
        if out_path is None:
            out_path_png = os.path.join(self.output_dir, f'risk_progression_fold{fold or 0}.png')
            out_path_csv = os.path.join(self.output_dir, f'risk_progression_fold{fold or 0}.csv')
        else:
            out_path_png = out_path
            out_path_csv = os.path.splitext(out_path)[0] + '.csv'

        # 保存CSV以便离线查看
        try:
            import csv
            with open(out_path_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                header = ['record_id', 'duration', 'event', 'risk_score']
                writer.writerow(header)
                for i in range(len(risk_scores)):
                    rid = record_ids[i] if (record_ids is not None and i < len(record_ids)) else i
                    dur = (y_subset[i,0] if (y_subset is not None and i < len(y_subset)) else '')
                    ev = (y_subset[i,1] if (y_subset is not None and i < len(y_subset)) else '')
                    writer.writerow([rid, dur, ev, float(risk_scores[i])])
            self.logger.info(f'风险进展CSV已保存: {out_path_csv}')
        except Exception as e:
            self.logger.warning(f'保存风险进展CSV失败: {e}')

        # 尝试画出简单的条形图/折线图（如果 matplotlib 可用）
        try:
            # delegate to plotting module if available
            from .plotting import plot_risk_progression as plotting_risk_progression
            try:
                res = plotting_risk_progression(X_subset, y_subset, record_ids, out_path)
                return res
            except TypeError:
                # try alternate signature
                res = plotting_risk_progression(risk_scores, y_subset, record_ids, out_path)
                return res
        except Exception:
            pass

        # Fallback to matplotlib or CSV-only
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            x = list(range(len(risk_scores)))
            plt.plot(x, risk_scores, marker='o')
            plt.title('Risk Scores Progression')
            plt.xlabel('Sample Index')
            plt.ylabel('Risk Score')
            if record_ids is not None:
                try:
                    xticks = [str(r) for r in record_ids[:len(x)]]
                    plt.xticks(x, xticks, rotation=45)
                except Exception:
                    pass
            plt.tight_layout()
            plt.savefig(out_path_png)
            plt.close()
            self.logger.info(f'简单风险进展图已保存: {out_path_png}')
            return out_path_png
        except Exception as e:
            self.logger.warning(f'绘制风险图失败（可能缺少绘图库）: {e}')
            return out_path_csv
