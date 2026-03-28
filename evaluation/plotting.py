"""
Functions for creating and saving visualizations of model performance and predictions.
"""
import logging
import numpy as np
import pandas as pd
import os
from lifelines import KaplanMeierFitter
import torch
import json
from datetime import datetime, timedelta
from configs.default_config import get_config
import re
import shutil
# Force non-interactive matplotlib backend to avoid display attempts on headless systems.
# Set the environment variable before any matplotlib import so the Agg backend is used.
try:
    os.environ.setdefault('MPLBACKEND', 'Agg')
except Exception:
    pass

# Plotting availability: prefer matplotlib; if unavailable, try plotly+kaleido; otherwise provide CSV/HTML fallback
MATPLOTLIB_AVAILABLE = False
plt = None
mcolors = None
sns = None
try:
    # Late import of matplotlib.pyplot using Agg backend
    import matplotlib
    try:
        matplotlib.use(os.environ.get('MPLBACKEND', 'Agg'))
    except Exception:
        # ignore if backend cannot be set (fallback will try to import)
        pass
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    try:
        # Ensure interactive mode is off and show() does nothing to avoid display attempts
        try:
            plt.ioff()
        except Exception:
            pass
        try:
            plt.show = lambda *args, **kwargs: None
        except Exception:
            pass
    except Exception:
        pass
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    logging.getLogger(__name__).warning(f"Matplotlib/seaborn import failed: {e}")

# For this workspace we force matplotlib-only plotting. If matplotlib not available, functions will log and skip.
PLOTLY_AVAILABLE = False
HAS_PLOTTING = MATPLOTLIB_AVAILABLE


def _ensure_matplotlib():
    """Ensure matplotlib.pyplot is importable and available as `plt` at runtime.

    Some execution environments partially import matplotlib or have broken
    system libraries which make the module unavailable at import time. This
    helper attempts a late import and updates module-level flags. Returns the
    plt module if available, otherwise None.
    """
    global plt, MATPLOTLIB_AVAILABLE, HAS_PLOTTING
    try:
        if MATPLOTLIB_AVAILABLE and plt is not None:
            return plt
        # try late import
        import matplotlib.pyplot as mpl_plt
        plt = mpl_plt
        MATPLOTLIB_AVAILABLE = True
        HAS_PLOTTING = True
        return plt
    except Exception as e:
        logging.getLogger(__name__).warning(f"Late matplotlib import failed: {e}")
        MATPLOTLIB_AVAILABLE = False
        HAS_PLOTTING = False
        plt = None
        return None

# Default results dir from global config
_GLOBAL_CFG = get_config()
DEFAULT_RESULTS_DIR = getattr(_GLOBAL_CFG, 'results_dir', 'results')


def _ensure_durations_in_hours(durations, cfg=None, name='durations'):
    """Ensure durations array is in hours.

    Heuristic: if median duration > 1000, assume seconds and convert (/3600).
    Returns numpy array of floats (hours).
    """
    import numpy as _np
    logger = logging.getLogger(__name__)
    try:
        arr = _np.asarray(durations, dtype=float)
    except Exception:
        try:
            arr = _np.array([float(x) for x in durations], dtype=float)
        except Exception:
            return _np.asarray(durations)
    if arr.size == 0:
        return arr
    med = float(_np.nanmedian(arr))
    # 1) If clearly in seconds (very large median), convert seconds->hours
    if med > 1000:
        logger.info(f"Converting {name} from seconds->hours (median={med})")
        try:
            arr = arr / 3600.0
        except Exception:
            pass

    # 2) Handle mixed semantics where some values are fractions of the prediction window
    #    (e.g. values in (0,~1]) and others are absolute hours (e.g. 24.0).
    #    If cfg provided, use its prediction_window_hours, else try global config.
    try:
        pred_w = None
        if cfg is not None:
            try:
                pred_w = float(getattr(cfg, 'data').sequence_generation.prediction_window_hours)
            except Exception:
                # attempt dict-like access
                try:
                    pred_w = float(cfg['data']['sequence_generation']['prediction_window_hours'])
                except Exception:
                    pred_w = None
        if pred_w is None:
            try:
                pred_w = float(getattr(_GLOBAL_CFG, 'data').sequence_generation.prediction_window_hours)
            except Exception:
                pred_w = None
    except Exception:
        pred_w = None

    if pred_w is not None:
        try:
            # treat values that are positive and <=1.0 as fraction-of-window and convert to hours
            # NOTE: previous heuristic used <=1.5 which incorrectly classified "days" (~1.4) or
            # small-hour values as fractions and caused values like 1.4167 -> 34 hours when
            # multiplied by prediction_window_hours=24. Tighten to <=1.0 to avoid such misclassification.
            frac_mask = (arr > 0) & (arr <= 1.0)
            if frac_mask.any():
                logger.info(f"Converting {name} fraction-of-window -> hours using prediction_window_hours={pred_w} for {int(frac_mask.sum())} values")
                arr = arr.copy()
                arr[frac_mask] = arr[frac_mask] * float(pred_w)
        except Exception:
            pass

    return arr


def _durations_in_hours_clamped(durations, cfg=None, name='durations'):
    """Convert durations to hours and clamp to prediction window.

    - Uses _ensure_durations_in_hours for unit normalization
    - Clamps to [0, prediction_window_hours] if cfg (or _GLOBAL_CFG) provides it
    """
    import numpy as _np
    logger = logging.getLogger(__name__)
    arr = _ensure_durations_in_hours(durations, cfg=cfg, name=name)
    pred_w = None
    try:
        if cfg is not None:
            try:
                pred_w = float(getattr(cfg, 'data').sequence_generation.prediction_window_hours)
            except Exception:
                try:
                    pred_w = float(cfg['data']['sequence_generation']['prediction_window_hours'])
                except Exception:
                    pred_w = None
        if pred_w is None:
            try:
                pred_w = float(getattr(_GLOBAL_CFG, 'data').sequence_generation.prediction_window_hours)
            except Exception:
                pred_w = None
    except Exception:
        pred_w = None
    if pred_w is not None:
        try:
            arr = _np.asarray(arr, dtype=float)
            arr = _np.minimum(_np.maximum(arr, 0.0), float(pred_w))
        except Exception:
            logger.warning(f"Failed clamping {name} to prediction window; leaving as-is")
    return arr


def _require_plotting_or_skip(func):
    def wrapper(*args, **kwargs):
        if not HAS_PLOTTING:
            logging.getLogger(__name__).warning(f"Skipping plotting function {func.__name__} because no plotting backend is available. Will save CSV/HTML fallbacks where possible.")
            # Attempt fallback behavior if function provides data saving (many functions call pandas/json saves internally). Otherwise return None.
            return func(*args, _allow_fallback=True, **kwargs) if 'kwargs' in locals() else None
        return func(*args, **kwargs)
    return wrapper

# Utility: save a matplotlib figure or use plotly to save an equivalent
def _save_figure(fig=None, save_path=None, dpi=200):
    """Save current matplotlib figure to save_path using plt.savefig.

    This helper assumes figure has been created via matplotlib plt and is active.
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.getLogger(__name__).warning("Cannot save figure: matplotlib not available")
        return False
    try:
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"_save_figure failed: {e}")
        try:
            plt.close()
        except Exception:
            pass
        return False

# Convenience: when matplotlib not available, helper to save simple line plots with plotly
def _plot_line_from_series(x, ys, labels, title, xlabel, ylabel, save_path, dpi=200):
    """Simple matplotlib line plot for multiple series and save with plt.savefig."""
    if not MATPLOTLIB_AVAILABLE:
        logging.getLogger(__name__).warning("Cannot plot line: matplotlib not available")
        return False
    try:
        plt.figure(figsize=(10, 6))
        for y, label in zip(ys, labels):
            plt.plot(list(x), list(y), marker='o', label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        ok = _save_figure(save_path=save_path, dpi=dpi)
        return ok
    except Exception as e:
        logging.getLogger(__name__).warning(f"_plot_line_from_series failed: {e}")
        try:
            plt.close()
        except Exception:
            pass
        return False

# Helper to ensure output dir exists; if not provided use DEFAULT_RESULTS_DIR/model_name
def _resolve_output_dir(output_dir, model_name):
    if not output_dir:
        output_dir = os.path.join(DEFAULT_RESULTS_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _parse_record_id(record_id):
    """从 record_id 或文件名中解析出信息。

    返回一个字典，包含可能的字段：
      - 'ar': 活动区标识符字符串，例如 'ar393' 或 'AR393'
      - 'start': 起始时间，datetime 对象（如果能解析）
      - 'end': 结束时间，datetime 对象（如果能解析）
      - 'flare_class': 如果文件名包含 M/X/B 等级则返回（例如 'M1.1'）
      - 'role': Primary/Secondary 等
      - 'raw': 原始 record_id 字符串

    解析逻辑：
      - 支持 tuple/list（优先使用其中的字符串/时间）
      - 支持文件名格式：M1.1@1332:Primary_ar393_s2011-02-27T16:24:00_e2011-02-28T04:12:00.csv
      - 支持多种分隔符和常见时间格式（T 或 空格，含秒）
    """
    info = {'ar': None, 'start': None, 'end': None, 'flare_class': None, 'role': None, 'raw': None}
    if record_id is None:
        return info
    # If record_id is already a metadata dict (from subsequence generator), accept it
    if isinstance(record_id, dict):
        # ensure keys exist and cast start/end to datetime if needed
        out = info.copy()
        out.update({k: record_id.get(k) for k in out.keys() if k in record_id})
        
        # 优先使用subseq_start作为start时间，如果没有则使用start
        if 'subseq_start' in record_id and record_id['subseq_start'] is not None:
            out['start'] = record_id['subseq_start']
        
        # coerce start/end if strings
        try:
            if out.get('start') is not None and not isinstance(out.get('start'), datetime):
                out['start'] = datetime.fromisoformat(str(out['start'])) if 'T' in str(out['start']) or ' ' in str(out['start']) else pd.to_datetime(out['start'])
        except Exception:
            out['start'] = out.get('start')
        try:
            if out.get('end') is not None and not isinstance(out.get('end'), datetime):
                out['end'] = datetime.fromisoformat(str(out['end'])) if 'T' in str(out['end']) or ' ' in str(out['end']) else pd.to_datetime(out['end'])
        except Exception:
            out['end'] = out.get('end')
        return out
    try:
        # Normalize to string
        if isinstance(record_id, (list, tuple)):
            # try to pick sensible values from tuple
            s = None
            for v in record_id:
                if isinstance(v, str):
                    s = v
                    break
            if s is None:
                s = str(record_id)
        else:
            s = str(record_id)
        info['raw'] = s

        # 1) 提取 flare class 前缀，如 M1.1 或 X2.3
        m_fc = re.search(r'\b([M|X|C|B]\d+(?:\.\d+)?)\b', s, flags=re.IGNORECASE)
        if m_fc:
            info['flare_class'] = m_fc.group(1)

        # 2) 提取 role (Primary/Secondary)
        m_role = re.search(r'\b(Primary|Secondary)\b', s, flags=re.IGNORECASE)
        if m_role:
            info['role'] = m_role.group(1)

        # 3) 提取 AR id (ar123 或 AR123) 或形如 '@4629_ar1999' 的复合模式
        # 优先提取包含数字的最后一组作为最具体的 AR 标识，并标准化为小写 'ar<digits>'。
        try:
            m_ar_all = re.findall(r'ar\D*?(\d+)', s, flags=re.IGNORECASE)
            if m_ar_all:
                info['ar'] = f'ar{int(m_ar_all[-1])}'
            else:
                m_ar = re.search(r'\b([aA][rR]?\d+)\b', s)
                if m_ar:
                    info['ar'] = m_ar.group(1).lower()
        except Exception:
            # fallback: leave as None
            pass

        # 4) 提取 start/end 时间戳，支持 T 或 空格，带或不带秒
        #    匹配 sYYYY-MM-DDTHH:MM:SS 或 sYYYY-MM-DD HH:MM:SS
        m_times = re.findall(r'([se])(?P<t>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', s)
        # m_times 为列表 of tuples like [('s', '2011-02-27T16:24:00'), ('e', '2011-02-28T04:12:00')]
        for prefix, timestr in m_times:
            try:
                dt = datetime.fromisoformat(timestr.replace('T', ' '))
            except Exception:
                try:
                    dt = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    dt = None
            if dt is None:
                continue
            if prefix == 's':
                info['start'] = dt
            elif prefix == 'e':
                info['end'] = dt

        # 5) 如果没有 s/e 格式，但存在两个 ISO 时间字符串，则第一为 start, 第二为 end
        if info['start'] is None or info['end'] is None:
            m_iso = re.findall(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', s)
            if len(m_iso) >= 2:
                try:
                    st = datetime.fromisoformat(m_iso[0].replace('T', ' '))
                    ed = datetime.fromisoformat(m_iso[1].replace('T', ' '))
                    if info['start'] is None:
                        info['start'] = st
                    if info['end'] is None:
                        info['end'] = ed
                except Exception:
                    pass

        # 6) as fallback, try to parse trailing timestamp digits like ..._20110227_162400
        if info['start'] is None:
            m_digits = re.search(r'(20\d{2}[01]\d[0-3]\d[_T\-]?\d{2}:?\d{2}:?\d{2})', s)
            if m_digits:
                ts = m_digits.group(1).replace('_', '').replace('-', '').replace(':', '')
                try:
                    dt = datetime.strptime(ts, '%Y%m%d%H%M%S')
                    info['start'] = dt
                except Exception:
                    pass

        return info
    except Exception:
        info['raw'] = str(record_id)
        return info

def auto_organize_results_by_model_times(output_dir):
    """自动将已有结果文件按照模型文件的修改时间搬入对应 fold 子目录。
    规则：查找 output_dir 下的 fold_{k}_model.pth 文件，按其 mtime 排序。对每个可移动的图像/数据文件，找到最近一个 mtime <= 文件mtime 的 model 文件并将该文件移动到对应的 fold_k 子目录。
    不会移动 experiment.log 与 config.json，且会跳过已经在 fold_* 子目录中的文件。
    """
    try:
        if not os.path.isdir(output_dir):
            return
        # find model files
        model_files = []
        for fn in os.listdir(output_dir):
            if fn.startswith('fold_') and fn.endswith('.pth'):
                path = os.path.join(output_dir, fn)
                m = re.match(r'fold_(\d+)_model\.pth', fn)
                if m:
                    fold = int(m.group(1))
                else:
                    # try extract digit
                    fold = int(re.findall(r'\d+', fn)[0]) if re.findall(r'\d+', fn) else 0
                model_files.append((path, fold, os.path.getmtime(path)))
        if not model_files:
            return
        # sort by mtime ascending
        model_files.sort(key=lambda x: x[2])
        # find candidate files in root to move
        candidates = []
        for fn in os.listdir(output_dir):
            full = os.path.join(output_dir, fn)
            if os.path.isdir(full):
                continue
            if fn in ('experiment.log', 'config.json'):
                continue
            if fn.endswith('.pth'):
                # skip model files
                continue
            candidates.append(full)
        for cand in candidates:
            mtime = os.path.getmtime(cand)
            assigned = None
            for path, fold, mt in reversed(model_files):
                if mt <= mtime:
                    assigned = fold
                    break
            if assigned is None:
                # if no model earlier, assign to latest fold
                assigned = model_files[-1][1]
            dest_dir = os.path.join(output_dir, f'fold_{assigned}')
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.move(cand, dest_dir)
            except Exception:
                # fallback to copy+remove
                try:
                    shutil.copy2(cand, dest_dir)
                    os.remove(cand)
                except Exception:
                    pass
    except Exception:
        logging.getLogger(__name__).warning('auto_organize_results_by_model_times failed', exc_info=True)

def plot_training_history(history, model_name, output_dir):
    """绘制训练和验证损失曲线"""
    logger = logging.getLogger(__name__)
    if history is None or (hasattr(history, 'empty') and history.empty):
        logger.warning("无法绘制训练历史：历史数据为空")
        return

    # Matplotlib path
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(12, 6))
            for metric in history.columns:
                if metric.startswith('val_'):
                    continue
                plt.plot(history[metric], label=f'Train {metric}')
                if f'val_{metric}' in history.columns:
                    plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{model_name} Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(output_dir, f'{model_name}_training_history.png')
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Training history plot saved to: {save_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib plotting failed: {e}")

    # Plotly support removed: always save CSV fallback for training history
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'{model_name}_training_history.csv')
        history.to_csv(csv_path, index=False)
        logger.info(f"Training history CSV saved to: {csv_path}")
        return
    except Exception as e:
        logger.warning(f"Failed to save training history CSV fallback: {e}")
        return

    logger.warning("无法绘制训练历史：没有可用的绘图后端")
    return
    """
    Publication-quality training curves: larger fonts, SVG output, annotated best validation epoch.
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("Skipping publication training curves: plotting backend unavailable")
        return
    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
        if val_losses:
            plt.plot(epochs[:len(val_losses)], val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'{model_name} - Training Loss', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.25)
        if val_losses and val_c_indices:
            try:
                best_idx = int(np.nanargmax(val_c_indices))
                best_val = val_c_indices[best_idx]
                if best_idx < len(val_losses):
                    plt.annotate(f'Best val C-index={best_val:.3f}', xy=(best_idx+1, val_losses[best_idx]), xytext=(best_idx+1, max(val_losses)+0.1*(max(val_losses)-min(train_losses))), arrowprops=dict(arrowstyle='->', color='black'))
            except Exception:
                pass
        plt.legend(fontsize=12)
        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, f'{model_name}_training_loss_pub.svg')
        png_path = os.path.join(output_dir, f'{model_name}_training_loss_pub.png')
        plt.tight_layout()
        plt.savefig(svg_path)
        plt.savefig(png_path, dpi=dpi)
        plt.close()
        logger.info(f'Publication training curves saved to: {svg_path} and {png_path}')
    except Exception as e:
        logger.warning(f'plot_publication_training_curves failed: {e}')


def plot_survival_curves_by_risk_group(y_test_df, risk_groups, model_name, output_dir, survival_funcs_df=None):
    logger = logging.getLogger(__name__)
    # Matplotlib path
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(1, 2, 1)
            kmf_high = KaplanMeierFitter()
            kmf_low = KaplanMeierFitter()
            high_risk_mask = (risk_groups == 'High Risk')
            low_risk_mask = (risk_groups == 'Low Risk')
            
            # 准备log-rank检验数据
            logrank_p_value = None
            logrank_statistic = None
            
            if high_risk_mask.any() and low_risk_mask.any():
                durations_high = y_test_df.loc[high_risk_mask, 'duration']
                events_high = y_test_df.loc[high_risk_mask, 'event']
                mask_high = np.isfinite(durations_high) & np.isfinite(events_high)
                durations_high = durations_high[mask_high]
                events_high = events_high[mask_high]
                durations_high = _ensure_durations_in_hours(np.asarray(durations_high), cfg=_GLOBAL_CFG, name='plot_survival_high')
                
                durations_low = y_test_df.loc[low_risk_mask, 'duration']
                events_low = y_test_df.loc[low_risk_mask, 'event']
                mask_low = np.isfinite(durations_low) & np.isfinite(events_low)
                durations_low = durations_low[mask_low]
                events_low = events_low[mask_low]
                durations_low = _ensure_durations_in_hours(np.asarray(durations_low), cfg=_GLOBAL_CFG, name='plot_survival_low')
                
                # 执行log-rank检验
                try:
                    from lifelines.statistics import logrank_test
                    logrank_result = logrank_test(durations_high, durations_low, events_high, events_low)
                    logrank_p_value = logrank_result.p_value
                    logrank_statistic = logrank_result.test_statistic
                    logger.info(f"Log-rank test: statistic={logrank_statistic:.4f}, p-value={logrank_p_value:.4f}")
                except Exception as e:
                    logger.warning(f"Log-rank test failed: {e}")
                
                # 拟合KM曲线
                kmf_high.fit(durations_high, events_high, label='High Risk (KM)')
                kmf_high.plot_survival_function(ax=ax1, ci_show=False)
                
                kmf_low.fit(durations_low, events_low, label='Low Risk (KM)')
                kmf_low.plot_survival_function(ax=ax1, ci_show=False)
            elif high_risk_mask.any():
                durations = y_test_df.loc[high_risk_mask, 'duration']
                events = y_test_df.loc[high_risk_mask, 'event']
                mask = np.isfinite(durations) & np.isfinite(events)
                durations = durations[mask]
                events = events[mask]
                durations = _ensure_durations_in_hours(np.asarray(durations), cfg=_GLOBAL_CFG, name='plot_survival_high')
                kmf_high.fit(durations, events, label='High Risk (KM)')
                kmf_high.plot_survival_function(ax=ax1, ci_show=False)
            elif low_risk_mask.any():
                durations = y_test_df.loc[low_risk_mask, 'duration']
                events = y_test_df.loc[low_risk_mask, 'event']
                mask = np.isfinite(durations) & np.isfinite(events)
                durations = durations[mask]
                events = events[mask]
                durations = _ensure_durations_in_hours(np.asarray(durations), cfg=_GLOBAL_CFG, name='plot_survival_low')
                kmf_low.fit(durations, events, label='Low Risk (KM)')
                kmf_low.plot_survival_function(ax=ax1, ci_show=False)
            
            # 设置标题，包含log-rank检验结果
            title = 'Kaplan-Meier Curves by Risk Group'
            if logrank_p_value is not None:
                significance = "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
                title += f'\nLog-rank test: p={logrank_p_value:.4f} ({significance})'
            
            ax1.set_title(title)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Survival Probability')
            ax1.legend()
            ax1.grid(True)
            ax2 = plt.subplot(1, 2, 2)
            if survival_funcs_df is not None and not survival_funcs_df.empty:
                ymins, ymaxs = [], []
                if high_risk_mask.any():
                    mean_surv_high = survival_funcs_df[high_risk_mask].mean(axis=0)
                    ax2.plot(mean_surv_high.index, mean_surv_high.values, label='High Risk (Pred Mean Survival)')
                    ymins.append(float(np.nanmin(mean_surv_high.values)))
                    ymaxs.append(float(np.nanmax(mean_surv_high.values)))
                if low_risk_mask.any():
                    mean_surv_low = survival_funcs_df[low_risk_mask].mean(axis=0)
                    ax2.plot(mean_surv_low.index, mean_surv_low.values, label='Low Risk (Pred Mean Survival)')
                    ymins.append(float(np.nanmin(mean_surv_low.values)))
                    ymaxs.append(float(np.nanmax(mean_surv_low.values)))
                ax2.set_title(f'Predicted Mean Survival')
                # Auto-scale y-axis to include all and add small margins
                if len(ymins) > 0:
                    y_min = max(0.0, min(ymins) - 0.02)
                    y_max = min(1.0, max(ymaxs) + 0.02)
                    ax2.set_ylim(y_min, y_max)
            else:
                ax2.text(0.5, 0.5, 'Not Available', horizontalalignment='center', verticalalignment='center')
                ax2.set_title(f'Prediction not provided')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Survival Probability')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{model_name}_survival_curves_by_risk_group.png')
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Survival curves by risk group saved to: {save_path}")
            
            # 保存log-rank检验结果到文件
            if logrank_p_value is not None:
                logrank_results = {
                    'logrank_statistic': float(logrank_statistic),
                    'logrank_p_value': float(logrank_p_value),
                    'significance': "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
                }
                import json
                logrank_path = os.path.join(output_dir, f'{model_name}_logrank_test_results.json')
                with open(logrank_path, 'w') as f:
                    json.dump(logrank_results, f, indent=2)
                logger.info(f"Log-rank test results saved to: {logrank_path}")
            
            return logrank_p_value, logrank_statistic
        except Exception as e:
            logger.warning(f"Matplotlib survival plotting failed: {e}")

    # Plotly support removed: CSV fallbacks handled below if matplotlib is not available

    # Fallback: save CSVs
    try:
        os.makedirs(output_dir, exist_ok=True)
        y_test_df.to_csv(os.path.join(output_dir, f'{model_name}_y_test.csv'), index=False)
        if survival_funcs_df is not None:
            survival_funcs_df.to_csv(os.path.join(output_dir, f'{model_name}_survival_funcs.csv'))
        pd.Series(risk_groups).to_csv(os.path.join(output_dir, f'{model_name}_risk_groups.csv'), index=False)
        logger.info(f"Saved survival data CSV fallbacks to: {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to save CSV fallback for survival plots: {e}")


def plot_publication_risk_distribution(risk_scores, events, model_name, output_dir, dpi=300):
    """
    Publication-style risk distribution: violin/KDE split by event type, medians annotated, optional Mann-Whitney p-value.
    """
    logger = logging.getLogger(__name__)
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        logger.warning('Skipping publication risk distribution: plotting backend unavailable')
        return
    try:
        try:
            from scipy.stats import mannwhitneyu, gaussian_kde
            has_mwu = True
        except Exception:
            has_mwu = False
        event_mask = np.asarray(events) == 1
        event_scores = np.asarray(risk_scores)[event_mask]
        censored_scores = np.asarray(risk_scores)[~event_mask]
        plt_local.figure(figsize=(8, 6))
        try:
            if sns is not None:
                data = pd.DataFrame({'risk': np.concatenate([event_scores, censored_scores]) if len(event_scores)+len(censored_scores)>0 else [], 'type': ['event']*len(event_scores) + ['censored']*len(censored_scores)})
                sns.violinplot(x='type', y='risk', data=data, palette=['#d62728', '#1f77b4'], inner=None)
                sns.stripplot(x='type', y='risk', data=data, color='k', size=2, jitter=0.15, alpha=0.4)
            else:
                xs = np.linspace(min(risk_scores), max(risk_scores), 200)
                if len(event_scores) > 1:
                    kde_e = gaussian_kde(event_scores)
                    plt_local.fill_between(xs, kde_e(xs), 0, alpha=0.4, color='#d62728', label='Events')
                if len(censored_scores) > 1:
                    kde_c = gaussian_kde(censored_scores)
                    plt_local.fill_between(xs, kde_c(xs), 0, alpha=0.4, color='#1f77b4', label='Censored')
        except Exception:
            plt_local.hist(event_scores, bins=30, density=True, alpha=0.6, color='#d62728', label='Events')
            plt_local.hist(censored_scores, bins=30, density=True, alpha=0.6, color='#1f77b4', label='Censored')
        med_e = np.median(event_scores) if len(event_scores) else np.nan
        med_c = np.median(censored_scores) if len(censored_scores) else np.nan
        plt_local.axvline(med_e, color='#800000', linestyle='--', linewidth=1.8, label=f'Events median={med_e:.3f}')
        plt_local.axvline(med_c, color='#003366', linestyle='--', linewidth=1.8, label=f'Censored median={med_c:.3f}')
        ptxt = ''
        if has_mwu and len(event_scores) > 0 and len(censored_scores) > 0:
            try:
                stat, p = mannwhitneyu(event_scores, censored_scores, alternative='two-sided')
                ptxt = f' Mann-Whitney p={p:.3e}'
            except Exception:
                ptxt = ''
        plt_local.xlabel('Risk Score', fontsize=13)
        plt_local.ylabel('Density', fontsize=13)
        plt_local.title(f'Risk Score Distribution{ptxt}', fontsize=15)
        plt_local.legend()
        plt_local.grid(alpha=0.2)
        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, f'{model_name}_risk_distribution_pub.svg')
        png_path = os.path.join(output_dir, f'{model_name}_risk_distribution_pub.png')
        plt_local.tight_layout()
        plt_local.savefig(svg_path)
        plt_local.savefig(png_path, dpi=dpi)
        plt_local.close()
        logger.info(f'Publication risk distribution plots saved to: {svg_path} and {png_path}')
    except Exception as e:
        logger.warning(f'plot_publication_risk_distribution failed: {e}')


def plot_feature_importance_barh(feature_importance, model_name, output_dir, top_k=30, dpi=300):
    """Horizontal sorted bar chart for feature importance suitable for papers."""
    logger = logging.getLogger(__name__)
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        logger.warning('Skipping feature importance plot: plotting backend unavailable')
        return
    try:
        if isinstance(feature_importance, dict):
            feature_importance = pd.Series(feature_importance)
        if isinstance(feature_importance, (list, np.ndarray)):
            feature_importance = pd.Series(feature_importance)
        fi = feature_importance.dropna()
        fi = fi.sort_values(ascending=True)
        if top_k and len(fi) > top_k:
            fi = fi[-top_k:]
        plt_local.figure(figsize=(8, max(4, 0.2 * len(fi))))
        plt_local.barh(fi.index.astype(str), fi.values, color='#2ca02c')
        plt_local.xlabel('Importance', fontsize=13)
        plt_local.yticks(fontsize=11)
        plt_local.title(f'Feature Importance', fontsize=15)
        plt_local.grid(axis='x', alpha=0.2)
        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, f'{model_name}_feature_importance_pub.svg')
        png_path = os.path.join(output_dir, f'{model_name}_feature_importance_pub.png')
        plt_local.tight_layout()
        plt_local.savefig(svg_path)
        plt_local.savefig(png_path, dpi=dpi)
        plt_local.close()
        logger.info(f'Publication feature importance saved to: {svg_path} and {png_path}')
    except Exception as e:
        logger.warning(f'plot_feature_importance_barh failed: {e}')

def plot_feature_importance_extended(feature_importance_dict, model_name, output_dir, top_k=30, dpi=300):
    """
    在 plot_feature_importance_barh 基础上，额外生成三张更有深度的可视化：
      1. 带数值标注、按重要性排序、使用 Viridis 色阶的增强水平柱状图
      2. 展示特征重要性分布的蜡烛/带状图（用 barh + 误差棒模拟散点分布感）
      3. Top-K 特征的雷达图

    Args:
        feature_importance_dict: Dict[str, float]  特征名 -> 重要性分数
        model_name: 模型名称（用于图表标题和文件名）
        output_dir: 图表保存目录
        top_k: 最多展示的特征数量
        dpi: 分辨率
    """
    logger = logging.getLogger(__name__)
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        return
    try:
        import numpy as _np
        import pandas as _pd
        import seaborn as _sns

        fi = _pd.Series(feature_importance_dict).dropna().sort_values(ascending=True)
        
        # Ensure we show all features if requested (top_k=0 or None), otherwise use top_k
        if top_k is None or top_k <= 0:
            top_k = len(fi)
            
        if len(fi) > top_k:
            fi = fi[-top_k:]

        if len(fi) == 0:
            return

        names = list(fi.index.astype(str))
        scores = list(fi.values)
        os.makedirs(output_dir, exist_ok=True)

        # ---- 1. 增强柱状图（Viridis 渐变色 + 数值标注）----
        try:
            # Adjust figure height based on number of features (approx 0.35 inch per feature)
            fig_h = max(5, len(names) * 0.35 + 1.5)
            fig, ax = plt_local.subplots(figsize=(10, fig_h))
            cmap = plt_local.cm.viridis
            colors = cmap(_np.linspace(0.15, 0.85, len(names)))
            bars = ax.barh(names, scores, color=colors, alpha=0.85)
            for bar in bars:
                w = bar.get_width()
                ax.text(w, bar.get_y() + bar.get_height() / 2,
                        f'  {w:.4f}', ha='left', va='center', fontsize=8.5)
            ax.set_xlabel('Importance Score (Permutation C-Index Drop)', fontsize=12)
            ax.set_title(f'Feature Importance', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.25, linestyle='--')
            plt_local.tight_layout()
            plt_local.savefig(os.path.join(output_dir, f'{model_name}_fi_ranked_annotated.png'), dpi=dpi, bbox_inches='tight')
            plt_local.close()
            logger.info(f'feature importance ranked annotated chart saved: {len(names)} features')
        except Exception as _e:
            logger.warning(f'fi_ranked_annotated failed: {_e}')
            plt_local.close()

        # ---- 2. Top-K 雷达图 ----
        top_k_radar = min(8, len(names))
        if top_k_radar >= 3:
            try:
                # 按重要性从大到小取 Top-K
                top_names = list(reversed(names))[:top_k_radar]
                top_scores = list(reversed(scores))[:top_k_radar]
                # 归一化到 [0, 1]
                max_s = max(top_scores) if max(top_scores) > 0 else 1.0
                top_scores_norm = [s / max_s for s in top_scores]

                angles = _np.linspace(0, 2 * _np.pi, top_k_radar, endpoint=False).tolist()
                angles += angles[:1]
                vals = top_scores_norm + top_scores_norm[:1]

                fig, ax = plt_local.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                ax.plot(angles, vals, 'o-', linewidth=2.0, color='#1f77b4', alpha=0.85)
                ax.fill(angles, vals, alpha=0.18, color='#1f77b4')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(top_names, fontsize=10)
                ax.set_title(f'{model_name} — Top {top_k_radar} Feature Importance Radar Chart', pad=20, fontsize=12, fontweight='bold')
                plt_local.tight_layout()
                plt_local.savefig(os.path.join(output_dir, f'{model_name}_fi_radar.png'), dpi=dpi, bbox_inches='tight')
                plt_local.close()
                logger.info('feature importance radar chart saved')
            except Exception as _e:
                logger.warning(f'fi_radar failed: {_e}')
                plt_local.close()

        # ---- 3. 归一化热力图（单方法，但展示每个特征绝对重要性色块）----
        try:
            fi_norm = fi / fi.max() if fi.max() > 0 else fi
            # 构造 DataFrame，列=model_name 便于未来多模型对比延伸
            df_heat = _pd.DataFrame({model_name: fi_norm.values}, index=fi_norm.index)
            fig, ax = plt_local.subplots(figsize=(max(3, 1.5), max(4, len(names) * 0.42)))
            _sns.heatmap(df_heat, annot=True, fmt='.3f', cmap='YlOrRd',
                        cbar_kws={'label': 'Normalized Importance'}, ax=ax)
            ax.set_title(f'{model_name} — Feature Importance Heatmap', fontsize=11, fontweight='bold')
            plt_local.tight_layout()
            plt_local.savefig(os.path.join(output_dir, f'{model_name}_fi_heatmap.png'), dpi=dpi, bbox_inches='tight')
            plt_local.close()
            logger.info('feature importance heatmap saved')
        except Exception as _e:
            logger.warning(f'fi_heatmap failed: {_e}')
            plt_local.close()

    except Exception as e:
        logger.warning(f'plot_feature_importance_extended failed: {e}')

def plot_survival_curves_by_risk_group_deephit(y_test_df, risk_groups, model_name, output_dir, survival_funcs_df):
    """
    为DeepHit模型绘制KM生存曲线和模型预测的平均生存曲线（阶梯图）。
    """
    logger = logging.getLogger(__name__)
    # ensure matplotlib is available at call time
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        logger.warning("无法绘制生存曲线：matplotlib 不可用，保存 CSV 备份数据")
        try:
            os.makedirs(output_dir, exist_ok=True)
            y_test_df.to_csv(os.path.join(output_dir, f'{model_name}_y_test_fallback.csv'), index=False)
            if survival_funcs_df is not None:
                survival_funcs_df.to_csv(os.path.join(output_dir, f'{model_name}_survival_funcs_fallback.csv'))
            pd.Series(risk_groups).to_csv(os.path.join(output_dir, f'{model_name}_risk_groups_fallback.csv'), index=False)
        except Exception as e:
            logger.warning(f"保存 CSV 备份数据失败: {e}")
        return

    plt_local.figure(figsize=(12, 6))

    # --- 1. 绘制 Kaplan-Meier 曲线 (与原函数相同) ---
    ax1 = plt.subplot(1, 2, 1)
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    high_risk_mask = (risk_groups == 'High Risk')
    low_risk_mask = (risk_groups == 'Low Risk')
    
    # 准备log-rank检验数据
    logrank_p_value = None
    logrank_statistic = None
    
    if high_risk_mask.any() and low_risk_mask.any():
        durations_high = y_test_df.loc[high_risk_mask, 'duration']
        events_high = y_test_df.loc[high_risk_mask, 'event']
        mask_high = np.isfinite(durations_high) & np.isfinite(events_high)
        durations_high = durations_high[mask_high]
        events_high = events_high[mask_high]
        
        durations_low = y_test_df.loc[low_risk_mask, 'duration']
        events_low = y_test_df.loc[low_risk_mask, 'event']
        mask_low = np.isfinite(durations_low) & np.isfinite(events_low)
        durations_low = durations_low[mask_low]
        events_low = events_low[mask_low]
        
        # 执行log-rank检验
        try:
            from lifelines.statistics import logrank_test
            logrank_result = logrank_test(durations_high, durations_low, events_high, events_low)
            logrank_p_value = logrank_result.p_value
            logrank_statistic = logrank_result.test_statistic
            logger.info(f"Log-rank test: statistic={logrank_statistic:.4f}, p-value={logrank_p_value:.4f}")
        except Exception as e:
            logger.warning(f"Log-rank test failed: {e}")
        
        # 拟合KM曲线
        kmf_high.fit(durations_high, events_high, label='High Risk (KM)')
        kmf_high.plot_survival_function(ax=ax1, ci_show=False)
        
        kmf_low.fit(durations_low, events_low, label='Low Risk (KM)')
        kmf_low.plot_survival_function(ax=ax1, ci_show=False)
    elif high_risk_mask.any():
        durations = y_test_df.loc[high_risk_mask, 'duration']
        events = y_test_df.loc[high_risk_mask, 'event']
        mask = np.isfinite(durations) & np.isfinite(events)
        durations = durations[mask]
        events = events[mask]
        kmf_high.fit(durations, events, label='High Risk (KM)')
        kmf_high.plot_survival_function(ax=ax1, ci_show=False)
    elif low_risk_mask.any():
        durations = y_test_df.loc[low_risk_mask, 'duration']
        events = y_test_df.loc[low_risk_mask, 'event']
        mask = np.isfinite(durations) & np.isfinite(events)
        durations = durations[mask]
        events = events[mask]
        kmf_low.fit(durations, events, label='Low Risk (KM)')
        kmf_low.plot_survival_function(ax=ax1, ci_show=False)
    
    # 设置标题，包含log-rank检验结果
    title = 'Kaplan-Meier Curves by Risk Group'
    if logrank_p_value is not None:
        significance = "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
        title += f'\nLog-rank test: p={logrank_p_value:.4f} ({significance})'
    
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Survival Probability')
    ax1.legend()
    ax1.grid(True)

    # --- 2. 绘制模型预测的平均生存曲线 (阶梯图) ---
    ax2 = plt.subplot(1, 2, 2)
    if not survival_funcs_df.empty:
        ymins, ymaxs = [], []
        if high_risk_mask.any():
            mean_surv_high = survival_funcs_df[high_risk_mask].mean(axis=0)
            ax2.step(mean_surv_high.index.to_numpy(), mean_surv_high.values, where='post', label='High Risk (Pred Mean Survival)')
            ymins.append(float(np.nanmin(mean_surv_high.values)))
            ymaxs.append(float(np.nanmax(mean_surv_high.values)))
        if low_risk_mask.any():
            mean_surv_low = survival_funcs_df[low_risk_mask].mean(axis=0)
            ax2.step(mean_surv_low.index.to_numpy(), mean_surv_low.values, where='post', label='Low Risk (Pred Mean Survival)')
            ymins.append(float(np.nanmin(mean_surv_low.values)))
            ymaxs.append(float(np.nanmax(mean_surv_low.values)))
        ax2.set_title(f'{model_name} - Predicted Mean Survival (Discrete)')
        if len(ymins) > 0:
            y_min = max(0.0, min(ymins) - 0.02)
            y_max = min(1.0, max(ymaxs) + 0.02)
            ax2.set_ylim(y_min, y_max)
    else:
        ax2.text(0.5, 0.5, 'Not Available', horizontalalignment='center', verticalalignment='center')
        ax2.set_title(f'{model_name} - Prediction not provided')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Survival Probability')
    ax2.legend()
    ax2.grid(True)

    plt_local.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_survival_curves_by_risk_group.png')
    try:
        plt_local.savefig(save_path)
        plt_local.close()
        logger.info(f"DeepHit survival curves by risk group saved to: {save_path}")
        
        # 保存log-rank检验结果到文件
        if logrank_p_value is not None:
            logrank_results = {
                'logrank_statistic': float(logrank_statistic),
                'logrank_p_value': float(logrank_p_value),
                'significance': "***" if logrank_p_value < 0.001 else "**" if logrank_p_value < 0.01 else "*" if logrank_p_value < 0.05 else "ns"
            }
            import json
            logrank_path = os.path.join(output_dir, f'{model_name}_logrank_test_results.json')
            with open(logrank_path, 'w') as f:
                json.dump(logrank_results, f, indent=2)
            logger.info(f"Log-rank test results saved to: {logrank_path}")
        
        return logrank_p_value, logrank_statistic
    except Exception as e:
        logger.warning(f"保存 DeepHit 生存曲线图片失败: {e}")
        return None, None

def plot_deephit_probability_distribution(pred_probs, model_name, output_dir, fold=None):
    """
    为DeepHit模型的预测绘制概率分布。
    Args:
        pred_probs (np.ndarray): 模型的概率输出，形状为 (n_samples, n_time_bins) 或 (n_samples, n_events, n_time_bins)。
        model_name (str): 模型的名称。
        output_dir (str): 保存绘图的目录。
    """
    if MATPLOTLIB_AVAILABLE:
        try:
            arr = np.array(pred_probs)
            if arr.ndim == 2:
                n_samples, n_bins = arr.shape
                event_to_plot = 0
                sample_indices = np.random.choice(n_samples, size=min(n_samples, 5), replace=False)
                ys = [arr[idx, :].tolist() for idx in sample_indices]
                labels = [f'Sample {idx}' for idx in sample_indices]
                x = list(range(arr.shape[-1]))
                fname = f"{model_name}_probability_distribution"
                if fold is not None:
                    fname += f"_fold{fold}"
                fname += '.png'
                save_path = os.path.join(output_dir, fname)
                _plot_line_from_series(x, ys, labels, f'{model_name} - Probability Distribution', 'Time Bins', 'Probability', save_path)
                return
            elif arr.ndim == 3:
                n_samples, n_events, n_bins = arr.shape
                sample_indices = np.random.choice(n_samples, size=min(n_samples, 5), replace=False)
                ys = [arr[idx, 0, :].tolist() for idx in sample_indices]
                labels = [f'Sample {idx}' for idx in sample_indices]
                x = list(range(n_bins))
                fname = f"{model_name}_probability_distribution"
                if fold is not None:
                    fname += f"_fold{fold}"
                fname += '.png'
                save_path = os.path.join(output_dir, fname)
                _plot_line_from_series(x, ys, labels, f'{model_name} - Probability Distribution', 'Time Bins', 'Probability', save_path)
                return
        except Exception as e:
            logging.getLogger(__name__).warning(f"Plotly deephit probability plotting failed: {e}")
    else:
        logging.getLogger(__name__).warning("Cannot plot deephit probability distribution: no plotting backend available")
        # save raw data
        try:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{model_name}_probability_distribution.npy"), pred_probs)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save deephit probability fallback data: {e}")


def plot_individual_survival_curves(survival_funcs_df, y_test_df, model_name, output_dir, n_samples=10, is_discrete=False, record_ids=None, fold=None):
    """
    绘制个体生存曲线，并在每个样本的生存曲线上标注真实事件时间（如果有），图例显示record_id。
    合并了原individual_survival_curves和survival_examples的功能。
    """
    logger = logging.getLogger(__name__)
    if MATPLOTLIB_AVAILABLE:
        try:
            # 选择更多样本进行展示，平衡事件和删失样本
            plt.figure(figsize=(12, 8))
            # determine candidate indices
            total_n = len(survival_funcs_df)
            ev_idx = [i for i, e in enumerate(y_test_df['event'].values) if int(e) == 1]
            cens_idx = [i for i, e in enumerate(y_test_df['event'].values) if int(e) == 0]
            chosen = []
            
            # 改进的选择策略：平衡事件和删失样本，最多选择6个样本
            max_samples = min(6, total_n)
            n_ev = min(len(ev_idx), max_samples // 2)
            n_cens = max_samples - n_ev
            
            # 选择事件样本
            if len(ev_idx) > 0 and n_ev > 0:
                chosen.extend(list(np.random.choice(ev_idx, size=n_ev, replace=False)))
            
            # 选择删失样本
            if len(cens_idx) > 0 and n_cens > 0:
                chosen.extend(list(np.random.choice(cens_idx, size=min(n_cens, len(cens_idx)), replace=False)))
            
            # fallback: if still empty, pick first up to max_samples
            if not chosen:
                chosen = list(range(min(max_samples, total_n)))

            times = survival_funcs_df.columns.astype(float)
            durations_arr = _ensure_durations_in_hours(y_test_df['duration'].values, cfg=_GLOBAL_CFG, name='y_test_df.duration')
            cmap = plt.get_cmap('tab10')
            ax = plt.gca()
            for i, idx in enumerate(chosen):
                surv = survival_funcs_df.iloc[idx].values
                mask = np.isfinite(surv) & np.isfinite(times)
                svals = surv[mask]
                tvals = times[mask]
                color = cmap(i % 10)
                # 构建图例标签，包含索引、AR信息和事件状态
                ev = int(y_test_df.iloc[idx]['event']) if 'event' in y_test_df.columns else 0
                event_status = "Event" if ev == 1 else "Censored"
                lbl = f'Idx {idx} ({event_status})'
                
                if record_ids is not None and len(record_ids) > idx:
                    try:
                        info = _parse_record_id(record_ids[idx])
                        ar = info.get('ar') or info.get('raw')
                        start_dt = info.get('start')
                        lbl += f' | AR:{ar}'
                        if start_dt is not None:
                            lbl += f' | {start_dt.strftime("%m-%d %H:%M")}'
                    except Exception:
                        pass
                # 改为生存曲线 S(t)
                ax.plot(tvals, svals, color=color, linewidth=2, label=lbl)
                # annotate event time for event samples
                ev = int(y_test_df.iloc[idx]['event']) if 'event' in y_test_df.columns else 0
                if ev == 1:
                    dur = durations_arr[idx] if idx < len(durations_arr) else None
                    if dur is not None and np.isfinite(dur):
                        ax.axvline(x=dur, color=color, linestyle='--', alpha=0.8)
                        # small text near the bottom aligned with the line
                        ax.text(dur, 0.05 + 0.02*i, f'Event {dur:.1f}h (Idx {idx})', color=color, rotation=90, va='bottom', ha='right', fontsize=8)
            ax.set_title(f'Individual Survival Curves (Event & Censored)')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Survival Probability')
            ax.set_xlim(left=0)
            # Auto-scale y-axis: if survival curves are concentrated near 1, zoom into top
            try:
                sel_surv_matrix = survival_funcs_df.iloc[chosen].values if len(chosen) > 0 else np.array([])
                sel_surv_matrix = np.asarray(sel_surv_matrix, dtype=float)
                if sel_surv_matrix.size > 0:
                    finite_mask = np.isfinite(sel_surv_matrix)
                    if finite_mask.any():
                        cur_min = float(np.nanmin(sel_surv_matrix[finite_mask]))
                        cur_max = float(np.nanmax(sel_surv_matrix[finite_mask]))
                        # 生存概率通常在 [0,1]，给出小幅边距
                        bottom = max(0.0, cur_min - 0.02)
                        top = min(1.02, cur_max + 0.02)
                        ax.set_ylim(bottom, top)
                    else:
                        ax.set_ylim(0, 1.02)
                else:
                    ax.set_ylim(0, 1.02)
            except Exception:
                ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
            fname = f'{model_name}_individual_survival_curves'
            if fold is not None:
                fname += f'_fold{fold}'
            fname += '.png'
            save_path = os.path.join(output_dir, fname)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Individual survival curves saved to: {save_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib individual survival plotting failed: {e}")
    # Plotly support removed: save CSV fallback for individual survival curves
    try:
        os.makedirs(output_dir, exist_ok=True)
        survival_funcs_df.iloc[:n_samples].to_csv(os.path.join(output_dir, f'{model_name}_individual_survival_curves.csv'))
        logger.info(f"Saved survival curves CSV to: {output_dir}")
        return
    except Exception as e:
        logger.warning(f"Failed to save survival curves CSV fallback: {e}")
    else:
        logger.warning("无法绘制个体生存曲线：缺少绘图支持")
        try:
            os.makedirs(output_dir, exist_ok=True)
            survival_funcs_df.iloc[:n_samples].to_csv(os.path.join(output_dir, f'{model_name}_individual_survival_curves.csv'))
            logger.info(f"Saved survival curves CSV to: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save survival curves CSV fallback: {e}")


def plot_roc_curves(roc_results, output_dir, model_name="Model", figsize=(12, 8)):
    logger = logging.getLogger(__name__)
    if MATPLOTLIB_AVAILABLE:
        # ...existing matplotlib code...
        try:
            plt.figure(figsize=figsize)
            colors = plt.cm.viridis(np.linspace(0, 1, len(roc_results)))
            for i, (time_point, roc_data) in enumerate(roc_results.items()):
                if roc_data['fpr'] is not None and roc_data['tpr'] is not None:
                    plt.plot(roc_data['fpr'], roc_data['tpr'], color=colors[i], label=f't={time_point:.1f} (AUC={roc_data["auc"]:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'Time-Dependent ROC Curves', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'roc_curves_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"ROC曲线已保存到: {output_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib ROC plotting failed: {e}")
    # Plotly support removed: save CSV fallbacks for ROC curves
    try:
        os.makedirs(output_dir, exist_ok=True)
        for t, roc_data in roc_results.items():
            if roc_data.get('fpr') is not None and roc_data.get('tpr') is not None:
                pd.DataFrame({'fpr': roc_data['fpr'], 'tpr': roc_data['tpr']}).to_csv(os.path.join(output_dir, f'roc_curve_t{t:.2f}_{model_name}.csv'), index=False)
        logger.info(f"Saved ROC CSVs to {output_dir}")
        return
    except Exception as e:
        logger.warning(f"Failed to save ROC CSV fallbacks: {e}")
    else:
        logger.warning("无法绘制ROC曲线：缺少绘图支持")


def plot_ar_subsample_colormap(survival_funcs_df, y_test_df, record_ids, ar_id, model_name, output_dir, fold=None, cmap_name='plasma'):
    """
    为一个指定 AR（活动区）绘制该 AR 的所有子样本的生存曲线集合，并按样本距离事件的时间排序，用颜色映射从黄->红->紫（远->近->过事件后）。

    Args:
        survival_funcs_df: DataFrame, each行是样本, 每列为时间点
        y_test_df: DataFrame with 'duration' and 'event'
        record_ids: list-like of record metadata (dicts or strings)
        ar_id: string AR id to filter (e.g., 'ar393')
        model_name, output_dir, fold: saving params
    """
    logger = logging.getLogger(__name__)
    try:
        # 集中filter: 找到属于 ar_id 的索引
        ar_indices = []
        for idx, rid in enumerate(record_ids):
            try:
                info = _parse_record_id(rid)
                if info.get('ar') is not None and str(info.get('ar')).lower() == str(ar_id).lower():
                    ar_indices.append(idx)
            except Exception:
                continue

        if not ar_indices:
            logger.warning(f'No samples found for AR {ar_id}')
            return

        # 提取对应的生存曲线与事件时间
        subset_surv = survival_funcs_df.iloc[ar_indices]
        subset_y = y_test_df.iloc[ar_indices]
        # 计算距离事件的 signed distance: 如果事件==1, distance = duration (正，时间到事件前)；如果 event==0 (censored), distance = duration * -1 (表示右侧)
        distances = []
        for i, row in subset_y.iterrows():
            # 优先使用明确的小时字段，向后兼容 'duration'
            if 'duration_hours' in row and not pd.isna(row.get('duration_hours')):
                d = float(row.get('duration_hours', np.nan))
            else:
                d = float(row.get('duration', np.nan))
            ev = int(row.get('event', 0))
            # 为可视化，我们希望距离事件由大到小，并把发生后（duration < 0 不应出现）单独标注
            # 这里distance定义为: time_to_event = duration if ev==1 else duration (still positive)
            distances.append(d)
        distances = np.array(distances)

        # 排序：按 distance 从大到小（最远在前），但事件发生的样本可被着色为更暖色
        order = np.argsort(distances)[::-1]
        ordered_surv = subset_surv.iloc[order]
        ordered_dist = distances[order]

        # 构建颜色映射：将距离归一化到[0,1]，然后使用 matplotlib colormap（如果不可用，使用简单渐变）
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            cmap = cm.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=ordered_dist.min(), vmax=ordered_dist.max())
            colors = [cmap(norm(v)) for v in ordered_dist]
        except Exception:
            # fallback simple mapping: yellow->red->purple using linear mix
            colors = []
            for v in ordered_dist:
                frac = 0.0 if ordered_dist.max()==ordered_dist.min() else (v - ordered_dist.min()) / (ordered_dist.max() - ordered_dist.min())
                # fraction 0->1 maps yellow->red->purple
                if frac < 0.5:
                    # yellow->red
                    r = frac*2
                    g = 1 - frac*2
                    b = 0.2
                else:
                    # red->purple
                    r = 1 - (frac-0.5)*2*0.2
                    g = 0.0
                    b = (frac-0.5)*2
                colors.append((r, g, b))

        # 绘图
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 8))
            times = ordered_surv.columns.astype(float)
            # 自适应纵轴缩放：累计风险曲线通常集中在低值，计算该AR下的上界
            cumrisk_all = 1.0 - np.asarray(ordered_surv.values, dtype=float)
            finite_mask = np.isfinite(cumrisk_all)
            ymax = 1.0
            if cumrisk_all.size > 0 and finite_mask.any():
                try:
                    q95 = float(np.nanpercentile(cumrisk_all[finite_mask], 95))
                except Exception:
                    q95 = float(np.nanmax(cumrisk_all[finite_mask]))
                ymax = max(0.1, min(1.0, q95 * 1.1))
            for idx_row, (_, row) in enumerate(ordered_surv.iterrows()):
                cumrisk = 1.0 - np.asarray(row.values, dtype=float)
                plt.plot(times, cumrisk, color=colors[idx_row], alpha=0.9)
            plt.title(f'{model_name} - AR {ar_id} Subsample Cumulative Risk Curves (colored by time-to-event)')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Risk')
            try:
                plt.ylim(0, ymax)
            except Exception:
                pass
            sm = None
            try:
                import matplotlib as mpl
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                plt.colorbar(sm, label='Time-to-event (hours)')
            except Exception:
                pass
            fname = f'{model_name}_AR_{ar_id}_subsamples'
            if fold is not None:
                fname += f'_fold{fold}'
            fname += '.png'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
            plt.close()
            logger.info(f'Saved AR subsample colormap to {os.path.join(output_dir, fname)}')
            return

    # Plotly removed: directly save CSV fallback (below) if matplotlib path not used

        # fallback save CSV
        try:
            os.makedirs(output_dir, exist_ok=True)
            ordered_surv.to_csv(os.path.join(output_dir, f'{model_name}_AR_{ar_id}_subsamples.csv'))
            logger.info(f'Saved AR subsample CSV to {output_dir}')
        except Exception as e:
            logger.warning(f'Failed to save AR subsample CSV fallback: {e}')

    except Exception as e:
        logger.warning(f'plot_ar_subsample_colormap failed: {e}', exc_info=True)


def plot_auc_over_time(auc_scores, output_dir, model_name="Model", figsize=(10, 6), mean_ci=None, time_stats=None):
    logger = logging.getLogger(__name__)
    if MATPLOTLIB_AVAILABLE:
        # ...existing matplotlib code...
        try:
            plt.figure(figsize=figsize)
            times = sorted(auc_scores.keys())
            aucs = [auc_scores[t] for t in times]
            plt.plot(times, aucs, 'b-o', linewidth=2, markersize=6, label=f'Tim-SFSA')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
            # 如果提供了 mean_ci，则绘制均值带阴影
            if mean_ci is not None:
                low, high = mean_ci
                plt.fill_between([min(times), max(times)], low, high, color='r', alpha=0.12, label=f'Mean AUC 95% CI [{low:.3f},{high:.3f}]')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('AUC', fontsize=12)
            plt.title(f'AUC over Time', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0.0, 1.0)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'auc_over_time_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"AUC时间曲线已保存到: {output_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib AUC plotting failed: {e}")
    # Plotly support removed: save CSV fallback for AUC over time
    try:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({'time': list(auc_scores.keys()), 'auc': list(auc_scores.values())}).to_csv(os.path.join(output_dir, f'auc_over_time_{model_name.lower().replace(" ", "_")}.csv'), index=False)
        logger.info(f"Saved AUC over time CSV to {output_dir}")
        return
    except Exception as e:
        logger.warning(f"Failed to save AUC over time CSV fallback: {e}")
    else:
        logger.warning("无法绘制AUC时间曲线：缺少绘图支持")


def plot_roc_analysis(predictions, durations, events, output_dir, model_name="Model", 
                     time_points=None, figsize=(15, 10)):
    """
    综合ROC分析，包括ROC曲线、AUC时间曲线和性能指标。
    
    Args:
        predictions (np.array): 模型预测的风险分数
        durations (np.array): 事件时间
        events (np.array): 事件指示器
        output_dir (str): 输出目录
        model_name (str): 模型名称
        time_points (list): 时间点列表
        figsize (tuple): 图形大小
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制ROC分析图：缺少绘图支持")
        return
    
    try:
        from .metrics import calculate_time_dependent_roc, calculate_auc_at_multiple_times, calculate_integrated_auc
        
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
        
        # 计算ROC相关指标
        roc_results = calculate_time_dependent_roc(predictions, durations, events, time_points)
        auc_scores = calculate_auc_at_multiple_times(predictions, durations, events, time_points)
        integrated_auc = calculate_integrated_auc(predictions, durations, events, time_points)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'ROC Analysis - {model_name}', fontsize=16)
        
        # 1. ROC曲线
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(roc_results)))
        for i, (time_point, roc_data) in enumerate(roc_results.items()):
            if roc_data['fpr'] is not None and roc_data['tpr'] is not None:
                ax1.plot(roc_data['fpr'], roc_data['tpr'], 
                        color=colors[i], 
                        label=f't={time_point:.1f} (AUC={roc_data["auc"]:.3f})',
                        linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Time-Dependent ROC Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. AUC时间曲线
        ax2 = axes[0, 1]
        times = sorted(auc_scores.keys())
        aucs = [auc_scores[t] for t in times]
        ax2.plot(times, aucs, 'b-o', linewidth=2, markersize=6)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('AUC')
        ax2.set_title(f'AUC over Time (iAUC={integrated_auc:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.0)
        
        # 3. 风险分数分布
        ax3 = axes[1, 0]
        event_mask = events == 1
        ax3.hist(predictions[~event_mask], bins=30, alpha=0.7, label='Censored', density=True)
        ax3.hist(predictions[event_mask], bins=30, alpha=0.7, label='Events', density=True)
        ax3.set_xlabel('Risk Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Risk Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标总结（增加 C-index 如可计算）
        ax4 = axes[1, 1]
        ax4.axis('off')
        # 计算统计信息
        mean_auc = np.mean(list(auc_scores.values()))
        std_auc = np.std(list(auc_scores.values()))
        min_auc = np.min(list(auc_scores.values()))
        max_auc = np.max(list(auc_scores.values()))

        # 尝试计算C-index（如果可用的实现）
        cindex_txt = 'N/A'
        try:
            # prefer lifelines concordance_index if available
            try:
                from lifelines.utils import concordance_index
                cidx = concordance_index(durations, -predictions, events)
                cindex_txt = f'{cidx:.3f}'
            except Exception:
                # fallback: try sklearn's concordance index implementation if present in metrics module
                try:
                    from .metrics import concordance_index as ci_fn
                    cidx = ci_fn(predictions, durations, events)
                    cindex_txt = f'{cidx:.3f}'
                except Exception:
                    cindex_txt = 'N/A'
        except Exception:
            cindex_txt = 'N/A'

        metrics_text = f"""
        ROC Analysis Summary

        Integrated AUC: {integrated_auc:.3f}
        Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}
        AUC Range: [{min_auc:.3f}, {max_auc:.3f}]
        C-index: {cindex_txt}

        Sample Statistics:
        Total Samples: {len(predictions)}
        Events: {int(np.sum(events))}
        Event Rate: {np.mean(events):.3f}

        Risk Score Statistics:
        Mean: {np.mean(predictions):.3f}
        Std: {np.std(predictions):.3f}
        Min: {np.min(predictions):.3f}
        Max: {np.max(predictions):.3f}
        """

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'roc_analysis_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"ROC分析图已保存到: {output_path}")
        
    except Exception as e:
        logging.error(f"绘制ROC分析时出错: {e}")
        return None
    
def plot_risk_progression(risk_scores, y_true, output_path, fold=None):
    """
    绘制样本风险进展图，标注删失/事件，事件样本标记真实事件时间。
    Args:
        risk_scores: 风险分数数组
        y_true: 标签数组，[:,0]=时间, [:,1]=事件(1=事件,0=删失)
        output_path: 输出文件路径（不含fold编号）
        fold: 当前fold编号（可选）
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制风险进展图：缺少绘图支持")
        return
    
    plt.figure(figsize=(12, 8))
    # 分离事件和删失样本
    event_mask = y_true[:, 1] == 1
    censored_mask = y_true[:, 1] == 0
    # 绘制删失样本（蓝色水平线）
    for i in np.where(censored_mask)[0]:
        duration, event = y_true[i]
        risk = risk_scores[i]
        # 过滤nan/inf
        if not (np.isfinite(duration) and np.isfinite(risk)):
            continue
        plt.plot([0, duration], [risk, risk], color='blue', alpha=0.6, linewidth=1.5)
        plt.scatter([duration], [risk], color='blue', marker='o', s=30, alpha=0.8)
    # 绘制事件样本（红色水平线）
    for i in np.where(event_mask)[0]:
        duration, event = y_true[i]
        risk = risk_scores[i]
        if not (np.isfinite(duration) and np.isfinite(risk)):
            continue
        plt.plot([0, duration], [risk, risk], color='red', alpha=0.8, linewidth=2)
        plt.scatter([duration], [risk], color='red', marker='x', s=80, linewidth=3, alpha=0.9)
    plt.scatter([], [], color='blue', marker='o', s=30, label='censored samples', alpha=0.8)
    plt.scatter([], [], color='red', marker='x', s=80, linewidth=3, label='event samples', alpha=0.9)
    plt.xlabel('Time (hours)')
    plt.ylabel('Risk Score')
    title = 'Risk Progression Plot'
    if fold is not None:
        title += f' (Fold {fold})'
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    if fold is not None:
        output_path = output_path.replace('.png', f'_fold{fold}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"风险进展图已保存到: {output_path}")

def plot_event_probability_curves(pred_probs, durations, events, record_ids, model_name, output_dir, n_samples=5, fold=None):
    """
    绘制事件概率随时间变化的曲线，支持活动区和事件点标注。
    pred_probs: (n_samples, n_time_bins)
    durations: (n_samples,)
    events: (n_samples,)
    record_ids: (n_samples,)
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制事件概率曲线：缺少绘图支持")
        return
    
    logger = logging.getLogger(__name__)
    
    # 检查pred_probs的形状
    if pred_probs.ndim == 1:
        logger.warning("pred_probs是一维数组，无法绘制事件概率曲线")
        return
    
    n_total = pred_probs.shape[0]
    sample_indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    # build time axis in hours matching prediction window
    try:
        cfg = get_config()
        total_hours = float(cfg.data.sequence_generation.prediction_window_hours)
    except Exception:
        total_hours = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
    
    # 确保pred_probs有正确的形状
    if pred_probs.shape[1] <= 0:
        logger.warning("pred_probs的时间维度无效，无法绘制事件概率曲线")
        return
        
    time_bins = np.linspace(0.0, total_hours, pred_probs.shape[1])
    # normalize durations to hours
    durations = _durations_in_hours_clamped(durations, cfg=_GLOBAL_CFG, name='durations')
    # 1) 组合图（便于概览）
    plt.figure(figsize=(15, 3 * len(sample_indices)))
    for i, idx in enumerate(sample_indices):
        ax = plt.subplot(len(sample_indices), 1, i + 1)
        ax.plot(time_bins, pred_probs[idx], marker='o', label=f'Sample {idx}')
        ar_str = f"AR: {record_ids[idx]}" if record_ids is not None else ""
        title = f"Sample {idx} | {ar_str}"
        if events[idx] == 1:
            try:
                dur = float(durations[idx]) if idx < len(durations) else None
            except Exception:
                dur = None
            if dur is not None and np.isfinite(dur):
                ax.axvline(x=dur, color='red', linestyle='--', alpha=0.7, label=f'Actual Event Time: {dur:.2f}')
        ax.set_title(title)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Event Probability')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    fname = f'{model_name}_event_probability_curves'
    if fold is not None:
        fname += f'_fold{fold}'
    fname += '.png'
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Event probability curves saved to: {save_path}")

    # 2) 分开画：为每个选中样本单独保存一张图
    try:
        for idx in sample_indices:
            fig = plt.figure(figsize=(8, 4))
            ax = plt.gca()
            ax.plot(time_bins, pred_probs[idx], marker='o', label=f'Sample {idx}')
            if events[idx] == 1:
                try:
                    dur = float(durations[idx]) if idx < len(durations) else None
                except Exception:
                    dur = None
                if dur is not None and np.isfinite(dur):
                    ax.axvline(x=dur, color='red', linestyle='--', alpha=0.7, label=f'Actual Event Time: {dur:.2f}')
            ar_str = f"AR: {record_ids[idx]}" if record_ids is not None else ""
            ax.set_title(f"Sample {idx} | {ar_str}")
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Event Probability')
            ax.legend()
            ax.grid(True)
            fname_single = f"{model_name}_event_prob_sample_{idx}"
            if fold is not None:
                fname_single += f"_fold{fold}"
            fname_single += ".png"
            save_single = os.path.join(output_dir, fname_single)
            plt.tight_layout()
            plt.savefig(save_single)
            plt.close(fig)
        logger.info("Saved individual event probability curve images for selected samples.")
    except Exception as e:
        logger.warning(f"Failed to save separate event probability figures: {e}")


def plot_cumulative_risk_examples(survival_funcs_df, durations, events, record_ids, model_name, output_dir, n_examples=6, fold=None):
    """
    改为绘制生存曲线 S(t) 示例（替换原累计风险）。选择 4-8 个样本，事件/删失各占一半（若可用）。
    对事件样本，用与曲线相同颜色的虚线标出真实事件时间；在图例显示 AR 信息。
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制累计风险示例：缺少绘图支持")
        return
    # select samples (prefer balanced events/censored) and plot all on one axis for comparison
    n = min(n_examples, len(survival_funcs_df))
    ev_idx = [i for i, e in enumerate(events) if int(e) == 1]
    cens_idx = [i for i, e in enumerate(events) if int(e) == 0]
    chosen = []
    # prefer half events, half censored when available
    n_ev = min(len(ev_idx), n // 2)
    n_cens = n - n_ev
    if n_ev > 0:
        chosen.extend(list(np.random.choice(ev_idx, size=n_ev, replace=False)))
    if n_cens > 0 and len(cens_idx) > 0:
        chosen.extend(list(np.random.choice(cens_idx, size=min(n_cens, len(cens_idx)), replace=False)))
    if not chosen:
        logger.warning('No samples selected for cumulative risk examples')
        return

    times = survival_funcs_df.columns.astype(float)
    durations = _durations_in_hours_clamped(durations, cfg=_GLOBAL_CFG, name='durations')
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    ax = plt.gca()
    for i, idx in enumerate(chosen):
        surv = survival_funcs_df.iloc[idx].values
        color = cmap(i % 10)
        ax.plot(times, surv, color=color, linewidth=2, label=f'Idx {idx}')
        if int(events[idx]) == 1:
            try:
                t = float(durations[idx])
            except Exception:
                t = None
            if t is not None and np.isfinite(t):
                ax.axvline(x=t, color=color, linestyle='--', alpha=0.8)
        rid = record_ids[idx] if record_ids is not None else None
        info = _parse_record_id(rid) if rid is not None else {}
        ar = info.get('ar') or info.get('raw') or str(rid)
        # update last plotted line's label to include AR
        ax.lines[-1].set_label(f'Idx {idx} | AR:{ar}')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Survival Probability')
    try:
        sel = survival_funcs_df.iloc[chosen].values.astype(float)
        finite = np.isfinite(sel)
        if finite.any():
            y_min = max(0.0, float(np.nanmin(sel[finite])) - 0.02)
            y_max = min(1.0, float(np.nanmax(sel[finite])) + 0.02)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.02)
    except Exception:
        ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fname = f'{model_name}_survival_examples'
    if fold is not None:
        fname += f'_fold{fold}'
    fname += '.png'
    save_path = os.path.join(output_dir, fname)
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Cumulative risk examples saved to: {save_path}')
    except Exception as e:
        logger.warning(f'Failed saving cumulative risk examples: {e}')


def plot_risk_timecourse_examples(pred_probs, durations, events, record_ids, model_name, output_dir, n_examples=4, fold=None, time_resolution_minutes=30, continuous_interpolation=True, display_mode='mass'):
    """
    绘制事件概率随时间的时间序列（以半小时为步长或连续插值）。
    pred_probs: (n_samples, n_time_bins) - probability mass per discrete bin (DeepHit)
    如果 continuous_interpolation=True，会对离散概率做线性插值以得到更平滑的连续曲线。
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制风险时间序列示例：缺少绘图支持")
        return
    # 检查pred_probs的维度
    if pred_probs.ndim == 1:
        logger.warning("plot_risk_timecourse_examples: pred_probs是一维数组，无法绘制风险时间序列")
        return
    
    # Combine selected samples on a single axis for easier comparison，同时也为每个样本单独出图
    n = min(n_examples, pred_probs.shape[0])
    ev_idx = [i for i, e in enumerate(events) if int(e) == 1]
    cens_idx = [i for i, e in enumerate(events) if int(e) == 0]
    chosen = []
    # try to pick at least 2 events and rest censored
    if len(ev_idx) >= 2:
        chosen.extend(list(np.random.choice(ev_idx, size=2, replace=False)))
    else:
        chosen.extend(ev_idx)
    rem = n - len(chosen)
    if len(cens_idx) >= rem:
        chosen.extend(list(np.random.choice(cens_idx, size=rem, replace=False)))
    else:
        chosen.extend(cens_idx[:rem])

    try:
        n_bins = pred_probs.shape[1]
        cfg = get_config()
        total_hours = float(cfg.data.sequence_generation.prediction_window_hours)
        times = np.linspace(0.0, total_hours, n_bins)
    except Exception:
        times = np.arange(pred_probs.shape[1])
    durations = _durations_in_hours_clamped(durations, cfg=_GLOBAL_CFG, name='durations')

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    ax = plt.gca()
    global_ymax = 0.0
    for i, idx in enumerate(chosen):
        color = cmap(i % 10)
        prob = np.asarray(pred_probs[idx], dtype=float)
        # If pred_probs is a single risk score (1D), skip per-time plotting
        if prob.ndim == 0 or prob.size == 1:
            # plot as horizontal line (risk score) for context
            rs = float(prob.ravel()[0])
            ax.axhline(y=rs, color=color, linestyle='-', linewidth=2, label=f'Idx {idx} (risk={rs:.3f})')
            # mark event time if available
            if int(events[idx]) == 1:
                try:
                    t = float(durations[idx])
                except Exception:
                    t = None
                if t is not None and np.isfinite(t):
                    ax.axvline(x=t, color=color, linestyle='--', alpha=0.8)
            continue

        # pred_probs expected as probability mass per bin: p_k = P(T in bin k)
        p = prob.copy()
        # 归一化为概率质量函数
        p = np.maximum(p, 0.0)
        total = float(np.nansum(p))
        if np.isfinite(total) and total > 0:
            p = p / total

        # cumulative distribution and previous survival S(t_{k-1})
        cdf = np.cumsum(p)
        cdf_prev = np.concatenate(([0.0], cdf[:-1]))
        S_prev = 1.0 - cdf_prev
        # avoid division by zero
        S_prev = np.maximum(S_prev, 1e-8)

        # 统一输出离散 hazard：y = p_k / S_{k-1}，并换算为每小时速率
        y_hazard = p / S_prev
        try:
            cfg = get_config()
            total_hours = float(cfg.data.sequence_generation.prediction_window_hours)
            delta_t = float(total_hours) / max(1, len(times))
        except Exception:
            delta_t = 1.0
        yvals = y_hazard / max(delta_t, 1e-8)
        ylabel = 'Conditional hazard rate (per hour)'

        # 记录全局 ymax 以用于合图轴缩放
        try:
            # 使用全局最大值，确保整条曲线可见
            local_max = float(np.nanmax(yvals)) if np.size(yvals) else 0.0
            if np.isfinite(local_max):
                global_ymax = max(global_ymax, local_max)
        except Exception:
            pass

        if continuous_interpolation:
            from scipy.interpolate import interp1d
            f = interp1d(times, yvals, kind='linear', bounds_error=False, fill_value=(yvals[0], yvals[-1]))
            fine_t = np.linspace(times.min(), times.max(), max(200, len(times)*20))
            ax.plot(fine_t, f(fine_t), color=color, linewidth=2, label=f'Idx {idx}')
        else:
            ax.plot(times, yvals, marker='o', color=color, label=f'Idx {idx}')

        # annotate event time
        if int(events[idx]) == 1:
            try:
                t = float(durations[idx])
            except Exception:
                t = None
            if t is not None and np.isfinite(t):
                ax.axvline(x=t, color=color, linestyle='--', alpha=0.8)
                ax.text(t, np.nanmax(yvals) * 0.05, f'Event {t:.1f}h', color=color, rotation=90, va='bottom', ha='right', fontsize=8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(ylabel if 'ylabel' in locals() else 'Predicted Probability / Risk')
    # 为合图自适应纵轴：顶端取全局最大值按时间单位换算后的 110% 或至少 0.05
    try:
        top = max(0.05, float(global_ymax) * 1.1) if np.isfinite(global_ymax) else 0.1
        ax.set_ylim(0, top)
    except Exception:
        ax.set_ylim(bottom=0)
    # leave top autoscaled for readability
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    fname = f'{model_name}_hazard_timecourse_examples'
    if fold is not None:
        fname += f'_fold{fold}'
    fname += '.png'
    save_path = os.path.join(output_dir, fname)
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Risk timecourse examples saved to: {save_path}')
    except Exception as e:
        logger.warning(f'Failed saving risk timecourse examples: {e}')

    # 另存：每个样本单独成图，便于查看
    try:
        for i, idx in enumerate(chosen):
            color = cmap(i % 10)
            prob = np.asarray(pred_probs[idx], dtype=float)
            if prob.ndim == 0 or prob.size == 1:
                # 单一 risk 分数情况，跳过单图
                continue
            p = np.maximum(prob.copy(), 0.0)
            total = float(np.nansum(p))
            if np.isfinite(total) and total > 0:
                p = p / total
            cdf = np.cumsum(p)
            cdf_prev = np.concatenate(([0.0], cdf[:-1]))
            S_prev = 1.0 - cdf_prev
            S_prev = np.maximum(S_prev, 1e-8)
            y_hazard = p / S_prev
            try:
                cfg = get_config()
                total_hours = float(cfg.data.sequence_generation.prediction_window_hours)
                delta_t = float(total_hours) / max(1, len(times))
            except Exception:
                delta_t = 1.0
            yvals = y_hazard / max(delta_t, 1e-8)
            ylabel = 'Conditional hazard rate (per hour)'
            if continuous_interpolation:
                from scipy.interpolate import interp1d
                f = interp1d(times, yvals, kind='linear', bounds_error=False, fill_value=(yvals[0], yvals[-1]))
                fine_t = np.linspace(times.min(), times.max(), max(200, len(times)*20))
                plt.figure(figsize=(8, 4))
                ax_i = plt.gca()
                ax_i.plot(fine_t, f(fine_t), color=color, linewidth=2)
            else:
                plt.figure(figsize=(8, 4))
                ax_i = plt.gca()
                ax_i.plot(times, yvals, marker='o', color=color)
            # 事件时间虚线
            if int(events[idx]) == 1:
                try:
                    t = float(durations[idx])
                except Exception:
                    t = None
                if t is not None and np.isfinite(t):
                    ax_i.axvline(x=t, color=color, linestyle='--', alpha=0.8)
            # y 轴自适应：顶部取该样本 y 的 110% 或至少 0.05
            try:
                ymax_i = float(np.nanmax(yvals)) if np.size(yvals) else 0.0
                top_i = max(0.05, ymax_i * 1.1)
                ax_i.set_ylim(0, top_i)
            except Exception:
                ax_i.set_ylim(bottom=0)
            ax_i.set_xlabel('Time (hours)')
            ax_i.set_ylabel(ylabel)
            plt.tight_layout()
            ind_save = os.path.join(output_dir, f"{model_name}_risk_timecourse_sample_{idx}.png")
            try:
                plt.savefig(ind_save, dpi=300, bbox_inches='tight')
            finally:
                plt.close()
    except Exception as e:
        logger.warning(f'Failed saving individual risk timecourse plots: {e}')

def plot_calibration_curves(predicted_probabilities, observed_events, model_name, output_dir, n_bins=10):
    """绘制校准曲线图"""
    if not HAS_PLOTTING:
        logging.warning("无法绘制校准曲线：缺少绘图支持")
        return
    
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    prob_true, prob_pred = calibration_curve(observed_events, predicted_probabilities, n_bins=n_bins, strategy='uniform')
    ax1.plot(prob_pred, prob_true, "s-", label=f'{model_name}')
    ax2.hist(predicted_probabilities, range=(0, 1), bins=n_bins, label=model_name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plots (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_calibration_curve.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Calibration curve saved to: {save_path}")

def plot_feature_interactions(df, feature1, feature2, target, model_name, output_dir):
    """绘制特征交互作用图"""
    if not HAS_PLOTTING:
        logging.warning("无法绘制特征交互作用图：缺少绘图支持")
        return
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue=target, palette='viridis', alpha=0.6)
    plt.title(f'Interaction between {feature1} and {feature2} for {model_name}')
    
    save_path = os.path.join(output_dir, f'{model_name}_interaction_{feature1}_{feature2}.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Feature interaction plot saved to: {save_path}")

def plot_feature_importance(feature_importance, model_name, output_dir):
    """
    绘制特征重要性图。
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制特征重要性图：缺少绘图支持")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance.index, feature_importance.values)
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.grid(True)
    
    save_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Feature importance plot saved to: {save_path}")

def plot_ar_survival_curves(survival_probs, durations, events, record_ids, output_dir, fold=None, n_ar=6, random_seed=42, model_name="", output_all_ars=False, subdir="ar_survival_curves"):
    """
    随机抽取n_ar个活动区，画每个活动区所有子样本的生存曲线，按离事件发生时间从远到近排序，颜色黄到红渐变，事件样本标记真实事件时间。
    Args:
        survival_probs: [N, T] 每个子样本的生存概率曲线
        durations: [N] 每个子样本的持续时间
        events: [N] 每个子样本的事件标记(1=事件,0=删失)
        record_ids: [N] 每个子样本的record_id
        output_dir: 输出目录
        fold: 当前fold编号（可选）
        n_ar: 随机抽取的活动区数量
        random_seed: 随机种子
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制活动区生存曲线：缺少绘图支持")
        return
    
    np.random.seed(random_seed)
    # record_ids may be strings or metadata dicts; extract AR identifiers robustly
    ar_list = []
    for rid in record_ids:
        try:
            info = _parse_record_id(rid)
            ar_val = info.get('ar') or info.get('raw') or str(rid)
        except Exception:
            ar_val = str(rid)
        ar_list.append(ar_val)
    ar_arr = np.array(ar_list, dtype=object)
    unique_ars = np.unique(ar_arr)
    if output_all_ars:
        selected_ars = unique_ars
    else:
        if len(unique_ars) > n_ar:
            selected_ars = np.random.choice(unique_ars, size=n_ar, replace=False)
        else:
            selected_ars = unique_ars
    logging.info(f"抽取的活动区: {selected_ars}")

    # 统一单位与裁剪
    durations = _durations_in_hours_clamped(durations, cfg=_GLOBAL_CFG, name='ar_survival_curves.durations')
    events = np.asarray(events)

    for ar in selected_ars:
        idx = np.where(ar_arr == ar)[0]
        if len(idx) == 0:
            continue
        # 使用生存时间进行排序与着色：删失样本通常等于预测窗口（已通过 clamp 保证）
        dur_local = np.asarray(durations[idx], dtype=float)
        # 从长到短排序（曲线绘制顺序从长到短）
        order = np.argsort(dur_local)[::-1]
        idx = idx[order]
        dur_sorted = dur_local[order]

        # 颜色映射：固定在预测窗口 [0, pred_w] 范围，避免颜色条超过48小时
        try:
            cfg = get_config()
            _pred_w_color = float(cfg.data.sequence_generation.prediction_window_hours)
        except Exception:
            _pred_w_color = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
        cmap = plt.get_cmap('viridis')
        try:
            _cap_max = float(_pred_w_color)
            # 不允许颜色条超过48小时
            if _cap_max > 48.0:
                _cap_max = 48.0
            norm = mcolors.Normalize(vmin=0.0, vmax=_cap_max, clip=True)
        except Exception:
            # 回退到基于数据的范围
            vmin = float(np.nanmin(dur_sorted)) if np.isfinite(np.nanmin(dur_sorted)) else 0.0
            vmax = float(np.nanmax(dur_sorted)) if np.isfinite(np.nanmax(dur_sorted)) else 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        plt.figure(figsize=(10, 6))
        y_min, y_max = 1.0, 0.0
        # 若 survival_probs 是 DataFrame，优先使用列作为时间轴
        if hasattr(survival_probs, 'columns'):
            t_axis = np.asarray(survival_probs.columns, dtype=float)
            # 裁剪时间轴到预测窗口
            try:
                cfg = get_config()
                pred_w2 = float(cfg.data.sequence_generation.prediction_window_hours)
            except Exception:
                pred_w2 = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
            mask_t2 = np.isfinite(t_axis) & (t_axis <= pred_w2)
            t_axis = t_axis[mask_t2] if mask_t2.any() else t_axis
        else:
            # 非 DataFrame 情况：使用预测窗口做线性映射到"小时"刻度
            try:
                cfg = get_config()
                _pred_w_guess = float(cfg.data.sequence_generation.prediction_window_hours)
            except Exception:
                _pred_w_guess = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
            T = int(np.asarray(survival_probs).shape[1])
            t_axis = np.linspace(0.0, _pred_w_guess, T, dtype=float)

        # 强制限制横轴在预测窗口内，避免超过预测窗口的时间显示
        try:
            cfg = get_config()
            _pred_w_plot = float(cfg.data.sequence_generation.prediction_window_hours)
        except Exception:
            _pred_w_plot = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
        try:
            plt.xlim(0.0, _pred_w_plot)
        except Exception:
            pass

        for i, j in enumerate(idx):
            color = cmap(norm(dur_sorted[i]))
            srow = survival_probs.iloc[j].values if hasattr(survival_probs, 'iloc') else np.asarray(survival_probs[j], dtype=float)
            if 'mask_t2' in locals() and mask_t2.any() and hasattr(survival_probs, 'iloc'):
                svals = srow[mask_t2]
            else:
                svals = np.asarray(srow, dtype=float)
            plt.plot(t_axis, svals, color=color, alpha=0.95)
            try:
                y_min = min(y_min, float(np.nanmin(svals)))
                y_max = max(y_max, float(np.nanmax(svals)))
            except Exception:
                pass
        # 添加颜色条，指示生存时间（小时）
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # 为避免颜色条因空数组自动扩展，提供与上限一致的示意数组
        try:
            import numpy as _np
            _demo_arr = _np.linspace(0.0, _cap_max, 256, dtype=float)
            sm.set_array(_demo_arr)
        except Exception:
            sm.set_array([0.0, 1.0])
        try:
            # 使用固定边界，彻底锁定颜色条范围
            _boundaries = _np.linspace(0.0, _cap_max, 257) if '_np' in locals() else None
        except Exception:
            _boundaries = None
        if _boundaries is not None:
            cbar = plt.colorbar(sm, boundaries=_boundaries, extend='neither')
        else:
            cbar = plt.colorbar(sm, extend='neither')
        cbar.set_label('Survival time (hours)')
        try:
            # 再次设置范围，防止自动扩展
            sm.set_clim(0.0, _cap_max)
        except Exception:
            pass
        try:
            # 设置颜色条刻度到0/12/24/36/48（若上限<48，则均分为5段）
            import numpy as _np
            if _cap_max >= 48.0:
                ticks = [0.0, 12.0, 24.0, 36.0, 48.0]
            else:
                ticks = _np.linspace(0.0, _cap_max, 5)
            cbar.set_ticks(ticks)
            # 明确设置标签，防止自动格式化出现越界或非整数
            cbar.set_ticklabels([f"{int(t)}" for t in ticks])
        except Exception:
            pass
        # Annotate with AR and fold
        title_ar = ar if isinstance(ar, (str, int)) else str(ar)
        title = f'{title_ar} Survival Curves'
        if fold is not None:
            title += f' (fold{fold})'
        plt.title(title)
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        try:
            bottom = max(0.0, y_min - 0.02)
            top = min(1.0, y_max + 0.02)
            if np.isfinite(bottom) and np.isfinite(top) and top > bottom:
                plt.ylim(bottom, top)
        except Exception:
            pass
        # 再次强制横轴限制，防止绘制后自动缩放覆盖
        try:
            cfg = get_config()
            _pred_w_plot2 = float(cfg.data.sequence_generation.prediction_window_hours)
        except Exception:
            _pred_w_plot2 = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
        try:
            plt.xlim(0.0, _pred_w_plot2)
            # 设定关键刻度以强化48小时上限的可视化
            try:
                import numpy as _np
                tick_vals = _np.linspace(0.0, _pred_w_plot2, 5)
                plt.xticks(tick_vals)
            except Exception:
                pass
        except Exception:
            pass
        plt.tight_layout()
        # ensure ar string safe for filename
        ar_str = str(ar).replace('AR', '').replace('ar', '') if ar is not None else 'unknown'
        # include model_name if provided for easier identification
        prefix = (model_name + '_') if model_name else ''
        fname = f'{prefix}ar_survival_curves_AR{ar_str}'
        if fold is not None:
            fname += f'_fold{fold}'
        fname += '.png'
        # 保存到子目录，集中管理所有活动区图像
        save_dir = os.path.join(output_dir, subdir) if (subdir and len(str(subdir)) > 0) else output_dir
        save_path = os.path.join(save_dir, fname)
        os.makedirs(os.path.dirname(save_path) or save_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def plot_mean_cumulative_risk(survival_probs, model_name, output_dir, by_risk_group=None, fold=None):
    """
    改为绘制平均生存曲线：平均 S(t)。若 by_risk_group 提供，则分别绘制两组的平均生存曲线。
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制平均累计风险曲线：缺少绘图支持")
        return
    try:
        if hasattr(survival_probs, 'values'):
            surv_mat = np.asarray(survival_probs.values, dtype=float)
            times = np.asarray(survival_probs.columns, dtype=float)
        else:
            surv_mat = np.asarray(survival_probs, dtype=float)
            times = np.arange(surv_mat.shape[1], dtype=float)
        plt.figure(figsize=(8, 6))
        if by_risk_group is None:
            mean_surv = np.nanmean(surv_mat, axis=0)
            plt.plot(times, mean_surv, label='Mean survival', linewidth=2)
        else:
            by_risk_group = np.asarray(by_risk_group)
            mask_high = (by_risk_group == 'High Risk')
            mask_low = (by_risk_group == 'Low Risk')
            if np.any(mask_high):
                mean_surv_high = np.nanmean(surv_mat[mask_high], axis=0)
                plt.plot(times, mean_surv_high, label='High Risk (mean)', linewidth=2)
            if np.any(mask_low):
                mean_surv_low = np.nanmean(surv_mat[mask_low], axis=0)
                plt.plot(times, mean_surv_low, label='Low Risk (mean)', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        try:
            y_min = max(0.0, float(np.nanmin(surv_mat)) - 0.02)
            y_max = min(1.0, float(np.nanmax(surv_mat)) + 0.02)
            plt.ylim(y_min, y_max)
        except Exception:
            plt.ylim(0, 1.02)
        title = f'{model_name} - Mean Survival'
        if fold is not None:
            title += f' (fold{fold})'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        fname = f'{model_name}_mean_survival'
        if fold is not None:
            fname += f'_fold{fold}'
        fname += '.png'
        save_path = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(save_path) or output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Mean cumulative risk saved to: {save_path}")
    except Exception as e:
        logger.warning(f"绘制平均累计风险曲线失败: {e}")


def plot_ar_cumulative_risk_curves(survival_probs, durations, events, record_ids, output_dir, fold=None, n_ar=6, random_seed=42, model_name=""):
    """
    改为绘制按活动区分组的生存曲线 S(t)，并按距离事件的先后排序（删失放最后），颜色渐变便于区分。
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制活动区累计风险曲线：缺少绘图支持")
        return
    try:
        # 准备 AR 列表
        np.random.seed(random_seed)
        ar_list = []
        for rid in record_ids:
            try:
                info = _parse_record_id(rid)
                ar_val = info.get('ar') or info.get('raw') or str(rid)
            except Exception:
                ar_val = str(rid)
            ar_list.append(ar_val)
        ar_arr = np.array(ar_list, dtype=object)
        unique_ars = np.unique(ar_arr)
        if len(unique_ars) > n_ar:
            selected_ars = np.random.choice(unique_ars, size=n_ar, replace=False)
        else:
            selected_ars = unique_ars

        # 时间轴
        if hasattr(survival_probs, 'columns'):
            times = np.asarray(survival_probs.columns, dtype=float)
            surv = np.asarray(survival_probs.values, dtype=float)
        else:
            surv = np.asarray(survival_probs, dtype=float)
            times = np.arange(surv.shape[1], dtype=float)

        # 将时间轴裁剪到预测窗口内
        try:
            cfg = get_config()
            pred_w = float(cfg.data.sequence_generation.prediction_window_hours)
        except Exception:
            pred_w = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
        if times.ndim == 1 and times.size == surv.shape[1]:
            mask_t = np.isfinite(times) & (times <= pred_w)
            if np.any(mask_t):
                times = times[mask_t]
                surv = surv[:, mask_t]

        # 统一单位并裁剪，防御性处理
        durations = _durations_in_hours_clamped(durations, cfg=_GLOBAL_CFG, name='ar_survival_curves_legacy.durations')
        events = np.asarray(events)
        for ar in selected_ars:
            idx = np.where(ar_arr == ar)[0]
            if len(idx) == 0:
                continue
            # 使用裁剪后的时长排序（从长到短）
            dur_local = np.asarray(durations[idx], dtype=float)
            order = np.argsort(dur_local)[::-1]
            idx = idx[order]
            dur_for_color = dur_local[order]

            # 颜色映射：固定到预测窗口上限（不超过48h）
            cmap = plt.get_cmap('viridis')  # viridis: 小值较暗，大值较亮
            try:
                cfg = get_config()
                _pred_w_color2 = float(cfg.data.sequence_generation.prediction_window_hours)
            except Exception:
                _pred_w_color2 = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
            _cap_max2 = 48.0 if _pred_w_color2 > 48.0 else _pred_w_color2
            norm = mcolors.Normalize(vmin=0.0, vmax=_cap_max2, clip=True)
            plt.figure(figsize=(10, 6))
            for i, j in enumerate(idx):
                color = cmap(norm(dur_for_color[i]))
                svals = np.asarray(surv[j], dtype=float)
                plt.plot(times, svals, color=color, alpha=0.95)
            title_ar = ar if isinstance(ar, (str, int)) else str(ar)
            title = f'Activity Region {title_ar} Survival Curves'
            if fold is not None:
                title += f' (fold{fold})'
            plt.title(title)
            plt.xlabel('Time (hours)')
            plt.ylabel('Survival Probability')
            try:
                plt.xlim(0, pred_w)
            except Exception:
                pass
            try:
                y_min = max(0.0, float(np.nanmin(surv[idx])) - 0.02)
                y_max = min(1.0, float(np.nanmax(surv[idx])) + 0.02)
                plt.ylim(y_min, y_max)
            except Exception:
                plt.ylim(0, 1.02)
            # 加颜色条，说明颜色代表的生存时长（固定0-48或预测窗口）
            try:
                import numpy as _np
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                demo = _np.linspace(0.0, _cap_max2, 256, dtype=float)
                sm.set_array(demo)
                boundaries = _np.linspace(0.0, _cap_max2, 257)
                cbar = plt.colorbar(sm, boundaries=boundaries, extend='neither')
                sm.set_clim(0.0, _cap_max2)
                cbar.set_label('Duration until event (hours)')
                ticks = [0.0, 12.0, 24.0, 36.0, 48.0] if _cap_max2 >= 48.0 else _np.linspace(0.0, _cap_max2, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{int(t)}" for t in ticks])
            except Exception:
                pass
            plt.tight_layout()
            ar_str = str(ar).replace('AR', '').replace('ar', '') if ar is not None else 'unknown'
            prefix = (model_name + '_') if model_name else ''
            fname = f'{prefix}ar_survival_curves_AR{ar_str}'
            if fold is not None:
                fname += f'_fold{fold}'
            fname += '.png'
            save_path = os.path.join(output_dir, fname)
            os.makedirs(os.path.dirname(save_path) or output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.warning(f"绘制活动区累计风险曲线失败: {e}")


def plot_training_curves(train_losses, val_losses, train_c_indices=None, val_c_indices=None, model_name="", output_dir=""):
    """
    绘制训练过程中的损失和C-index曲线
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制训练曲线：缺少绘图支持")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 损失曲线
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    if val_losses is not None:
        axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # C-index曲线
    if train_c_indices is not None:
        axes[1].plot(train_c_indices, label='Training C-index', color='blue')
    if val_c_indices is not None:
        axes[1].plot(val_c_indices, label='Validation C-index', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('C-index')
    axes[1].set_title(f'{model_name} - C-index Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"训练曲线图已保存到: {save_path}")

def plot_risk_distribution_by_event_type(risk_scores, events, model_name, output_dir):
    """
    按事件类型绘制风险分数分布
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制风险分布图：缺少绘图支持")
        return
    
    # 分离事件和删失样本
    event_risks = risk_scores[events == 1]
    censored_risks = risk_scores[events == 0]

    # 如果 matplotlib 可用且 plt 对象存在，使用 matplotlib 绘图
    if HAS_PLOTTING and plt is not None:
        plt.figure(figsize=(10, 6))
        # 绘制直方图
        if len(event_risks) > 0:
            plt.hist(event_risks, bins=30, alpha=0.7, label='event samples', color='red', density=True)
        if len(censored_risks) > 0:
            plt.hist(censored_risks, bins=30, alpha=0.7, label='censored samples', color='blue', density=True)

        plt.xlabel('Risk Score')
        plt.ylabel('Density')
        plt.title(f'Risk Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f'{model_name}_risk_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"风险分布图已保存到: {save_path}")
        return

    # Plotly removed: save CSV fallback for risk distributions
    try:
        os.makedirs(output_dir, exist_ok=True)
        if len(event_risks) > 0:
            pd.DataFrame({'event_risks': event_risks}).to_csv(os.path.join(output_dir, f'{model_name}_event_risks_fallback.csv'), index=False)
        if len(censored_risks) > 0:
            pd.DataFrame({'censored_risks': censored_risks}).to_csv(os.path.join(output_dir, f'{model_name}_censored_risks_fallback.csv'), index=False)
        logging.info(f"Saved risk distribution CSV fallbacks to: {output_dir}")
    except Exception:
        logging.warning('无法绘制或保存风险分布：无可用绘图后端或写入失败')

def plot_survival_time_distribution(durations, events, model_name, output_dir):
    """
    绘制生存时间分布
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制生存时间分布图：缺少绘图支持")
        return

    # convert durations to canonical hours
    try:
        durations = _ensure_durations_in_hours(durations, cfg=_GLOBAL_CFG, name='plot_survival_time_distribution')
    except Exception:
        durations = np.asarray(durations)

    # Prefer matplotlib if available and plt is a valid module
    if MATPLOTLIB_AVAILABLE and plt is not None:
        try:
            plt.figure(figsize=(12, 5))

            # 分离事件和删失样本
            event_times = durations[events == 1]
            censored_times = durations[events == 0]

            # 左侧图：展示事件样本的计数直方图（更直观），并在条形上标注计数
            plt.subplot(1, 2, 1)
            if len(event_times) > 0:
                # choose bin edges that align with typical observation windows if possible
                bins = np.histogram_bin_edges(event_times, bins='auto')
                counts, edges, patches = plt.hist(event_times, bins=bins, alpha=0.9, color='red')
                plt.xlabel('Event Time (hours)')
                plt.ylabel('Count (events)')
                plt.title('Event Time Histogram')
                # annotate counts on bars
                for rect, c in zip(patches, counts):
                    h = rect.get_height()
                    if h > 0:
                        plt.text(rect.get_x() + rect.get_width() / 2, h, f'{int(c)}', ha='center', va='bottom', fontsize=8)
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No event samples', ha='center', va='center')

            # 绘制箱线图
            plt.subplot(1, 2, 2)
            data_to_plot = []
            labels = []
            if len(event_times) > 0:
                data_to_plot.append(event_times)
                labels.append('event samples')
            if len(censored_times) > 0:
                data_to_plot.append(censored_times)
                labels.append('censored samples')

            if data_to_plot:
                plt.boxplot(data_to_plot, labels=labels)
                plt.ylabel('Survival Time (hours)')
                plt.title('Survival Time Boxplot')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{model_name}_survival_time_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"生存时间分布图已保存到: {save_path}")
            return
        except Exception as e:
            logging.getLogger(__name__).warning(f"Matplotlib plotting failed for survival time distribution: {e}")

    # Plotly fallback: produce HTML + attempt PNG
    # Plotly support removed: save CSV fallback for survival time distribution
    try:
        os.makedirs(output_dir, exist_ok=True)
        event_times = durations[events == 1]
        censored_times = durations[events == 0]
        pd.DataFrame({'event_times': event_times}).to_csv(os.path.join(output_dir, f'{model_name}_event_times_fallback.csv'), index=False)
        pd.DataFrame({'censored_times': censored_times}).to_csv(os.path.join(output_dir, f'{model_name}_censored_times_fallback.csv'), index=False)
        logging.getLogger(__name__).info(f"Saved survival time CSVs to {output_dir}")
        return
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save survival time CSV fallbacks: {e}")

    logging.getLogger(__name__).warning("无法绘制生存时间分布图：没有可用的绘图后端")

def plot_feature_correlation_heatmap(feature_data, feature_names, model_name, output_dir):
    """
    绘制特征相关性热力图
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制特征相关性热力图：缺少绘图支持")
        return
    
    plt.figure(figsize=(12, 10))
    
    # 计算相关性矩阵
    if feature_data.ndim == 3:
        # 如果是3D数据，取最后一个时间步
        feature_data_2d = feature_data[:, -1, :]
    else:
        feature_data_2d = feature_data
    
    corr_matrix = np.corrcoef(feature_data_2d.T)
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title(f'{model_name} - Feature Correlation Heatmap')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{model_name}_feature_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Feature correlation heatmap saved to: {save_path}")

def plot_model_comparison(metrics_dict, output_dir):
    """
    绘制不同模型的性能对比图
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制模型对比图：缺少绘图支持")
        return
    
    if not metrics_dict:
        return
    
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        axes[i].bar(models, values, color=['blue', 'red', 'green', 'orange'][:len(models)])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Training history plot saved to: {save_path}")

def plot_confidence_intervals(survival_curves, confidence_level=0.95, model_name="", output_dir=""):
    """
    绘制生存曲线的置信区间
    """
    # ensure matplotlib available now
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        logging.getLogger(__name__).warning("无法绘制置信区间图：matplotlib 不可用，保存 CSV 备份")
        try:
            os.makedirs(output_dir, exist_ok=True)
            survival_curves.to_csv(os.path.join(output_dir, f'{model_name}_survival_curves_fallback.csv'))
        except Exception as e:
            logging.getLogger(__name__).warning(f"保存 survival_curves 备份 CSV 失败: {e}")
        return

    if survival_curves.empty:
        return

    plt_local.figure(figsize=(10, 6))
    
    # 计算置信区间
    mean_curve = survival_curves.mean(axis=0)
    std_curve = survival_curves.std(axis=0)
    
    # 计算置信区间
    z_score = 1.96  # 95% 置信区间
    ci_lower = mean_curve - z_score * std_curve / np.sqrt(len(survival_curves))
    ci_upper = mean_curve + z_score * std_curve / np.sqrt(len(survival_curves))
    
    time_points = survival_curves.columns
    
    # 绘制置信区间
    try:
        plt_local.fill_between(time_points, ci_lower, ci_upper, alpha=0.3, label=f'{confidence_level*100}% Confidence Interval')
        plt_local.plot(time_points, mean_curve, 'b-', linewidth=2, label='Average Survival Curve')
    except Exception as e:
        logging.getLogger(__name__).warning(f"绘制置信区间时出错: {e}")
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Survival Probability')
    plt.title(f'{model_name} - Survival Curve Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'{model_name}_confidence_intervals.png')
    try:
        plt_local.savefig(save_path, dpi=300, bbox_inches='tight')
        plt_local.close()
        logging.info(f"置信区间图已保存到: {save_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"保存置信区间图失败: {e}")

def plot_enhanced_training_curves(train_losses, val_losses, val_c_indices, learning_rates, model_name="", output_dir=""):
    logger = logging.getLogger(__name__)
    if MATPLOTLIB_AVAILABLE:
        # ...existing matplotlib code...
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{model_name} - Enhanced Training Curves', fontsize=16, fontweight='bold')
            epochs = range(1, len(train_losses) + 1)
            ax1 = axes[0, 0]
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            if val_losses:
                val_epochs = range(1, len(val_losses) + 1)
                ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2 = axes[0, 1]
            if val_c_indices:
                val_c_epochs = range(1, len(val_c_indices) + 1)
                ax2.plot(val_c_epochs, val_c_indices, 'g-', linewidth=2, marker='o', markersize=4)
                ax2.set_title('Validation C-index')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('C-index')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)
            ax3 = axes[1, 0]
            if learning_rates:
                ax3.plot(epochs, learning_rates, 'purple', linewidth=2)
                ax3.set_title('Learning Rate Schedule')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
            ax4 = axes[1, 1]
            if val_losses and len(train_losses) == len(val_losses):
                loss_ratios = [t/v if v > 0 else 1 for t, v in zip(train_losses, val_losses)]
                ax4.plot(epochs, loss_ratios, 'orange', linewidth=2)
                ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Loss')
                ax4.set_title('Train/Validation Loss Ratio')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss Ratio')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(output_dir, f'{model_name}_enhanced_training_curves_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Enhanced training curves saved to: {save_path}")
            history_data = {'epochs': list(range(1, len(train_losses)+1)), 'train_losses': train_losses, 'val_losses': val_losses, 'val_c_indices': val_c_indices, 'learning_rates': learning_rates}
            history_path = os.path.join(output_dir, f'{model_name}_training_history_{timestamp}.json')
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Training history data saved to: {history_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib enhanced plotting failed: {e}")
    else:
        # Plotly removed: fallback to CSV and matplotlib-only plotting handled above; ensure history JSON saved
        try:
            os.makedirs(output_dir, exist_ok=True)
            history_data = {'epochs': list(range(1, len(train_losses) + 1)), 'train_losses': train_losses, 'val_losses': val_losses, 'val_c_indices': val_c_indices, 'learning_rates': learning_rates}
            history_path = os.path.join(output_dir, f'{model_name}_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            logging.getLogger(__name__).info(f"Saved training history JSON to: {history_path}")
            return
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save enhanced training history JSON: {e}")


def plot_model_performance_comparison(metrics_dict, output_dir):
    logger = logging.getLogger(__name__)
    if not metrics_dict:
        logger.warning("没有指标数据，无法绘制性能对比图")
        return
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    # Matplotlib path
    if MATPLOTLIB_AVAILABLE:
        try:
            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]
            for i, metric in enumerate(metrics):
                values = [metrics_dict[model].get(metric, 0) for model in models]
                values = np.array(values)
                mask = np.isfinite(values)
                values_plot = values[mask]
                models_plot = np.array(models)[mask]
                axes[i].bar(models_plot, values_plot, color=['blue', 'red', 'green', 'orange'][:len(values_plot)])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                for j, v in enumerate(values_plot):
                    axes[i].text(j, v + max(values_plot)*0.01, f'{v:.3f}', ha='center', va='bottom')
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'model_performance_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"模型对比图已保存到: {save_path}")
            return
        except Exception as e:
            logger.warning(f"Matplotlib model comparison plotting failed: {e}")
    else:
        # Plotly removed: save CSV comparison table for offline plotting
        try:
            os.makedirs(output_dir, exist_ok=True)
            comp_df = pd.DataFrame({m: {model: metrics_dict[model].get(m, np.nan) for model in models} for m in metrics})
            comp_df.to_csv(os.path.join(output_dir, 'model_performance_comparison_fallback.csv'))
            logging.getLogger(__name__).info(f"Saved model performance comparison CSV to: {output_dir}")
            return
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save model comparison CSV fallback: {e}")


def plot_regression_analysis(pred_event_times, true_event_times, model_name, output_dir, record_ids=None):
    """
    回归性分析：预测事件时间 vs 真实事件时间，输出MSE、准确度等。
    pred_event_times, true_event_times: 只包含事件样本
    """
    if not HAS_PLOTTING:
        logging.warning("无法绘制回归分析图：缺少绘图支持")
        return
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(true_event_times, pred_event_times)
    r2 = r2_score(true_event_times, pred_event_times)
    acc = np.mean(np.abs(pred_event_times - true_event_times) < 1.0)  # 误差小于1小时的比例
    plt.figure(figsize=(8, 8))
    plt.scatter(true_event_times, pred_event_times, c='b', alpha=0.6, label='Samples')
    plt.plot([true_event_times.min(), true_event_times.max()], [true_event_times.min(), true_event_times.max()], 'r--', label='Ideal (y=x)')
    plt.xlabel('True Event Time')
    plt.ylabel('Predicted Event Time')
    plt.title(f'{model_name} - Event Time Regression\nMSE={mse:.2f}, R2={r2:.2f}, Acc(<1h)={acc:.2%}')
    if record_ids is not None:
        for i in range(len(true_event_times)):
            plt.annotate(str(record_ids[i]), (true_event_times[i], pred_event_times[i]), fontsize=8, alpha=0.7)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f'{model_name}_event_time_regression.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Event time regression plot saved to: {save_path}")
    # 返回统计指标
    return {'mse': float(mse), 'r2': float(r2), 'acc_1h': float(acc)}

def plot_hazard_curves_examples(survival_funcs, durations, events, record_ids, model_name, output_dir, n_examples=4, fold=None, subdir='hazard_curves', hazard_scale=1.0, clip_quantile=99.0, smooth_window=None, show_cumhaz=False):
    """
    从离散生存曲线 S(t_k) 计算样本危险函数 h_k = (S_{k-1} - S_k) / S_{k-1}，并按每小时速率输出与平滑显示。
    - 对每个样本：h_rate_k = h_k / Delta_t（Delta_t 由预测窗口 / bin 数推得）
    - 使用 Savitzky-Golay 或移动平均平滑，避免末端尖峰与视觉噪声
    - 输出一个合图（多样本叠加）与每个样本单图，保存到子目录
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制危险函数曲线：缺少绘图支持")
        return

    import numpy as _np

    try:
        n = min(int(n_examples), len(survival_funcs))
    except Exception:
        n = min(4, int(_np.asarray(survival_funcs).shape[0]))

    # 选择样本：优先2事件+2删失
    try:
        events = _np.asarray(events).astype(int)
        ev_idx = [i for i, e in enumerate(events) if int(e) == 1]
        cens_idx = [i for i, e in enumerate(events) if int(e) == 0]
    except Exception:
        ev_idx, cens_idx = [], list(range(n))
    chosen = []
    if len(ev_idx) >= 2:
        chosen.extend(list(_np.random.choice(ev_idx, size=2, replace=False)))
    else:
        chosen.extend(ev_idx)
    rem = n - len(chosen)
    if rem > 0:
        if len(cens_idx) >= rem:
            chosen.extend(list(_np.random.choice(cens_idx, size=rem, replace=False)))
        else:
            chosen.extend(cens_idx[:rem])
    chosen = sorted(set(chosen))
    if len(chosen) == 0:
        chosen = list(range(n))

    # 时间轴（小时）与 Delta_t
    try:
        if hasattr(survival_funcs, 'columns'):
            t_axis = _np.asarray(survival_funcs.columns, dtype=float)
            T = len(t_axis)
            total_hours = float(get_config().data.sequence_generation.prediction_window_hours)
            diffs = _np.diff(t_axis)
            delta_t = float(_np.median(diffs)) if diffs.size > 0 and _np.isfinite(_np.median(diffs)) else float(total_hours) / max(1, T)
        else:
            T = int(_np.asarray(survival_funcs).shape[1])
            total_hours = float(get_config().data.sequence_generation.prediction_window_hours)
            t_axis = _np.linspace(0.0, total_hours, T)
            delta_t = float(total_hours) / max(1, T)
    except Exception:
        T = int(_np.asarray(survival_funcs).shape[1])
        t_axis = _np.arange(T, dtype=float)
        delta_t = 1.0

    def _to_hazard_rate_from_survival(S_row: _np.ndarray) -> _np.ndarray:
        S_row = _np.asarray(S_row, dtype=float)
        S_row = _np.clip(S_row, 0.0, 1.0)
        S_prev = _np.concatenate(([1.0], S_row[:-1]))
        eps = 1e-6
        S_prev = _np.maximum(S_prev, eps)
        h_discrete = _np.maximum(S_prev - S_row, 0.0) / S_prev
        h_rate = h_discrete / max(delta_t, eps)
        finite_vals = h_rate[_np.isfinite(h_rate)]
        try:
            q = float(clip_quantile)
        except Exception:
            q = 99.0
        if finite_vals.size > 0 and q is not None and q > 0:
            pq = float(_np.percentile(finite_vals, min(99.9, max(50.0, q))))
            if _np.isfinite(pq) and pq > 0:
                h_rate = _np.clip(h_rate, 0.0, pq)
        return h_rate

    def _smooth(y: _np.ndarray) -> _np.ndarray:
        y = _np.asarray(y, dtype=float)
        try:
            from scipy.signal import savgol_filter
            if smooth_window is not None:
                win = int(max(5, smooth_window))
                if win % 2 == 0:
                    win += 1
                win = min(win, T - (1 - T % 2)) if T > 5 else 5
            else:
                win = max(5, (T // 12) * 2 + 1)
            win = min(win, T - (1 - T % 2)) if T > 5 else 5
            if win % 2 == 0:
                win = max(5, win - 1)
            poly = 2 if win >= 7 else 1
            return savgol_filter(y, window_length=win, polyorder=poly, mode='interp')
        except Exception:
            k = max(3, T // 20)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            ypad = _np.pad(y, (pad, pad), mode='edge')
            kernel = _np.ones(k) / float(k)
            return _np.convolve(ypad, kernel, mode='valid')

    save_dir = os.path.join(output_dir, subdir) if (subdir and len(str(subdir)) > 0) else output_dir
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception:
        pass

    # 合图
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    for i, idx in enumerate(chosen):
        color = cmap(i % 10)
        S = survival_funcs.iloc[idx].values if hasattr(survival_funcs, 'iloc') else _np.asarray(survival_funcs[idx], dtype=float)
        h_rate = _to_hazard_rate_from_survival(S)
        h_smooth = _smooth(h_rate)
        if hazard_scale and float(hazard_scale) != 1.0:
            h_smooth_plot = h_smooth * float(hazard_scale)
        else:
            h_smooth_plot = h_smooth
        plt.plot(t_axis, h_smooth_plot, color=color, linewidth=2.0, label=f"idx={idx}")
        try:
            if int(events[idx]) == 1:
                t_ev = float(durations[idx])
                if _np.isfinite(t_ev):
                    plt.axvline(x=t_ev, color=color, linestyle='--', alpha=0.7)
        except Exception:
            pass
    plt.xlabel('Time (hours)')
    ylabel = 'Hazard rate h(t) (per hour)'
    try:
        if hazard_scale and float(hazard_scale) != 1.0:
            ylabel += f' × {int(hazard_scale)}'
    except Exception:
        pass
    plt.ylabel(ylabel)
    title = f'{model_name} Hazard Curves'
    if fold is not None:
        title += f' (fold{fold})'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name}_hazard_examples{('_fold'+str(fold)) if fold is not None else ''}.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Hazard examples saved: {out_path}")

    # 单样本图与CSV
    for i, idx in enumerate(chosen):
        S = survival_funcs.iloc[idx].values if hasattr(survival_funcs, 'iloc') else _np.asarray(survival_funcs[idx], dtype=float)
        h_rate = _to_hazard_rate_from_survival(S)
        h_smooth = _smooth(h_rate)
        # PNG
        plt.figure(figsize=(8, 4))
        if hazard_scale and float(hazard_scale) != 1.0:
            h_plot = h_smooth * float(hazard_scale)
        else:
            h_plot = h_smooth
        plt.plot(t_axis, h_plot, color='C0', linewidth=2.0)
        try:
            if int(events[idx]) == 1:
                t_ev = float(durations[idx])
                if _np.isfinite(t_ev):
                    plt.axvline(x=t_ev, color='C1', linestyle='--', alpha=0.9, label='event time')
        except Exception:
            pass
        plt.xlabel('Time (hours)')
        plt.ylabel(ylabel)
        plt.title(f'idx={idx} hazard')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"{model_name}_hazard_idx{idx}{('_fold'+str(fold)) if fold is not None else ''}.png"
        fpath = os.path.join(save_dir, fname)
        plt.savefig(fpath)
        plt.close()
        # CSV
        try:
            import csv as _csv
            csv_path = os.path.join(save_dir, f"{model_name}_hazard_idx{idx}{('_fold'+str(fold)) if fold is not None else ''}.csv")
            with open(csv_path, 'w', newline='') as cf:
                w = _csv.writer(cf)
                w.writerow(['time_hours', 'hazard_rate_per_hour', f'hazard_rate_smoothed{("_scaled" if (hazard_scale and float(hazard_scale) != 1.0) else "") }'])
                scale_val = float(hazard_scale) if (hazard_scale and float(hazard_scale) != 1.0) else 1.0
                for t, hv, hs in zip(t_axis, h_rate, h_smooth):
                    hv_out = float(hv) * scale_val if _np.isfinite(hv) else ''
                    hs_out = float(hs) * scale_val if _np.isfinite(hs) else ''
                    w.writerow([float(t), hv_out, hs_out])
        except Exception:
            pass

    # 可选：累计风险 H(t) = -log S(t)
    if show_cumhaz:
        try:
            plt.figure(figsize=(12, 6))
            for i, idx in enumerate(chosen):
                color = cmap(i % 10)
                S = survival_funcs.iloc[idx].values if hasattr(survival_funcs, 'iloc') else _np.asarray(survival_funcs[idx], dtype=float)
                S = _np.clip(S, 1e-8, 1.0)
                H = -_np.log(S)
                plt.plot(t_axis, H, color=color, linewidth=2.0, label=f"idx={idx}")
            plt.xlabel('Time (hours)')
            plt.ylabel('Cumulative hazard H(t)')
            title2 = f'{model_name} Cumulative Hazard'
            if fold is not None:
                title2 += f' (fold{fold})'
            plt.title(title2)
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2, fontsize=9)
            plt.tight_layout()
            out_path2 = os.path.join(save_dir, f"{model_name}_cumulative_hazard{('_fold'+str(fold)) if fold is not None else ''}.png")
            plt.savefig(out_path2)
            plt.close()
            logger.info(f"Cumulative hazard saved: {out_path2}")
        except Exception as e:
            logger.warning(f'Failed plotting cumulative hazard: {e}')
        except Exception:
            pass

def plot_non_cumulative_risk_per_minute_examples(survival_funcs_df, durations, events, record_ids, model_name, output_dir, n_examples=6, fold=None, subdir='per_minute_risk', log_scale=True, smooth_minutes=None, aggregate_minutes=None):
    """
    从生存曲线 S(t) 计算每分钟非累计风险（事件概率密度离散近似）：
    - 将 S(t) 在每分钟网格上线性插值：t_m = 0, 1min, 2min, ... 直至预测窗口
    - 计算每分钟的非累计风险（概率质量） p_m = max(S(t_{m-1}) - S(t_m), 0)
    - 可选平滑：对 p_m 做分钟窗口的移动平均
    - 默认使用对数坐标展示，避免人为乘常数放大
    输出：
    - 合图（多样本叠加）
    - 每个样本单图与CSV（含时间(分钟)与每分钟概率）
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制每分钟非累计风险：缺少绘图支持")
        return

    import numpy as _np

    try:
        total_hours = float(get_config().data.sequence_generation.prediction_window_hours)
    except Exception:
        try:
            total_hours = float(_GLOBAL_CFG.data.sequence_generation.prediction_window_hours)
        except Exception:
            total_hours = float(survival_funcs_df.columns.astype(float).max()) if hasattr(survival_funcs_df, 'columns') else 24.0

    # 构造每分钟时间轴（单位：分钟）
    total_minutes = int(round(total_hours * 60.0))
    if total_minutes <= 0:
        total_minutes = 1
    t_minutes = _np.arange(0, total_minutes + 1, dtype=float)
    # 同时构造小时轴用于插值
    t_hours_grid = t_minutes / 60.0

    # 选择样本：尽量事件/删失各半
    try:
        events_arr = _np.asarray(events).astype(int)
        ev_idx = [i for i, e in enumerate(events_arr) if int(e) == 1]
        cens_idx = [i for i, e in enumerate(events_arr) if int(e) == 0]
    except Exception:
        ev_idx, cens_idx = [], list(range(len(survival_funcs_df)))
    n = min(int(n_examples), len(survival_funcs_df)) if hasattr(survival_funcs_df, 'iloc') else min(int(n_examples), int(_np.asarray(survival_funcs_df).shape[0]))
    chosen = []
    n_ev = min(len(ev_idx), n // 2)
    chosen.extend(ev_idx[:n_ev])
    n_rem = n - len(chosen)
    if n_rem > 0:
        chosen.extend(cens_idx[:n_rem])
    if len(chosen) == 0:
        chosen = list(range(n))

    # 基于活动区与开始时间排序，便于区分
    def _ar_start_key(idx):
        try:
            rid = record_ids[idx] if record_ids is not None else None
            info = _parse_record_id(rid)
            ar_val = str(info.get('ar') or '')
            st = info.get('start')
            ts = float(st.timestamp()) if (st is not None and hasattr(st, 'timestamp')) else _np.inf
            return (ar_val.lower(), ts, idx)
        except Exception:
            return ('', _np.inf, idx)
    try:
        chosen = sorted(chosen, key=_ar_start_key)
    except Exception:
        pass

    # 准备输出目录
    save_dir = os.path.join(output_dir, subdir) if (subdir and len(str(subdir)) > 0) else output_dir
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception:
        pass

    def _interp_S_to_minutes(S_row, t_cols_hours):
        S_row = _np.asarray(S_row, dtype=float)
        t_cols_hours = _np.asarray(t_cols_hours, dtype=float)
        # 保证单调边界与数值范围
        S_row = _np.clip(S_row, 0.0, 1.0)
        # 线性插值至每分钟小时轴
        try:
            S_min = _np.interp(t_hours_grid, t_cols_hours, S_row, left=1.0, right=float(S_row[-1]))
        except Exception:
            # 回退：假设等间隔
            T = len(S_row)
            t_lin = _np.linspace(0.0, total_hours, T)
            S_min = _np.interp(t_hours_grid, t_lin, S_row, left=1.0, right=float(S_row[-1]))
        # 差分得到每分钟概率质量
        S_prev = _np.concatenate(([1.0], S_min[:-1]))
        p_min = _np.maximum(S_prev - S_min, 0.0)
        return p_min

    def _smooth(y_minute, k_minutes):
        if k_minutes is None or k_minutes <= 1:
            return y_minute
        k = int(max(2, k_minutes))
        kernel = _np.ones(k, dtype=float) / float(k)
        pad = k // 2
        ypad = _np.pad(y_minute, (pad, pad), mode='edge')
        return _np.convolve(ypad, kernel, mode='valid')

    # 合图
    try:
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('tab10')
        t_plot = t_minutes  # 分钟
        # 提取列时间（小时）
        if hasattr(survival_funcs_df, 'columns'):
            t_cols = _np.asarray(survival_funcs_df.columns, dtype=float)
        else:
            T = int(_np.asarray(survival_funcs_df).shape[1])
            t_cols = _np.linspace(0.0, total_hours, T)
        # 为 AR 分配稳定颜色
        try:
            ar_list = []
            for idx in chosen:
                rid = record_ids[idx] if record_ids is not None else None
                info = _parse_record_id(rid)
                ar_val = info.get('ar') or info.get('raw') or str(rid)
                ar_list.append(str(ar_val))
            unique_ars = []
            for a in ar_list:
                if a not in unique_ars:
                    unique_ars.append(a)
            ar_to_color = {a: i for i, a in enumerate(unique_ars)}
        except Exception:
            ar_to_color = {}
        for i, idx in enumerate(chosen):
            # 颜色按 AR 分组，若失败则退化为索引色
            try:
                rid = record_ids[idx] if record_ids is not None else None
                info = _parse_record_id(rid)
                ar_val = info.get('ar') or info.get('raw') or str(rid)
                cidx = ar_to_color.get(str(ar_val), i)
                color = cmap(cidx % 10)
            except Exception:
                color = cmap(i % 10)
            S = survival_funcs_df.iloc[idx].values if hasattr(survival_funcs_df, 'iloc') else _np.asarray(survival_funcs_df[idx], dtype=float)
            p_min = _interp_S_to_minutes(S, t_cols)
            # K分钟聚合（求和，而非平均）以提高幅值与可读性
            if aggregate_minutes is not None and int(aggregate_minutes) > 1:
                k = int(aggregate_minutes)
                kernel = _np.ones(k, dtype=float)
                pad = k - 1
                ypad = _np.pad(p_min, (pad, 0), mode='edge')
                p_min = _np.convolve(ypad, kernel, mode='valid')[:len(p_min)]
            if smooth_minutes is not None and smooth_minutes > 1:
                p_min = _smooth(p_min, smooth_minutes)
            # 构造图例：AR 与开始时间
            try:
                rid = record_ids[idx] if record_ids is not None else None
                info = _parse_record_id(rid)
                ar_val = info.get('ar') or info.get('raw') or str(rid)
                st = info.get('start')
                st_txt = (st.strftime('%Y-%m-%d %H:%M') if st is not None and hasattr(st, 'strftime') else 'NA')
                label_txt = f"AR:{ar_val} | start:{st_txt}"
            except Exception:
                label_txt = f"idx={idx}"
            plt.plot(t_plot, p_min, color=color, linewidth=1.8, label=label_txt)
            try:
                if int(events_arr[idx]) == 1:
                    t_ev_min = float(durations[idx]) * 60.0
                    if _np.isfinite(t_ev_min):
                        plt.axvline(x=t_ev_min, color=color, linestyle='--', alpha=0.6)
            except Exception:
                pass
        plt.xlabel('Time (minutes)')
        plt.ylabel('Non-cumulative risk per minute (probability)')
        title = f'{model_name} Non-cumulative Risk (per-minute)'
        if fold is not None:
            title += f' (fold{fold})'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if log_scale:
            try:
                plt.yscale('log')
            except Exception:
                pass
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{model_name}_per_minute_risk{('_fold'+str(fold)) if fold is not None else ''}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Per-minute non-cumulative risk saved: {out_path}")
    except Exception as e:
        logger.warning(f"绘制每分钟非累计风险合图失败: {e}")

    # 单图与CSV
    try:
        if hasattr(survival_funcs_df, 'columns'):
            t_cols = _np.asarray(survival_funcs_df.columns, dtype=float)
        else:
            T = int(_np.asarray(survival_funcs_df).shape[1])
            t_cols = _np.linspace(0.0, total_hours, T)
        for idx in chosen:
            S = survival_funcs_df.iloc[idx].values if hasattr(survival_funcs_df, 'iloc') else _np.asarray(survival_funcs_df[idx], dtype=float)
            p_min = _interp_S_to_minutes(S, t_cols)
            if aggregate_minutes is not None and int(aggregate_minutes) > 1:
                k = int(aggregate_minutes)
                kernel = _np.ones(k, dtype=float)
                pad = k - 1
                ypad = _np.pad(p_min, (pad, 0), mode='edge')
                p_min = _np.convolve(ypad, kernel, mode='valid')[:len(p_min)]
            if smooth_minutes is not None and smooth_minutes > 1:
                p_min_s = _smooth(p_min, smooth_minutes)
            else:
                p_min_s = p_min
            # 图
            plt.figure(figsize=(8, 4))
            plt.plot(t_minutes, p_min_s, color='C0', linewidth=1.8)
            try:
                if int(events_arr[idx]) == 1:
                    t_ev_min = float(durations[idx]) * 60.0
                    if _np.isfinite(t_ev_min):
                        plt.axvline(x=t_ev_min, color='C1', linestyle='--', alpha=0.8)
            except Exception:
                pass
            plt.xlabel('Time (minutes)')
            plt.ylabel('Non-cumulative risk per minute (probability)')
            # 标题显示 AR 与开始时间
            try:
                rid = record_ids[idx] if record_ids is not None else None
                info = _parse_record_id(rid)
                ar_val = info.get('ar') or info.get('raw') or str(rid)
                st = info.get('start')
                st_txt = (st.strftime('%Y-%m-%d %H:%M') if st is not None and hasattr(st, 'strftime') else 'NA')
                ttl = f'AR:{ar_val} | start:{st_txt} | idx={idx}'
            except Exception:
                ttl = f'idx={idx} per-minute risk'
            plt.title(ttl)
            plt.grid(True, alpha=0.3)
            if log_scale:
                try:
                    plt.yscale('log')
                except Exception:
                    pass
            plt.tight_layout()
            f_png = os.path.join(save_dir, f"{model_name}_per_minute_risk_idx{idx}{('_fold'+str(fold)) if fold is not None else ''}.png")
            plt.savefig(f_png, dpi=300, bbox_inches='tight')
            plt.close()
            # CSV
            try:
                import csv as _csv
                f_csv = os.path.join(save_dir, f"{model_name}_per_minute_risk_idx{idx}{('_fold'+str(fold)) if fold is not None else ''}.csv")
                with open(f_csv, 'w', newline='') as cf:
                    w = _csv.writer(cf)
                    # 附带 AR 与开始时间便于外部分析
                    try:
                        rid = record_ids[idx] if record_ids is not None else None
                        info = _parse_record_id(rid)
                        ar_val = info.get('ar') or info.get('raw') or str(rid)
                        st = info.get('start')
                        st_txt = (st.isoformat() if st is not None and hasattr(st, 'isoformat') else '')
                    except Exception:
                        ar_val, st_txt = '', ''
                    w.writerow(['time_min', 'p_per_min', 'p_per_min_smoothed', 'AR', 'start_iso'])
                    for t, pv, pvs in zip(t_minutes, p_min, p_min_s):
                        w.writerow([float(t), float(pv) if _np.isfinite(pv) else '', float(pvs) if _np.isfinite(pvs) else '', ar_val, st_txt])
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"生成每分钟非累计风险单图失败: {e}")

# --- New: Time-wise separation visualization (events vs censored) ---
def plot_timewise_separation(survival_funcs_df=None, pred_probs=None, events=None, model_name="", output_dir="", fold=None, smooth=False):
    """
    Visualize the model's continuous predictive ability across time by contrasting
    per-time risk between event and non-event samples.

    Inputs (one of):
    - survival_funcs_df: DataFrame shape (n_samples, n_times) for S(t)
    - pred_probs: np.ndarray/DataFrame shape (n_samples, n_times) for per-time event probability mass p(t)
    - events: array-like shape (n_samples,), 1 for event, 0 for censored

    Outputs:
    - Saves two figures under output_dir:
      1) mean per-time risk for events vs censored (with +/-1 SEM bands)
      2) per-time ROC AUC separating events vs censored using per-time risk values
    """
    logger = logging.getLogger(__name__)
    if not HAS_PLOTTING:
        logger.warning("无法绘制time-wise分离图：缺少绘图支持")
        return

    import numpy as _np
    import matplotlib.pyplot as plt
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    if events is None:
        logger.warning("plot_timewise_separation: 缺少 events，跳过绘图。")
        return

    # Build risk matrix R(t):
    # - If pred_probs provided, treat as per-time event probability mass p(t)
    # - Else if survival provided, derive p(t) = S(t_{k-1}) - S(t_k)
    risk_mat = None
    t_axis = None
    try:
        ev_arr = _np.asarray(events).astype(int)
    except Exception:
        ev_arr = _np.asarray(events)

    if pred_probs is not None:
        try:
            P = _np.asarray(pred_probs, dtype=float)
            if P.ndim == 1:
                P = P.reshape(1, -1)
            P = _np.clip(P, 0.0, 1.0)
            risk_mat = P
        except Exception as e:
            logger.warning(f"plot_timewise_separation: pred_probs 解析失败: {e}")
            risk_mat = None
        # time axis (hours) unknown here; try to infer from config or fallback to bins index
        try:
            total_hours = float(get_config().data.sequence_generation.prediction_window_hours)
            T = risk_mat.shape[1]
            t_axis = _np.linspace(0.0, total_hours, T)
        except Exception:
            t_axis = _np.arange(risk_mat.shape[1], dtype=float)
    elif survival_funcs_df is not None:
        try:
            S = _np.asarray(survival_funcs_df.values, dtype=float)
            S = _np.clip(S, 0.0, 1.0)
            # p(t_k) = S(t_{k-1}) - S(t_k), with S(t_{-1}) := 1
            S_prev = _np.concatenate([_np.ones((S.shape[0], 1), dtype=float), S[:, :-1]], axis=1)
            risk_mat = _np.maximum(S_prev - S, 0.0)
            t_axis = _np.asarray(survival_funcs_df.columns, dtype=float)
        except Exception as e:
            logger.warning(f"plot_timewise_separation: survival_funcs_df 解析失败: {e}")
            return
    else:
        logger.warning("plot_timewise_separation: 既没有 survival_funcs_df 也没有 pred_probs，跳过。")
        return

    # Optionally smooth along time for readability (simple moving average)
    if smooth:
        try:
            k = max(3, int(risk_mat.shape[1] // 50))
            if k % 2 == 0:
                k += 1
            kernel = _np.ones(k, dtype=float) / float(k)
            pad = k // 2
            risk_mat_pad = _np.pad(risk_mat, ((0, 0), (pad, pad)), mode='edge')
            risk_mat = _np.apply_along_axis(lambda v: _np.convolve(v, kernel, mode='valid'), axis=1, arr=risk_mat_pad)
        except Exception:
            pass

    # Split by event status
    try:
        # 确保ev_arr和risk_mat的维度匹配
        if len(ev_arr) != risk_mat.shape[0]:
            logger.warning(f"plot_timewise_separation: events长度({len(ev_arr)})与risk_mat行数({risk_mat.shape[0]})不匹配，跳过绘图")
            return
        ev_mask = (ev_arr == 1)
        ce_mask = (ev_arr == 0)
    except Exception as e:
        logger.warning(f"plot_timewise_separation: 事件状态分割失败: {e}")
        return

    def _sem(x, axis=0):
        x = _np.asarray(x, dtype=float)
        n = _np.sum(_np.isfinite(x), axis=axis)
        std = _np.nanstd(x, axis=axis)
        with _np.errstate(divide='ignore', invalid='ignore'):
            return _np.where(n > 0, std / _np.sqrt(_np.maximum(n, 1)), _np.nan)

    # Figure 1: mean risk curves with SEM bands
    try:
        plt.figure(figsize=(12, 6))
        mu_ev = _np.nanmean(risk_mat[ev_mask], axis=0) if _np.any(ev_mask) else _np.full((risk_mat.shape[1],), _np.nan)
        mu_ce = _np.nanmean(risk_mat[ce_mask], axis=0) if _np.any(ce_mask) else _np.full((risk_mat.shape[1],), _np.nan)
        se_ev = _sem(risk_mat[ev_mask], axis=0) if _np.any(ev_mask) else _np.full((risk_mat.shape[1],), _np.nan)
        se_ce = _sem(risk_mat[ce_mask], axis=0) if _np.any(ce_mask) else _np.full((risk_mat.shape[1],), _np.nan)
        plt.plot(t_axis, mu_ev, color='C1', label='Event (mean)', linewidth=2.0)
        plt.fill_between(t_axis, mu_ev - se_ev, mu_ev + se_ev, color='C1', alpha=0.2)
        plt.plot(t_axis, mu_ce, color='C0', label='Censored (mean)', linewidth=2.0)
        plt.fill_between(t_axis, mu_ce - se_ce, mu_ce + se_ce, color='C0', alpha=0.2)
        ttl = f"{model_name} Time-wise Risk (event vs censored)"
        if fold is not None:
            ttl += f" (fold{fold})"
        plt.title(ttl)
        plt.xlabel('Time (hours)')
        plt.ylabel('Per-time risk (probability mass)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        f1 = os.path.join(output_dir, f"{model_name}_timewise_risk_mean{('_fold'+str(fold)) if fold is not None else ''}.png")
        plt.savefig(f1, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Time-wise mean risk plot saved: {f1}")
    except Exception as e:
        logger.warning(f"绘制time-wise均值曲线失败: {e}")

    # Figure 2: per-time ROC AUC between groups (if sklearn available)
    if roc_auc_score is not None:
        try:
            y = ev_arr.astype(int)
            auc_t = []
            for j in range(risk_mat.shape[1]):
                xj = risk_mat[:, j]
                # require at least 2 classes present
                if _np.unique(y[_np.isfinite(xj)]).size < 2:
                    auc_t.append(_np.nan)
                    continue
                try:
                    auc_val = roc_auc_score(y[_np.isfinite(xj)], xj[_np.isfinite(xj)])
                except Exception:
                    auc_val = _np.nan
                auc_t.append(float(auc_val) if _np.isfinite(auc_val) else _np.nan)
            auc_t = _np.asarray(auc_t, dtype=float)
            plt.figure(figsize=(12, 4))
            plt.plot(t_axis, auc_t, color='purple', linewidth=2.0)
            plt.axhline(0.5, color='gray', linestyle='--', alpha=0.6)
            ttl2 = f"{model_name} Per-time ROC AUC"
            if fold is not None:
                ttl2 += f" (fold{fold})"
            plt.title(ttl2)
            plt.xlabel('Time (hours)')
            plt.ylabel('AUC(t)')
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            f2 = os.path.join(output_dir, f"{model_name}_timewise_auc{('_fold'+str(fold)) if fold is not None else ''}.png")
            plt.savefig(f2, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Per-time AUC plot saved: {f2}")
        except Exception as e:
            logger.warning(f"计算/绘制 per-time AUC 失败: {e}")
    else:
        logger.info("sklearn 不可用，跳过 per-time AUC 曲线绘制。")


def plot_ar_dynamic_prediction_with_features(X_test, durations_for_plots, events, survival_funcs_df, pmf_T_numpy, risk_scores, record_ids, feature_names, model_name, output_dir, dpi=300, raw_samples=None, feature_importances=None, sample_indices=None):
    """
    此函数用于可视化特定样本的动态预测结果：
    1. 自动筛选代表性样本（最高/最低风险及随机抽取）。
    2. 提取并展示历史特征轨迹（归一化）与真实未来轨迹（原始值）。
    3. 叠加展示预测的存活曲线 (S(t)) 与概率密度函数 (PMF)。
    4. 标注事件发生或删失的时间点，并生成风险摘要。
    """
    import logging
    import os
    import numpy as _np
    import pandas as pd
    import matplotlib.patches as patches
    
    logger = logging.getLogger(__name__)
    plt_local = _ensure_matplotlib()
    if plt_local is None:
        return
        
    try:
        if hasattr(X_test, 'cpu'):
            X_test = X_test.detach().cpu().numpy()
            
        # --- 1. 样本选择逻辑 ---
        # --- 1. 样本选择逻辑 ---
        if sample_indices is not None:
            candidates = list(sample_indices)
        else:
            # 保证包含事件样本 (Selection logic rewritten to ensure event coverage)
            event_indices = _np.where(events == 1)[0]
            censored_indices = _np.where(events == 0)[0]
            
            candidates = []
            # 获取最高风险的5个事件样本
            if len(event_indices) > 0:
                e_risks = risk_scores[event_indices]
                candidates.extend(event_indices[_np.argsort(e_risks)[-5:]].tolist())
            
            # 获取最低风险的5个删失样本
            if len(censored_indices) > 0:
                c_risks = risk_scores[censored_indices]
                candidates.extend(censored_indices[_np.argsort(c_risks)[:5]].tolist())
            
            # 补充随机样本
            potential_random = [i for i in range(len(risk_scores)) if i not in candidates]
            if potential_random:
                _np.random.seed(42)
                candidates.extend(_np.random.choice(potential_random, min(5, len(potential_random)), replace=False).tolist())
            
            seen = set()
            candidates = [x for x in candidates if not (x in seen or seen.add(x))]
            
        for idx in candidates:
            idx = int(idx)
            surv_curve = survival_funcs_df.iloc[idx]
            time_bins = surv_curve.index.astype(float).values
            surv_vals = surv_curve.values
            
            # --- 2. 元数据处理与安全文件名生成 ---
            rec = record_ids[idx] if (record_ids is not None and len(record_ids) > idx) else {}
            ar_id = rec.get('ar') or rec.get('raw') or f'Sample_{idx}'
            ar_id_safe = "".join([c if c.isalnum() or c in "-_" else "_" for c in str(ar_id)])[:50]
            
            # --- 3. 特征重要性排序与提取 ---
            num_feats = X_test.shape[2]
            orig_feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(num_feats)]
            f_names = sorted(orig_feature_names, key=lambda x: feature_importances.get(x, 0.0) if feature_importances else 0.0, reverse=True)
            n_plot_feats = min(10, len(f_names), num_feats)
            f_names = f_names[:n_plot_feats]

            # --- 4. 提取历史与未来轨迹 ---
            seq_len = X_test.shape[1]
            real_feature_series = [] 
            past_feature_series = []
            real_future_found = False
            
            # Find raw parent sample to extract raw trajectories (Unify data source to match feature scales)
            parent = None
            if raw_samples is not None:
                record_id_val = rec.get('raw') or rec.get('ar') or ''
                ar_raw_str = str(record_id_val).lower().strip()
                # Try exact match first, then partial
                parent = next((s for s in raw_samples if ar_raw_str == str(s.get('record_id', '')).lower().strip()), None)
                if parent is None:
                    parent = next((s for s in raw_samples if ar_raw_str in str(s.get('record_id', '')).lower()), None)

            if parent is not None and rec.get('subseq_start'):
                try:
                    df_features, ts_raw = parent['features'], pd.to_datetime(parent['timestamps_list'])
                    T_start_pd = pd.to_datetime(rec['subseq_start'])
                    
                    # Find T0 (The end of the input sequence)
                    # The subsequence starts at T_start_pd and has seq_len steps. 
                    # We assume standard cadence to find the index of the last input step T0.
                    idx_start = (ts_raw - T_start_pd).dt.total_seconds().abs().argmin()
                    idx_T0 = idx_start + seq_len - 1
                    
                    if idx_T0 >= len(ts_raw):
                        idx_T0 = len(ts_raw) - 1
                        
                    T0_pd = ts_raw.iloc[idx_T0]
                    dt_hours_all = (ts_raw - T0_pd).dt.total_seconds() / 3600.0
                    
                    # 4.1 Define masks for visualization
                    # History: show roughly 1-2x the sequence length worth of history for context
                    hist_start_idx = max(0, idx_T0 - int(seq_len * 1.5))
                    hist_mask = (dt_hours_all <= 0) & (dt_hours_all >= dt_hours_all.iloc[hist_start_idx])
                    
                    # Future: show full prediction window
                    pred_window = max(time_bins)
                    future_mask = (dt_hours_all >= 0.0) & (dt_hours_all <= pred_window + 2.0)
                    
                    for fn in f_names:
                        base_fn = fn.split('_')[0]
                        col = base_fn if base_fn in df_features.columns else (fn if fn in df_features.columns else None)
                        if col:
                            past_feature_series.append((fn, dt_hours_all[hist_mask].values, df_features[col].values[hist_mask]))
                            real_feature_series.append((fn, dt_hours_all[future_mask].values, df_features[col].values[future_mask]))
                        else:
                            past_feature_series.append((fn, None, None))
                            real_feature_series.append((fn, None, None))
                    real_future_found = True
                except Exception as e:
                    logger.debug(f"Raw trajectory extraction failed for {ar_id_safe}: {e}")

            # Fallback: 如果没找到 parent，尽量把 index 转为 hours (Assume standard 12min cadence if unknown)
            if not real_future_found:
                # 尝试从数据推断步长
                cadence_hours = 0.2 # 默认SWAN 12分钟
                past_times = _np.linspace(-seq_len * cadence_hours, 0, seq_len)
                for f_i, fn in enumerate(f_names):
                    feat_idx = orig_feature_names.index(fn)
                    past_feature_series.append((fn, past_times, X_test[idx, :, feat_idx]))
                    real_feature_series.append((fn, None, None))

            # --- 5. 绘图布局与轴优化 ---
            n_rows = n_plot_feats + 1
            colors = plt_local.cm.tab10.colors
            x_max_plot = max([_np.max(rt) for _, rt, rv in real_feature_series if rt is not None] + [0]) if real_future_found else min(max(time_bins), durations_for_plots[idx] + 2)
            
            fig, axes = plt_local.subplots(n_rows, 1, figsize=(12, 0.9 * n_plot_feats + 4.5), sharex=True, gridspec_kw={'height_ratios': [1.0]*n_plot_feats + [4.0]})
            
            for f_i, fn in enumerate(f_names):
                ax_f = axes[f_i]
                c_feat = colors[f_i % 10]
                
                # 绘制历史 (dt <= 0)
                _, p_t, p_v = past_feature_series[f_i]
                if p_v is not None:
                    ax_f.plot(p_t, p_v, color='#7f7f7f', alpha=0.6, linestyle='-', linewidth=1.5, label='History')
                
                # 绘制未来真实轨迹 (dt >= 0)
                all_vals = []
                if p_v is not None: all_vals.append(p_v)
                
                if real_future_found and len(real_feature_series) > f_i:
                    _, rt, rv = real_feature_series[f_i]
                    if rv is not None:
                        ax_f.plot(rt, rv, color=c_feat, marker='o', markersize=3, linewidth=2, label='True Future')
                        all_vals.append(rv)
                
                # 动态调整 Y 轴范围
                if all_vals:
                    concat_v = _np.concatenate([v.flatten() for v in all_vals])
                    v_min, v_max = _np.nanmin(concat_v), _np.nanmax(concat_v)
                    v_range = v_max - v_min
                    if v_range == 0: v_range = 1.0
                    ax_f.set_ylim(v_min - 0.15*v_range, v_max + 0.15*v_range)
                
                ax_f.axvline(0, color='black', alpha=0.3)
                ax_f.set_ylabel(fn[:15], fontsize=8, fontweight='bold', rotation=0, labelpad=40)
                ax_f.grid(True, alpha=0.2)

            # --- 6. 存活曲线与 PMF 绘制 ---
            ax_surv = axes[-1]
            ax_surv.plot(time_bins, surv_vals, color='#d62728', linewidth=3, label='Survival $S(t)$')
            ax_surv.fill_between(time_bins, surv_vals, 0, color='#d62728', alpha=0.1)
            
            if pmf_T_numpy is not None:
                ax_pmf = ax_surv.twinx()
                ax_pmf.plot(time_bins, pmf_T_numpy[idx], color='#1f77b4', alpha=0.6, label='Prob Density (PMF)')
                ax_pmf.fill_between(time_bins, 0, pmf_T_numpy[idx], color='#1f77b4', alpha=0.1)
                ax_pmf.set_ylabel('Density', color='#1f77b4')

            # 标注事件状态
            actual_t = durations_for_plots[idx]
            is_event = int(events[idx]) == 1
            ax_surv.axvline(actual_t, color='black', linestyle='--' if is_event else '-.', alpha=0.8)
            ax_surv.text(actual_t, 0.5, ' EVENT' if is_event else ' CENSORED', fontweight='bold')
            
            # 风险摘要 Badge
            risk_pct = (_np.sum(risk_scores <= risk_scores[idx]) / len(risk_scores)) * 100
            perf_text = f"Risk Score: {risk_scores[idx]:.4f}\nPercentile: {risk_pct:.1f}%\nTime: {actual_t:.1f}h"
            ax_surv.text(0.98, 0.95, perf_text, transform=ax_surv.transAxes, bbox=dict(facecolor='white', alpha=0.8), ha='right', va='top', family='monospace')

            # Focus view on prediction window but keep visible history
            t_min_plot = min([_np.min(p_t) for _, p_t, p_v in past_feature_series if p_t is not None] + [-6]) 
            ax_surv.set_xlim(t_min_plot, x_max_plot)
            ax_surv.set_xlabel('Hours from $T_0$ (Now)')
            fig.suptitle(f'Trajectory Analysis: {ar_id_safe} | Model: {model_name}', fontsize=14, fontweight='bold')
            
            os.makedirs(output_dir, exist_ok=True)
            status_str = "event" if is_event else "censored"
            plt_local.savefig(os.path.join(output_dir, f'{model_name}_{status_str}_{ar_id_safe}.png'), dpi=dpi, bbox_inches='tight')
            plt_local.close(fig)
            
        logger.info(f"Dynamic Trajectory plots saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot dynamic trajectories: {e}", exc_info=True)


# Ensure exported names exist for imports elsewhere in the codebase
try:
    # If functions were defined earlier, ensure they are present
    __all__ = [
    'plot_training_history', 'plot_survival_curves_by_risk_group', 'plot_survival_curves_by_risk_group_deephit',
    'plot_deephit_probability_distribution', 'plot_individual_survival_curves', 'plot_roc_curves', 'plot_auc_over_time',
    'plot_enhanced_training_curves', 'plot_model_performance_comparison', 'plot_deephit_probability_distribution',
    'plot_event_probability_curves', 'plot_calibration_curves', 'plot_feature_interactions', 'plot_feature_importance',
    'plot_training_curves', 'plot_risk_distribution_by_event_type', 'plot_survival_time_distribution',
    'plot_feature_correlation_heatmap', 'plot_model_comparison', 'plot_confidence_intervals', 'plot_ar_survival_curves',
    'plot_regression_analysis', 'plot_publication_training_curves', 'plot_publication_risk_distribution', 'plot_feature_importance_barh',
    'plot_hazard_curves_examples', 'plot_non_cumulative_risk_per_minute_examples', 'plot_timewise_separation',
    'plot_ar_dynamic_prediction_with_features'
    ]
except Exception:
    pass
