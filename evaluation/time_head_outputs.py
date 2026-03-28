"""Utility helpers for generating time-head style evaluation artefacts.

This module centralises the logic that was previously embedded inside
``main._evaluate_time_head_predictions`` so that other components (such as
the prediction methods comparison workflow) can reuse the same visualisation
and reporting pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional
import re

import numpy as np
import pandas as pd

# Ensure matplotlib uses a non-interactive backend before importing pyplot
try:
    import matplotlib

    try:
        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    except Exception:  # pragma: no cover - defensive
        pass

    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - headless fall-back
    matplotlib = None
    plt = None
    logging.getLogger(__name__).warning(
        "Matplotlib backend unavailable (%s); skipping time-head style plots.", exc
    )


LOGGER = logging.getLogger(__name__)


def _ensure_valid_arrays(
    true_times: np.ndarray,
    events: np.ndarray,
    preds: np.ndarray,
    eval_events_only: bool,
) -> Optional[np.ndarray]:
    """Return a boolean mask of valid samples or ``None`` when empty."""

    if true_times.size == 0 or preds.size == 0 or events.size == 0:
        return None

    n = min(true_times.shape[0], preds.shape[0], events.shape[0])
    true_times = true_times[:n]
    events = events[:n]
    preds = preds[:n]

    mask = np.isfinite(true_times) & np.isfinite(preds)
    if eval_events_only:
        mask &= (events == 1)
        mask &= (true_times > 0)
    else:
        mask &= (true_times >= 0)

    return mask if np.any(mask) else None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression-style diagnostics used by time-head evaluations."""

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(
            np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1e-6, None))) * 100.0
        )
    bias = float(np.mean(y_pred - y_true))

    if y_true.shape[0] > 1:
        try:
            corr = float(np.corrcoef(y_pred, y_true)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        except Exception:  # pragma: no cover - defensive
            corr = 0.0
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    else:
        corr = 0.0
        r2 = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
        "correlation": float(corr),
        "r2": float(r2),
        "n_samples": int(y_true.shape[0]),
    }


def generate_time_head_style_outputs(
    output_dir: str,
    split: str,
    true_times: np.ndarray,
    events: np.ndarray,
    preds: np.ndarray,
    *,
    model_name: str = "TimeHead",
    file_tag: str = "time_head",
    eval_events_only: bool = True,
    record_ids: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    """Generate metrics, comparison tables and plots matching time-head outputs.

    Returns the computed metrics when successful, otherwise ``None``.
    """

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:  # pragma: no cover - IO guard
        LOGGER.warning("Failed to create output directory %s: %s", output_dir, exc)
        return None

    true_times = np.asarray(true_times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float).reshape(-1)

    valid_mask = _ensure_valid_arrays(true_times, events, preds, eval_events_only)
    if valid_mask is None:
        LOGGER.warning(
            "[TimeHeadStyle] No valid samples for %s (%s); skipping outputs.",
            model_name,
            file_tag,
        )
        return None

    # Align arrays to the valid mask
    true_used = true_times[: valid_mask.shape[0]][valid_mask].astype(float)
    pred_used = preds[: valid_mask.shape[0]][valid_mask].astype(float)

    # 如果提供了 record_ids，则仅保留每个活动区（AR）的第一个滑动窗样本
    if record_ids is not None:
        try:
            rids = np.asarray(record_ids).reshape(-1)[: valid_mask.shape[0]][valid_mask]
            # 提取标准化的 ar key（如 'ar123'），若无法解析则使用原始字符串
            ar_keys = []
            for v in rids:
                try:
                    s = str(v)
                    m = re.search(r'ar\D*?(\d+)', s, flags=re.IGNORECASE)
                    if m:
                        ar_keys.append(f"ar{int(m.group(1))}")
                        continue
                    m2 = re.search(r'\b([aA][rR]?\d+)\b', s)
                    if m2:
                        ar_keys.append(m2.group(1).lower())
                    else:
                        ar_keys.append(s)
                except Exception:
                    ar_keys.append(str(v))

            seen = set()
            keep_indices = []
            for i, k in enumerate(ar_keys):
                if k not in seen:
                    seen.add(k)
                    keep_indices.append(i)

            if len(keep_indices) > 0:
                keep_mask = np.zeros_like(true_used, dtype=bool)
                keep_mask[np.array(keep_indices, dtype=int)] = True
                true_used = true_used[keep_mask]
                pred_used = pred_used[keep_mask]
            else:
                LOGGER.debug("record_ids provided but no AR keys extracted; leaving samples unchanged")
        except Exception as exc:
            LOGGER.warning("Failed to filter by record_ids for AR-first-sample logic: %s", exc)

    metrics = _compute_metrics(true_used, pred_used)

    metrics_path = os.path.join(output_dir, f"{split}_{file_tag}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    comp_df = pd.DataFrame(
        {
            "true_time": true_used,
            "pred_time": pred_used,
            "absolute_error": np.abs(pred_used - true_used),
            "relative_error_percent": np.abs(
                (pred_used - true_used) / np.clip(true_used, 1e-6, None)
            )
            * 100.0,
            "relative_error_percent_signed": (
                (pred_used - true_used) / np.clip(true_used, 1e-6, None)
            ) * 100.0,  # 非绝对值的相对误差（可以有正负）
            "bias": pred_used - true_used,
        }
    )

    comp_path = os.path.join(output_dir, f"{split}_{file_tag}_comparison_table.csv")
    comp_df.to_csv(comp_path, index=False)

    predictions_path = os.path.join(output_dir, f"{split}_{file_tag}_predictions.csv")
    comp_df[["true_time", "pred_time"]].to_csv(predictions_path, index=False)

    if plt is None:  # pragma: no cover - plotting disabled
        return metrics

    try:
        max_axis = float(max(np.max(true_used), np.max(pred_used)))

        plt.figure(figsize=(8, 7))
        plt.scatter(true_used, pred_used, s=12, alpha=0.6)
        plt.plot([0, max_axis], [0, max_axis], "r--", lw=2)
        plt.xlabel("True Time (h)")
        plt.ylabel("Predicted Time (h)")
        plt.title(f"{model_name} Pred vs True ({split})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_pred_vs_true.png"),
            dpi=300,
        )
        plt.close()

        errors = pred_used - true_used
        plt.figure(figsize=(8, 7))
        plt.hist(np.abs(errors), bins=30, alpha=0.8, color="tab:blue", edgecolor="black")
        plt.axvline(metrics["mae"], color="red", ls="--", lw=2, label=f"MAE={metrics['mae']:.2f}h")
        plt.xlabel("Absolute Error (h)")
        plt.ylabel("Count")
        plt.title(f"{model_name} Error Distribution ({split})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_error_distribution.png"),
            dpi=300,
        )
        plt.close()

        plt.figure(figsize=(8, 7))
        plt.hist(true_used, bins=20, alpha=0.6, label="True", density=True, color="tab:red")
        plt.hist(pred_used, bins=20, alpha=0.6, label="Pred", density=True, color="tab:blue")
        plt.xlabel("Time (h)")
        plt.ylabel("Density")
        plt.title(f"{model_name} Time Distribution ({split})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_time_distribution.png"),
            dpi=300,
        )
        plt.close()

        plt.figure(figsize=(8, 7))
        plt.scatter(pred_used, errors, s=12, alpha=0.6)
        plt.axhline(0, color="red", ls="--", lw=2)
        plt.xlabel("Predicted (h)")
        plt.ylabel("Residual (Pred-True) (h)")
        plt.title(f"{model_name} Residuals ({split})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_residuals.png"),
            dpi=300,
        )
        plt.close()

        # 偏差分布图（非绝对值的偏差，类似绝对误差分布图但显示正负）
        # 使用残差（pred - true），单位是小时，可以有正负值
        errors = pred_used - true_used  # 残差，单位是小时
        # 过滤无效值
        valid_errors = errors[np.isfinite(errors)]
        
        plt.figure(figsize=(8, 7))
        
        # 绘制直方图（类似绝对误差分布图）
        n, bins, patches = plt.hist(valid_errors, bins=30, alpha=0.8, color="tab:green", 
                                     edgecolor="black", linewidth=0.5)
        
        # 添加正态分布拟合曲线
        try:
            from scipy import stats
            # 计算正态分布参数
            mu, sigma = np.mean(valid_errors), np.std(valid_errors)
            # 生成平滑的x轴数据，覆盖整个绘图范围(-50到50)以补全曲线
            x_fit = np.linspace(-50, 50, 400)
            # 计算正态分布PDF
            y_fit = stats.norm.pdf(x_fit, mu, sigma)
            # 将PDF转换为计数尺度（与直方图匹配）
            bin_width = bins[1] - bins[0]
            y_fit_scaled = y_fit * len(valid_errors) * bin_width
            # 绘制平滑曲线
            plt.plot(x_fit, y_fit_scaled, 'g-', linewidth=2.5, 
                    label=f'Normal Fit (μ={mu:.2f}h, σ={sigma:.2f}h)', alpha=0.9)
        except Exception:
            # 如果scipy不可用，尝试使用KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(valid_errors)
                x_fit = np.linspace(-50, 50, 400)
                y_fit = kde(x_fit)
                # 转换为计数尺度
                bin_width = bins[1] - bins[0]
                y_fit_scaled = y_fit * len(valid_errors) * bin_width
                plt.plot(x_fit, y_fit_scaled, 'g-', linewidth=2.5, 
                        label='KDE Fit', alpha=0.9)
            except Exception:
                pass  # 如果都不可用，只显示直方图
        
        # 添加0误差线（完美预测）
        plt.axvline(0, color="black", ls="-", lw=2, alpha=0.7, label="Perfect Prediction")
        
        # 固定横轴范围为-50到50
        plt.xlim(-50, 50)
        
        plt.xlabel("Error (h)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"{model_name} Error Distribution ({split})", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_relative_error_distribution.png"),
            dpi=300,
        )
        plt.close()

        # 相对误差 vs 真实时间散点图（使用非绝对值的相对误差）
        relative_errors_signed = comp_df["relative_error_percent_signed"].values
        # 确保true_used和relative_errors_signed长度匹配
        mask = np.isfinite(relative_errors_signed)
        if len(mask) == len(true_used):
            true_for_plot = true_used[mask]
            rel_errors_for_plot = relative_errors_signed[mask]
        else:
            # 如果长度不匹配，直接使用所有有效值
            true_for_plot = true_used
            rel_errors_for_plot = relative_errors_signed[np.isfinite(relative_errors_signed)]
        
        plt.figure(figsize=(8, 7))
        plt.scatter(true_for_plot, rel_errors_for_plot, s=12, alpha=0.6, color="tab:purple")
        # 使用平均绝对相对误差作为参考线
        mean_abs_rel_error = np.mean(np.abs(rel_errors_for_plot))
        plt.axhline(mean_abs_rel_error, color="red", ls="--", lw=2, 
                   label=f"Mean |Rel Error|={mean_abs_rel_error:.2f}%")
        plt.axhline(-mean_abs_rel_error, color="red", ls="--", lw=2, alpha=0.5)
        plt.axhline(0, color="black", ls="-", lw=2, alpha=0.7, label="Perfect Prediction")
        plt.xlabel("True Time (h)", fontsize=12)
        plt.ylabel("Relative Error (%)", fontsize=12)
        plt.title(f"{model_name} Relative Error vs True Time ({split})", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{split}_{file_tag}_relative_error_vs_true.png"),
            dpi=300,
        )
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting guard
        LOGGER.warning("[TimeHeadStyle] Plotting failed for %s: %s", model_name, exc)

    return metrics


