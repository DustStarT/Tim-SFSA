"""Censor-aware discrimination, calibration and cluster-bootstrap metrics."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-8


def trapezoidal_integral(values, coordinates) -> float:
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def _km_curve(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    times = np.unique(durations)
    survival = 1.0
    values = []
    for time in times:
        at_risk = np.sum(durations >= time)
        failures = np.sum(events & np.isclose(durations, time, rtol=0.0, atol=1e-10))
        if at_risk > 0:
            survival *= 1.0 - failures / at_risk
        values.append(survival)
    return times, np.asarray(values, dtype=float)


def _step_value(times: np.ndarray, values: np.ndarray, query) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    positions = np.searchsorted(times, query, side="right") - 1
    result = np.ones_like(query, dtype=float)
    valid = positions >= 0
    result[valid] = values[positions[valid]]
    return result


def km_survival_at(duration, event, query) -> np.ndarray:
    """Kaplan--Meier event-free survival evaluated at one or more times."""
    return _step_value(*_km_curve(duration, np.asarray(event, dtype=bool)), query)


class IPCW:
    def __init__(self, train_duration: np.ndarray, train_event: np.ndarray):
        self.times, self.survival = _km_curve(train_duration, ~np.asarray(train_event, dtype=bool))

    def g(self, time) -> np.ndarray:
        return np.clip(_step_value(self.times, self.survival, time), EPS, 1.0)

    def horizon_weights(self, duration, event, horizon):
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=bool)
        case = event & (duration <= horizon)
        control = duration > horizon
        known = case | control
        weights = np.zeros_like(duration, dtype=float)
        weights[case] = 1.0 / self.g(duration[case])
        weights[control] = 1.0 / self.g(np.asarray([horizon]))[0]
        labels = case.astype(int)
        return labels, known, weights


def harrell_c_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    try:
        from sksurv.metrics import concordance_index_censored
        return float(concordance_index_censored(
            np.asarray(event, dtype=bool), np.asarray(duration, dtype=float), np.asarray(risk, dtype=float)
        )[0])
    except Exception:
        concordant = comparable = tied = 0.0
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=bool)
        risk = np.asarray(risk, dtype=float)
        for index in np.where(event)[0]:
            later = np.where(duration > duration[index])[0]
            comparable += len(later)
            concordant += np.sum(risk[index] > risk[later])
            tied += np.sum(risk[index] == risk[later])
        return float((concordant + 0.5 * tied) / comparable) if comparable else float("nan")


def uno_c_index(train_duration, train_event, duration, event, risk) -> float:
    try:
        from sksurv.metrics import concordance_index_ipcw
        y_train = np.array(
            list(zip(np.asarray(train_event, dtype=bool), np.asarray(train_duration, dtype=float))),
            dtype=[("event", "?"), ("time", "<f8")],
        )
        y_test = np.array(
            list(zip(np.asarray(event, dtype=bool), np.asarray(duration, dtype=float))),
            dtype=[("event", "?"), ("time", "<f8")],
        )
        tau = min(float(np.max(train_duration)), float(np.max(duration))) - 1e-8
        return float(concordance_index_ipcw(y_train, y_test, np.asarray(risk), tau=tau)[0])
    except Exception:
        censor = IPCW(train_duration, train_event)
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=bool)
        risk = np.asarray(risk, dtype=float)
        numerator = denominator = 0.0
        for index in np.where(event)[0]:
            later = np.where(duration > duration[index])[0]
            if not len(later):
                continue
            weight = 1.0 / float(censor.g([duration[index]])[0] ** 2)
            score = (risk[index] > risk[later]).astype(float)
            score += 0.5 * (risk[index] == risk[later])
            numerator += weight * score.sum()
            denominator += weight * len(later)
        return float(numerator / denominator) if denominator else float("nan")


def comparable_pair_decomposition(duration, event, risk) -> dict:
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    buckets = {
        "event_event": [0.0, 0.0, 0.0],
        "event_censored": [0.0, 0.0, 0.0],
    }
    for index in np.where(event)[0]:
        later = np.where(duration > duration[index])[0]
        for other in later:
            key = "event_event" if event[other] else "event_censored"
            buckets[key][0] += 1.0
            buckets[key][1] += float(risk[index] > risk[other])
            buckets[key][2] += float(risk[index] == risk[other])
    result = {}
    total = sum(value[0] for value in buckets.values())
    for key, (count, concordant, tied) in buckets.items():
        result[f"{key}_pairs"] = int(count)
        result[f"{key}_fraction"] = float(count / total) if total else float("nan")
        result[f"{key}_c_index"] = float((concordant + 0.5 * tied) / count) if count else float("nan")
    return result


def weighted_classification_metrics(labels, probability, known, weights, threshold) -> dict:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    mask = np.asarray(known, dtype=bool)
    weight = np.asarray(weights, dtype=float)
    prediction = probability >= threshold
    tp = float(weight[mask & (labels == 1) & prediction].sum())
    tn = float(weight[mask & (labels == 0) & ~prediction].sum())
    fp = float(weight[mask & (labels == 0) & prediction].sum())
    fn = float(weight[mask & (labels == 1) & ~prediction].sum())
    tpr = tp / max(tp + fn, EPS)
    tnr = tn / max(tn + fp, EPS)
    precision = tp / max(tp + fp, EPS)
    recall = tpr
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    result = {
        "threshold": float(threshold), "tp_ipcw": tp, "tn_ipcw": tn,
        "fp_ipcw": fp, "fn_ipcw": fn, "tss": tpr + tnr - 1.0,
        "hss2": 2.0 * (tp * tn - fn * fp) / hss_den if hss_den > 0 else float("nan"),
        "far": fp / max(tp + fp, EPS), "precision": precision, "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, EPS),
        "alert_fraction": float(np.mean(prediction[mask])) if mask.any() else float("nan"),
        "known_count": int(mask.sum()),
    }
    if len(np.unique(labels[mask])) == 2:
        result["auc"] = float(roc_auc_score(labels[mask], probability[mask], sample_weight=weight[mask]))
        result["average_precision"] = float(average_precision_score(
            labels[mask], probability[mask], sample_weight=weight[mask]
        ))
    else:
        result["auc"] = result["average_precision"] = float("nan")
    return result


def choose_thresholds(train_duration, train_event, val_duration, val_event, val_survival, horizons):
    censor = IPCW(train_duration, train_event)
    thresholds = {}
    for index, horizon in enumerate(horizons):
        labels, known, weights = censor.horizon_weights(val_duration, val_event, horizon)
        probability = 1.0 - val_survival[:, index]
        candidates = np.unique(np.concatenate([
            np.linspace(0.01, 0.99, 99), probability[known]
        ]))
        best = (-float("inf"), 0.5)
        for threshold in candidates:
            tss = weighted_classification_metrics(
                labels, probability, known, weights, float(threshold)
            )["tss"]
            candidate = (tss, -abs(float(threshold) - 0.5))
            if candidate > (best[0], -abs(best[1] - 0.5)):
                best = (tss, float(threshold))
        thresholds[float(horizon)] = best[1]
    return thresholds


def ipcw_brier(censor: IPCW, duration, event, survival_probability, horizon) -> float:
    labels, known, weights = censor.horizon_weights(duration, event, horizon)
    observed_survival = 1 - labels
    error = weights * (observed_survival - np.asarray(survival_probability)) ** 2
    return float(error.sum() / len(error))


def calibration_at_horizon(
    train_duration, train_event, duration, event, event_probability, horizon, bins=10
) -> tuple[dict, pd.DataFrame]:
    censor = IPCW(train_duration, train_event)
    labels, known, weights = censor.horizon_weights(duration, event, horizon)
    probability = np.clip(np.asarray(event_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    known_probability = probability[known]
    edges = np.unique(np.quantile(known_probability, np.linspace(0.0, 1.0, bins + 1)))
    rows = []
    if len(edges) >= 2:
        bin_index = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, len(edges) - 2)
        for index in range(len(edges) - 1):
            mask = known & (bin_index == index)
            if not mask.any():
                continue
            observed = float(np.average(labels[mask], weights=weights[mask]))
            predicted = float(np.average(probability[mask], weights=weights[mask]))
            rows.append({
                "bin": index, "count": int(mask.sum()), "predicted_event_probability": predicted,
                "observed_event_probability_ipcw": observed,
            })
    curve = pd.DataFrame(rows)
    ece = float(sum(
        row["count"] * abs(row["predicted_event_probability"] - row["observed_event_probability_ipcw"])
        for row in rows
    ) / max(1, sum(row["count"] for row in rows)))
    intercept = slope = float("nan")
    if known.sum() >= 10 and len(np.unique(labels[known])) == 2:
        logit = np.log(probability[known] / (1.0 - probability[known])).reshape(-1, 1)
        calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        calibration.fit(logit, labels[known], sample_weight=weights[known])
        intercept = float(calibration.intercept_[0])
        slope = float(calibration.coef_[0, 0])
    return {"ece": ece, "calibration_intercept": intercept, "calibration_slope": slope}, curve


def evaluate_survival_predictions(
    train_duration: np.ndarray,
    train_event: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    survival: np.ndarray,
    horizons: list[float],
    thresholds: dict[float, float],
) -> tuple[dict, pd.DataFrame]:
    censor = IPCW(train_duration, train_event)
    km_times, km_survival = _km_curve(train_duration, train_event)
    metrics = {
        "harrell_c_index": harrell_c_index(duration, event, risk),
        "uno_c_index": uno_c_index(train_duration, train_event, duration, event, risk),
        **comparable_pair_decomposition(duration, event, risk),
    }
    calibration_rows = []
    brier_values, auc_values = [], []
    for index, horizon in enumerate(horizons):
        probability = 1.0 - survival[:, index]
        labels, known, weights = censor.horizon_weights(duration, event, horizon)
        classification = weighted_classification_metrics(
            labels, probability, known, weights, thresholds[float(horizon)]
        )
        model_brier = ipcw_brier(censor, duration, event, survival[:, index], horizon)
        km_reference = float(_step_value(km_times, km_survival, [horizon])[0])
        reference_brier = ipcw_brier(
            censor, duration, event, np.full(len(duration), km_reference), horizon
        )
        calibration, curve = calibration_at_horizon(
            train_duration, train_event, duration, event, probability, horizon
        )
        prefix = f"{int(horizon)}h"
        for key, value in classification.items():
            metrics[f"{prefix}_{key}"] = value
        metrics[f"{prefix}_brier"] = model_brier
        metrics[f"{prefix}_km_reference_brier"] = reference_brier
        metrics[f"{prefix}_bss"] = 1.0 - model_brier / reference_brier if reference_brier > EPS else float("nan")
        metrics[f"{prefix}_predicted_survival_mean"] = float(np.mean(survival[:, index]))
        metrics[f"{prefix}_observed_survival_km"] = float(
            _step_value(*_km_curve(duration, event), [horizon])[0]
        )
        for key, value in calibration.items():
            metrics[f"{prefix}_{key}"] = value
        if not curve.empty:
            curve.insert(0, "horizon_hours", horizon)
            calibration_rows.append(curve)
        brier_values.append(model_brier)
        auc_values.append(classification["auc"])
    horizon_array = np.asarray(horizons, dtype=float)
    span = float(horizon_array[-1] - horizon_array[0]) if len(horizon_array) > 1 else 1.0
    metrics["integrated_brier_score"] = float(
        trapezoidal_integral(brier_values, horizon_array) / span
    )
    finite_auc = np.isfinite(auc_values)
    metrics["integrated_auc"] = float(
        trapezoidal_integral(
            np.asarray(auc_values)[finite_auc], horizon_array[finite_auc]
        )
        / max(EPS, horizon_array[finite_auc][-1] - horizon_array[finite_auc][0])
    ) if finite_auc.sum() >= 2 else float("nan")
    curve_frame = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    return metrics, curve_frame


def cluster_bootstrap(
    groups: np.ndarray,
    replicates: int,
    seed: int,
    evaluator: Callable[[np.ndarray], dict],
) -> pd.DataFrame:
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    group_indices = {group: np.where(groups == group)[0] for group in unique_groups}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        values = evaluator(indices)
        rows.append({"replicate": replicate, **values})
    return pd.DataFrame(rows)


def summarise_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        if column == "replicate":
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "metric": column, "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "ci_lower_2.5": float(values.quantile(0.025)),
            "ci_upper_97.5": float(values.quantile(0.975)),
        })
    return pd.DataFrame(rows)
