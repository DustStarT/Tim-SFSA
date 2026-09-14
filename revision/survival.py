"""Exact Efron partial likelihood and baseline survival estimation."""

from __future__ import annotations

import numpy as np
import torch


def efron_cox_loss(
    log_risk: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Efron log partial likelihood over the complete risk set."""

    eta = log_risk.reshape(-1).to(dtype=torch.float64)
    time = durations.reshape(-1).to(device=eta.device, dtype=torch.float64)
    event = events.reshape(-1).to(device=eta.device) > 0.5
    event_times = torch.unique(time[event], sorted=True)
    if event_times.numel() == 0:
        raise ValueError("Cox loss requires at least one observed event")
    log_likelihood = eta.new_zeros(())
    event_count = eta.new_zeros(())
    tiny = torch.finfo(torch.float64).tiny
    for event_time in event_times:
        deaths = event & (time == event_time)
        risk_set = time >= event_time
        d = int(deaths.sum().item())
        if d == 0:
            continue
        risk_eta = eta[risk_set]
        death_eta = eta[deaths]
        shift = torch.max(risk_eta)
        risk_sum = torch.exp(risk_eta - shift).sum()
        death_sum = torch.exp(death_eta - shift).sum()
        log_likelihood = log_likelihood + death_eta.sum()
        for tied_index in range(d):
            denominator = risk_sum - (float(tied_index) / float(d)) * death_sum
            log_likelihood = log_likelihood - (torch.log(torch.clamp(denominator, min=tiny)) + shift)
        event_count = event_count + d
    return (-log_likelihood / event_count).to(dtype=log_risk.dtype)


def efron_baseline_cumulative_hazard(
    log_risk: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate baseline cumulative hazard on the natural training cohort."""

    eta = np.asarray(log_risk, dtype=np.float64).reshape(-1)
    time = np.asarray(durations, dtype=np.float64).reshape(-1)
    event = np.asarray(events).reshape(-1).astype(bool)
    event_times = np.unique(time[event])
    increments = []
    for event_time in event_times:
        deaths = event & np.isclose(time, event_time, rtol=0.0, atol=1e-10)
        risk_set = time >= event_time
        d = int(deaths.sum())
        shift = float(np.max(eta[risk_set]))
        risk_sum = float(np.exp(eta[risk_set] - shift).sum())
        death_sum = float(np.exp(eta[deaths] - shift).sum())
        increment_scaled = 0.0
        for tied_index in range(d):
            denominator = risk_sum - (tied_index / d) * death_sum
            increment_scaled += 1.0 / max(denominator, np.finfo(float).tiny)
        increments.append(increment_scaled * np.exp(-shift))
    return event_times.astype(np.float64), np.cumsum(np.asarray(increments, dtype=np.float64))


def predict_survival(
    log_risk: np.ndarray,
    baseline_times: np.ndarray,
    baseline_cumulative_hazard: np.ndarray,
    evaluation_times: np.ndarray,
) -> np.ndarray:
    eta = np.asarray(log_risk, dtype=np.float64).reshape(-1)
    eval_times = np.asarray(evaluation_times, dtype=np.float64).reshape(-1)
    positions = np.searchsorted(baseline_times, eval_times, side="right") - 1
    base = np.zeros_like(eval_times, dtype=np.float64)
    valid = positions >= 0
    base[valid] = baseline_cumulative_hazard[positions[valid]]
    relative_risk = np.exp(np.clip(eta, -50.0, 50.0))
    survival = np.exp(-relative_risk[:, None] * base[None, :])
    return np.clip(survival, 0.0, 1.0)
