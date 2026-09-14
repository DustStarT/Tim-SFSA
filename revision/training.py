"""Resumable neural training, including exact chunked full-risk-set Cox updates."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .survival import efron_cox_loss


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def batched_predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _exact_chunked_cox_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    device: torch.device,
    batch_size: int,
    gradient_clip: float,
) -> float:
    """One exact global Cox update without retaining all forward graphs.

    The first pass obtains dL/d(eta) over the complete risk set.  A second pass
    propagates those exact score gradients through each covariate chunk.  RNG
    states are replayed so dropout masks match both passes.
    """

    model.train()
    score_chunks = []
    rng_states = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            rng_states.append(capture_rng_state())
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            score_chunks.append(model(batch).reshape(-1).detach())
    score_leaf = torch.cat(score_chunks).to(dtype=torch.float32).requires_grad_(True)
    duration_tensor = torch.as_tensor(duration, dtype=torch.float64, device=device)
    event_tensor = torch.as_tensor(event, dtype=torch.float64, device=device)
    loss = efron_cox_loss(score_leaf, duration_tensor, event_tensor)
    loss.backward()
    score_gradient = score_leaf.grad.detach()

    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for chunk_index, start in enumerate(range(0, len(x), batch_size)):
        restore_rng_state(rng_states[chunk_index])
        batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        scores = model(batch).reshape(-1)
        end = offset + len(scores)
        scores.backward(score_gradient[offset:end])
        offset = end
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    return float(loss.detach().cpu())


def _cox_validation_loss(
    model: nn.Module, x: np.ndarray, duration: np.ndarray, event: np.ndarray,
    device: torch.device, batch_size: int,
) -> float:
    scores = batched_predict(model, x, device, batch_size)
    with torch.no_grad():
        loss = efron_cox_loss(
            torch.as_tensor(scores, dtype=torch.float32, device=device),
            torch.as_tensor(duration, dtype=torch.float64, device=device),
            torch.as_tensor(event, dtype=torch.float64, device=device),
        )
    return float(loss.cpu())


def _load_checkpoint(path: Path, device: torch.device) -> dict | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_cox_model(
    model: nn.Module,
    train: dict,
    validation: dict,
    output_dir: Path,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    gradient_clip: float,
    resume: bool,
) -> tuple[nn.Module, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last_checkpoint.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(2, patience // 4), factor=0.5)
    model.to(device)
    start_epoch, best_loss, stale = 0, float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    checkpoint = _load_checkpoint(checkpoint_path, device) if resume else None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        stale = int(checkpoint["stale"])
        best_state = checkpoint["best_model"]
        history = checkpoint.get("history", [])
        restore_rng_state(checkpoint.get("rng"))
    for epoch in range(start_epoch, max_epochs):
        train_loss = _exact_chunked_cox_step(
            model, optimizer, train["x"], train["duration"], train["event"],
            device, batch_size, gradient_clip,
        )
        validation_loss = _cox_validation_loss(
            model, validation["x"], validation["duration"], validation["event"],
            device, batch_size,
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-6
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_efron_loss": train_loss,
            "validation_efron_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "best_model": best_state,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_loss": best_loss, "stale": stale, "history": history,
            "rng": capture_rng_state(),
        }, checkpoint_path)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "history": history, "best_loss": best_loss}, output_dir / "model.pt")
    return model, history


def horizon_targets(duration: np.ndarray, event: np.ndarray, horizons: list[float]):
    duration = np.asarray(duration)[:, None]
    event = np.asarray(event).astype(bool)[:, None]
    horizon = np.asarray(horizons, dtype=float)[None, :]
    target = (event & (duration <= horizon)).astype(np.float32)
    known = ((duration > horizon) | (event & (duration <= horizon))).astype(np.float32)
    return target, known


def train_lstm_classifier(
    model: nn.Module,
    train: dict,
    validation: dict,
    horizons: list[float],
    output_dir: Path,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    resume: bool,
) -> tuple[nn.Module, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last_checkpoint.pt"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(2, patience // 4), factor=0.5)
    train_target, train_known = horizon_targets(train["duration"], train["event"], horizons)
    val_target, val_known = horizon_targets(validation["duration"], validation["event"], horizons)
    x_train = torch.as_tensor(train["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    m_train = torch.as_tensor(train_known, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(validation["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val_target, dtype=torch.float32, device=device)
    m_val = torch.as_tensor(val_known, dtype=torch.float32, device=device)
    # Natural event prevalence is retained; validation thresholds handle the
    # operating trade-off without distorting predicted probabilities.
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    start_epoch, best_loss, stale = 0, float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    checkpoint = _load_checkpoint(checkpoint_path, device) if resume else None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        stale = int(checkpoint["stale"])
        best_state = checkpoint["best_model"]
        history = checkpoint.get("history", [])
        restore_rng_state(checkpoint.get("rng"))
    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_raw = criterion(model(x_train), y_train)
        train_loss = (train_raw * m_train).sum() / torch.clamp(m_train.sum(), min=1.0)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_raw = criterion(model(x_val), y_val)
            val_loss = (val_raw * m_val).sum() / torch.clamp(m_val.sum(), min=1.0)
        validation_loss = float(val_loss.cpu())
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_masked_bce": float(train_loss.detach().cpu()),
            "validation_masked_bce": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "best_model": best_state,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_loss": best_loss, "stale": stale, "history": history,
            "rng": capture_rng_state(),
        }, checkpoint_path)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "history": history, "best_loss": best_loss}, output_dir / "model.pt")
    return model, history


def transfer_encoder(classifier: nn.Module, risk_model: nn.Module) -> dict:
    source = classifier.encoder.state_dict()
    missing, unexpected = risk_model.encoder.load_state_dict(source, strict=True)
    return {"missing": list(missing), "unexpected": list(unexpected), "transferred": len(source)}
