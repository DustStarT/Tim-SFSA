"""Training-only preprocessing that retains every requested physical feature."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SIGNED_LOG_FEATURES


class FeaturePreprocessor:
    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.medians: np.ndarray | None = None
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.audit: list[dict] = []

    def _transform_shape(self, x: np.ndarray) -> np.ndarray:
        transformed = np.asarray(x, dtype=np.float64).copy()
        transformed[~np.isfinite(transformed)] = np.nan
        for index, name in enumerate(self.feature_names):
            if name in SIGNED_LOG_FEATURES:
                values = transformed[..., index]
                transformed[..., index] = np.sign(values) * np.log1p(np.abs(values))
        return transformed

    def fit(self, x: np.ndarray) -> "FeaturePreprocessor":
        raw = np.asarray(x, dtype=np.float64)
        if raw.ndim != 3 or raw.shape[2] != len(self.feature_names):
            raise ValueError("Expected X with shape [samples, timesteps, features]")
        transformed = self._transform_shape(raw)
        flat = transformed.reshape(-1, transformed.shape[-1])
        self.medians = np.zeros(flat.shape[1], dtype=np.float64)
        self.centers = np.zeros(flat.shape[1], dtype=np.float64)
        self.scales = np.ones(flat.shape[1], dtype=np.float64)
        self.audit = []
        for index, name in enumerate(self.feature_names):
            raw_values = raw[..., index].reshape(-1)
            finite_raw = raw_values[np.isfinite(raw_values)]
            values = flat[:, index]
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            filled = np.where(np.isfinite(values), values, median)
            q25, q75 = np.percentile(filled, [25.0, 75.0])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            self.medians[index] = median
            self.centers[index] = float(np.median(filled))
            self.scales[index] = scale
            self.audit.append({
                "feature": name,
                "raw_min": float(np.min(finite_raw)) if finite_raw.size else None,
                "raw_max": float(np.max(finite_raw)) if finite_raw.size else None,
                "missing_count": int(raw_values.size - finite_raw.size),
                "missing_rate": float(1.0 - finite_raw.size / max(1, raw_values.size)),
                "transformation": "signed_log1p" if name in SIGNED_LOG_FEATURES else "identity",
                "imputation_median_transformed": median,
                "robust_center": self.centers[index],
                "robust_scale_iqr": scale,
            })
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.medians is None or self.centers is None or self.scales is None:
            raise RuntimeError("FeaturePreprocessor must be fitted on training data first")
        transformed = self._transform_shape(x)
        for index in range(transformed.shape[-1]):
            values = transformed[..., index]
            values = np.where(np.isfinite(values), values, self.medians[index])
            transformed[..., index] = (values - self.centers[index]) / self.scales[index]
        if not np.isfinite(transformed).all():
            raise RuntimeError("Non-finite values remain after preprocessing")
        return transformed.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_names": self.feature_names,
            "medians": self.medians.tolist(),
            "centers": self.centers.tolist(),
            "scales": self.scales.tolist(),
        }
        (directory / "preprocessor.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        import pandas as pd
        pd.DataFrame(self.audit).to_csv(directory / "feature_audit.csv", index=False)

    @classmethod
    def load(cls, path: Path) -> "FeaturePreprocessor":
        state = json.loads(path.read_text(encoding="utf-8"))
        instance = cls(state["feature_names"])
        instance.medians = np.asarray(state["medians"], dtype=np.float64)
        instance.centers = np.asarray(state["centers"], dtype=np.float64)
        instance.scales = np.asarray(state["scales"], dtype=np.float64)
        return instance
