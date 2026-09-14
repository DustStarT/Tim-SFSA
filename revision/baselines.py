"""Censor-aware horizon-specific Logistic and SVM baselines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .training import horizon_targets


class PenalizedLinearCox:
    """Deterministic L2-penalized linear Cox model with Efron ties.

    The previous baseline used AdamW with the neural-model learning rate and
    remained far from convergence after 150 epochs.  This convex baseline is
    optimized directly with L-BFGS and uses the prediction-origin feature row.
    """

    def __init__(self, l2_penalty: float = 1e-4, max_iterations: int = 1000):
        self.l2_penalty = float(l2_penalty)
        self.max_iterations = int(max_iterations)
        self.coef_: np.ndarray | None = None
        self.fit_result_: dict = {}

    @staticmethod
    def _static(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("Linear Cox input must have shape [sample, time, feature]")
        return values[:, -1, :]

    def _objective(self, beta, x, duration, event):
        eta = x @ beta
        shift = float(np.max(eta))
        exp_eta = np.exp(np.clip(eta - shift, -700.0, 0.0))

        order = np.argsort(-duration, kind="mergesort")
        sorted_time = duration[order]
        sorted_exp = exp_eta[order]
        sorted_weighted_x = sorted_exp[:, None] * x[order]
        cumulative_risk = np.cumsum(sorted_exp)
        cumulative_weighted_x = np.cumsum(sorted_weighted_x, axis=0)

        log_likelihood = 0.0
        gradient = np.zeros_like(beta, dtype=np.float64)
        event_count = int(event.sum())
        if event_count == 0:
            raise ValueError("Linear Cox fit requires at least one observed event")

        for event_time in np.unique(duration[event]):
            deaths = event & np.isclose(duration, event_time, rtol=0.0, atol=1e-10)
            death_count = int(deaths.sum())
            risk_count = int(np.searchsorted(-sorted_time, -event_time, side="right"))
            risk_sum = float(cumulative_risk[risk_count - 1])
            risk_weighted_x = cumulative_weighted_x[risk_count - 1]
            death_sum = float(exp_eta[deaths].sum())
            death_weighted_x = (exp_eta[deaths, None] * x[deaths]).sum(axis=0)

            log_likelihood += float(eta[deaths].sum())
            gradient += x[deaths].sum(axis=0)
            for tied_index in range(death_count):
                fraction = tied_index / death_count
                denominator = max(
                    risk_sum - fraction * death_sum, np.finfo(np.float64).tiny
                )
                log_likelihood -= np.log(denominator) + shift
                gradient -= (
                    risk_weighted_x - fraction * death_weighted_x
                ) / denominator

        penalty = 0.5 * self.l2_penalty * float(beta @ beta)
        loss = -log_likelihood / event_count + penalty
        loss_gradient = -gradient / event_count + self.l2_penalty * beta
        return float(loss), loss_gradient

    def fit(self, x: np.ndarray, duration: np.ndarray, event: np.ndarray):
        static = self._static(x)
        duration = np.asarray(duration, dtype=np.float64).reshape(-1)
        event = np.asarray(event, dtype=bool).reshape(-1)
        if len(static) != len(duration) or len(static) != len(event):
            raise ValueError("Linear Cox covariates and outcomes have different lengths")
        result = minimize(
            self._objective,
            np.zeros(static.shape[1], dtype=np.float64),
            args=(static, duration, event),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
            raise RuntimeError(f"Linear Cox optimization failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=np.float64)
        self.fit_result_ = {
            "optimizer": "scipy.optimize.L-BFGS-B",
            "tie_method": "Efron",
            "l2_penalty": self.l2_penalty,
            "max_iterations": self.max_iterations,
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "converged": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "gradient_max_abs": float(np.max(np.abs(result.jac))),
        }
        if not result.success:
            raise RuntimeError(
                "Linear Cox optimizer did not converge: "
                f"status={result.status}, message={result.message}"
            )
        return self

    def predict_log_risk(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Linear Cox model is not fitted")
        return self._static(x) @ self.coef_

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class HorizonClassifiers:
    def __init__(self, family: str, horizons: list[float], seed: int):
        if family not in {"logistic", "svm"}:
            raise ValueError("family must be logistic or svm")
        self.family = family
        self.horizons = list(horizons)
        self.seed = int(seed)
        self.models = []

    def _new_model(self, labels: np.ndarray):
        if len(np.unique(labels)) < 2:
            return DummyClassifier(strategy="constant", constant=int(labels[0]))
        if self.family == "logistic":
            return LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
                class_weight=None, random_state=self.seed,
            )
        return SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True,
            class_weight=None, random_state=self.seed,
        )

    def fit(self, x: np.ndarray, duration: np.ndarray, event: np.ndarray):
        static = np.asarray(x)[:, -1, :]
        target, known = horizon_targets(duration, event, self.horizons)
        self.models = []
        for index in range(len(self.horizons)):
            mask = known[:, index].astype(bool)
            labels = target[mask, index].astype(int)
            if not len(labels):
                raise RuntimeError(f"No known training labels at horizon {self.horizons[index]}")
            model = self._new_model(labels)
            model.fit(static[mask], labels)
            self.models.append(model)
        return self

    def predict_event_probability(self, x: np.ndarray) -> np.ndarray:
        static = np.asarray(x)[:, -1, :]
        columns = []
        for model in self.models:
            probability = model.predict_proba(static)
            classes = list(model.classes_)
            columns.append(probability[:, classes.index(1)] if 1 in classes else np.zeros(len(static)))
        # Cumulative event probability must not decrease with a longer horizon.
        return np.maximum.accumulate(np.column_stack(columns), axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
