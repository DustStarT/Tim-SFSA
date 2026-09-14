"""Training and evaluation of one complete revision experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import HorizonClassifiers, PenalizedLinearCox
from .config import (
    FEATURES_24, LEGACY_DROPPED_FEATURES, ExperimentSpec, PipelineConfig,
)
from .data import arrays_from_samples
from .importance import (
    ar_block_permutation_importance, correlated_group_permutation_importance,
    correlation_diagnostics,
)
from .metrics import (
    IPCW, choose_thresholds, cluster_bootstrap, evaluate_survival_predictions,
    comparable_pair_decomposition, harrell_c_index, ipcw_brier, km_survival_at,
    summarise_bootstrap, trapezoidal_integral, uno_c_index,
    weighted_classification_metrics,
)
from .models import MultiHorizonLSTMClassifier, TimSFSARiskModel
from .preprocessing import FeaturePreprocessor
from .survival import efron_baseline_cumulative_hazard, predict_survival
from .training import (
    batched_predict, resolve_device, seed_everything, train_cox_model,
    train_lstm_classifier, transfer_encoder,
)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, default=str), encoding="utf-8")


EXPERIMENT_IMPLEMENTATION_VERSION = {
    "linear_cox": 2,
    "default": 1,
}


def _experiment_signature(
    split: dict[str, list[dict]], split_mode: str, spec: ExperimentSpec,
    feature_names: list[str], seed: int, depth: int, config: PipelineConfig,
    compute_importance: bool,
) -> str:
    payload = {
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION.get(
            spec.family, EXPERIMENT_IMPLEMENTATION_VERSION["default"]
        ),
        "split_mode": split_mode,
        "spec": spec.__dict__,
        "feature_names": list(feature_names),
        "seed": int(seed),
        "depth": int(depth),
        "compute_importance": bool(compute_importance),
        "training_config": {
            key: config.as_dict()[key] for key in (
                "horizons_hours", "hidden_size", "connector_size", "dropout",
                "connector_dropout", "learning_rate", "weight_decay", "max_epochs",
                "pretrain_epochs", "patience", "gradient_clip", "inference_batch_size",
                "bootstrap_replicates", "permutation_repeats", "correlation_threshold",
                "deterministic",
            )
        },
        "samples": {
            name: [sample["sample_id"] for sample in rows]
            for name, rows in split.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_completion_is_reusable(split_mode: str, spec: ExperimentSpec) -> bool:
    # The transferred server run predates experiment signatures.  Its official
    # artifacts are reusable except for the demonstrably unconverged linear Cox
    # baseline.  Chronological legacy artifacts are never reused because the
    # boundary-HARP policy changed in this revision.
    return split_mode.startswith("official") and spec.family != "linear_cox"


def prepare_arrays(
    split: dict[str, list[dict]], feature_names: list[str], include_bc_history: bool,
    output_dir: Path,
) -> tuple[dict[str, dict], FeaturePreprocessor, dict[str, dict]]:
    raw = {
        name: arrays_from_samples(rows, feature_names, include_bc_history=include_bc_history)
        for name, rows in split.items()
    }
    preprocessor = FeaturePreprocessor(raw["train"]["feature_names"])
    processed = {}
    for name, arrays in raw.items():
        processed[name] = dict(arrays)
        processed[name]["x"] = (
            preprocessor.fit_transform(arrays["x"])
            if name == "train" else None
        )
    for name in ("validation", "test"):
        processed[name]["x"] = preprocessor.transform(raw[name]["x"])
    preprocessor.save(output_dir / "preprocessing")
    return processed, preprocessor, raw


def _balanced_training(train: dict, target_event_ratio: float, seed: int) -> dict:
    event_indices = np.where(train["event"] == 1)[0]
    censored_indices = np.where(train["event"] == 0)[0]
    target_censored = int(round(len(event_indices) * (1.0 - target_event_ratio) / target_event_ratio))
    if target_censored >= len(censored_indices):
        return train
    rng = np.random.default_rng(seed)
    selected = np.sort(np.concatenate([
        event_indices, rng.choice(censored_indices, size=target_censored, replace=False)
    ]))
    result = {}
    for key, value in train.items():
        if isinstance(value, np.ndarray) and len(value) == len(train["event"]):
            result[key] = value[selected]
        elif isinstance(value, pd.DataFrame) and len(value) == len(train["event"]):
            result[key] = value.iloc[selected].reset_index(drop=True)
        else:
            result[key] = value
    return result


def _classification_survival(probability: np.ndarray) -> np.ndarray:
    cumulative = np.maximum.accumulate(np.clip(probability, 0.0, 1.0), axis=1)
    return 1.0 - cumulative


def _train_predict(
    spec: ExperimentSpec,
    arrays: dict[str, dict],
    config: PipelineConfig,
    depth: int,
    seed: int,
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, np.ndarray], object | None]:
    device = resolve_device(config.device)
    n_features = arrays["train"]["x"].shape[-1]
    horizons = config.horizons_hours

    if spec.family in {"logistic", "svm"}:
        model = HorizonClassifiers(spec.family, horizons, seed).fit(
            arrays["train"]["x"], arrays["train"]["duration"], arrays["train"]["event"]
        )
        model.save(output_dir / "model.joblib")
        predictions = {}
        for split_name, values in arrays.items():
            probability = model.predict_event_probability(values["x"])
            predictions[split_name] = {
                "survival": _classification_survival(probability),
                "risk": probability.mean(axis=1),
            }
        return predictions, model

    if spec.family == "linear_cox":
        obsolete_training = output_dir / "training"
        for obsolete in (
            output_dir / "training_history.csv",
            obsolete_training / "last_checkpoint.pt",
            obsolete_training / "model.pt",
        ):
            if obsolete.is_file():
                obsolete.unlink()
        if obsolete_training.is_dir():
            try:
                obsolete_training.rmdir()
            except OSError:
                # Preserve any unrecognized file rather than recursively
                # deleting data that this migration did not create.
                pass
        model = PenalizedLinearCox(l2_penalty=config.weight_decay).fit(
            arrays["train"]["x"], arrays["train"]["duration"], arrays["train"]["event"]
        )
        model.save(output_dir / "model.joblib")
        _write_json(output_dir / "linear_cox_fit.json", model.fit_result_)
        natural_train_risk = model.predict_log_risk(arrays["train"]["x"])
        baseline_times, baseline_hazard = efron_baseline_cumulative_hazard(
            natural_train_risk, arrays["train"]["duration"], arrays["train"]["event"]
        )
        np.savez(
            output_dir / "natural_train_baseline_hazard.npz",
            event_times=baseline_times, cumulative_hazard=baseline_hazard,
        )
        predictions = {}
        for split_name, values in arrays.items():
            risk = natural_train_risk if split_name == "train" else model.predict_log_risk(
                values["x"]
            )
            predictions[split_name] = {
                "risk": risk,
                "survival": predict_survival(
                    risk, baseline_times, baseline_hazard,
                    np.asarray(horizons, dtype=float),
                ),
            }
        return predictions, model

    if spec.family == "lstm_classifier":
        model = MultiHorizonLSTMClassifier(
            n_features=n_features, n_horizons=len(horizons),
            hidden_size=config.hidden_size, depth=depth, dropout=config.dropout,
            connector_size=config.connector_size,
        )
        model, history = train_lstm_classifier(
            model, arrays["train"], arrays["validation"], horizons,
            output_dir / "training", device=device,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            max_epochs=config.pretrain_epochs, patience=config.patience, resume=resume,
        )
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        predictions = {}
        for split_name, values in arrays.items():
            logits = batched_predict(model, values["x"], device, config.inference_batch_size)
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            predictions[split_name] = {
                "survival": _classification_survival(probability),
                "risk": probability.mean(axis=1),
            }
        return predictions, model

    training_values = arrays["train"]
    if spec.balanced_event_ratio is not None:
        training_values = _balanced_training(training_values, spec.balanced_event_ratio, seed)

    model = TimSFSARiskModel(
        n_features=n_features, hidden_size=config.hidden_size, depth=depth,
        dropout=config.dropout, connector_size=config.connector_size,
        connector_dropout=config.connector_dropout, use_lstm=spec.use_lstm,
        use_connector=spec.use_connector, linear_head=spec.linear_head,
    )
    transfer_report = None
    if spec.use_lstm and spec.use_pretraining:
        classifier = MultiHorizonLSTMClassifier(
            n_features=n_features, n_horizons=len(horizons),
            hidden_size=config.hidden_size, depth=depth, dropout=config.dropout,
            connector_size=config.connector_size,
        )
        classifier, classifier_history = train_lstm_classifier(
            classifier, training_values, arrays["validation"], horizons,
            output_dir / "pretraining", device=device,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            max_epochs=config.pretrain_epochs, patience=config.patience, resume=resume,
        )
        pd.DataFrame(classifier_history).to_csv(
            output_dir / "pretraining_history.csv", index=False
        )
        transfer_report = transfer_encoder(classifier, model)
    model, history = train_cox_model(
        model, training_values, arrays["validation"], output_dir / "training",
        device=device, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, max_epochs=config.max_epochs,
        patience=config.patience, batch_size=config.inference_batch_size,
        gradient_clip=config.gradient_clip, resume=resume,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    if transfer_report is not None:
        _write_json(output_dir / "pretraining_transfer.json", transfer_report)

    # Baseline hazard is always estimated on the complete, natural-rate train set.
    natural_train_risk = batched_predict(
        model, arrays["train"]["x"], device, config.inference_batch_size
    )
    baseline_times, baseline_hazard = efron_baseline_cumulative_hazard(
        natural_train_risk, arrays["train"]["duration"], arrays["train"]["event"]
    )
    np.savez(
        output_dir / "natural_train_baseline_hazard.npz",
        event_times=baseline_times, cumulative_hazard=baseline_hazard,
    )
    predictions = {}
    for split_name, values in arrays.items():
        risk = natural_train_risk if split_name == "train" else batched_predict(
            model, values["x"], device, config.inference_batch_size
        )
        predictions[split_name] = {
            "risk": risk,
            "survival": predict_survival(
                risk, baseline_times, baseline_hazard,
                np.asarray(horizons, dtype=float),
            ),
        }
    return predictions, model


def _bootstrap_values(
    indices: np.ndarray,
    arrays: dict[str, dict],
    predictions: dict[str, np.ndarray],
    horizons: list[float],
    thresholds: dict[float, float],
) -> dict:
    train = arrays["train"]
    test = arrays["test"]
    duration = test["duration"][indices]
    event = test["event"][indices]
    risk = predictions["risk"][indices]
    survival = predictions["survival"][indices]
    censor = IPCW(train["duration"], train["event"])
    result = {
        "harrell_c_index": harrell_c_index(duration, event, risk),
        "uno_c_index": uno_c_index(
            train["duration"], train["event"], duration, event, risk
        ),
        **comparable_pair_decomposition(duration, event, risk),
    }
    briers, aucs = [], []
    for horizon_index, horizon in enumerate(horizons):
        labels, known, weights = censor.horizon_weights(duration, event, horizon)
        classification = weighted_classification_metrics(
            labels, 1.0 - survival[:, horizon_index], known, weights,
            thresholds[float(horizon)],
        )
        brier = ipcw_brier(censor, duration, event, survival[:, horizon_index], horizon)
        km_reference_survival = float(km_survival_at(
            train["duration"], train["event"], [horizon]
        )[0])
        reference_brier = ipcw_brier(
            censor, duration, event,
            np.full(len(duration), km_reference_survival), horizon,
        )
        prefix = f"{int(horizon)}h"
        for key in (
            "auc", "average_precision", "tss", "hss2", "far", "precision",
            "recall", "f1", "alert_fraction",
        ):
            result[f"{prefix}_{key}"] = classification[key]
        result[f"{prefix}_brier"] = brier
        result[f"{prefix}_km_reference_brier"] = reference_brier
        result[f"{prefix}_bss"] = (
            1.0 - brier / reference_brier if reference_brier > 1e-8 else float("nan")
        )
        result[f"{prefix}_predicted_survival_mean"] = float(
            np.mean(survival[:, horizon_index])
        )
        result[f"{prefix}_observed_survival_km"] = float(
            km_survival_at(duration, event, [horizon])[0]
        )
        briers.append(brier)
        aucs.append(classification["auc"])
    result["integrated_brier_score"] = float(
        trapezoidal_integral(briers, horizons) / (horizons[-1] - horizons[0])
    )
    finite_auc = np.isfinite(aucs)
    result["integrated_auc"] = float(
        trapezoidal_integral(
            np.asarray(aucs)[finite_auc], np.asarray(horizons)[finite_auc]
        )
        / max(1e-8, np.asarray(horizons)[finite_auc][-1] - np.asarray(horizons)[finite_auc][0])
    ) if finite_auc.sum() >= 2 else float("nan")
    return result


def _write_secondary_analyses(
    arrays: dict[str, dict], predictions: dict[str, dict], thresholds: dict[float, float],
    horizons: list[float], output_dir: Path,
) -> None:
    test = arrays["test"]
    event_duration = test["duration"][test["event"] == 1]
    lead_time = {
        "event_count": int(len(event_duration)),
        "median_hours": float(np.median(event_duration)) if len(event_duration) else None,
        "q1_hours": float(np.quantile(event_duration, 0.25)) if len(event_duration) else None,
        "q3_hours": float(np.quantile(event_duration, 0.75)) if len(event_duration) else None,
    }
    _write_json(output_dir / "lead_time_summary.json", lead_time)

    censor = IPCW(arrays["train"]["duration"], arrays["train"]["event"])
    sensitivity_rows = []
    for index, horizon in enumerate(horizons):
        labels, known, weights = censor.horizon_weights(
            test["duration"], test["event"], horizon
        )
        probability = 1.0 - predictions["test"]["survival"][:, index]
        validation_probability = 1.0 - predictions["validation"]["survival"][:, index]
        threshold_rows = [("primary_tss", None, thresholds[float(horizon)])]
        for target_load in (0.01, 0.025, 0.05, 0.10, 0.20):
            load_threshold = float(np.quantile(validation_probability, 1.0 - target_load))
            threshold_rows.append(("validation_alert_load", target_load, load_threshold))
        for threshold_source, target_load, threshold in threshold_rows:
            threshold = float(np.clip(threshold, 0.001, 0.999))
            values = weighted_classification_metrics(
                labels, probability, known, weights, threshold
            )
            sensitivity_rows.append({
                "horizon_hours": horizon,
                "threshold_source": threshold_source,
                "target_validation_alert_fraction": target_load,
                "achieved_validation_alert_fraction": float(
                    np.mean(validation_probability >= threshold)
                ),
                **values,
            })
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "alert_load_sensitivity.csv", index=False
    )

    risk = predictions["test"]["risk"]
    quantiles = np.unique(np.quantile(risk, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    risk_bin = np.clip(np.digitize(risk, quantiles[1:-1], right=True), 0, len(quantiles) - 2)
    rows = []
    for index in range(len(quantiles) - 1):
        mask = risk_bin == index
        if not mask.any():
            continue
        rows.append({
            "risk_quantile_bin": index + 1, "count": int(mask.sum()),
            "risk_min": float(risk[mask].min()), "risk_max": float(risk[mask].max()),
            "event_fraction_observed": float(test["event"][mask].mean()),
            "duration_median_hours": float(np.median(test["duration"][mask])),
        })
    pd.DataFrame(rows).to_csv(output_dir / "risk_quantile_sensitivity.csv", index=False)


def _ph_diagnostic(arrays: dict[str, dict], predictions: dict[str, dict], output: Path) -> None:
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
        frame = pd.DataFrame({
            "duration": arrays["train"]["duration"],
            "event": arrays["train"]["event"],
            "risk_score": predictions["train"]["risk"],
        })
        diagnostic = CoxPHFitter(penalizer=1e-6).fit(
            frame, duration_col="duration", event_col="event"
        )
        test = proportional_hazard_test(diagnostic, frame, time_transform="rank")
        result = {
            "purpose": "diagnostic only; this auxiliary coefficient is not used for prediction",
            "risk_score_coefficient": float(diagnostic.params_["risk_score"]),
            "schoenfeld_rank_p_value": float(test.summary.loc["risk_score", "p"]),
            "schoenfeld_rank_statistic": float(test.summary.loc["risk_score", "test_statistic"]),
        }
    except Exception as exc:
        result = {"status": "unavailable", "error": repr(exc)}
    _write_json(output, result)


def run_experiment(
    split: dict[str, list[dict]],
    split_mode: str,
    spec: ExperimentSpec,
    feature_names: list[str],
    seed: int,
    depth: int,
    config: PipelineConfig,
    output_dir: Path,
    *,
    resume: bool,
    compute_importance: bool = False,
) -> dict:
    completion = output_dir / "experiment_complete.json"
    result_path = output_dir / "metrics.json"
    signature = _experiment_signature(
        split, split_mode, spec, feature_names, seed, depth, config,
        compute_importance,
    )
    required_artifacts = [
        completion, result_path, output_dir / "test_predictions.npz",
        output_dir / "ar_cluster_bootstrap_summary.csv",
    ]
    if spec.family == "linear_cox":
        required_artifacts.extend([
            output_dir / "model.joblib",
            output_dir / "linear_cox_fit.json",
            output_dir / "natural_train_baseline_hazard.npz",
        ])
    if compute_importance:
        required_artifacts.append(output_dir / "validation_ar_block_importance.csv")
    required_complete = all(path.exists() for path in required_artifacts)
    if resume and required_complete:
        completion_state = json.loads(completion.read_text(encoding="utf-8"))
        stored_signature = completion_state.get("signature")
        if completion_state.get("status") == "complete" and (
            stored_signature == signature or (
                stored_signature is None
                and _legacy_completion_is_reusable(split_mode, spec)
            )
        ):
            return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(seed, config.deterministic)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, preprocessor, raw_arrays = prepare_arrays(
        split, feature_names, spec.include_bc_history, output_dir
    )
    _write_json(output_dir / "experiment_spec.json", {
        **spec.__dict__, "seed": seed, "depth": depth,
        "split_mode": split_mode, "feature_names": preprocessor.feature_names,
    })
    predictions, fitted_model = _train_predict(
        spec, arrays, config, depth, seed, output_dir, resume
    )
    thresholds = choose_thresholds(
        arrays["train"]["duration"], arrays["train"]["event"],
        arrays["validation"]["duration"], arrays["validation"]["event"],
        predictions["validation"]["survival"], config.horizons_hours,
    )
    _write_json(output_dir / "validation_thresholds.json", thresholds)
    validation_metrics, validation_calibration = evaluate_survival_predictions(
        arrays["train"]["duration"], arrays["train"]["event"],
        arrays["validation"]["duration"], arrays["validation"]["event"],
        predictions["validation"]["risk"], predictions["validation"]["survival"],
        config.horizons_hours, thresholds,
    )
    test_metrics, test_calibration = evaluate_survival_predictions(
        arrays["train"]["duration"], arrays["train"]["event"],
        arrays["test"]["duration"], arrays["test"]["event"],
        predictions["test"]["risk"], predictions["test"]["survival"],
        config.horizons_hours, thresholds,
    )
    if not validation_calibration.empty:
        validation_calibration.to_csv(output_dir / "validation_calibration.csv", index=False)
    if not test_calibration.empty:
        test_calibration.to_csv(output_dir / "test_calibration.csv", index=False)

    for split_name in ("train", "validation", "test"):
        np.savez_compressed(
            output_dir / f"{split_name}_predictions.npz",
            duration=arrays[split_name]["duration"], event=arrays[split_name]["event"],
            groups=arrays[split_name]["groups"], risk=predictions[split_name]["risk"],
            survival=predictions[split_name]["survival"],
            horizons=np.asarray(config.horizons_hours),
        )
        arrays[split_name]["metadata"].to_csv(
            output_dir / f"{split_name}_prediction_metadata.csv", index=False
        )

    bootstrap = cluster_bootstrap(
        arrays["test"]["groups"], config.bootstrap_replicates, seed,
        lambda indices: _bootstrap_values(
            indices, arrays, predictions["test"], config.horizons_hours, thresholds
        ),
    )
    bootstrap.to_csv(output_dir / "ar_cluster_bootstrap.csv", index=False)
    summarise_bootstrap(bootstrap).to_csv(
        output_dir / "ar_cluster_bootstrap_summary.csv", index=False
    )
    _write_secondary_analyses(
        arrays, predictions, thresholds, config.horizons_hours, output_dir
    )
    _ph_diagnostic(arrays, predictions, output_dir / "ph_diagnostic.json")

    if compute_importance and fitted_model is not None and spec.family == "cox_neural":
        device = resolve_device(config.device)
        predict = lambda values: batched_predict(
            fitted_model, values, device, config.inference_batch_size
        )
        importance = ar_block_permutation_importance(
            predict, arrays["validation"]["x"], arrays["validation"]["duration"],
            arrays["validation"]["event"], arrays["validation"]["groups"],
            arrays["train"]["duration"], arrays["train"]["event"],
            preprocessor.feature_names, config.permutation_repeats, seed,
        )
        importance.to_csv(output_dir / "validation_ar_block_importance.csv", index=False)
        correlations, clusters = correlation_diagnostics(
            raw_arrays["train"]["x"], raw_arrays["train"]["feature_names"],
            LEGACY_DROPPED_FEATURES, config.correlation_threshold,
        )
        correlations.to_csv(output_dir / "legacy_feature_correlation_explanation.csv", index=False)
        _write_json(output_dir / "correlated_feature_groups.json", clusters)
        group_importance = correlated_group_permutation_importance(
            predict, arrays["validation"]["x"], arrays["validation"]["duration"],
            arrays["validation"]["event"], arrays["validation"]["groups"],
            arrays["train"]["duration"], arrays["train"]["event"],
            preprocessor.feature_names, clusters, config.permutation_repeats, seed + 10000,
        )
        if not group_importance.empty:
            group_importance.to_csv(
                output_dir / "validation_correlated_group_importance.csv", index=False
            )

    result = {
        "experiment": spec.name, "split_mode": split_mode, "seed": seed,
        "depth": depth, "feature_count": len(preprocessor.feature_names),
        "train_sample_count": len(arrays["train"]["event"]),
        "validation_sample_count": len(arrays["validation"]["event"]),
        "test_sample_count": len(arrays["test"]["event"]),
        "validation": validation_metrics, "test": test_metrics,
    }
    _write_json(result_path, result)
    _write_json(completion, {
        "status": "complete",
        "signature": signature,
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION.get(
            spec.family, EXPERIMENT_IMPLEMENTATION_VERSION["default"]
        ),
    })
    return _json_value(result)
