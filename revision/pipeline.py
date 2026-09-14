"""Top-level resumable orchestration for the reviewer revision."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CHRONO_EXPERIMENT_NAMES, FEATURES_24, LEGACY_18_FEATURES,
    OFFICIAL_EXPERIMENTS, ExperimentSpec, PipelineConfig,
)
from .data import (
    load_cached_event_dataset, load_event_dataset, make_splits, save_event_dataset,
    save_split_exclusions, save_split_manifest,
)
from .experiments import run_experiment
from .state import (
    StageTracker, atomic_json, capture_environment, code_fingerprint,
    dataset_fingerprint,
)
from .synthetic import make_synthetic_event_samples
from .reporting import create_revision_figures


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(tracker: StageTracker, name: str, action, artifacts: list[Path], resume: bool):
    if resume and tracker.is_complete(name, artifacts):
        return
    tracker.start(name)
    try:
        details = action() or {}
        missing = [str(path) for path in artifacts if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Stage {name!r} returned without required artifacts: {missing}"
            )
        tracker.complete(name, artifacts, details)
    except Exception as exc:
        tracker.fail(name, exc)
        raise


def _depth_selection(
    official_split: dict[str, list[dict]], config: PipelineConfig,
    output_dir: Path, resume: bool,
) -> int:
    rows = []
    # Never expose partition 5 while choosing model depth.
    tuning_split = {
        "train": official_split["train"],
        "validation": official_split["validation"],
        "test": official_split["validation"],
    }
    tuning_config = copy.deepcopy(config)
    tuning_config.bootstrap_replicates = min(20, config.bootstrap_replicates)
    tuning_config.permutation_repeats = 1
    spec = next(item for item in OFFICIAL_EXPERIMENTS if item.name == "tim_sfsa")
    for depth in config.depth_candidates:
        for seed in config.seeds:
            result = run_experiment(
                tuning_split, "official_validation_depth_tuning", spec, FEATURES_24,
                seed, depth, tuning_config, output_dir / f"depth_{depth}" / f"seed_{seed}",
                resume=resume, compute_importance=False,
            )
            rows.append({
                "depth": depth, "seed": seed,
                "validation_integrated_brier_score": result["validation"]["integrated_brier_score"],
                "validation_uno_c_index": result["validation"]["uno_c_index"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "depth_selection_runs.csv", index=False)
    summary = frame.groupby("depth", as_index=False).agg(
        mean_ibs=("validation_integrated_brier_score", "mean"),
        std_ibs=("validation_integrated_brier_score", "std"),
        mean_uno_c_index=("validation_uno_c_index", "mean"),
        std_uno_c_index=("validation_uno_c_index", "std"),
    )
    minimum = float(summary["mean_ibs"].min())
    eligible = summary[summary["mean_ibs"] <= minimum + 0.005].copy()
    eligible.sort_values(["mean_uno_c_index", "depth"], ascending=[False, True], inplace=True)
    selected = int(eligible.iloc[0]["depth"])
    summary["selected"] = summary["depth"] == selected
    summary.to_csv(output_dir / "depth_selection_summary.csv", index=False)
    atomic_json(output_dir / "selected_depth.json", {
        "selected_depth": selected,
        "rule": "minimum mean validation IBS; within 0.005 prefer higher mean Uno C-index, then shallower depth",
    })
    return selected


def _derive_validation_pruned(official_dir: Path, seeds: list[int]) -> list[str]:
    frames = []
    for seed in seeds:
        path = official_dir / "tim_sfsa" / f"seed_{seed}" / "validation_ar_block_importance.csv"
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("feature", as_index=False).agg(
        importance_mean=("importance_mean", "mean"),
        importance_seed_std=("importance_mean", "std"),
    )
    removed = summary.loc[summary["importance_mean"] <= 0.0, "feature"].tolist()
    retained = [feature for feature in FEATURES_24 if feature not in removed]
    if not retained:
        retained = [summary.sort_values("importance_mean", ascending=False).iloc[0]["feature"]]
        removed = [feature for feature in FEATURES_24 if feature not in retained]
    summary["validation_pruned"] = summary["feature"].isin(removed)
    summary.to_csv(official_dir / "validation_feature_pruning.csv", index=False)
    atomic_json(official_dir / "validation_pruned_features.json", {
        "selection_data": "official partition 4 only",
        "criterion": "mean AR-block permutation importance across seeds <= 0",
        "retained": retained, "removed": removed,
        "test_partition_used": False,
    })
    return retained


def _run_official_experiments(
    split: dict[str, list[dict]], selected_depth: int, config: PipelineConfig,
    output_dir: Path, resume: bool,
) -> dict:
    results = []
    main_spec = next(spec for spec in OFFICIAL_EXPERIMENTS if spec.name == "tim_sfsa")
    for seed in config.seeds:
        results.append(run_experiment(
            split, "official", main_spec, FEATURES_24, seed, selected_depth, config,
            output_dir / main_spec.name / f"seed_{seed}", resume=resume,
            compute_importance=True,
        ))
    pruned = _derive_validation_pruned(output_dir, config.seeds)
    for spec in OFFICIAL_EXPERIMENTS:
        if spec.name == "tim_sfsa":
            continue
        if spec.feature_set == "legacy18":
            features = LEGACY_18_FEATURES
        elif spec.feature_set == "validation_pruned":
            features = pruned
        else:
            features = FEATURES_24
        for seed in config.seeds:
            results.append(run_experiment(
                split, "official", spec, features, seed, selected_depth, config,
                output_dir / spec.name / f"seed_{seed}", resume=resume,
                compute_importance=False,
            ))
    details = {
        "experiment_runs": len(results),
        "validation_pruned_feature_count": len(pruned),
    }
    atomic_json(output_dir / "official_experiments_summary.json", details)
    return details


def _run_chronological_experiments(
    split: dict[str, list[dict]], selected_depth: int, config: PipelineConfig,
    output_dir: Path, resume: bool,
) -> dict:
    specs = [spec for spec in OFFICIAL_EXPERIMENTS if spec.name in CHRONO_EXPERIMENT_NAMES]
    results = []
    for spec in specs:
        for seed in config.seeds:
            results.append(run_experiment(
                split, "chronological", spec, FEATURES_24, seed, selected_depth, config,
                output_dir / spec.name / f"seed_{seed}", resume=resume,
                compute_importance=False,
            ))
    details = {"experiment_runs": len(results)}
    atomic_json(output_dir / "chronological_experiments_summary.json", details)
    return details


def _flatten_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((run_dir / "experiments").glob("*/*/seed_*/metrics.json")):
        result = _read_json(path)
        base = {
            "split_mode": result["split_mode"], "experiment": result["experiment"],
            "seed": result["seed"], "depth": result["depth"],
            "feature_count": result["feature_count"],
        }
        for metric, value in result["test"].items():
            rows.append({**base, "metric": metric, "value": value})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    numeric = long.dropna(subset=["value"]).copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    summary = numeric.groupby(
        ["split_mode", "experiment", "feature_count", "metric"], as_index=False
    ).agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
    return long, summary


def _placeholder_map(summary: pd.DataFrame, selected_depth: int) -> dict:
    values = {}
    for _, row in summary.iterrows():
        key = "METRIC_{}_{}_{}".format(
            str(row["split_mode"]).upper(), str(row["experiment"]).upper(),
            str(row["metric"]).upper().replace(".", "_").replace("-", "_"),
        )
        mean = row["mean"]
        std = row["std"]
        values[key] = None if pd.isna(mean) else (
            f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f} ± {std:.4f}"
        )
    values.update({
        "SELECTED_LSTM_DEPTH": str(selected_depth),
        "TABLE_EVENT_SAMPLE_AUDIT": "data/event_sample_summary.json and data/event_sample_audit.csv",
        "TABLE_CATALOG_CASES_4536_3784": "data/catalog_cases_4536_3784.csv",
        "TABLE_COMPARABLE_PAIRS": "aggregate/seed_summary.csv (event_event/event_censored metrics)",
        "TABLE_PRIMARY_RESULTS": "aggregate/seed_summary.csv",
        "TABLE_ABLATIONS": "aggregate/seed_summary.csv",
        "TABLE_OPERATIONAL_METRICS": "aggregate/seed_summary.csv (12/24/48/72/144h)",
        "TABLE_BRIER_BSS": "aggregate/seed_summary.csv (Brier, KM reference, BSS)",
        "TABLE_CALIBRATION": "experiments/official/tim_sfsa/seed_*/test_calibration.csv",
        "TABLE_BALANCING_ABLATION": "aggregate/seed_summary.csv (tim_sfsa vs balanced22)",
        "TABLE_PRETRAINING_ABLATION": "aggregate/seed_summary.csv (tim_sfsa vs no_pretraining)",
        "TABLE_FEATURE_SET_ABLATION": "aggregate/seed_summary.csv (tim_sfsa vs legacy18 vs validation_pruned)",
        "TABLE_LEGACY_CORRELATIONS": "experiments/official/tim_sfsa/seed_*/legacy_feature_correlation_explanation.csv",
        "TABLE_GROUP_IMPORTANCE": "experiments/official/tim_sfsa/seed_*/validation_correlated_group_importance.csv",
        "TABLE_RISK_QUANTILE_SENSITIVITY": "experiments/official/tim_sfsa/seed_*/risk_quantile_sensitivity.csv",
        "TABLE_LEAD_TIME": "experiments/official/tim_sfsa/seed_*/lead_time_summary.json",
        "TABLE_PH_DIAGNOSTIC": "experiments/official/tim_sfsa/seed_*/ph_diagnostic.json",
        "TABLE_CLUSTER_BOOTSTRAP": "experiments/*/*/seed_*/ar_cluster_bootstrap_summary.csv",
        "TABLE_PAIRED_MODEL_DIFFERENCES": "aggregate/paired_model_differences.csv",
        "TABLE_LINEAR_COX_RERUN": "aggregate/seed_summary.csv (official linear_cox)",
        "TABLE_CHRONOLOGICAL_RESULTS": "aggregate/seed_summary.csv (chronological)",
        "FIGURE_CALIBRATION": "figures/calibration_12_24_48h.png",
        "FIGURE_FEATURE_IMPORTANCE": "figures/feature_importance.png",
        "FIGURE_BRIER_BSS": "figures/brier_vs_km_reference.png",
        "FIGURE_REVISED_OVERVIEW": "figures/event_anchored_sampling.png",
    })
    return values


def _paired_model_differences(run_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """Summarise paired AR-bootstrap improvements over comparator models."""
    metric_directions = {
        "harrell_c_index": "primary_minus_comparator",
        "uno_c_index": "primary_minus_comparator",
        "integrated_auc": "primary_minus_comparator",
        "integrated_brier_score": "comparator_minus_primary",
        "24h_tss": "primary_minus_comparator",
        "24h_hss2": "primary_minus_comparator",
        "24h_bss": "primary_minus_comparator",
        "48h_tss": "primary_minus_comparator",
        "48h_hss2": "primary_minus_comparator",
        "48h_bss": "primary_minus_comparator",
    }
    rows = []
    experiments_root = run_dir / "experiments"
    for split_dir in sorted(experiments_root.glob("*")):
        if not split_dir.is_dir():
            continue
        primary_dir = split_dir / "tim_sfsa"
        if not primary_dir.is_dir():
            continue
        for comparator_dir in sorted(split_dir.iterdir()):
            if not comparator_dir.is_dir() or comparator_dir.name == "tim_sfsa":
                continue
            for seed in seeds:
                primary_path = primary_dir / f"seed_{seed}" / "ar_cluster_bootstrap.csv"
                comparator_path = (
                    comparator_dir / f"seed_{seed}" / "ar_cluster_bootstrap.csv"
                )
                if not primary_path.exists() or not comparator_path.exists():
                    continue
                primary = pd.read_csv(primary_path).set_index("replicate")
                comparator = pd.read_csv(comparator_path).set_index("replicate")
                common = primary.index.intersection(comparator.index)
                for metric, direction in metric_directions.items():
                    if metric not in primary.columns or metric not in comparator.columns:
                        continue
                    if direction == "primary_minus_comparator":
                        delta = primary.loc[common, metric] - comparator.loc[common, metric]
                    else:
                        delta = comparator.loc[common, metric] - primary.loc[common, metric]
                    delta = pd.to_numeric(delta, errors="coerce").dropna()
                    if delta.empty:
                        continue
                    rows.append({
                        "split_mode": split_dir.name,
                        "primary": "tim_sfsa",
                        "comparator": comparator_dir.name,
                        "seed": seed,
                        "metric": metric,
                        "improvement_definition": direction,
                        "bootstrap_replicates": len(delta),
                        "mean_improvement": float(delta.mean()),
                        "std_improvement": float(delta.std(ddof=1)),
                        "ci_lower_2.5": float(delta.quantile(0.025)),
                        "ci_upper_97.5": float(delta.quantile(0.975)),
                    })
    return pd.DataFrame(rows)


def run_pipeline(config: PipelineConfig, *, resume: bool, synthetic: bool = False) -> Path:
    config.normalise()
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = StageTracker(run_dir)
    project_root = Path(__file__).resolve().parents[1]
    config_path = run_dir / "resolved_config.json"
    resolved_config = config.as_dict()
    if resume and config_path.exists() and _read_json(config_path) != resolved_config:
        raise RuntimeError("Resolved configuration changed; use --fresh for a new run")
    atomic_json(config_path, resolved_config)
    atomic_json(run_dir / "environment.json", capture_environment())
    code_path = run_dir / "code_fingerprint.json"
    code_state = code_fingerprint(project_root)
    if resume and code_path.exists() and _read_json(code_path).get("digest") != code_state["digest"]:
        previous_code = _read_json(code_path)
        migration_path = project_root / "revision" / "resume_migration.json"
        migration = _read_json(migration_path) if migration_path.exists() else {}
        if (
            previous_code.get("digest") != migration.get("previous_digest")
            or code_state["digest"] != migration.get("replacement_digest")
        ):
            raise RuntimeError("Code fingerprint changed; use --fresh for a new run")
        history_path = run_dir / "code_fingerprint_history.json"
        history = _read_json(history_path) if history_path.exists() else []
        history.append({
            "previous": previous_code,
            "replacement": code_state,
            "reason": migration.get("reason", "Approved code-fingerprint migration"),
        })
        atomic_json(history_path, history)
    atomic_json(code_path, code_state)
    atomic_json(run_dir / "experiment_manifest.json", {
        "official": [spec.__dict__ for spec in OFFICIAL_EXPERIMENTS],
        "chronological": [
            spec.__dict__ for spec in OFFICIAL_EXPERIMENTS
            if spec.name in CHRONO_EXPERIMENT_NAMES
        ],
        "seeds": config.seeds,
        "depth_candidates": config.depth_candidates,
        "selection_split": "official partition 4",
        "test_selection_prohibited": True,
    })
    fingerprint = {"synthetic": True} if synthetic else dataset_fingerprint(config.data_root)
    fingerprint_path = run_dir / "data_fingerprint.json"
    if resume and fingerprint_path.exists():
        previous = _read_json(fingerprint_path)
        if previous.get("digest") != fingerprint.get("digest") or previous.get("synthetic") != fingerprint.get("synthetic"):
            raise RuntimeError("Data fingerprint changed; use --fresh for a new run")
    atomic_json(fingerprint_path, fingerprint)

    data_dir = run_dir / "data"
    def prepare_data():
        if synthetic:
            samples = make_synthetic_event_samples(config.seeds[0])
            audit = pd.DataFrame({"status": ["synthetic"] * len(samples)})
        else:
            samples, audit = load_event_dataset(config)
        save_event_dataset(samples, audit, data_dir)
        return {"samples": len(samples)}
    reuse_cached_data = resume
    if resume and tracker.is_complete(
        "prepare_event_data",
        [data_dir / "event_samples.pkl", data_dir / "event_sample_audit.csv"],
    ):
        try:
            load_cached_event_dataset(data_dir)
        except (EOFError, OSError, ValueError, RuntimeError, pickle.UnpicklingError):
            # A present but incomplete transfer must never be mistaken for a
            # valid checkpoint.  Rebuild it from the fingerprinted raw data.
            reuse_cached_data = False
    _stage(
        tracker, "prepare_event_data", prepare_data,
        [data_dir / "event_samples.pkl", data_dir / "event_sample_audit.csv"],
        reuse_cached_data,
    )
    samples, _ = load_cached_event_dataset(data_dir)

    official_split = make_splits(samples, "official")
    save_split_manifest(official_split, "official", data_dir / "official_split_manifest.csv")
    depth_dir = run_dir / "depth_selection"
    def tune_depth():
        selected = _depth_selection(official_split, config, depth_dir, resume)
        return {"selected_depth": selected}
    _stage(
        tracker, "depth_selection", tune_depth,
        [depth_dir / "selected_depth.json", depth_dir / "depth_selection_summary.csv"], resume,
    )
    selected_depth = int(_read_json(depth_dir / "selected_depth.json")["selected_depth"])

    experiments_root = run_dir / "experiments"
    if config.splits in {"all", "official"}:
        _stage(
            tracker, "official_experiments",
            lambda: _run_official_experiments(
                official_split, selected_depth, config, experiments_root / "official", resume
            ),
            [
                experiments_root / "official" / "validation_pruned_features.json",
                experiments_root / "official" / "official_experiments_summary.json",
            ], resume,
        )
    if config.splits in {"all", "chronological"}:
        chronological_split, chronological_exclusions = make_splits(
            samples, "chronological", return_exclusions=True
        )
        save_split_manifest(
            chronological_split, "chronological", data_dir / "chronological_split_manifest.csv"
        )
        save_split_exclusions(
            chronological_exclusions, "chronological",
            data_dir / "chronological_split_exclusions.csv",
        )
        _stage(
            tracker, "chronological_experiments",
            lambda: _run_chronological_experiments(
                chronological_split, selected_depth, config,
                experiments_root / "chronological", resume,
            ),
            [
                experiments_root / "chronological" / "tim_sfsa"
                / f"seed_{config.seeds[0]}" / "metrics.json",
                experiments_root / "chronological" / "chronological_experiments_summary.json",
            ],
            resume,
        )

    aggregate_dir = run_dir / "aggregate"
    def aggregate():
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        long, summary = _flatten_results(run_dir)
        long.to_csv(aggregate_dir / "all_seed_metrics.csv", index=False)
        summary.to_csv(aggregate_dir / "seed_summary.csv", index=False)
        try:
            (aggregate_dir / "seed_summary.md").write_text(
                summary.to_markdown(index=False), encoding="utf-8"
            )
            (aggregate_dir / "seed_summary.tex").write_text(
                summary.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
            )
        except Exception:
            pass
        placeholders = _placeholder_map(summary, selected_depth)
        atomic_json(aggregate_dir / "response_placeholders.json", placeholders)
        (aggregate_dir / "response_placeholders.md").write_text(
            "\n".join(f"- `{{{{{key}}}}}`: {value}" for key, value in placeholders.items()),
            encoding="utf-8",
        )
        paired = _paired_model_differences(run_dir, config.seeds)
        paired.to_csv(aggregate_dir / "paired_model_differences.csv", index=False)
        figures = create_revision_figures(run_dir, config.seeds)
        return {
            "metric_rows": len(long), "summary_rows": len(summary),
            "figures": [str(path) for path in figures],
            "paired_difference_rows": len(paired),
        }
    _stage(
        tracker, "aggregate", aggregate,
        [
            aggregate_dir / "seed_summary.csv",
            aggregate_dir / "response_placeholders.json",
            aggregate_dir / "paired_model_differences.csv",
        ], resume,
    )
    return run_dir
