"""Dataset loading, event-sample materialisation and leakage-safe splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_prep.data_manager import DataManager
from data_prep.subsequence_generator import EventSampleConfig, generate_event_samples
from .config import FEATURES_24, HISTORY_FEATURES, PipelineConfig


def load_event_dataset(config: PipelineConfig) -> tuple[list[dict], pd.DataFrame]:
    manager = DataManager({
        "data_dir": config.data_root,
        "source_mode": "raw_harp",
        "require_flare_catalog": True,
        "allow_raw_timestamp_event_fallback": False,
        "partitions_to_process": [1, 2, 3, 4, 5],
        "specified_features": FEATURES_24,
        "include_nf_data": False,
    })
    parents = manager.load_and_merge_data()
    sample_cfg = EventSampleConfig(
        history_hours=config.history_hours,
        cadence_minutes=config.cadence_minutes,
        history_steps=config.history_steps,
        max_gap_minutes=config.max_gap_minutes,
        flare_history_hours=config.flare_history_hours,
    )
    samples: list[dict] = []
    audit_rows: list[dict] = []
    for parent in parents:
        generated, audit = generate_event_samples(
            parent, active_feature_names=FEATURES_24, config=sample_cfg, return_audit=True
        )
        samples.extend(generated)
        audit_rows.extend(audit)
    audit_rows.extend({
        "status": "dropped",
        "reason": "unmatched_major_catalog_event",
        **item,
    } for item in manager.unmatched_major_events)
    if not samples:
        raise RuntimeError("No valid event-anchored samples were generated")
    audit_df = pd.DataFrame([_flatten_audit_row(row) for row in audit_rows])
    return samples, audit_df


def _flatten_audit_row(row: dict) -> dict:
    flat = dict(row)
    for key, value in list(flat.items()):
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, default=str, ensure_ascii=False)
        elif isinstance(value, pd.Timestamp):
            flat[key] = value.isoformat()
    return flat


def save_event_dataset(samples: list[dict], audit: pd.DataFrame, directory: Path) -> None:
    import pickle

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "event_samples.pkl").open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    audit.to_csv(directory / "event_sample_audit.csv", index=False)
    if not audit.empty:
        text = audit.astype(str).agg(" ".join, axis=1)
        case_mask = text.str.contains(r"(?<!\d)(?:4536|3784)(?!\d)", regex=True, na=False)
        audit.loc[case_mask].to_csv(directory / "catalog_cases_4536_3784.csv", index=False)
    else:
        pd.DataFrame().to_csv(directory / "catalog_cases_4536_3784.csv", index=False)
    target_raw_duplicate_count = sum(
        int(sample["record_id"].get("duplicate_occurrence_count", 0))
        for sample in samples
    )
    covered_raw_duplicate_count = sum(
        int(event.get("duplicate_occurrence_count", 0))
        for sample in samples
        for event in sample["record_id"].get("covered_events", [])
    )
    target_catalog_duplicate_count = sum(
        int(sample["record_id"].get("catalog_duplicate_row_count", 0))
        for sample in samples
    )
    covered_catalog_duplicate_count = sum(
        int(event.get("catalog_duplicate_row_count", 0))
        for sample in samples
        for event in sample["record_id"].get("covered_events", [])
    )
    durations = np.asarray([sample["duration"] for sample in samples], dtype=float)
    audit_status_counts = (
        audit["status"].fillna("missing").astype(str).value_counts().to_dict()
        if "status" in audit.columns else {}
    )
    audit_reason_counts = (
        audit.loc[audit.get("status", "") == "dropped", "reason"]
        .fillna("unspecified").astype(str).value_counts().to_dict()
        if {"status", "reason"}.issubset(audit.columns) else {}
    )
    summary = {
        "sample_count": len(samples),
        "event_count": int(sum(int(sample["event"]) for sample in samples)),
        "censored_count": int(sum(1 - int(sample["event"]) for sample in samples)),
        "active_region_count": len({group_id(sample) for sample in samples}),
        "covered_event_count": int(sum(
            len(sample["record_id"].get("covered_events", [])) for sample in samples
        )),
        "duplicate_raw_label_occurrence_count": int(
            target_raw_duplicate_count + covered_raw_duplicate_count
        ),
        "duplicate_catalog_row_count": int(
            target_catalog_duplicate_count + covered_catalog_duplicate_count
        ),
        "catalog_conflict_event_count": int(sum(
            bool(sample["record_id"].get("catalog_conflict", False))
            + sum(bool(event.get("catalog_conflict", False)) for event in sample["record_id"].get("covered_events", []))
            for sample in samples
        )),
        "unmatched_major_catalog_event_count": int(
            audit_reason_counts.get("unmatched_major_catalog_event", 0)
        ),
        "audit_status_counts": {key: int(value) for key, value in audit_status_counts.items()},
        "dropped_reason_counts": {key: int(value) for key, value in audit_reason_counts.items()},
        "partition_sample_counts": {
            str(partition): int(sum(
                sample["record_id"].get("partition") == partition for sample in samples
            ))
            for partition in (1, 2, 3, 4, 5)
        },
        "duration_min_hours": float(np.min(durations)),
        "duration_q1_hours": float(np.quantile(durations, 0.25)),
        "duration_median_hours": float(np.median(durations)),
        "duration_q3_hours": float(np.quantile(durations, 0.75)),
        "duration_max_hours": float(np.max(durations)),
    }
    (directory / "event_sample_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def load_cached_event_dataset(directory: Path) -> tuple[list[dict], pd.DataFrame]:
    import pickle

    with (directory / "event_samples.pkl").open("rb") as handle:
        samples = pickle.load(handle)
    audit = pd.read_csv(directory / "event_sample_audit.csv")
    return samples, audit


def group_id(sample: dict) -> str:
    record = sample.get("record_id", {})
    return str(record.get("harp_id") or record.get("raw") or record.get("ar"))


def prediction_start(sample: dict) -> pd.Timestamp:
    value = sample.get("record_id", {}).get("prediction_start")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Sample {sample.get('sample_id')} has no valid prediction_start")
    return pd.Timestamp(parsed)


def _chronological_band(sample: dict) -> str | None:
    year = prediction_start(sample).year
    if 2010 <= year <= 2014:
        return "train"
    if year == 2015:
        return "validation"
    if 2016 <= year <= 2018:
        return "test"
    return None


def make_splits(
    samples: list[dict], mode: str, *, return_exclusions: bool = False,
):
    exclusions: list[dict] = []
    if mode == "official":
        split = {
            "train": [s for s in samples if s["record_id"].get("partition") in {1, 2, 3}],
            "validation": [s for s in samples if s["record_id"].get("partition") == 4],
            "test": [s for s in samples if s["record_id"].get("partition") == 5],
        }
    elif mode == "chronological":
        split = {"train": [], "validation": [], "test": []}
        grouped: dict[str, list[dict]] = {}
        for sample in samples:
            grouped.setdefault(group_id(sample), []).append(sample)

        for group, rows in grouped.items():
            bands = {_chronological_band(sample) for sample in rows}
            eligible_bands = {band for band in bands if band is not None}
            if len(eligible_bands) > 1:
                for sample in rows:
                    exclusions.append({
                        "sample": sample,
                        "reason": "active_region_crosses_chronological_boundary",
                        "group_bands": ",".join(sorted(eligible_bands)),
                    })
                continue
            for sample in rows:
                band = _chronological_band(sample)
                if band is None:
                    exclusions.append({
                        "sample": sample,
                        "reason": "forecast_origin_outside_2010_2018",
                        "group_bands": "",
                    })
                else:
                    split[band].append(sample)
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    for name, rows in split.items():
        if not rows:
            raise RuntimeError(f"{mode} split produced an empty {name} set")
    _assert_group_disjoint(split, mode)
    if return_exclusions:
        return split, exclusions
    return split


def _assert_group_disjoint(split: dict[str, list[dict]], mode: str) -> None:
    groups = {name: {group_id(sample) for sample in rows} for name, rows in split.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = groups[left] & groups[right]
        if overlap:
            raise RuntimeError(
                f"{mode} split leaks {len(overlap)} active regions between {left} and {right}: "
                f"{sorted(overlap)[:10]}"
            )


def save_split_manifest(split: dict[str, list[dict]], mode: str, output: Path) -> None:
    rows = []
    for split_name, samples in split.items():
        for sample in samples:
            record = sample["record_id"]
            rows.append({
                "split_mode": mode,
                "split": split_name,
                "sample_id": sample["sample_id"],
                "group_id": group_id(sample),
                "ar": record.get("ar"),
                "harp_id": record.get("harp_id"),
                "partition": record.get("partition"),
                "prediction_start": record.get("prediction_start"),
                "duration_hours": sample["duration"],
                "event": sample["event"],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def save_split_exclusions(exclusions: list[dict], mode: str, output: Path) -> None:
    rows = []
    for item in exclusions:
        sample = item["sample"]
        record = sample["record_id"]
        rows.append({
            "split_mode": mode,
            "reason": item["reason"],
            "group_bands": item.get("group_bands", ""),
            "sample_id": sample["sample_id"],
            "group_id": group_id(sample),
            "ar": record.get("ar"),
            "harp_id": record.get("harp_id"),
            "partition": record.get("partition"),
            "prediction_start": record.get("prediction_start"),
            "duration_hours": sample["duration"],
            "event": sample["event"],
        })
    columns = [
        "split_mode", "reason", "group_bands", "sample_id", "group_id", "ar",
        "harp_id", "partition", "prediction_start", "duration_hours", "event",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)


def arrays_from_samples(
    samples: Iterable[dict],
    feature_names: list[str],
    *,
    include_bc_history: bool = False,
) -> dict[str, np.ndarray | list[str] | pd.DataFrame]:
    rows = list(samples)
    if not rows:
        raise ValueError("Cannot create arrays from an empty sample collection")
    base_indices = [FEATURES_24.index(name) for name in feature_names]
    x = np.stack([
        np.asarray(sample["features"], dtype=np.float64)[:, base_indices] for sample in rows
    ])
    names = list(feature_names)
    if include_bc_history:
        history = np.asarray([
            [sample.get("causal_flare_history", {}).get(name, 0.0) for name in HISTORY_FEATURES]
            for sample in rows
        ], dtype=np.float64)
        history = np.repeat(history[:, None, :], x.shape[1], axis=1)
        x = np.concatenate([x, history], axis=2)
        names.extend(HISTORY_FEATURES)
    metadata = pd.DataFrame([{
        "sample_id": sample["sample_id"],
        "group_id": group_id(sample),
        "prediction_start": prediction_start(sample),
        "event_time": sample["record_id"].get("event_time"),
        "event_id": sample["record_id"].get("event_id"),
        "event_class": sample["record_id"].get("event_class"),
        "partition": sample["record_id"].get("partition"),
    } for sample in rows])
    return {
        "x": x,
        "duration": np.asarray([sample["duration"] for sample in rows], dtype=np.float64),
        "event": np.asarray([sample["event"] for sample in rows], dtype=np.int64),
        "groups": metadata["group_id"].astype(str).to_numpy(),
        "feature_names": names,
        "metadata": metadata,
    }
