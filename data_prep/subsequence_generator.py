"""Event-anchored sample construction for Tim-SFSA.

The former implementation produced overlapping sliding windows and assigned
every non-event window the same 48-hour censoring time.  The revised design
creates one sample at observation start and one after each eligible M/X flare.
Follow-up ends at the next M/X flare or the actual HARP observation end.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventSampleConfig:
    history_hours: float = 4.0
    cadence_minutes: float = 12.0
    history_steps: int = 20
    max_gap_minutes: float = 24.0
    flare_history_hours: float = 24.0


def _timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _normalise_events(events: Iterable[Any], observation_end: pd.Timestamp) -> list[dict]:
    deduplicated: dict[Any, dict] = {}
    for raw in events or []:
        event = dict(raw) if isinstance(raw, dict) else {"time": raw}
        event_time = _timestamp(event.get("time"))
        if event_time is None or event_time > observation_end:
            continue
        event["time"] = event_time
        event_id = event.get("flare_id")
        key = ("id", str(event_id)) if event_id not in (None, "") else (
            str(event.get("flare_class", "")), event_time.isoformat()
        )
        if key not in deduplicated:
            event.setdefault("source_occurrence_count", 1)
            event.setdefault("duplicate_occurrence_count", 0)
            deduplicated[key] = event
            continue
        retained = deduplicated[key]
        retained["source_occurrence_count"] = int(
            retained.get("source_occurrence_count", 1)
        ) + int(event.get("source_occurrence_count", 1))
        retained["duplicate_occurrence_count"] = int(
            retained.get("duplicate_occurrence_count", 0)
        ) + int(event.get("duplicate_occurrence_count", 0)) + 1
    return sorted(deduplicated.values(), key=lambda item: item["time"])


def _record_metadata(sample: dict) -> dict:
    raw = str(sample.get("record_id", ""))
    match = re.findall(r"ar\D*?(\d+)", raw, flags=re.IGNORECASE)
    normalised_ar = f"ar{int(match[-1])}" if match else raw.lower()
    return {
        "raw": sample.get("record_id"),
        "ar": sample.get("ar_id") or normalised_ar,
        "harp_id": sample.get("harp_id"),
        "partition": sample.get("partition"),
        "observation_start": _timestamp(sample.get("start_time")),
        "observation_end": _timestamp(sample.get("end_time")),
        "source_file": sample.get("source_file"),
    }


def _event_summary(event: dict) -> dict:
    return {
        "event_id": event.get("flare_id"),
        "event_time": event.get("time"),
        "event_class": event.get("flare_class"),
        "event_source": event.get("source"),
        "catalog_match": bool(event.get("catalog_match", False)),
        "raw_label_timestamp": event.get("raw_label_timestamp"),
        "catalog_time_offset_hours": event.get("catalog_time_offset_hours"),
        "sol_standard": event.get("sol_standard"),
        "catalog_noaa_active_region": event.get("noaa_active_region"),
        "catalog_row_count": int(event.get("catalog_row_count", 1)),
        "catalog_duplicate_row_count": int(event.get("catalog_duplicate_row_count", 0)),
        "catalog_conflict": bool(event.get("catalog_conflict", False)),
        "catalog_alternate_records": event.get("catalog_alternate_records", []),
        "raw_label": event.get("raw_label"),
        "raw_labels": event.get("raw_labels", [event.get("raw_label")]),
        "source_occurrence_count": int(event.get("source_occurrence_count", 1)),
        "duplicate_occurrence_count": int(event.get("duplicate_occurrence_count", 0)),
    }


def _causal_flare_history(
    events: list[dict], prediction_start: pd.Timestamp, lookback_hours: float
) -> dict[str, float]:
    lower = prediction_start - pd.Timedelta(hours=float(lookback_hours))
    result = {"B_COUNT_24H": 0.0, "C_COUNT_24H": 0.0}
    for event in events:
        event_time = event["time"]
        if not (lower < event_time <= prediction_start):
            continue
        flare_class = str(event.get("flare_class", event.get("label_prefix", ""))).upper()
        if flare_class.startswith("B"):
            result["B_COUNT_24H"] += 1.0
        elif flare_class.startswith("C"):
            result["C_COUNT_24H"] += 1.0
    return result


def generate_event_samples(
    sample_dict: dict,
    active_feature_names: list[str] | None = None,
    config: EventSampleConfig | dict | None = None,
    *,
    return_audit: bool = False,
):
    """Create event-anchored survival samples for one HARP.

    Flares occurring while the four-hour covariate history is collected are
    recorded as covered events and skipped as targets.
    """

    if config is None:
        cfg = EventSampleConfig()
    elif isinstance(config, EventSampleConfig):
        cfg = config
    else:
        cfg = EventSampleConfig(**{
            key: value for key, value in dict(config).items()
            if key in EventSampleConfig.__dataclass_fields__
        })

    audit: list[dict] = []
    features = sample_dict.get("features")
    timestamps_raw = sample_dict.get("timestamps_list", [])
    if features is None:
        result = []
        audit.append({"status": "dropped", "reason": "missing_features"})
        return (result, audit) if return_audit else result

    features_df = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    if active_feature_names:
        features_df = features_df.reindex(columns=list(active_feature_names))
    timestamps = pd.Series([_timestamp(value) for value in timestamps_raw], name="timestamp")
    if len(features_df) != len(timestamps):
        result = []
        audit.append({
            "status": "dropped", "reason": "feature_timestamp_length_mismatch",
            "feature_rows": len(features_df), "timestamp_rows": len(timestamps),
        })
        return (result, audit) if return_audit else result

    aligned = features_df.reset_index(drop=True).copy()
    aligned.insert(0, "__timestamp__", timestamps)
    aligned.dropna(subset=["__timestamp__"], inplace=True)
    aligned.sort_values("__timestamp__", inplace=True)
    aligned.drop_duplicates(subset=["__timestamp__"], keep="first", inplace=True)
    aligned.reset_index(drop=True, inplace=True)
    if aligned.empty:
        result = []
        audit.append({"status": "dropped", "reason": "no_valid_timestamps"})
        return (result, audit) if return_audit else result

    observation_start = pd.Timestamp(aligned["__timestamp__"].iloc[0])
    observation_end = pd.Timestamp(aligned["__timestamp__"].iloc[-1])
    major_events = _normalise_events(sample_dict.get("major_events", []), observation_end)
    history_events = _normalise_events(
        sample_dict.get("flare_history_events", major_events), observation_end
    )
    record_base = _record_metadata(sample_dict)
    record_base["observation_start"] = observation_start
    record_base["observation_end"] = observation_end

    samples: list[dict] = []
    anchor = observation_start
    iteration = 0
    max_iterations = len(major_events) + 2
    while iteration < max_iterations:
        history_end_exclusive = anchor + pd.Timedelta(hours=float(cfg.history_hours))
        eligible = aligned[
            (aligned["__timestamp__"] >= anchor)
            & (aligned["__timestamp__"] < history_end_exclusive)
        ].iloc[: int(cfg.history_steps)]
        if len(eligible) < int(cfg.history_steps):
            partial_times = eligible["__timestamp__"].reset_index(drop=True)
            partial_gaps = partial_times.diff().dropna()
            partial_max_gap = (
                partial_gaps.max() if not partial_gaps.empty else pd.Timedelta(0)
            )
            allowed_gap = pd.Timedelta(minutes=float(cfg.max_gap_minutes))
            reason = "history_gap" if partial_max_gap > allowed_gap else "incomplete_history"
            audit.append({
                **record_base, "status": "dropped", "reason": reason,
                "anchor_time": anchor, "available_steps": int(len(eligible)),
                "required_steps": int(cfg.history_steps),
                "max_gap_minutes": partial_max_gap.total_seconds() / 60.0,
            })
            break

        history_times = eligible["__timestamp__"].reset_index(drop=True)
        first_delay = history_times.iloc[0] - anchor
        gaps = history_times.diff().dropna()
        max_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)
        median_gap = gaps.median() if not gaps.empty else pd.Timedelta(0)
        history_span = history_times.iloc[-1] - history_times.iloc[0]
        allowed_gap = pd.Timedelta(minutes=float(cfg.max_gap_minutes))
        nominal_gap = pd.Timedelta(minutes=float(cfg.cadence_minutes))
        minimum_span = nominal_gap * max(0, int(cfg.history_steps) - 2)
        cadence_invalid = (
            median_gap < nominal_gap / 2
            or median_gap > allowed_gap
            or history_span < minimum_span
        )
        if first_delay > allowed_gap or max_gap > allowed_gap or cadence_invalid:
            audit.append({
                **record_base, "status": "dropped", "reason": "history_gap",
                "anchor_time": anchor,
                "first_delay_minutes": first_delay.total_seconds() / 60.0,
                "max_gap_minutes": max_gap.total_seconds() / 60.0,
                "median_gap_minutes": median_gap.total_seconds() / 60.0,
                "history_span_minutes": history_span.total_seconds() / 60.0,
            })
            break

        prediction_start = pd.Timestamp(history_times.iloc[-1])
        if prediction_start >= observation_end:
            audit.append({
                **record_base, "status": "dropped", "reason": "nonpositive_followup",
                "anchor_time": anchor, "prediction_start": prediction_start,
            })
            break

        covered_events = [
            event for event in major_events if anchor < event["time"] <= prediction_start
        ]
        target = next(
            (event for event in major_events if prediction_start < event["time"] <= observation_end),
            None,
        )
        followup_end = target["time"] if target is not None else observation_end
        duration_hours = (followup_end - prediction_start).total_seconds() / 3600.0
        if duration_hours <= 0:
            audit.append({
                **record_base, "status": "dropped", "reason": "nonpositive_duration",
                "anchor_time": anchor, "prediction_start": prediction_start,
                "followup_end": followup_end,
            })
            break

        target_summary = _event_summary(target) if target is not None else {
            "event_id": None, "event_time": None, "event_class": None,
            "event_source": None, "sol_standard": None, "raw_label": None,
            "catalog_match": None, "raw_label_timestamp": None,
            "catalog_time_offset_hours": None,
            "catalog_noaa_active_region": None, "catalog_row_count": 0,
            "catalog_duplicate_row_count": 0, "catalog_conflict": False,
            "catalog_alternate_records": [],
            "raw_labels": [], "source_occurrence_count": 0,
            "duplicate_occurrence_count": 0,
        }
        sample_id = f"{record_base.get('harp_id') or record_base.get('raw')}__event_anchor_{iteration:03d}"
        record_meta = {
            **record_base, "sample_id": sample_id, "interval_index": iteration,
            "anchor_time": anchor, "history_start": pd.Timestamp(history_times.iloc[0]),
            "history_end": prediction_start, "prediction_start": prediction_start,
            "followup_end": followup_end,
            "censor_reason": None if target is not None else "observation_end",
            "covered_events": [_event_summary(event) for event in covered_events],
            **target_summary,
        }
        samples.append({
            "sample_id": sample_id,
            "features": eligible.drop(columns=["__timestamp__"]).to_numpy(dtype=np.float32),
            "feature_names": list(eligible.columns[1:]),
            "history_timestamps": [pd.Timestamp(value) for value in history_times],
            "duration": float(duration_hours), "duration_hours": float(duration_hours),
            "duration_units": "hours", "event": int(target is not None),
            "record_id": record_meta,
            "causal_flare_history": _causal_flare_history(
                history_events, prediction_start, cfg.flare_history_hours
            ),
        })
        audit.append({
            **record_meta, "status": "created", "event": int(target is not None),
            "duration_hours": float(duration_hours),
            "covered_event_count": len(covered_events),
        })

        if target is None:
            break
        anchor = pd.Timestamp(target["time"])
        iteration += 1

    if iteration >= max_iterations:
        audit.append({
            **record_base, "status": "dropped", "reason": "iteration_guard",
            "config": asdict(cfg),
        })
    return (samples, audit) if return_audit else samples
