import numpy as np
import pandas as pd
import pytest

from data_prep.subsequence_generator import EventSampleConfig, generate_event_samples
from data_prep.data_manager import DataManager
from revision.config import FEATURES_24


def parent(hours=20, events=None, gap_index=None):
    timestamps = list(pd.date_range("2014-01-01", periods=int(hours * 5) + 1, freq="12min"))
    if gap_index is not None:
        timestamps.pop(gap_index)
        timestamps.pop(gap_index)
    features = pd.DataFrame(
        np.arange(len(timestamps) * len(FEATURES_24), dtype=float).reshape(len(timestamps), -1),
        columns=FEATURES_24,
    )
    return {
        "record_id": "ar123", "harp_id": "123", "partition": 1,
        "features": features, "timestamps_list": [str(value) for value in timestamps],
        "start_time": timestamps[0], "end_time": timestamps[-1],
        "major_events": events or [], "flare_history_events": events or [],
    }


def event(event_id, hour):
    return {
        "flare_id": event_id, "flare_class": "M1.0",
        "time": pd.Timestamp("2014-01-01") + pd.Timedelta(hours=hour),
        "source": "test_catalog", "sol_standard": f"SOL-{event_id}",
    }


def test_multiple_events_covered_and_terminal_censor():
    samples, audit = generate_event_samples(
        parent(events=[event(1, 2), event(2, 6), event(3, 8), event(4, 14)]),
        FEATURES_24, EventSampleConfig(), return_audit=True,
    )
    assert [sample["event"] for sample in samples] == [1, 1, 0]
    assert [sample["record_id"]["event_id"] for sample in samples] == [2, 4, None]
    assert [item["event_id"] for item in samples[0]["record_id"]["covered_events"]] == [1]
    assert [item["event_id"] for item in samples[1]["record_id"]["covered_events"]] == [3]
    assert samples[-1]["record_id"]["censor_reason"] == "observation_end"
    assert all(sample["features"].shape == (20, 24) for sample in samples)
    assert any(row["status"] == "created" for row in audit)


def test_no_event_creates_one_actual_end_censor():
    samples = generate_event_samples(parent(hours=10), FEATURES_24)
    assert len(samples) == 1
    assert samples[0]["event"] == 0
    expected = 10.0 - 19 * 12.0 / 60.0
    assert samples[0]["duration"] == pytest.approx(expected)


def test_event_at_forecast_origin_is_covered_not_target():
    origin_hour = 19 * 12.0 / 60.0
    samples = generate_event_samples(
        parent(events=[event(1, origin_hour), event(2, 8)]), FEATURES_24
    )
    assert samples[0]["record_id"]["event_id"] == 2
    assert samples[0]["record_id"]["covered_events"][0]["event_id"] == 1


def test_duplicate_event_id_is_deduplicated():
    duplicate = event(2, 6)
    samples = generate_event_samples(
        parent(events=[duplicate, dict(duplicate), event(3, 12)]), FEATURES_24
    )
    assert [sample["record_id"]["event_id"] for sample in samples if sample["event"]] == [2, 3]
    assert samples[0]["record_id"]["source_occurrence_count"] == 2
    assert samples[0]["record_id"]["duplicate_occurrence_count"] == 1


def test_incomplete_post_event_history_is_audited():
    samples, audit = generate_event_samples(
        parent(hours=7, events=[event(1, 6)]), FEATURES_24,
        return_audit=True,
    )
    assert len(samples) == 1 and samples[0]["event"] == 1
    assert audit[-1]["reason"] == "incomplete_history"


def test_history_gap_drops_anchor():
    samples, audit = generate_event_samples(
        parent(hours=10, gap_index=10), FEATURES_24, return_audit=True
    )
    assert samples == []
    assert audit[-1]["reason"] == "history_gap"


def test_no_positive_followup_is_dropped():
    samples, audit = generate_event_samples(
        parent(hours=19 * 12.0 / 60.0), FEATURES_24, return_audit=True
    )
    assert samples == []
    assert audit[-1]["reason"] == "nonpositive_followup"


def test_only_a_real_sol_locator_is_accepted_as_sol_identifier():
    assert DataManager._sol_identifier(pd.Series({"event_id": "8898"})) == ""
    assert DataManager._sol_identifier(
        pd.Series({"event_id": "SOL2014-09-10T17:45"})
    ) == "SOL2014-09-10T17:45"
