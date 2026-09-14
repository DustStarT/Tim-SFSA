from pathlib import Path
import os

import pandas as pd
import pytest

from revision.data import make_splits
from revision.state import StageTracker
from revision.synthetic import make_synthetic_event_samples


def test_official_and_chronological_splits_are_group_disjoint():
    samples = make_synthetic_event_samples()
    for mode in ("official", "chronological"):
        split = make_splits(samples, mode)
        groups = {
            name: {sample["record_id"]["harp_id"] for sample in rows}
            for name, rows in split.items()
        }
        assert not groups["train"] & groups["validation"]
        assert not groups["train"] & groups["test"]
        assert not groups["validation"] & groups["test"]


def test_chronological_split_purges_complete_boundary_crossing_harp():
    samples = make_synthetic_event_samples()
    crossing_group = samples[0]["record_id"]["harp_id"]
    same_group = [
        sample for sample in samples
        if sample["record_id"]["harp_id"] == crossing_group
    ]
    same_group[0]["record_id"]["prediction_start"] = pd.Timestamp("2014-12-31 23:48")
    same_group[1]["record_id"]["prediction_start"] = pd.Timestamp("2015-01-01 04:00")

    split, exclusions = make_splits(
        samples, "chronological", return_exclusions=True
    )

    assert all(
        sample["record_id"]["harp_id"] != crossing_group
        for rows in split.values() for sample in rows
    )
    excluded = [
        item for item in exclusions
        if item["sample"]["record_id"]["harp_id"] == crossing_group
    ]
    assert len(excluded) == len(same_group)
    assert {item["reason"] for item in excluded} == {
        "active_region_crosses_chronological_boundary"
    }


def test_stage_tracker_requires_artifact_for_resume(tmp_path: Path):
    tracker = StageTracker(tmp_path)
    artifact = tmp_path / "artifact.txt"
    tracker.start("stage")
    tracker.complete("stage", [artifact])
    assert not tracker.is_complete("stage", [artifact])
    artifact.write_text("ok", encoding="utf-8")
    assert tracker.is_complete("stage", [artifact])


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_REVISION_INTEGRATION") != "1",
    reason="set RUN_REVISION_INTEGRATION=1 to run the full synthetic matrix",
)
def test_synthetic_pipeline_can_resume(tmp_path: Path):
    pytest.importorskip("torch")
    from revision.config import PipelineConfig
    from revision.pipeline import run_pipeline
    run_dir = tmp_path / "run"
    config = PipelineConfig(
        data_root=str(tmp_path), run_dir=str(run_dir), splits="all",
        device="cpu", quick=True,
    )
    run_pipeline(config, resume=False, synthetic=True)
    run_pipeline(config, resume=True, synthetic=True)
    assert (run_dir / "aggregate" / "seed_summary.csv").exists()
    assert (run_dir / "experiments" / "official" / "tim_sfsa" / "seed_2345" / "metrics.json").exists()
    assert (run_dir / "experiments" / "chronological" / "tim_sfsa" / "seed_2345" / "metrics.json").exists()
