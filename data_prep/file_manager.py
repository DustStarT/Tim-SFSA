"""Locate raw SWAN-SF HARP partition files."""

from __future__ import annotations

import glob
import os


def resolve_raw_swan_dir(data_dir):
    """Return the directory containing ``partition1`` through ``partition5``."""
    candidates = (data_dir, os.path.join(data_dir, "SWAN"))
    for candidate in candidates:
        partition1 = os.path.join(candidate, "partition1")
        if os.path.isdir(partition1) and glob.glob(os.path.join(partition1, "*.csv")):
            return candidate
    return None


def get_raw_harp_files(data_dir, partition):
    raw_root = resolve_raw_swan_dir(data_dir)
    if raw_root is None:
        raise FileNotFoundError(
            "Could not find raw SWAN-SF HARP files. Expected partition*/*.csv "
            f"under {data_dir!r} or its SWAN subdirectory."
        )
    partition_dir = os.path.join(raw_root, f"partition{int(partition)}")
    if not os.path.isdir(partition_dir):
        raise FileNotFoundError(f"Missing raw SWAN-SF partition: {partition_dir}")
    files = sorted(glob.glob(os.path.join(partition_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw HARP CSV files found in {partition_dir}")
    return files
