"""Atomic stage state, fingerprints and environment capture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def dataset_fingerprint(data_root: str) -> dict:
    root = Path(data_root)
    candidates = []
    for pattern in ("partition*/*.csv", "SWAN/partition*/*.csv", "integrated_flare_data/*.csv"):
        candidates.extend(root.glob(pattern))
    parent_catalog = root.parent / "integrated_flare_data" / "goes_flares_integrated.csv"
    if parent_catalog.is_file():
        candidates.append(parent_catalog)
    rows = []
    for path in sorted(set(candidates)):
        stat = path.stat()
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path.resolve())
        rows.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "algorithm": "sha256(path,size,mtime_ns manifest)",
        "digest": hashlib.sha256(payload).hexdigest(),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
    }


def code_fingerprint(project_root: Path) -> dict:
    digest = hashlib.sha256()
    files = []
    for path in sorted(project_root.rglob("*.py")):
        if "results" in path.parts or "__pycache__" in path.parts:
            continue
        content = path.read_bytes()
        # POSIX separators make the same source tree hash identically on the
        # Linux training server and a Windows review workstation.
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(content)
        files.append(path.relative_to(project_root).as_posix())
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "files": files}


def capture_environment() -> dict:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=False,
            capture_output=True, text=True, timeout=60,
        ).stdout.splitlines()
    except Exception:
        freeze = []
    return {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": freeze,
    }


class StageTracker:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "pipeline_state.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"created_utc": utc_now(), "stages": {}}

    def is_complete(self, stage: str, required: list[Path] | None = None) -> bool:
        entry = self.state["stages"].get(stage, {})
        return entry.get("status") == "complete" and all(
            path.exists() for path in (required or [])
        )

    def start(self, stage: str) -> None:
        self.state["stages"][stage] = {"status": "running", "started_utc": utc_now()}
        atomic_json(self.path, self.state)

    def complete(self, stage: str, artifacts: list[Path] | None = None, details: dict | None = None) -> None:
        entry = self.state["stages"].setdefault(stage, {})
        entry.update({
            "status": "complete", "completed_utc": utc_now(),
            "artifacts": [str(path) for path in (artifacts or [])],
            "details": details or {},
        })
        atomic_json(self.path, self.state)

    def fail(self, stage: str, error: Exception) -> None:
        entry = self.state["stages"].setdefault(stage, {})
        entry.update({"status": "failed", "failed_utc": utc_now(), "error": repr(error)})
        atomic_json(self.path, self.state)
