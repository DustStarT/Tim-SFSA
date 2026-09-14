"""Single command for all revised training, experiments and evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the event-anchored Tim-SFSA reviewer revision pipeline."
    )
    parser.add_argument("--data-root", required=True, help="SWAN-SF root directory")
    parser.add_argument("--run-dir", required=True, help="Output directory")
    parser.add_argument("--splits", choices=["all", "official", "chronological"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume completed stages and epoch checkpoints")
    mode.add_argument("--fresh", action="store_true", help="Start in a new timestamped directory if run-dir exists")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--quick", action="store_true", help="Two-epoch smoke-test configuration")
    parser.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    from revision.config import PipelineConfig
    from revision.pipeline import run_pipeline
    run_dir = Path(args.run_dir).expanduser()
    if args.fresh and run_dir.exists() and any(run_dir.iterdir()):
        run_dir = run_dir.with_name(
            f"{run_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    elif not args.resume and not args.fresh and (run_dir / "pipeline_state.json").exists():
        raise SystemExit("Existing pipeline state found. Use --resume or --fresh.")
    config = PipelineConfig(
        data_root=args.data_root, run_dir=str(run_dir), splits=args.splits,
        device=args.device, quick=args.quick,
    )
    completed = run_pipeline(config, resume=args.resume, synthetic=args.synthetic)
    print(f"Revision pipeline complete: {completed}")


if __name__ == "__main__":
    main()
