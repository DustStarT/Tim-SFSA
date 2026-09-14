#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/home/lxc/Solar_Flare/data/SWAN}"
RUN_DIR="${RUN_DIR:-/mnt/disk16T/lxc/Tim-SFSA_v2/result/run_all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
SPLITS="${SPLITS:-all}"
RUN_MODE="${RUN_MODE:-resume}"

case "$RUN_MODE" in
  resume)
    MODE_FLAG="--resume"
    ;;
  fresh)
    MODE_FLAG="--fresh"
    ;;
  *)
    echo "RUN_MODE must be 'resume' or 'fresh'" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" -u run_revision_pipeline.py \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --splits "$SPLITS" \
  --device "$DEVICE" \
  "$MODE_FLAG"
