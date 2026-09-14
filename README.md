# Tim-SFSA reviewer-revision pipeline

This repository is the cleaned public code release corresponding to the paper draft in `Deep Survival Analysis of Solar Flare Forecasting for Continuous Risk Forecasting`.

This is the first revised version. It mainly improves the data partitioning strategy and expands a set of comparison and ablation experiments.

The chronological experiment uses forecast-origin years: 11 training and three
validation censored intervals cross year cutoffs. It is a chronological
sensitivity analysis, not a strictly prospective backtest. The returned
`environment.json` records Python 3.7.12/PyTorch 1.13.1; the YAML below defines
a separate reproduction environment, not the historical runtime of every fit.

This repository implements the event-anchored revision of Tim-SFSA. Each HARP
contributes a four-hour, 20-step history at observation start and after eligible
M/X events. Follow-up ends at the next catalog-resolved M/X flare or the actual
observation end. There are no sliding samples and no fixed 48-hour censoring.

The primary model uses all 24 SHARP parameters. The former correlation-reduced
18-feature set is retained only as an ablation.

## Environment

```bash
conda env create -f environment_revision.yml
conda activate tim-sfsa-revision
```

CUDA 11.8 is selected for compatibility with the external GTX 1070-class
server. Change the PyTorch CUDA build in the environment file if the external
driver requires a different supported runtime.

## Complete run

```bash
python run_revision_pipeline.py \
  --data-root /path/to/SWAN \
  --run-dir /path/to/revision_run \
  --splits all \
  --resume
```

For the documented external server paths, the equivalent resumable wrapper is:

```bash
bash run_revision_server.sh
```

Override paths without editing it, for example
`DATA_ROOT=/data/SWAN RUN_DIR=/runs/revision bash run_revision_server.sh`.
Use `RUN_MODE=fresh` for a new timestamped run.

- `--resume` verifies the input fingerprint, skips complete stages and restores
  model, optimizer, scheduler, epoch and random-number states.
- The known interrupted pre-fix run is migrated automatically: compatible
  official artifacts are retained, the unconverged linear-Cox baseline is
  superseded, and chronological evaluation resumes after purging HARPs that
  cross year boundaries. Unrelated code changes are still rejected.
- `--fresh` creates a new timestamped output directory if `--run-dir` is not
  empty.
- `--splits official` or `--splits chronological` runs one evaluation scheme;
  `all` runs both.
- `--quick` is a two-epoch smoke test and is not a reportable experiment.

The full run performs three-seed depth selection, the official 1–3/4/5 split,
origin-year chronological sensitivity, all baselines and ablations, censor-aware
metrics, calibration, 1,000 active-region cluster bootstrap replicates,
AR-block feature importance and response-placeholder generation.

## Important outputs

```text
result/
  data/                         event samples, split manifests, catalog audits
  depth_selection/              1/2/3/7-layer validation selection
  experiments/official/         full experiment matrix
  experiments/chronological/    main model and core baselines
  aggregate/                    three-seed tables and placeholder map
  figures/                      sampling, calibration, Brier and importance figures
  pipeline_state.json           resumable stage state
```

The chronological manifest has a companion
`data/chronological_split_exclusions.csv`. A HARP with forecast origins on both
sides of a 2015 or 2016 boundary is excluded as a whole, preserving both strict
calendar ordering and HARP disjointness.

Deterministic CUDA execution defaults to
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before PyTorch is imported.

## Tests

```bash
pytest -m "not integration"
```

The slow synthetic end-to-end/resume test is opt-in:

```bash
RUN_REVISION_INTEGRATION=1 pytest -m integration
```

`main.py` is a compatibility shim to the new runner. Old duplicate runners,
fixed-window generation, post-hoc time-head outputs and log-rank workflow entry
points have been removed.
