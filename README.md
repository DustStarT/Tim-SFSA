# Tim-SFSA((Temporally-dependent Solar Flare Survival Analysis)) model

This repository is the cleaned public code release corresponding to the paper in `Deep Survival Analysis of Solar Flare Forecasting for Continuous Risk Forecasting`.

The project implements a sequence-based deep survival analysis solar-flare modeling pipeline with:

- two-stage training: classification pretraining followed by survival modeling
- LSTM and Transformer sequence encoders
- DeepSurv, DeepHit, CoxPH, and CoxKAN downstream survival backbones
- preprocessing, evaluation, plotting, and survival-time post-processing utilities

## Repository Layout

```text
FlameShadowModel/
|- main.py                         # main training / evaluation entrypoint
|- run_two_stage.py                # lightweight wrapper for quick launches
|- configs/
|  |- default_config.py            # default configuration
|  `- experiments/                 # curated public presets
|- data_prep/                      # raw data discovery and subsequence generation
|- preprocessing/                  # preprocessing and feature selection
|- models/                         # sequence encoders and survival backbones
|- evaluation/                     # metrics, evaluator, plotting, comparisons
`- utils/                          # training, calibration, plotting, misc helpers
```

## Data Layout

By default the code looks for data under `data/SWAN`.

Expected layout:

```text
data/SWAN/
|- partition1/
|  |- FL/*.csv
|  `- NF/*.csv
|- partition2/
|- partition3/
|- partition4/
`- partition5/
```

Compressed `partition*_instances.tar.gz` files are also supported and will be extracted automatically when needed.

## Setup

Recommended: create a fresh Python environment and install the core dependencies manually.

```bash
pip install torch numpy pandas scikit-learn lifelines PyYAML easydict scipy matplotlib seaborn joblib statsmodels
```

Optional packages may be needed for some extended utilities or bundled third-party baseline code.

Before running experiments, make sure `cfg.data.data_dir` points to your local SWAN dataset root. The default public value is now the relative path `data/SWAN`.

## Quick Start

Run a lightweight classification-only example:

```bash
python main.py --preset classification_only_example
```

Run a lightweight two-stage example:

```bash
python main.py --preset two_stage_example
```

Or use the convenience wrapper:

```bash
python run_two_stage.py --mode two_stage --output_dir results/my_run
```

All experiment outputs, checkpoints, and plots are written under `results/`.

## Notes for the Public Release

- No dataset is bundled in this repository.
- No author-local pretrained checkpoint paths are assumed anymore.
