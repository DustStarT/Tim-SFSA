"""Decision-complete configuration for the reviewer revision experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FEATURES_24 = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45",
    "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY",
    "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX", "R_VALUE",
]

LEGACY_DROPPED_FEATURES = [
    "TOTBSQ", "USFLUX", "TOTUSJZ", "TOTPOT", "MEANGBZ", "MEANSHR",
]
LEGACY_18_FEATURES = [name for name in FEATURES_24 if name not in LEGACY_DROPPED_FEATURES]

SIGNED_LOG_FEATURES = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "TOTFY", "TOTFX", "MEANPOT", "R_VALUE",
]
HISTORY_FEATURES = ["B_COUNT_24H", "C_COUNT_24H"]
HORIZONS_HOURS = [12.0, 24.0, 48.0, 72.0, 144.0]
SEEDS = [2345, 2346, 2347]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str = "cox_neural"
    use_lstm: bool = True
    use_connector: bool = True
    use_pretraining: bool = True
    balanced_event_ratio: float | None = None
    feature_set: str = "all24"
    include_bc_history: bool = False
    linear_head: bool = False


OFFICIAL_EXPERIMENTS = [
    ExperimentSpec("tim_sfsa"),
    ExperimentSpec("logistic", family="logistic", use_lstm=False, use_connector=False, use_pretraining=False),
    ExperimentSpec("svm", family="svm", use_lstm=False, use_connector=False, use_pretraining=False),
    ExperimentSpec("lstm_classifier", family="lstm_classifier"),
    ExperimentSpec("linear_cox", family="linear_cox", use_lstm=False, use_connector=False, use_pretraining=False, linear_head=True),
    ExperimentSpec("lstm_linear_cox", linear_head=True),
    ExperimentSpec("no_lstm", use_lstm=False, use_pretraining=False),
    ExperimentSpec("no_pretraining", use_pretraining=False),
    ExperimentSpec("no_connector", use_connector=False),
    ExperimentSpec("balanced22", balanced_event_ratio=0.22),
    ExperimentSpec("legacy18", feature_set="legacy18"),
    ExperimentSpec("validation_pruned", feature_set="validation_pruned"),
    ExperimentSpec("bc_history", include_bc_history=True),
]

CHRONO_EXPERIMENT_NAMES = {
    "tim_sfsa", "logistic", "svm", "lstm_classifier", "linear_cox"
}


@dataclass
class PipelineConfig:
    data_root: str
    run_dir: str
    splits: str = "all"
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))
    horizons_hours: list[float] = field(default_factory=lambda: list(HORIZONS_HOURS))
    history_hours: float = 4.0
    cadence_minutes: float = 12.0
    history_steps: int = 20
    max_gap_minutes: float = 24.0
    flare_history_hours: float = 24.0
    depth_candidates: list[int] = field(default_factory=lambda: [1, 2, 3, 7])
    hidden_size: int = 256
    connector_size: int = 512
    dropout: float = 0.3
    connector_dropout: float = 0.4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 150
    pretrain_epochs: int = 100
    patience: int = 25
    gradient_clip: float = 5.0
    inference_batch_size: int = 512
    bootstrap_replicates: int = 1000
    permutation_repeats: int = 20
    correlation_threshold: float = 0.90
    num_workers: int = 0
    device: str = "auto"
    deterministic: bool = True
    quick: bool = False

    def normalise(self) -> "PipelineConfig":
        self.data_root = str(Path(self.data_root).expanduser().resolve())
        self.run_dir = str(Path(self.run_dir).expanduser().resolve())
        if self.splits not in {"all", "official", "chronological"}:
            raise ValueError("splits must be one of: all, official, chronological")
        if self.quick:
            self.seeds = self.seeds[:1]
            self.depth_candidates = [1]
            self.max_epochs = min(self.max_epochs, 2)
            self.pretrain_epochs = min(self.pretrain_epochs, 2)
            self.patience = min(self.patience, 2)
            self.bootstrap_replicates = min(self.bootstrap_replicates, 20)
            self.permutation_repeats = min(self.permutation_repeats, 2)
            self.hidden_size = min(self.hidden_size, 16)
            self.connector_size = min(self.connector_size, 32)
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
