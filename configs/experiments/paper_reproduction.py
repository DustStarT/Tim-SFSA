"""Paper-oriented preset with a clean public-facing name."""

from configs.default_config import get_config as get_default_config


def get_config(base_cfg=None):
    cfg = get_default_config() if base_cfg is None else base_cfg

    cfg.run_mode = "two_stage"
    cfg.model.name = "DeepSurv"
    cfg.model.use_lstm = True
    cfg.model.encoder.type = "lstm"

    cfg.model.two_stage.enabled = True
    cfg.model.two_stage.classification_stage.enabled = True
    cfg.model.two_stage.survival_stage.enabled = True
    cfg.model.two_stage.survival_stage.pretrained_lstm_path = ""
    cfg.model.checkpoint_path = ""

    cfg.results_dir = "results/paper_reproduction"
    return cfg


if __name__ == "__main__":
    from main import run_two_stage_experiment

    run_two_stage_experiment(get_config())
