"""Minimal two-stage preset for public release demos."""

from configs.default_config import get_config as get_default_config


def get_config(base_cfg=None):
    cfg = get_default_config() if base_cfg is None else base_cfg

    cfg.run_mode = "two_stage"
    cfg.model.two_stage.enabled = True
    cfg.model.two_stage.classification_stage.enabled = True
    cfg.model.two_stage.survival_stage.enabled = True
    cfg.model.two_stage.survival_stage.pretrained_lstm_path = ""

    cfg.model.two_stage.classification_stage.num_epochs = 50
    cfg.model.two_stage.classification_stage.learning_rate = 1e-3
    cfg.model.two_stage.classification_stage.batch_size = 64
    cfg.model.two_stage.classification_stage.early_stopping_patience = 15
    cfg.model.two_stage.classification_stage.save_best_model = True

    cfg.model.two_stage.survival_stage.load_pretrained_lstm = True
    cfg.model.two_stage.survival_stage.freeze_lstm = False
    cfg.model.two_stage.survival_stage.fine_tune_epochs = 30
    cfg.model.two_stage.survival_stage.fine_tune_lr = 5e-5

    cfg.model.name = "DeepSurv"
    cfg.model.use_lstm = True
    cfg.model.lstm.hidden_size = 64
    cfg.model.lstm.num_lstm_layers = 2
    cfg.model.lstm.dropout_rate = 0.3
    cfg.model.lstm.bidirectional = True
    cfg.model.lstm.use_attention = True

    cfg.training.num_epochs = 100
    cfg.training.learning_rate = 1e-4
    cfg.training.batch_size = 64
    cfg.training.early_stopping_patience = 20

    cfg.results_dir = "results/two_stage_example"
    return cfg


if __name__ == "__main__":
    from main import run_two_stage_experiment

    run_two_stage_experiment(get_config())
