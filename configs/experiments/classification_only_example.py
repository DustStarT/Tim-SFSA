"""Minimal classification-only preset for public release demos."""

from configs.default_config import get_config as get_default_config


def get_config(base_cfg=None):
    cfg = get_default_config() if base_cfg is None else base_cfg

    cfg.run_mode = "classification_only"
    cfg.model.two_stage.enabled = True
    cfg.model.two_stage.classification_stage.enabled = True
    cfg.model.two_stage.survival_stage.enabled = False

    cfg.model.two_stage.classification_stage.num_epochs = 100
    cfg.model.two_stage.classification_stage.learning_rate = 1e-3
    cfg.model.two_stage.classification_stage.batch_size = 64
    cfg.model.two_stage.classification_stage.early_stopping_patience = 20
    cfg.model.two_stage.classification_stage.save_best_model = True

    cfg.model.lstm.hidden_size = 64
    cfg.model.lstm.num_lstm_layers = 2
    cfg.model.lstm.dropout_rate = 0.3
    cfg.model.lstm.bidirectional = True
    cfg.model.lstm.use_attention = True

    cfg.results_dir = "results/classification_only_example"
    return cfg


if __name__ == "__main__":
    from main import run_classification_experiment

    run_classification_experiment(get_config())
