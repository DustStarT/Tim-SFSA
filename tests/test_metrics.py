import numpy as np

from revision.metrics import IPCW, ipcw_brier, weighted_classification_metrics


def test_ipcw_brier_reduces_to_binary_brier_without_censoring():
    train_duration = np.array([2.0, 8.0, 20.0, 25.0])
    train_event = np.ones(4, dtype=int)
    censor = IPCW(train_duration, train_event)
    duration = np.array([2.0, 20.0])
    event = np.ones(2, dtype=int)
    survival = np.array([0.2, 0.8])
    score = ipcw_brier(censor, duration, event, survival, 12.0)
    expected = ((0.0 - 0.2) ** 2 + (1.0 - 0.8) ** 2) / 2.0
    assert np.isclose(score, expected)


def test_weighted_operational_metrics_have_expected_perfect_values():
    result = weighted_classification_metrics(
        labels=np.array([1, 1, 0, 0]),
        probability=np.array([0.9, 0.8, 0.2, 0.1]),
        known=np.ones(4, dtype=bool), weights=np.ones(4), threshold=0.5,
    )
    assert result["tss"] == 1.0
    assert result["hss2"] == 1.0
    assert result["far"] == 0.0
    assert result["f1"] == 1.0
