import numpy as np
import torch

from revision.survival import (
    efron_baseline_cumulative_hazard, efron_cox_loss, predict_survival,
)
from revision.training import horizon_targets
from revision.training import _exact_chunked_cox_step


def test_efron_loss_matches_direct_formula_with_ties():
    eta = torch.tensor([0.2, -0.1, 0.5, 0.0], dtype=torch.float64)
    duration = torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    event = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    risk_all = torch.exp(eta).sum()
    tied = torch.exp(eta[:2]).sum()
    ll_time_1 = eta[:2].sum() - torch.log(risk_all) - torch.log(risk_all - 0.5 * tied)
    risk_time_2 = torch.exp(eta[2:]).sum()
    ll_time_2 = eta[2] - torch.log(risk_time_2)
    expected = -(ll_time_1 + ll_time_2) / 3.0
    assert torch.allclose(efron_cox_loss(eta, duration, event), expected, atol=1e-10)


def test_baseline_and_predicted_survival_are_monotone():
    eta = np.array([0.2, -0.1, 0.5, 0.0])
    duration = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 1, 0])
    times, hazard = efron_baseline_cumulative_hazard(eta, duration, event)
    survival = predict_survival(eta, times, hazard, np.array([0.5, 1.0, 2.0, 3.0]))
    assert np.all(np.diff(hazard) >= 0)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    assert np.all((survival >= 0) & (survival <= 1))


def test_censor_before_horizon_is_unknown_not_negative():
    target, known = horizon_targets(
        np.array([5.0, 5.0, 20.0]), np.array([1, 0, 0]), [12.0]
    )
    assert target[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert known[:, 0].tolist() == [1.0, 0.0, 1.0]


def test_two_pass_chunked_gradient_matches_full_risk_set():
    x = np.array([[[0.2]], [[-0.1]], [[0.5]], [[0.0]]], dtype=np.float32)
    duration = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 1, 0])
    direct = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1, bias=False))
    chunked = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1, bias=False))
    chunked.load_state_dict(direct.state_dict())
    direct_loss = efron_cox_loss(
        direct(torch.tensor(x)).reshape(-1), torch.tensor(duration), torch.tensor(event)
    )
    direct_loss.backward()
    expected = direct[1].weight.grad.clone()
    optimizer = torch.optim.SGD(chunked.parameters(), lr=0.0)
    _exact_chunked_cox_step(
        chunked, optimizer, x, duration, event, torch.device("cpu"),
        batch_size=2, gradient_clip=100.0,
    )
    assert torch.allclose(chunked[1].weight.grad, expected, atol=1e-6)
