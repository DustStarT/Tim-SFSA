import numpy as np

from revision.baselines import PenalizedLinearCox


def test_penalized_linear_cox_converges_and_orders_simple_risk():
    origin = np.asarray([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    x = np.repeat(origin[:, None, None], 20, axis=1)
    duration = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.asarray([1, 1, 1, 1, 0, 0])

    model = PenalizedLinearCox(l2_penalty=1e-3).fit(x, duration, event)
    risk = model.predict_log_risk(x)

    assert model.fit_result_["converged"]
    assert model.fit_result_["gradient_max_abs"] < 1e-5
    assert np.all(np.diff(risk) < 0.0)


def test_penalized_linear_cox_gradient_matches_finite_difference_with_ties():
    model = PenalizedLinearCox(l2_penalty=0.02)
    x = np.asarray([
        [0.2, -0.4], [0.7, 0.1], [-0.3, 0.8], [0.1, 0.2], [-0.6, -0.2],
    ])
    duration = np.asarray([1.0, 1.0, 2.0, 3.0, 4.0])
    event = np.asarray([1, 1, 1, 0, 0], dtype=bool)
    beta = np.asarray([0.15, -0.25])
    loss, gradient = model._objective(beta, x, duration, event)
    epsilon = 1e-6
    numerical = np.empty_like(beta)
    for index in range(len(beta)):
        step = np.zeros_like(beta)
        step[index] = epsilon
        plus = model._objective(beta + step, x, duration, event)[0]
        minus = model._objective(beta - step, x, duration, event)[0]
        numerical[index] = (plus - minus) / (2.0 * epsilon)

    assert np.isfinite(loss)
    assert np.allclose(gradient, numerical, atol=1e-6)
