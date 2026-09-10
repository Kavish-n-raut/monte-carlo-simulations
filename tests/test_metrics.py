import numpy as np

from data.processor import annualised_return, covariance_matrix, log_returns
from models.portfolio_sim import simulate_portfolio
from risk.metrics import (
    compare_var_methods,
    expected_shortfall,
    maximum_drawdown,
    probability_of_loss,
    var_historical,
    var_monte_carlo,
    var_parametric,
)

INVESTMENT = 100_000.0
N_SIMS = 4_000
HORIZON = 1.0


def _sim(prices, weights):
    return simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=42)


def test_all_var_methods_return_non_negative_losses(prices, weights):
    lr = log_returns(prices)
    cov = covariance_matrix(lr)
    mu_port = float(weights @ annualised_return(lr).to_numpy())
    sigma_port = float(np.sqrt(weights @ cov.to_numpy() @ weights))
    pnl = _sim(prices, weights).pnl

    assert var_monte_carlo(pnl, 0.95) >= 0
    assert var_historical(lr, weights, INVESTMENT, 0.95, 252) >= 0
    assert var_parametric(mu_port, sigma_port, INVESTMENT, 0.95, HORIZON) >= 0


def test_var_increases_with_confidence(prices, weights):
    lr = log_returns(prices)
    cov = covariance_matrix(lr)
    mu_port = float(weights @ annualised_return(lr).to_numpy())
    sigma_port = float(np.sqrt(weights @ cov.to_numpy() @ weights))
    comp = compare_var_methods(_sim(prices, weights), prices, weights, mu_port, sigma_port, INVESTMENT, HORIZON)

    assert comp["confidence"] == [0.95, 0.99]
    for method in ("monte_carlo", "historical", "parametric"):
        assert comp[method][1] >= comp[method][0], method


def test_expected_shortfall_is_at_least_monte_carlo_var(prices, weights):
    pnl = _sim(prices, weights).pnl
    assert expected_shortfall(pnl, 0.95) >= var_monte_carlo(pnl, 0.95)


def test_historical_var_uses_weights(prices):
    lr = log_returns(prices)
    concentrated = var_historical(lr, np.array([1.0, 0.0, 0.0]), INVESTMENT, 0.95, 252)
    diversified = var_historical(lr, np.array([1 / 3, 1 / 3, 1 / 3]), INVESTMENT, 0.95, 252)
    # a single-asset book and an equal-weight book should not give the same VaR
    assert not np.isclose(concentrated, diversified)


def test_drawdown_and_probability_of_loss_are_bounded(prices, weights):
    sim = _sim(prices, weights)
    max_dd = maximum_drawdown(sim.portfolio_paths)
    p_loss = probability_of_loss(sim.final_values, INVESTMENT)
    assert 0.0 <= max_dd <= 1.0
    assert 0.0 <= p_loss <= 1.0
