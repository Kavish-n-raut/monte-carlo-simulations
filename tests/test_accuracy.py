"""Numerical-accuracy regression tests.

These pin the simulation to closed-form Geometric Brownian Motion / lognormal
theory. They guard against the double-counted-volatility class of bug (using the
covariance Cholesky together with a separate sigma scaling), which silently
understates all risk numbers while leaving structural invariants intact.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from data.processor import annualised_return, annualised_volatility, log_returns
from models.portfolio_sim import simulate_portfolio
from risk.metrics import expected_shortfall, var_monte_carlo

_PHI = NormalDist()


def _single_asset_prices(ann_drift=0.10, ann_vol=0.25, n_days=1000, seed=1, s0=100.0):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    lr = (ann_drift - 0.5 * ann_vol**2) * dt + ann_vol * math.sqrt(dt) * rng.standard_normal(n_days)
    return pd.DataFrame(s0 * np.exp(np.cumsum(lr)), columns=["AAA"],
                        index=pd.bdate_range("2020-01-01", periods=n_days))


def test_terminal_mean_and_std_match_lognormal():
    prices = _single_asset_prices(seed=1)
    lr = log_returns(prices)
    m = float(annualised_return(lr).iloc[0])
    s = float(annualised_volatility(lr).iloc[0])
    I, T, N = 100_000.0, 1.0, 100_000

    fv = simulate_portfolio(prices, np.array([1.0]), I, T, N, seed=7).final_values

    exp_mean = I * math.exp(m * T)
    exp_std = I * math.exp(m * T) * math.sqrt(math.exp(s * s * T) - 1)
    # If volatility were double-counted the std would be ~4x too small; 10% tol
    # passes the correct engine comfortably and fails the buggy one by a mile.
    assert abs(fv.mean() - exp_mean) / exp_mean < 0.02
    assert abs(fv.std() - exp_std) / exp_std < 0.10


def test_monte_carlo_var_matches_closed_form():
    prices = _single_asset_prices(seed=2)
    lr = log_returns(prices)
    m = float(annualised_return(lr).iloc[0])
    s = float(annualised_volatility(lr).iloc[0])
    I, T, N = 100_000.0, 1.0, 150_000
    pnl = simulate_portfolio(prices, np.array([1.0]), I, T, N, seed=9).pnl

    for c in (0.95, 0.99):
        z = _PHI.inv_cdf(1 - c)
        var_theory = I * (1 - math.exp((m - 0.5 * s * s) * T + s * math.sqrt(T) * z))
        var_sim = var_monte_carlo(pnl, c)
        assert abs(var_sim - var_theory) / var_theory < 0.05, c


def test_expected_shortfall_matches_closed_form():
    prices = _single_asset_prices(seed=3)
    lr = log_returns(prices)
    m = float(annualised_return(lr).iloc[0])
    s = float(annualised_volatility(lr).iloc[0])
    I, T, N = 100_000.0, 1.0, 150_000
    pnl = simulate_portfolio(prices, np.array([1.0]), I, T, N, seed=4).pnl

    for c in (0.95, 0.99):
        z = _PHI.inv_cdf(1 - c)
        es_theory = I * (1 - math.exp(m * T) * _PHI.cdf(z - s * math.sqrt(T)) / (1 - c))
        es_sim = expected_shortfall(pnl, c)
        assert abs(es_sim - es_theory) / es_theory < 0.06, c


def test_expected_value_independent_of_volatility():
    """GBM martingale property: E[S_T] does not depend on volatility."""
    lr = log_returns(_single_asset_prices(ann_vol=0.20, seed=5))
    m = float(annualised_return(lr).iloc[0])
    I, T, N = 100_000.0, 1.0, 120_000

    prices_lo = _single_asset_prices(ann_vol=0.15, seed=5)
    prices_hi = _single_asset_prices(ann_vol=0.45, seed=5)
    # same drift, different vol -> same expected terminal value
    lo = simulate_portfolio(prices_lo, np.array([1.0]), I, T, N, seed=6).final_values.mean()
    hi = simulate_portfolio(prices_hi, np.array([1.0]), I, T, N, seed=6).final_values.mean()
    m_lo = float(annualised_return(log_returns(prices_lo)).iloc[0])
    m_hi = float(annualised_return(log_returns(prices_hi)).iloc[0])
    assert abs(lo - I * math.exp(m_lo * T)) / I < 0.02
    assert abs(hi - I * math.exp(m_hi * T)) / I < 0.02
