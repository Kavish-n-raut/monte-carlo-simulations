"""Shared fixtures.

Tests use deterministic synthetic price data so they run offline and fast -
no yfinance calls.
"""

import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAA", "BBB", "CCC"]


def _synthetic_prices(n_days: int = 504, seed: int = 12345) -> pd.DataFrame:
    """Correlated equity-like price history from a GBM with fixed parameters."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, 0.40, 0.30], [0.40, 1.0, 0.35], [0.30, 0.35, 1.0]])
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n_days, len(TICKERS))) @ chol.T
    daily_mu = np.array([0.08, 0.10, 0.06]) / 252
    daily_sigma = np.array([0.20, 0.28, 0.18]) / np.sqrt(252)
    log_ret = daily_mu - 0.5 * daily_sigma**2 + daily_sigma * shocks
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    index = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=TICKERS)


@pytest.fixture
def prices() -> pd.DataFrame:
    return _synthetic_prices()


@pytest.fixture
def weights() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2])
