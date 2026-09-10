import numpy as np
import pandas as pd
from scipy.stats import norm
from models.portfolio_sim import SimulationResult
from data.processor import annualised_return, annualised_volatility, log_returns

# Convention: every VaR / Expected Shortfall function below returns a POSITIVE
# loss magnitude (dollars). A larger number means a worse loss, and a higher
# confidence level never produces a smaller number than a lower one.


def var_monte_carlo(pnl: np.ndarray, confidence: float) -> float:
    """Monte Carlo VaR: positive loss magnitude at the given confidence.

    ``pnl`` is the simulated profit-and-loss distribution over the full horizon
    (losses are negative). The loss quantile is the lower-tail percentile.
    """
    loss_quantile = np.percentile(pnl, (1 - confidence) * 100)
    return float(max(-loss_quantile, 0.0))


def var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    investment: float,
    confidence: float,
    horizon_days: int,
) -> float:
    """Historical VaR using the actual portfolio weights.

    A one-day VaR is estimated from the empirical distribution of weighted
    portfolio returns and scaled to ``horizon_days`` with the square-root-of-time
    rule. Returns a positive loss magnitude.
    """
    weights = np.asarray(weights, dtype=float)
    if returns.shape[1] != weights.shape[0]:
        raise ValueError("weights length must match the number of return series")
    portfolio_returns = returns.to_numpy() @ weights
    daily_pnl = portfolio_returns * investment
    daily_loss_quantile = np.percentile(daily_pnl, (1 - confidence) * 100)
    horizon_var = -daily_loss_quantile * np.sqrt(max(horizon_days, 1))
    return float(max(horizon_var, 0.0))


def var_parametric(mu: float, sigma: float, investment: float, confidence: float, T: float) -> float:
    """Variance-covariance (parametric normal) VaR as a positive loss magnitude."""
    z = norm.ppf(confidence)  # positive quantile, e.g. 1.645 at 95%
    var = investment * (z * sigma * np.sqrt(T) - mu * T)
    return float(max(var, 0.0))


def expected_shortfall(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): mean loss beyond VaR, as a positive magnitude."""
    loss_quantile = np.percentile(pnl, (1 - confidence) * 100)
    tail = pnl[pnl <= loss_quantile]
    if tail.size == 0:
        return float(max(-loss_quantile, 0.0))
    return float(max(-tail.mean(), 0.0))


def maximum_drawdown(portfolio_paths: np.ndarray) -> float:
    peaks = np.maximum.accumulate(portfolio_paths, axis=1)
    drawdowns = (peaks - portfolio_paths) / peaks
    return np.max(drawdowns)


def probability_of_loss(final_values: np.ndarray, investment: float) -> float:
    return np.sum(final_values < investment) / len(final_values)


def compare_var_methods(
    sim_result: SimulationResult,
    prices: pd.DataFrame,
    weights: np.ndarray,
    mu_port: float,
    sigma_port: float,
    investment: float,
    T: float,
) -> dict:
    """Compare Monte Carlo, Historical, and Parametric VaR at 95% and 99%.

    All three columns are positive loss magnitudes on the same horizon ``T``
    (years), so they are directly comparable.
    """
    confidences = [0.95, 0.99]
    returns = log_returns(prices)
    horizon_days = max(int(round(252 * T)), 1)

    mc_vars = [var_monte_carlo(sim_result.pnl, c) for c in confidences]
    hist_vars = [var_historical(returns, weights, investment, c, horizon_days) for c in confidences]
    param_vars = [var_parametric(mu_port, sigma_port, investment, c, T) for c in confidences]

    return {
        'confidence': confidences,
        'monte_carlo': mc_vars,
        'historical': hist_vars,
        'parametric': param_vars,
    }
