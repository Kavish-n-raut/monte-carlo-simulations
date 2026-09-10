import numpy as np
import pandas as pd
from dataclasses import dataclass
from data.processor import log_returns, annualised_return, annualised_volatility, covariance_matrix, cholesky_decomposition

@dataclass
class SimulationResult:
    portfolio_paths: np.ndarray
    final_values: np.ndarray
    pnl: np.ndarray
    asset_paths: np.ndarray

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    investment: float,
    T: float,
    n_sims: int,
    seed: int = None
) -> SimulationResult:
    if seed is not None:
        np.random.seed(seed)
        
    returns = log_returns(prices)
    mu = annualised_return(returns).values
    sigma = annualised_volatility(returns).values
    cov_mat = covariance_matrix(returns)
    L = cholesky_decomposition(cov_mat)
    
    n_assets = len(weights)
    n_steps = int(252 * T)
    dt = T / n_steps
    
    Z = np.random.standard_normal((n_assets, n_steps * n_sims))
    correlated_Z = L @ Z
    correlated_Z = correlated_Z.reshape((n_assets, n_sims, n_steps))
    
    asset_paths = np.zeros((n_assets, n_sims, n_steps + 1))
    initial_prices = prices.iloc[-1].values
    
    for i in range(n_assets):
        asset_paths[i, :, 0] = initial_prices[i]
        for t in range(1, n_steps + 1):
            drift = (mu[i] - 0.5 * sigma[i] ** 2) * dt
            diffusion = sigma[i] * np.sqrt(dt) * correlated_Z[i, :, t-1]
            asset_paths[i, :, t] = asset_paths[i, :, t-1] * np.exp(drift + diffusion)
            
    portfolio_paths = np.zeros((n_sims, n_steps + 1))
    for i in range(n_assets):
        units = (weights[i] * investment) / initial_prices[i]
        portfolio_paths += units * asset_paths[i]
        
    final_values = portfolio_paths[:, -1]
    pnl = final_values - investment
    
    return SimulationResult(portfolio_paths, final_values, pnl, asset_paths)