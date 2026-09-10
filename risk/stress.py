import numpy as np
import pandas as pd
from models.portfolio_sim import SimulationResult
from data.processor import annualised_return, annualised_volatility, covariance_matrix, log_returns

def run_stress_tests(
    prices: pd.DataFrame,
    weights: np.ndarray,
    investment: float,
    T: float,
    n_sims: int,
    scenarios: list[str] = ['crash', 'vol_shock', 'rate_shock', 'correlation_shock', 'recession']
) -> dict[str, SimulationResult]:
    
    results = {}
    returns = log_returns(prices)
    base_mu = annualised_return(returns).values
    base_sigma = annualised_volatility(returns).values
    base_cov = covariance_matrix(returns)
    initial_prices = prices.iloc[-1].values
    
    def sim_custom(mu, sigma, L_matrix, time_horizon):
        n_assets = len(weights)
        n_steps = int(252 * time_horizon)
        dt = time_horizon / n_steps
        Z = np.random.standard_normal((n_assets, n_steps * n_sims))
        correlated_Z = L_matrix @ Z
        correlated_Z = correlated_Z.reshape((n_assets, n_sims, n_steps))
        
        asset_paths = np.zeros((n_assets, n_sims, n_steps + 1))
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
            
        final_vals = portfolio_paths[:, -1]
        return SimulationResult(portfolio_paths, final_vals, final_vals - investment, asset_paths)

    for scenario in scenarios:
        mu = base_mu.copy()
        sigma = base_sigma.copy()
        horizon = T
        L = np.linalg.cholesky(base_cov.values)
        
        if scenario == 'crash':
            mu = mu - 0.30
            sigma = sigma * 2.0
        elif scenario == 'vol_shock':
            sigma = sigma * 3.0
        elif scenario == 'rate_shock':
            mu = mu - 0.10 # Assuming 200bps default
        elif scenario == 'correlation_shock':
            stressed_corr = np.full((len(weights), len(weights)), 0.90)
            np.fill_diagonal(stressed_corr, 1.0)
            D = np.diag(sigma)
            stressed_cov = D @ stressed_corr @ D
            L = np.linalg.cholesky(stressed_cov)
        elif scenario == 'recession':
            mu = mu - 0.20
            sigma = sigma * 1.5
            horizon = T * 1.5
            
        results[scenario] = sim_custom(mu, sigma, L, horizon)
        
    return results