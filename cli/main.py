import argparse
import numpy as np
from data.fetcher import fetch_prices
from data.processor import log_returns, annualised_return, covariance_matrix
from models.portfolio_sim import simulate_portfolio
from risk.metrics import compare_var_methods

def run_pipeline():
    parser = argparse.ArgumentParser(description="Monte Carlo Financial Risk Analytics")
    parser.add_argument('--tickers', nargs='+', required=True, help="List of ticker symbols")
    parser.add_argument('--investment', type=float, default=100000.0)
    parser.add_argument('--horizon', type=float, default=1.0)
    parser.add_argument('--sims', type=int, default=10000)
    parser.add_argument('--start', type=str, default='2021-01-01')
    parser.add_argument('--end', type=str, default='2026-01-01')
    args = parser.parse_args()

    weights = np.array([1/len(args.tickers)] * len(args.tickers))
    
    print(f"Fetching data for {args.tickers}...")
    prices = fetch_prices(args.tickers, args.start, args.end)
    
    print(f"Running {args.sims} Monte Carlo simulations...")
    sim_result = simulate_portfolio(prices, weights, args.investment, args.horizon, args.sims)
    
    returns = log_returns(prices)
    mu_port = np.dot(weights, annualised_return(returns))
    cov = covariance_matrix(returns)
    sigma_port = np.sqrt(weights.T @ cov @ weights)
    
    var_comparison = compare_var_methods(
        sim_result, prices, weights, mu_port, sigma_port, args.investment, args.horizon
    )
    print(f"\n--- VaR Comparison (loss over {args.horizon:g}y, USD) ---")
    print(f"{'Confidence':>10} | {'Monte Carlo':>14} | {'Historical':>14} | {'Parametric':>14}")
    for i, conf in enumerate(var_comparison['confidence']):
        print(
            f"{conf:>10.0%} | "
            f"{var_comparison['monte_carlo'][i]:>14,.2f} | "
            f"{var_comparison['historical'][i]:>14,.2f} | "
            f"{var_comparison['parametric'][i]:>14,.2f}"
        )
    print("-" * 62 + "\nIn-memory execution complete.")

if __name__ == "__main__":
    run_pipeline()