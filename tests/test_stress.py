import numpy as np

from models.portfolio_sim import simulate_portfolio
from risk.stress import run_stress_tests

INVESTMENT = 100_000.0
N_SIMS = 1_500
HORIZON = 1.0


def test_run_stress_tests_returns_all_default_scenarios(prices, weights):
    results = run_stress_tests(prices, weights, INVESTMENT, HORIZON, N_SIMS)
    assert set(results) == {"crash", "vol_shock", "rate_shock", "correlation_shock", "recession"}

    n_assets = prices.shape[1]
    for name, sim in results.items():
        assert sim.portfolio_paths.shape[0] == N_SIMS, name
        assert sim.asset_paths.shape[0] == n_assets, name
        assert (sim.asset_paths > 0).all(), name


def test_crash_scenario_is_worse_than_the_base_case(prices, weights):
    base = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=99)
    crash = run_stress_tests(prices, weights, INVESTMENT, HORIZON, N_SIMS)["crash"]
    assert crash.final_values.mean() < base.final_values.mean()
