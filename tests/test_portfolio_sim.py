import numpy as np

from models.portfolio_sim import simulate_portfolio

INVESTMENT = 100_000.0
N_SIMS = 2_000
HORIZON = 1.0


def test_simulation_output_shapes(prices, weights):
    sim = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=7)
    n_assets = prices.shape[1]
    n_steps = int(252 * HORIZON)

    assert sim.asset_paths.shape == (n_assets, N_SIMS, n_steps + 1)
    assert sim.portfolio_paths.shape == (N_SIMS, n_steps + 1)
    assert sim.final_values.shape == (N_SIMS,)
    assert sim.pnl.shape == (N_SIMS,)


def test_asset_paths_stay_positive_and_pnl_is_consistent(prices, weights):
    sim = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=7)
    assert (sim.asset_paths > 0).all()
    assert np.allclose(sim.pnl, sim.final_values - INVESTMENT)
    # portfolio starts at the investment amount
    assert np.allclose(sim.portfolio_paths[:, 0], INVESTMENT)


def test_simulation_is_deterministic_with_a_fixed_seed(prices, weights):
    a = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=123)
    b = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=123)
    assert np.array_equal(a.portfolio_paths, b.portfolio_paths)


def test_different_seeds_give_different_paths(prices, weights):
    a = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=1)
    b = simulate_portfolio(prices, weights, INVESTMENT, HORIZON, N_SIMS, seed=2)
    assert not np.array_equal(a.portfolio_paths, b.portfolio_paths)
