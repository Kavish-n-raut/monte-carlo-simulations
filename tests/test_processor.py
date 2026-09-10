import numpy as np

from data.processor import (
    annualised_return,
    annualised_volatility,
    cholesky_decomposition,
    correlation_matrix,
    covariance_matrix,
    log_returns,
)


def test_log_returns_drops_first_row_and_is_finite(prices):
    lr = log_returns(prices)
    assert lr.shape == (len(prices) - 1, prices.shape[1])
    assert np.isfinite(lr.to_numpy()).all()


def test_annualised_stats_have_expected_sign(prices):
    lr = log_returns(prices)
    assert (annualised_volatility(lr) > 0).all()
    # synthetic drifts are positive
    assert (annualised_return(lr) > 0).all()


def test_correlation_matrix_is_valid(prices):
    corr = correlation_matrix(log_returns(prices))
    values = corr.to_numpy()
    assert np.allclose(np.diag(values), 1.0)
    assert np.allclose(values, values.T)
    assert values.min() >= -1.0 and values.max() <= 1.0


def test_covariance_matrix_is_positive_definite(prices):
    cov = covariance_matrix(log_returns(prices))
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    chol = cholesky_decomposition(cov)  # raises LinAlgError if not PD
    assert chol.shape == (prices.shape[1], prices.shape[1])
