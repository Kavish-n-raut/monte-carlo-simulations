import numpy as np
import pandas as pd

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def annualised_return(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.mean() * 252

def annualised_volatility(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.std() * np.sqrt(252)

def covariance_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.cov() * 252

def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.corr()

def cholesky_decomposition(cov_matrix: pd.DataFrame) -> np.ndarray:
    return np.linalg.cholesky(cov_matrix.values)