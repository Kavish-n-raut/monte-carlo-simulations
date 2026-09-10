import numpy as np

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    Z = np.random.standard_normal((n_sims, n_steps))
    
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t-1]
        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)
        
    return paths