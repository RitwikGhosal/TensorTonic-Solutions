import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.asarray(x, dtype = float)
    alpha = (1.0 - ci) / 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size = (n_bootstrap, x.size))
    means = x[indices].mean(axis = 1)
    return {
        "bootstrap_mean" : float(means.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1-alpha)),
    }