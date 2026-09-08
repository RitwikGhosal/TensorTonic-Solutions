import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    k = np.asarray(k, dtype=int)
    pmf = (1 - p)**(k-1) * p
    return {"pmf": pmf, "mean": float(1.0/p)}