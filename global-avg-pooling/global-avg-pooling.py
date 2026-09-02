import numpy as np

def global_avg_pool(x: list) -> np.ndarray:
    """
    Returns a spatially averaged NumPy array with shape (C,) or (N, C).
    """
    x = np.asarray(x, dtype = float)
    return np.mean(x, axis = (-2, -1))