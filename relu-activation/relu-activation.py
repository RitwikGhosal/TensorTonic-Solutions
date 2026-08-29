import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype = float)
    return np.asarray(np.maximum(0.0, x)) 