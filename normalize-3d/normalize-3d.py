import numpy as np

def normalize_3d(v: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as v.
    """
    v = np.asarray(v, dtype = float)
    norms = np.sqrt(np.sum(v ** 2, axis = -1, keepdims = True))
    return np.divide(v, norms, out = np.zeros_like(v), where=norms!=0)