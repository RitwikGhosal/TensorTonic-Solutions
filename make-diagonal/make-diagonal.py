import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, N).
    """
    out = np.zeros((len(v),len(v)))
    for i in range(len(v)):
        out[i, i] = v[i]
    return out