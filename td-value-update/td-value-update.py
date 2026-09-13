import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as V.
    """
    V = np.asarray(V, dtype=float).copy()
    delta = r + gamma *V[s_next]
    V[s] += alpha * (delta - V[s])
    return V