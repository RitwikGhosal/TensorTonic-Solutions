import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g = np.asarray(g, dtype = float)
    return np.where(np.linalg.norm(g) > max_norm, g * (max_norm / np.linalg.norm(g)), g)