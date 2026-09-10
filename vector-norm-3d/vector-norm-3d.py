import numpy as np

def vector_norm_3d(v: list) -> float | np.ndarray:
    """
    Returns a float or a NumPy array.
    """
    values = np.asarray(v, dtype=float)
    norms = np.sqrt(np.sum(values ** 2, axis=-1))
    return float(norms) if norms.ndim == 0 else norms