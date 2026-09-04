import numpy as np

def streaming_minmax(D: int, batches: list, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with normalized_batches, min, and max.
    """
    r_min = np.full(D, np.inf, dtype = float)
    r_max = np.full(D, -np.inf, dtype = float)
    n_bs = []
    for b in batches:
        batch = np.asarray(b, dtype = float)
        r_min = np.minimum(r_min, np.min(batch, axis = 0))
        r_max = np.maximum(r_max, np.max(batch, axis = 0))
        scale = np.maximum(r_max - r_min, eps)
        n_bs.append((batch - r_min)/scale)
    return {"normalized_batches": n_bs, "min": r_min, "max": r_max}