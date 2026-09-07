import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.asarray(x, dtype = float)
    q = np.asarray(q, dtype = float)
    s = np.sort(x)
    pos = q/100.0 * (x.size - 1)
    l = np.floor(pos).astype(int)
    u = np.ceil(pos).astype(int)
    w = pos - l
    return (1 - w) * s[l] + w * s[u]