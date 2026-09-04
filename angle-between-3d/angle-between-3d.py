import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """
    first = np.asarray(v, dtype=float)
    second = np.asarray(w, dtype=float)
    first_norm = np.sqrt(np.sum(first ** 2))
    second_norm = np.sqrt(np.sum(second ** 2))
    if first_norm == 0 or second_norm == 0:
        return float(np.nan)
    cosine = np.dot(first, second) / (first_norm * second_norm)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))