import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    p = np.asarray(p, dtype = float)
    y = np.asarray(y, dtype = float)
    s = 2.0 * np.sum(p * y)
    out = (s + eps) / (np.sum(p) + np.sum(y) + eps)
    return float(1 - out)