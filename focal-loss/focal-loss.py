import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    pos = (1.0 - p)**gamma  * y * np.log(p)
    #neg = p ** gamma * (1.0 - y) * np.log1p(-p)
    neg = p ** gamma * (1.0 - y) * np.log(1-p)
    return float(np.mean(-(pos + neg)))