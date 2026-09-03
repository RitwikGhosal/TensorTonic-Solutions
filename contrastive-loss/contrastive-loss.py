import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)
    y = np.asarray(y, dtype = float)
    diff = a-b
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
    distances = np.linalg.norm(diff, axis = 1)
    pos_loss = y * distances**2
    neg_loss = (1.0 - y) * np.maximum(0.0, margin - distances) ** 2
    loss = pos_loss + neg_loss
    if reduction == "mean":
        return float(np.mean(loss))
    return float(np.sum(loss))
    
    