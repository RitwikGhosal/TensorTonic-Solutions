import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    anchor = np.atleast_2d(np.asarray(anchor, dtype = float))
    positive = np.atleast_2d(np.asarray(positive, dtype = float))
    negative = np.atleast_2d(np.asarray(negative, dtype = float))

    positive_distance = np.sum((anchor - positive)**2, axis = 1)
    negative_distance = np.sum((anchor - negative)**2, axis = 1)
    losses = np.maximum(0.0, positive_distance - negative_distance + margin)
    return float(np.mean(losses))