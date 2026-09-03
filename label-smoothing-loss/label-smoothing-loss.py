import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    K = len(predictions)
    loss = 0.0
    for i in range(K):
        q = (1.0 - epsilon + epsilon/K) if i == target else (epsilon/K)
        loss -= q * math.log(predictions[i])
    return loss