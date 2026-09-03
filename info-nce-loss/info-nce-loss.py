import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    Z1 = np.asarray(Z1, dtype = float)
    Z2 = np.asarray(Z2, dtype = float)
    logits = Z1 @ Z2.T / temperature
    shifted = logits - np.max(logits, axis = 1, keepdims = True)
    denom = np.log(np.sum(np.exp(shifted), axis = 1))
    loss = -np.diag(shifted) + denom
    return float(np.mean(loss))