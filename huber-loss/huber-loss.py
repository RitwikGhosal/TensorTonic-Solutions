import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true, dtype = np.float64)
    y_pred = np.asarray(y_pred, dtype = np.float64)
    return float(np.mean(np.where(np.abs(y_true - y_pred) <= delta , 0.5*(y_true-y_pred)**2, delta*(np.abs(y_true-y_pred) - 0.5 * delta))))