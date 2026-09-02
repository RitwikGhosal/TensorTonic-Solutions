import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    y_mean = np.mean(y_true)
    unexplained_variance = np.sum((y_true - y_pred)**2)
    total_variance = np.sum((y_true - y_mean)**2)
    if total_variance == 0:
        return 1.0 if unexplained_variance == 0 else 0.0
    return float(1 - (unexplained_variance/total_variance))