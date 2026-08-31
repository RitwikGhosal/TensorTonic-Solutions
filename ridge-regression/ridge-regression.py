import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    return w.tolist()