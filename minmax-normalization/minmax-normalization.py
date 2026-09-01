import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype = float)
    minimum = np.min(X, axis = axis, keepdims = True)
    maximum = np.max(X, axis = axis, keepdims = True)
    num = X - minimum
    denom = maximum - minimum
    safe_denom = np.where(denom > eps, denom, 1.0)
    return (num/safe_denom)