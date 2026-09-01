import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X, dtype = float)
    X_c = X - np.mean(X, axis = 0)
    return (X_c.T @ X_c)/(X.shape[0]-1) 