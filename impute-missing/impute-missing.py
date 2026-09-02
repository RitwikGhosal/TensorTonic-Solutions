import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        missing = np.isnan(X)
        observed = X[~missing]

        if observed.size == 0:
            X[missing] = 0.0
        else:
            if strategy == "mean":
                value = np.mean(observed)
            elif strategy == "median":
                value = np.median(observed)

            X[missing] = value

        return X

    for i in range(X.shape[1]):
        column = X[:, i]
        missing = np.isnan(column)
        observed = column[~missing]

        if observed.size == 0:
            X[missing, i] = 0.0
        else:
            if strategy == "mean":
                value = np.mean(observed)
            elif strategy == "median":
                value = np.median(observed)

            X[missing, i] = value

    return X
            
            
        