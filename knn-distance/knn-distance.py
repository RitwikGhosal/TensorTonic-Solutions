import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train, dtype = float)
    X_test = np.asarray(X_test, dtype = float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    dist = np.sqrt(np.sum((X_test[:, None,:] - X_train[None,:, :])**2, axis = 2))
    count = min(k, X_train.shape[0])
    nbh = np.argsort(dist, axis = 1, kind = "stable")[:, :count]
    if count < k:
        padding = np.full((X_test.shape[0], k - count), -1, dtype=int)
        nbh = np.concatenate((nbh, padding), axis=1)
    return nbh.astype(int)
    