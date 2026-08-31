import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype = float)
    n, d = X.shape
    mean = X.mean(axis = 0)
    X_c = X - mean
    cov = (X_c.T @ X_c) / (n - 1)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    W = evecs[:, :k]
    for j in range(k):
        max_idx = np.argmax(np.abs(W[:, j]))
        if W[max_idx, j] < 0:
            W[:, j] *= -1
    X_proj = X_c @ W
    return X_proj.tolist()
    