import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(len(X))
    np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start: start+batch_size]
        if drop_last and batch_indices.size < batch_size:
            break
        yield X[batch_indices], y[batch_indices]