import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x, dtype = float)
    if x.ndim == 1:
        numer = np.exp(x-max(x))
        denom = np.exp(x-max(x)).sum()
        return numer / denom
    numer = np.exp((x - np.max(x, axis=1, keepdims = True)))
    denom = np.exp((x - np.max(x, axis=1, keepdims = True))).sum(axis = 1, keepdims = True)
    return numer/denom