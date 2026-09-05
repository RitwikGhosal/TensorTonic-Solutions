import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x, dtype = float)
    W1 = np.array(W1, dtype = float)
    W2 = np.array(W2, dtype = float)
    x_ = x.copy()
    out = relu(x @ W1.T)
    out = out @ W2.T
    res = relu(out + x_)
    return [[round(float(value), 4) for value in row] for row in res]