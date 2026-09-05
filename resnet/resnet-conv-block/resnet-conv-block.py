import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.array(x, dtype = float)
    W1 = np.array(W1, dtype = float)
    W2 = np.array(W2, dtype = float)
    Ws = np.array(Ws, dtype = float)
    x_ = x @ Ws
    out = relu(x @ W1)
    out = out @ W2
    result = relu(out + x_)
    return [[round(float(value), 4) for value in row] for row in result]

