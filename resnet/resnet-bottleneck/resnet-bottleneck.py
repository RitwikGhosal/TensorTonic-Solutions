import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x, dtype = float)
    W1 = np.array(W1, dtype = float)
    W2 = np.array(W2, dtype = float)
    W3 = np.array(W3, dtype = float)
    if Ws is not None:
        Ws = np.array(Ws, dtype = float)
        x_ = x @ Ws
    else:
        x_ = x.copy()
    out = relu(x @ W1)
    out = relu(out @ W2)
    out = out @ W3
    result = relu(out + x_)
    return [[round(float(v),4) for v in row] for row in result]
        