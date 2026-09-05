import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.array(x, dtype = float)
    conv1 = np.array(conv1, dtype = float)
    fc = np.array(fc, dtype = float)
    x = np.maximum(0.0, x@conv1)
    blocks = [
        (np.array(W1_b1, dtype=float), np.array(W2_b1, dtype=float), None),
        (np.array(W1_b2, dtype=float), np.array(W2_b2, dtype=float), np.array(Ws_b2, dtype=float)),
    ]
    for W1, W2, Ws in blocks:
        identity = x.copy()
        if Ws is not None:
            identity = x @ Ws
        out = np.maximum(0, x @ W1)
        out = out @ W2
        x = np.maximum(0, out + identity)
    result = x @ fc
    return [[round(float(v), 4) for v in row] for row in result]