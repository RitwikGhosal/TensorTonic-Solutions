import numpy as np

def bn(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis = 0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def relu(x):
    return np.maximum(0.0, x)

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.array(x, dtype = float)
    W1 = np.array(W1, dtype = float)
    W2 = np.array(W2, dtype = float)
    gamma1 = np.array(gamma1, dtype = float)
    beta1 = np.array(beta1, dtype = float)
    gamma2 = np.array(gamma2, dtype = float)
    beta2 = np.array(beta2, dtype = float)
    x_ = x.copy()
    if mode == "post":
        out = x @ W1
        out = bn(out, gamma1, beta1)
        out = relu(out)
        out = out @ W2
        out = bn(out, gamma2, beta2)
        out = out + x_
        out = relu(out)
        return {'output': [[round(float(v), 4) for v in row] for row in out], 'mode': 'post'}
    else:
        out = bn(x, gamma1, beta1)
        out = relu(out)
        out = out @ W1
        out = bn(out, gamma2, beta2)
        out = relu(out)
        out = out @ W2
        out = out + x_
        return {"output": [[round(float(v), 4) for v in row] for row in out], "mode": "pre"}
        
        
    