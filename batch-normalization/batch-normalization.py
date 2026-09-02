import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype = float)
    beta = np.asarray(beta, dtype = float)
    if x.ndim == 2:
        axes = (0,)
        parameter_shape = (1, -1)
    else:
        axes = (0, 2, 3)
        parameter_shape = (1, -1, 1, 1)
    mean = np.mean(x, axis = axes, keepdims = True)
    variance = np.var(x, axis = axes, keepdims = True)
    normalized = (x - mean) / np.sqrt(variance + eps)
    return normalized * gamma.reshape(parameter_shape) + beta.reshape(parameter_shape)