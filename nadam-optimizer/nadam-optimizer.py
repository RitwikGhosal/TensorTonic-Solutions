import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * grad ** 2
    n_m = beta1 * new_m + (1.0 - beta1) * grad
    new_w = w - lr * n_m / (np.sqrt(new_v) + eps)
    return {"new_w": new_w, "new_m": new_m, "new_v": new_v}