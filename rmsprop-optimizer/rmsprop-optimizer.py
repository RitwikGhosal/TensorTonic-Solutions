import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w = np.array(w, dtype = float)
    g = np.array(g, dtype = float)
    s = np.array(s, dtype = float)
    s_t = beta * s + (1 - beta) * g**2
    w_t = w - (lr / (s_t + eps)**0.5) * g
    return np.round(w_t, 6).tolist(), np.round(s_t, 6).tolist()