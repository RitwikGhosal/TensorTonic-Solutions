import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype = float)
    q_safe = np.clip(q[p>0], eps, None)
    return float(np.sum(p[p>0] * np.log(p[p>0] / q_safe)))