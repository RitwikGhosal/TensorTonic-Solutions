import math
import numpy as np

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    n_x1 = np.linalg.norm(x1)
    n_x2 = np.linalg.norm(x2)
    c_x1x2 = np.dot(x1, x2) / (n_x1 * n_x2)
    if label == 1:
        return 1.0 - c_x1x2
    else:
        return max(0.0, c_x1x2 - margin)