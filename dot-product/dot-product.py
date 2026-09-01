import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    return float(np.dot(x, y))