import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    return np.sqrt(np.sum(np.square(np.diff((x, y), axis = 0))))