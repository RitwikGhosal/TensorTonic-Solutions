import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.asarray(A, dtype = float)
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A)