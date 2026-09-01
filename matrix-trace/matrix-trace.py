import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    A = np.asarray(A, dtype = float)
    out = 0.0
    for i in range(A.shape[0]):
        out += A[i, i]
    return out