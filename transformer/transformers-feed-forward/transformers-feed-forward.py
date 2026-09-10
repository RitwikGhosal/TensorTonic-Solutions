import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    return  np.dot(relu(np.dot(x, W1) + b1), W2) + b2