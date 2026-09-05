import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    r = x.copy().astype(float)
    for grad in gradients_F:
        r = r + np.dot(r, grad)
    return r

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    r = x.copy().astype(float)
    for grad in gradients_F:
        r = np.dot(r, grad)
    return r
    