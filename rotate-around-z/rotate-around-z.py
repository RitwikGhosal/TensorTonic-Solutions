import numpy as np

def rotate_around_z(points: list, theta: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as points.
    """
    values = np.asarray(points, dtype=float)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    x = values[..., 0]
    y = values[..., 1]
    z = values[..., 2]
    return np.stack((x * cosine - y * sine, x * sine + y * cosine, z), axis=-1)