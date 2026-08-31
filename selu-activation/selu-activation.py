import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    lam = 1.0507
    alpha = 1.6733
    x = np.asarray(x, dtype = float)
    return np.where(x > 0, lam * x, (lam * alpha * (np.exp(x) - 1)))