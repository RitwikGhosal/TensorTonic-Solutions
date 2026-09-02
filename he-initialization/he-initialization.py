import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    l = math.sqrt(6.0 / fan_in)
    return [[round(W[i][j] * 2 * l - l, 4) for j in range(len(W[0]))] for i in range(len(W))]