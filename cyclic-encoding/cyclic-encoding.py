import math

def cyclic_encoding(values: list, period: float) -> list:
    """
    Returns the sine and cosine encoding of every cyclic value.
    """
    result = []
    for v in values:
        ang = 2 * math.pi * v /period
        result.append([math.sin(ang), math.cos(ang)])
    return result