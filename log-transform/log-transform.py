import math

def log_transform(values: list) -> list:
    """
    Returns the log1p-transformed values rounded to four decimals.
    """
    return [round(float(math.log1p(v)), 4) for v in values]