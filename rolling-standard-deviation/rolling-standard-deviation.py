import math

def rolling_std(values: list, window_size: int) -> list:
    """
    Returns the population standard deviation of every complete window.
    """
    res = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i+window_size]
        mean = sum(window) / window_size
        var = sum((x - mean)**2 for x in window) / window_size
        res.append(math.sqrt(var))
    return res