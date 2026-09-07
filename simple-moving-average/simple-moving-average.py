import numpy as np
def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    result = []
    for i in range(len(values) - window_size + 1):
        result.append(np.mean(values[i:i+window_size]))
    return result