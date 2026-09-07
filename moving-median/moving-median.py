def moving_median(values, window_size):
    """
    Returns the median of every complete sliding window.
    """
    result = []
    for i in range(len(values) - window_size + 1):
        window = sorted(values[i:i + window_size])
        n = window_size
        if n % 2 == 1:
            result.append(float(window[n // 2]))
        else:
            result.append((window[n // 2 - 1] + window[n // 2]) / 2.0)
    return result
