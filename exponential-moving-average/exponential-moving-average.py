def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    result = []
    result.append(values[0])
    for i in range(1,len(values)):
        result.append(alpha * values[i] + (1-alpha) * result[i-1])
    return result