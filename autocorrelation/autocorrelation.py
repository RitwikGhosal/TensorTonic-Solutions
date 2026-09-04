def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    n = len(series)
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series)
    if variance == 0:
        return [1.0] + [0.0] * max_lag
    result = []
    for lag in range(max_lag +1):
        cov = sum((series[t] - mean) * (series[t + lag] - mean) for t in range(n - lag))
        result.append(cov / variance)
    return [round(value, 6) for value in result]