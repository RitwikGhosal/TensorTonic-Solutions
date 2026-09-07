def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    max_lag = max(lags)
    result = []
    for t in range(max_lag, len(series)):
        row = [series[t - lag] for lag in lags]
        result.append(row)
    return result
    