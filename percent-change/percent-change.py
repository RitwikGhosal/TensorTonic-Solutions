def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    result = []
    for i in range(len(series) - 1):
        result.append((series[i+1]- series[i])/series[i]) if series[i] !=0 else result.append(0.0) 
    return result