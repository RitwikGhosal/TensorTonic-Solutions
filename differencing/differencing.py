def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    
    while order > 0:
        res = []
        for i in range(len(series)-1):
            res.append(series[i+1] - series[i])
        order -= 1
        series = res
    return res
        
        