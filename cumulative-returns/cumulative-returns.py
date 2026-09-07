def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    result = []
    W = 1.0
    for r in returns:
        W *= (1 + r)
        result.append(W - 1)
    return result