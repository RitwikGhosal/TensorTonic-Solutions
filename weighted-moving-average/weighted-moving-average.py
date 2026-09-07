def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    weights_total = sum(weights)
    result = []
    for i in range(len(values) - len(weights) + 1):
        total  = sum(weights[j] * values[i+j] for j in range(len(weights)))
        result.append(total/weights_total)
    return result