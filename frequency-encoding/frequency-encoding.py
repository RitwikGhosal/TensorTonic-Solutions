def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return [counts[v] / len(values) for v in values]
    
    