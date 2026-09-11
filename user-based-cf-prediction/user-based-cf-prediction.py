def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    n, d = 0.0, 0.0
    for sim, rat in zip(similarities, ratings):
        if sim > 0:
            n += sim * rat
            d += sim
    if d == 0:
        return 0.0
    return n/d