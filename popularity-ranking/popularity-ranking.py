def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    result = []
    for avg_r, num_v in items:
        wr = (num_v / (num_v + min_votes)) * avg_r + (min_votes / (num_v + min_votes)) * global_mean
        result.append(wr)
    return result