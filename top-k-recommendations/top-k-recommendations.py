def top_k_recommendations(scores: list, rated_indices: list, k: int) -> list:
    """
    Returns the highest-scoring unrated item indices.
    """
    unrated = []
    for i in range(len(scores)):
        if i in rated_indices:
            continue 
        unrated.append(i)

    res = {}
    for a in unrated:
        res[a] = scores[a]

    sorted_res = dict(sorted(res.items(), key= lambda item: item[1], reverse = True))
    return list(sorted_res.keys())[:k]