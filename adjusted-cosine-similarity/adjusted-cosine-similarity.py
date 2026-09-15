def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    user_means = []
    for u in range(len(ratings_matrix)):
        rated = [r for r in ratings_matrix[u] if r != 0]
        user_means.append(sum(rated) / len(rated) if rated else 0.0)
    num = 0.0
    den_i = 0.0
    den_j = 0.0
    for u in range(len(ratings_matrix)):
        if ratings_matrix[u][item_i] == 0 or ratings_matrix[u][item_j] == 0:
            continue 
        ri = ratings_matrix[u][item_i] - user_means[u]
        rj = ratings_matrix[u][item_j] - user_means[u]
        num += ri * rj
        den_i += ri * ri
        den_j += rj * rj
    den = (den_i ** 0.5) * (den_j ** 0.5)
    if den == 0:
        return 0.0
    return num / den
    