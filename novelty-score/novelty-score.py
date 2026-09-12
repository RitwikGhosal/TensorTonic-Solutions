import math

def novelty_score(recommendations: list, item_counts: list, n_users: int) -> float:
    """
    Returns the average self-information of the recommended items.
    """
    out = 0.0
    for i in recommendations:
        div = item_counts[i] / n_users
        if div > 0:
            out += -math.log2(div)
    return out / len(recommendations) if recommendations else 0.0