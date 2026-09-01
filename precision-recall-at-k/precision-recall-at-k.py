def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    top_k = recommended[:k]
    relevant_items = set(relevant)
    i_ = sum(item in relevant_items for item in top_k)
    precision = i_ / k
    recall = i_ / len(relevant_items)
    return [precision, recall] 