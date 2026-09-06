import math

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    def dcg(scores):
        return sum((2 ** relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(scores[:k], start=1))
    actual = dcg(relevance_scores)
    ideal = dcg(sorted(relevance_scores, reverse=True))
    return actual / ideal if ideal else 0.0