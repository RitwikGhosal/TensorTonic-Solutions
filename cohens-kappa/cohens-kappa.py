def cohens_kappa(rater1: list, rater2: list) -> float:
    """
    Returns Cohen's kappa as a float.
    """
    n = len(rater1)
    po = sum(a == b for a, b in zip(rater1, rater2)) / n
    pe = 0.0
    for label in set(rater1) | set(rater2):
        pe += (rater1.count(label) / n) * (rater2.count(label) / n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
    