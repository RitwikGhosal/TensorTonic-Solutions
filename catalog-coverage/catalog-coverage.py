def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    res = []
    for i in recommendations:
        for j in i:
            res.append(j)

    return len(list(set(res))) / n_items