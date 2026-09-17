def interaction_features(X: list) -> list:
    """
    Returns original features followed by unique pairwise products.
    """
    res = []
    for row in X:
        inner = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                inner.append(row[i]*row[j]) 
        res.append(list(row) + inner)
    return res