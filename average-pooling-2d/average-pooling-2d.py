def average_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping average-pooled windows.
    """
    H, W = len(X), len(X[0])
    h_out = H // pool_size
    w_out = W // pool_size
    res = []
    for i in range(h_out):
        row = []
        for j in range(w_out):
            total = 0
            for pi in range(pool_size):
                for pj in range(pool_size):
                    total += X[i * pool_size + pi][j * pool_size + pj]
            row.append(total / (pool_size)**2)
        res.append(row)
    return res