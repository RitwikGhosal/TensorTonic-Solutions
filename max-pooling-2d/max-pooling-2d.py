def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    H, W = len(X), len(X[0])
    out_h = H // pool_size
    out_w = W // pool_size
    res=  []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_value = X[i *pool_size][j * pool_size]
            for pi in range(pool_size):
                for pj in range(pool_size):
                    val = X[i * pool_size + pi][j * pool_size + pj]
                    if val > max_value:
                        max_value = val
            row.append(max_value)
        res.append(row)
    return res
    