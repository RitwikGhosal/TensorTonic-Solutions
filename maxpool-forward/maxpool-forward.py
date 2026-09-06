def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H, W = len(X), len(X[0])
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    result = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = X[i * stride][j * stride]
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * stride + a][j * stride + b]
                    if val > max_val:
                        max_val = val
            row.append(max_val)
        result.append(row)
    return result