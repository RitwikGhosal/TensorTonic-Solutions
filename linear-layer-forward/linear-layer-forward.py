def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    n = len(X)
    return [[sum(X[i][k] * W[k][j] for k in range(len(X[0]))) + b[j] for j in range(len(W[0]))] for i in range(n)]
    