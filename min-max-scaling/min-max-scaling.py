def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    n_r = len(data)
    n_c = len(data[0])
    result = [[0.0] * n_c for _ in range(n_r)]
    for j in range(n_c):
        col = [data[i][j] for i in range(n_r)]
        col_min = min(col)
        col_max = max(col)
        diff = col_max - col_min
        for i in range(n_r):
            result[i][j] = (data[i][j] - col_min) / diff if diff !=0 else 0.0
    return result