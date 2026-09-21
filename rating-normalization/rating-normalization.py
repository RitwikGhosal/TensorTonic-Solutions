def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    res= []
    for row in matrix:
        new_row = [0] * len(row)
        c = 0 
        for j in range(len(row)): 
            if row[j]!=0:
                c += 1
        if c != 0:
            avg = sum(row) / c
        else:
            avg = 0.0
        for k in range(len(row)):
            if row[k] != 0:
                new_row[k] = row[k] - avg
        res.append(new_row)
    return res

        