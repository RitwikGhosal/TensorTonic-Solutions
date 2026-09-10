def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        for i in range(len(result)):
            row = result[i]

            rated = [v for v in row if v != 0]
            value = sum(rated) / len(rated) if rated else 0.0

            for j in range(len(row)):
                if row[j] == 0:
                    row[j] = value

    else:
        for col in range(len(result[0])):
            rated = [row[col] for row in result if row[col] != 0]
            value = sum(rated) / len(rated) if rated else 0.0

            for i in range(len(result)):
                if result[i][col] == 0:
                    result[i][col] = value

    return result