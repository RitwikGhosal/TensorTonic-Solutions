def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    return [[x ** p for p in range(degree + 1)] for x in values]