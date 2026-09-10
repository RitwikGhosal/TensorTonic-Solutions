def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    blocked = set(stopwords)
    return [token for token in tokens if token not in blocked]