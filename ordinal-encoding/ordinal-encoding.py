def ordinal_encoding(values: list, ordering: list) -> list:
    """
    Returns the ordinal index of every input value.
    """
    stoi = {s:i for i, s in enumerate(ordering)}
    return [stoi[v] for v in values]