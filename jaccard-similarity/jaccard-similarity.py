def jaccard_similarity(set_a: list, set_b: list) -> float:
    """
    Returns the Jaccard similarity of the two item collections.
    """
    a = set(set_a)
    b = set(set_b)
    
    union = a.union(b)
    if not union:
        return 0.0
        
    return len(a.intersection(b)) / len(union)