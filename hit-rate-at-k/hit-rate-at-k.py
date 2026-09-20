def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    
    found = []
    g =[]
    
    for j in ground_truth:
        g.append(j[0])
        
    for i in recommendations:
        for j in g:
            if j in i[:k] and i not in found:
                found.append(i)

    return (len(found) / len(recommendations))
        