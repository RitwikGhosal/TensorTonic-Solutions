def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    n = len(values)
    s = sorted(range(n), key = lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[s[j]] == values[s[i]]:
            j += 1
        avg_rank = sum(range(i+1, j+1))/ (j-i)
        for k in range(i, j):
            ranks[s[k]] = avg_rank
        i = j
    return ranks
    
    
    
    