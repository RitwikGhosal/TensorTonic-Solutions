def priority_replay_sample(priorities, alpha, beta):
    """
    Returns sampling probabilities and normalized importance weights.
    """
    n = len(priorities)
    powered = [p ** alpha for p in priorities]
    total = sum(powered)
    probs = [p / total for p in powered]
    raw_weights = [(n * pr) ** (-beta) for pr in probs]
    max_w = max(raw_weights)
    weights = [w / max_w for w in raw_weights]
    return [probs, weights]
