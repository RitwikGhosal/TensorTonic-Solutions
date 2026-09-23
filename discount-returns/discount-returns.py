def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    n = len(rewards)
    out = [0.0] * n
    running_return = 0.0

    for i in range(n - 1, -1, -1):
        running_return = rewards[i] + gamma * running_return
        out[i] = float(running_return)
        
    return out
        