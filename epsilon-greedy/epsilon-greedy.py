import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    rng = np.random.default_rng(seed)
    q_values = np.asarray(q_values, dtype = float)
    if rng.random() < epsilon:
        return int(rng.integers(q_values.size))
    return int(np.argmax(q_values))