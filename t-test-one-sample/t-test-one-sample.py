import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x, dtype = float)
    mean = np.mean(x)
    x_ = x - mean 
    std = np.sqrt(np.sum(x_**2) / (x.size - 1))
    if std == 0:
        diff = float(mean - mu0)
        if diff == 0:
            return 0.0
        return float(np.inf if diff > 0 else -np.inf)
    return float((mean - mu0) / (std / np.sqrt(x.size)))