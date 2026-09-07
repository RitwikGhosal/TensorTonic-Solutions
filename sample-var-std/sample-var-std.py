import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x, dtype = float)
    mean = np.mean(x)
    var = float(np.sum((x - mean)**2)/(len(x) - 1))
    std_dev = float(np.sqrt(var))
    return {"variance": var, "standard_deviation": std_dev}