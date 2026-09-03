import numpy as np

def compute_ddpm_loss(x_0: list, betas: list[float], t_values: list[int], epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean squared noise-prediction error.
    """
    tn = np.asarray(epsilon, dtype = np.float64)
    pn = np.asarray(epsilon_pred, dtype = np.float64)
    loss = np.mean((tn - pn)**2)
    return round(float(loss), 6) 