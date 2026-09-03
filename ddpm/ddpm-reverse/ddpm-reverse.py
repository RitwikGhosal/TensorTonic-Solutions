import numpy as np

def reverse_step(x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """
    now = np.asarray(x_t, dtype = np.float64)
    pred_noise = np.asarray(epsilon_pred, dtype = np.float64)
    beta_values = np.asarray(betas, dtype = np.float64)
    alpha_values = 1.0 - beta_values
    alpha_bar_values = np.cumprod(alpha_values)
    beta_t = beta_values[t-1]
    alpha_t = alpha_values[t-1]
    alpha_bar_t = alpha_bar_values[t-1]
    mean = (now - (beta_t / np.sqrt(1 - alpha_bar_t))*pred_noise) / np.sqrt(alpha_t)

    if t > 1:
        mean = mean + np.sqrt(beta_t)*np.asarray(z, dtype = np.float64)
    return np.round(mean, 4).tolist()