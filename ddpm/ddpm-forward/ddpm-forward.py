import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    alphas = 1.0 - np.asarray(betas, dtype = np.float64)
    return [round(float(value),6) for value in np.cumprod(alphas)]

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    clean = np.asarray(x_0, dtype = np.float64)
    noise = np.asarray(epsilon, dtype = np.float64)
    alphas = 1.0 - np.asarray(betas, dtype = np.float64)
    alpha_bar_t = np.cumprod(alphas)[t-1]
    x_t = np.sqrt(alpha_bar_t)*clean + np.sqrt(1.0 - alpha_bar_t)*noise
    return np.round(x_t, 4).tolist()