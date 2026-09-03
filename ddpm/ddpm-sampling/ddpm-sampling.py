import numpy as np

def ddpm_sample(x_T: list, betas: list[float], epsilon_preds: list, z_values: list) -> list:
    """
    Returns the final denoised sample rounded to four decimals.
    """
    x = np.asarray(x_T, dtype = np.float64)
    beta_values = np.asarray(betas, dtype = np.float64)
    alpha_values = 1.0 - beta_values
    alpha_bar_values = np.cumprod(alpha_values)

    for i, t in enumerate(range(len(beta_values), 0 , -1)):
        beta_t = beta_values[t-1]
        alpha_t = alpha_values[t-1]
        alpha_bar_t = alpha_bar_values[t-1]
        predicted_noise = np.asarray(epsilon_preds[i], dtype = np.float64)
        mean = (x - beta_t * predicted_noise/np.sqrt(1.0 - alpha_bar_t)) / np.sqrt(alpha_t)

        if t > 1:
            noise = np.asarray(z_values[i], dtype = np.float64)
            x = mean + np.sqrt(beta_t) * noise
        else:
            x = mean
    return np.round(x, 4).tolist()