import numpy as np

def kl_divergence(mu, log_var):
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
    return float(np.mean(kl))
