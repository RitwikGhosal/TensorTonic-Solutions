import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):

        self.W_mu = W_mu
        self.b_mu = b_mu
        self.W_logvar = W_logvar
        self.b_logvar = b_logvar
        self.W_dec = W_dec
        self.b_dec = b_dec
    
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns dict with "recon", "mu", "log_var".
        """
        mu = np.dot(x, self.W_mu) + self.b_mu
        log_var = np.dot(x, self.W_logvar) + self.b_logvar
        std = np.exp(0.5 * log_var)
        z = mu + std * epsilon
        recon = np.dot(z, self.W_dec) + self.b_dec
        return {"recon":recon, "mu": mu, "log_var":log_var}
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        return np.dot(z, self.W_dec) + self.b_dec 
