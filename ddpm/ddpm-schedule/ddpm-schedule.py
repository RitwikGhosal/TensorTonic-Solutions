import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> list[float]:
    """
    Returns the linear beta schedule rounded to six decimals.
    """
    out = np.linspace(beta_1, beta_T, T, dtype = np.float64)
    return [round(float(o),6) for o in out]

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> list[float]:
    """
    Returns the clipped cosine alpha-bar schedule rounded to six decimals.
    """
    pos = np.arange(T+1, dtype=  np.float64)/T
    f = np.cos((pos + s)/(1.0 + s) * np.pi /2.0)**2
    alpha_bars = np.clip(f[1:] / f[0], 0.0001, 0.9999)
    return [round(float(out), 6) for out in alpha_bars]

def alpha_bar_to_betas(alpha_bars: list[float]) -> list[float]:
    """
    Returns the recovered beta schedule rounded to six decimals.
    """
    curr = np.asarray(alpha_bars, dtype = np.float64)
    prev = np.concatenate(([1.0], curr[:-1]))
    betas = np.clip(1.0 - curr / prev, 0.0001, 0.9999)
    return [round(float(value), 6) for value in betas]