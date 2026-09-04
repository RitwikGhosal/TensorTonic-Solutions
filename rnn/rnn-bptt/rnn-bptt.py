import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray,
                     h_prev: np.ndarray, x_t: np.ndarray,
                     W_hh: np.ndarray, W_xh: np.ndarray) -> dict:
    """
    Returns the five float64 local RNN gradients.
    """
    da = dh_next*(1.0-h_t**2)
    return {"dh_prev":(da@W_hh).astype(np.float64),"dx":(da@W_xh).astype(np.float64),"dW_hh":(da.T@h_prev).astype(np.float64),"dW_xh":(da.T@x_t).astype(np.float64),"db_h":np.sum(da,axis=0).astype(np.float64)}