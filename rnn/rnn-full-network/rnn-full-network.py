import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, W_hy: np.ndarray, b_h: np.ndarray,
                b_y: np.ndarray) -> dict:
    """
    Returns outputs and final_hidden_state as float64 arrays.
    """
    h = h_0
    op = []
    for t in range(X.shape[1]):
        h = np.tanh(X[:, t, :]@W_xh.T + h@W_hh.T + b_h)
        op.append(h@W_hy.T + b_y)

    return {"outputs":np.stack(op,axis=1).astype(np.float64),"final_hidden_state":h.astype(np.float64)}