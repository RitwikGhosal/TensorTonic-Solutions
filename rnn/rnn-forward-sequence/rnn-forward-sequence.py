import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    states = []
    h = h_0
    for t in range(X.shape[1]):
        h = np.tanh(X[:, t, :]@W_xh.T + h@W_hh.T + b_h)
        states.append(h)
    return {"hidden_states":np.stack(states, axis=1).astype(np.float64), "final_hidden_state": h.astype(np.float64)}