import numpy as np

def rnn_step_backward(dh: list, cache: list) -> dict:
    """
    Returns a dictionary with dx_t, dh_prev, dW, dU, and db.
    """
    x_t, h_prev, h_t, W, U, _= [np.asarray(value, dtype = float) for value in cache]
    dh = np.asarray(dh, dtype = float)
    dz = dh * (1.0 - h_t ** 2)
    return {
        "dx_t": W.T @ dz,
        "dh_prev": U.T @ dz,
        "dW": np.outer(dz, x_t),
        "dU": np.outer(dz, h_prev),
        "db": dz.copy(),
    }