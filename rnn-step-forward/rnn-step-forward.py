import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    x_t = np.asarray(x_t, dtype = float)
    h_prev = np.asarray(h_prev, dtype = float)
    Wx = np.asarray(Wx, dtype = float)
    Wh = np.asarray(Wh, dtype = float)
    b = np.asarray(b, dtype = float)
    return np.tanh(x_t@Wx + h_prev@Wh + b)