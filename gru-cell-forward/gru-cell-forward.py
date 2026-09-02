import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x = np.asarray(x, dtype = float)
    h_prev = np.asarray(h_prev, dtype = float)
    parameters = {name : np.asarray(value, dtype=float) for name, value in params.items()}
    single_sample = x.ndim == 1
    if single_sample:
        x = x.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)

    def sigmoid(value):
        return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))

    z = sigmoid(x @ parameters["Wz"] + h_prev @ parameters["Uz"] + parameters["bz"])
    r = sigmoid(x @ parameters["Wr"] + h_prev @ parameters["Ur"] + parameters["br"])
    candidate = np.tanh(x @ parameters["Wh"] + (r * h_prev) @ parameters["Uh"] + parameters["bh"])
    hidden = (1.0 - z) * h_prev + z * candidate
    return hidden[0] if single_sample else hidden