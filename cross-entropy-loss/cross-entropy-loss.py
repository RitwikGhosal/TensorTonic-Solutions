import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:

    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = int)
    ind = np.arange(len(y_true))
    probab = y_pred[ind, y_true]
    return float(-np.mean(np.log(probab)))
                    
    