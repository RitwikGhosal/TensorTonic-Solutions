import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    cost = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return float(np.mean(cost))
    return float(np.sum(cost))