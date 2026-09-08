import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    result = {}
    for feature in train_dist:
        train = np.asarray(train_dist[feature], dtype = float) + eps
        serving = np.asarray(serving_dist[feature], dtype = float) + eps
        psi = round(float(np.sum((serving - train) * np.log(serving / train))), 6)
        result[feature] = {"psi": psi, "skewed": psi >= threshold}
    return result