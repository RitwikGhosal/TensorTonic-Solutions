import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1]]
    fpr = []
    tpr = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr.append(tp / positives)
        fpr.append(fp / negatives)

    return {
        "fpr": np.array(fpr),
        "tpr": np.array(tpr),
        "thresholds": thresholds
    }
    