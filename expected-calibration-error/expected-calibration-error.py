def expected_calibration_error(y_true: list, y_pred: list, n_bins: int) -> float:
    """
    Returns the calibration error as a float.
    """
    bins = [[] for _ in range(n_bins)]
    for true, pred in zip(y_true, y_pred):
        idx = min(int(pred * n_bins), n_bins -1)
        bins[idx].append((true, pred))
    error = 0.0
    for val in bins:
        if val:
            accuracy = sum(i for i, _ in val) / len(val)
            conf = sum(j for _, j in val) / len(val)
            error += len(val) / len(y_true) * abs(accuracy - conf)
    return error
    
    