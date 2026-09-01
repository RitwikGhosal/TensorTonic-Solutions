def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    true_p = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    false_ = len(y_true) - true_p
    denom = 2*true_p + 2*false_
    return round(2 * true_p/denom, 4)
             