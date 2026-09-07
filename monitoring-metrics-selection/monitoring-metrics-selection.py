def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    if system_type == "classification":
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
        tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2*precision*recall / (precision + recall) if precision + recall else 0.0
        return {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall, "f1": f1}
    if system_type == "regression":
        errors = [target - prediction for target, prediction in zip(y_true, y_pred)]
        return {"mae": sum(abs(error) for error in errors) / len(errors), "rmse": (sum(error ** 2 for error in errors) / len(errors)) ** 0.5}
    order = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)
    relevant_in_top = sum(y_true[i] == 1 for i in order[:3])
    total_relevant = sum(value == 1 for value in y_true)
    return {"precision_at_3": relevant_in_top / 3, "recall_at_3": relevant_in_top / total_relevant if total_relevant else 0.0}