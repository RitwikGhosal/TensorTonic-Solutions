def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize="none"):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = 0 if y_true.size == 0 else int(max(y_true.max(), y_pred.max())) + 1

    matrix = np.bincount(
        y_true * num_classes + y_pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    if normalize == "none":
        return matrix

    matrix = matrix.astype(float)

    if normalize == "true":
        s = matrix.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        s = matrix.sum(axis=0, keepdims=True)
    else:
        s = np.array([[matrix.sum()]])

    return np.divide(
        matrix,
        s,
        out=np.zeros_like(matrix),
        where=s != 0
    )
        