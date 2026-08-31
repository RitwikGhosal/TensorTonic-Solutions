import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    def node_(labels):
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / labels.size
        return (1.0 - np.sum(p**2))

    total = y_left.size + y_right.size
    if total == 0:
        return 0.0
    return float((y_left.size/total)*node_(y_left) + (y_right.size/total)*node_(y_right))
    