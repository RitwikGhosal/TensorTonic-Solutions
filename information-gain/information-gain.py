import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y, dtype = int)
    mask = np.asarray(split_mask, dtype=bool)

    def ent_(labels):
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts = True)
        p = counts / labels.size
        return (-np.sum(p * np.log2(p)))

    left = y[mask]
    right = y[~mask]

    if left.size == 0 or right.size == 0:
        return 0.0
    out = ((left.size / y.size) * ent_(left) + (right.size / y.size) * ent_(right))
    return float(ent_(y) - out)

    