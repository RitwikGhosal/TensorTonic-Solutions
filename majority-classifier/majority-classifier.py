import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    labels = np.asarray(y_train, dtype = int)
    samples = np.asarray(X_test)
    values, first_position, counts = np.unique(labels, return_index = True, return_counts = True)
    candidates = np.flatnonzero(counts == counts.max())
    majority = values[candidates[np.argmin(first_position[candidates])]]
    sample_count = samples.shape[0]
    return np.full(sample_count, majority, dtype = int)