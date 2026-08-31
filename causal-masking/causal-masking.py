import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.asarray(scores, dtype = float)
    mask = np.triu(np.ones_like(scores, dtype = bool), k=1)
    return np.where(mask, mask_value, scores)