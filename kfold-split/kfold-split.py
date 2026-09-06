import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.arange(N)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    folds = np.array_split(indices, k)
    res = []
    for idx, val in enumerate(folds):
        train = np.concatenate(folds[:idx] + folds[idx+1:])
        res.append({"train_idx": train.astype(int), "val_idx": val.astype(int)})
    return res