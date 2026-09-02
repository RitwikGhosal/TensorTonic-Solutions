import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    #tokens = np.asarray(tokens)
    #unique, counts = np.unique(tokens, return_counts = True)
    #count_map = dict(zip(unique, counts))
    #full_counts = np.array([count_map.get(cls, 0) for cls in vocab])
    #return full_counts 

    token_index = {token : idx for idx, token in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype = int)
    for token in tokens:
        if token in token_index:
            counts[token_index[token]] += 1
    return counts