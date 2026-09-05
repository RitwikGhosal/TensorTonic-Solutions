import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    vocab = sorted(set(tokens))
    token_index = {t : i for i,t in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)), dtype = int)
    for f, s in zip(tokens[:-1], tokens[1:]): 
        counts[token_index[f], token_index[s]] += 1
    prob = (counts + 1) / (counts.sum(axis =1, keepdims = True) + len(vocab))
    return {"vocab": vocab, "counts": counts, "probabilities": prob}
        