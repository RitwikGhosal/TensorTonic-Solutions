import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    if len(docs) == 0:
        return np.zeros(0, dtype= float)
    lengths = np.array([len(document) for document in docs], dtype = float)
    average_length = float(np.mean(lengths))
    frequencies = [Counter(document) for document in docs]
    document_frequnecy = Counter()
    for document in docs:
        document_frequnecy.update(set(document))
    scores = np.zeros(len(docs), dtype = float)
    for term in dict.fromkeys(query_tokens):
        frequency = document_frequnecy[term]
        if frequency == 0.0:
            continue 
        idf = math.log((len(docs) - frequency +0.5) / (frequency + 0.5) + 1.0)
        term_freqency = np.array([counts[term] for counts in frequencies], dtype = float)
        length_factor = 1.0 - b + b * lengths / average_length
        denominator = term_freqency + k1 * length_factor
        scores += idf * term_freqency * (k1 + 1.0) / denominator
    return scores
    
    