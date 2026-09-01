import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    vocab = sorted(set(word for doc in documents for word in doc.lower().split()))
    idx = {tok:pos for pos,tok in enumerate(vocab)}
    matrix = np.zeros((len(documents), len(vocab)), dtype = float)

    N = len(documents)
    df = Counter()
    
    for doc in documents:
        words =set(doc.lower().split()) 
        for word in words:
            df[word] += 1

    for i, doc in enumerate(documents):
        words = doc.lower().split()
        counts = Counter(words)

        for word, count in counts.items():
            tf = count / len(words)
            idf = math.log(N / df[word])
            matrix[i, idx[word]] = tf*idf

    return {
        "tfidf_matrix": matrix,
        "vocabulary": vocab
    }
    

    
    