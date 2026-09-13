import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    if len(candidate) == 0:
        return 0.0
    c = len(candidate)
    r = len(reference)
    if c <= r:
        bp = math.exp(1 - (r/c))
    else:
        bp = 1.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = []
        for i in range(len(candidate)- n +1):
            cand_ngrams.append(tuple(candidate[i:i+n]))
        ref_ngrams = []
        for i in range(len(reference)- n +1):
            ref_ngrams.append(tuple(reference[i:i+n]))
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue 
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        clipped = 0
        for ngram, count in cand_counts.items():
            clipped += min(count, ref_counts.get(ngram, 0))
        prec = clipped / len(cand_ngrams)
        precisions.append(prec)
    if any(p == 0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / len(precisions)
    bleu = bp * math.exp(log_mean)
    return float(bleu)