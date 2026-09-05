import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    log_sum = 0.0
    for i in range(len(actual_tokens)):
        p = prob_distributions[i][actual_tokens[i]]
        log_sum += math.log(p)
    H = -log_sum / len(actual_tokens)
    return round(math.exp(H), 4)
    