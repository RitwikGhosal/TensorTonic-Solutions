def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample, breaking ties with min().
    """
    n_samples = len(predictions[0])
    vote = []
    
    for i in range(n_samples):
        res = {}
        for tree_preds in predictions:
            j = tree_preds[i]
            res[j] = res.get(j, 0) + 1
            

        mx = max(res.values())
        vote.append(min(k for k, v in res.items() if v == mx))

    return vote

        