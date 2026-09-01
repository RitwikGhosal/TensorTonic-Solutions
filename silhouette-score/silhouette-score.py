import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.asarray(X, dtype = float)
    labels = np.asarray(labels)
    distances = np.sqrt(
        np.sum((X[:,None,:] - X[None,:,:])**2, axis = 2)
    )

    clusters, cluster_index = np.unique(labels, return_inverse = True)
    membership = labels[None, :] == clusters[:, None]
    cluster_sizes = membership.sum(axis = 1)
    mean_to_cluster = distances @ membership.T / cluster_sizes

    own_sizes = cluster_sizes[cluster_index]
    same_cluster = labels[:, None] == labels[None, :]
    within = (distances * same_cluster).sum(axis=1) / (own_sizes - 1)
    mean_to_cluster[np.arange(len(X)), cluster_index] = np.inf
    nearest_other = mean_to_cluster.min(axis=1)
    scores = (nearest_other - within) / np.maximum(within, nearest_other)
    return float(np.mean(scores))
    

    
    