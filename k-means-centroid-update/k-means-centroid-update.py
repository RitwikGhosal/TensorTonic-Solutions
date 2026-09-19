def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster. Safe version keeping your original style.
    """
    cluster = dict.fromkeys(range(k))

    for i, j in zip(points, assignments):
        if cluster[j] is None:
            cluster[j] = [i]
        else:
            cluster[j].append(i)   
    
    cluster_means = {}
    dim = len(points[0]) if points else 2
    
    for cluster_id, points_list in cluster.items():
        if points_list is None:
            cluster_means[cluster_id] = [0.0] * dim
            continue
            
        num_points = len(points_list)
        coordinate_sums = [sum(coords) for coords in zip(*points_list)]
        cluster_means[cluster_id] = [coord_sums/num_points for coord_sums in coordinate_sums]

    return list(cluster_means.values())

        
            