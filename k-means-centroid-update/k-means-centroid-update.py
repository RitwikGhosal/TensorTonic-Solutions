def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster. Safe against empty clusters.
    """
    cluster = {cluster_id: [] for cluster_id in range(k)}

    for i, j in zip(points, assignments):
        cluster[j].append(i)   
    
    cluster_means = {}
    dim = len(points[0]) if points else 2 
    
    for cluster_id, points_list in cluster.items():
        if not points_list:
            cluster_means[cluster_id] = [0.0] * dim
            continue
            
        num_points = len(points_list)
        coordinate_sums = [sum(coords) for coords in zip(*points_list)]
        cluster_means[cluster_id] = [coord_sum / num_points for coord_sum in coordinate_sums]

    return [cluster_means[i] for i in range(k)]
        
            