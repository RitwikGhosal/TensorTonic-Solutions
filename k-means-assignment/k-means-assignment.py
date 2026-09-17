def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    out = []
    dist = []
    for p in points:
        dist.append(sum((a-b)**2 for a, b in list(zip(p, centroids[0]))))
    for j in range(len(points)):
        best_i = 0  
        for i in range(1, len(centroids)):
            d = sum((a - b)**2 for a, b in list(zip(points[j], centroids[i]))) 
            if d < dist[j]:
                dist[j] = d  
                best_i = i   
        out.append(best_i)  
    return out