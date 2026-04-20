import numpy as np 

def sse(data, centroid):
    if len(data) == 0:
        return 0.0
    return np.sum((data - centroid)**2)

def assign_labels(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def bin_kmeans(data, max_iters=100):
    if len(data) < 2:
        return np.vstack([data, data]), np.zeros(len(data), dtype=int)
        
    indices = np.random.choice(len(data), 2, replace=False)
    centroids = data[indices]
    
    for _ in range(max_iters):
        labels = assign_labels(data, centroids)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(2)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels

def kmeans(X, k):
    clusters = [X]
    centroids = [np.mean(X, axis=0)]
    
    while len(clusters) < k:
        sses = [sse(cluster, centroid) for cluster, centroid in zip(clusters, centroids)]
        target_idx = np.argmax(sses)

        target_cluster = clusters.pop(target_idx)
        centroids.pop(target_idx)
        new_centroids, labels = bin_kmeans(target_cluster)

        for i in range(2):
            sub_cluster = target_cluster[labels == i]
            if len(sub_cluster) > 0:
                clusters.append(sub_cluster)
                centroids.append(new_centroids[i])
            else:
                clusters.append(target_cluster)
                centroids.append(new_centroids[i])
                
    return np.array(centroids)