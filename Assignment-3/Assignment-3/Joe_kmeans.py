import numpy as np 

def sse(data, centroid):
    """
    Calculates the Sum of Squared Errors (SSE) for a given cluster.
    Used to determine which cluster has the highest variance and should be split.
    """
    if len(data) == 0:
        return 0.0
    return np.sum((data - centroid)**2)

def assign_labels(data, centroids):
    """
    Assigns each data point to the nearest centroid using Euclidean distance.
    Utilizes NumPy broadcasting for vectorized distance calculation.
    """
    # Calculate distances from each point to each centroid: shape (n_samples, n_centroids)
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    # Return the index of the minimum distance for each point
    return np.argmin(distances, axis=1)

def bin_kmeans(data, max_iters=100):
    """
    Performs standard K-Means strictly for k=2.
    Used to bisect a target cluster into two smaller sub-clusters.
    """
    # Edge case: Cannot split a cluster with fewer than 2 points
    if len(data) < 2:
        return np.vstack([data, data]), np.zeros(len(data), dtype=int)
        
    # Randomly initialize 2 centroids from existing data points
    indices = np.random.choice(len(data), 2, replace=False)
    centroids = data[indices]
    
    for _ in range(max_iters):
        # Assign points to the nearest of the 2 centroids
        labels = assign_labels(data, centroids)
        
        # Calculate new centroids as the mean of assigned points
        # Fallback to the previous centroid if a cluster becomes empty
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(2)
        ])
        
        # Check for convergence (centroids stop moving)
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, labels

def kmeans(X, k):
    """
    Executes the divisive (bisecting) K-Means algorithm to find k centroids.
    Iteratively splits the cluster with the highest SSE until k clusters are formed.
    """
    # Initialize with all data points in a single cluster
    clusters = [X]
    centroids = [np.mean(X, axis=0)]
    
    # Continue bisecting until the target number of clusters (k) is reached
    while len(clusters) < k:
        # Calculate SSE for all current clusters
        sses = [sse(cluster, centroid) for cluster, centroid in zip(clusters, centroids)]
        
        # Identify the cluster with the highest SSE (most variance)
        target_idx = np.argmax(sses)

        # Remove the target cluster and its centroid from the active lists
        target_cluster = clusters.pop(target_idx)
        centroids.pop(target_idx)
        
        # Bisect the target cluster into two new sub-clusters
        new_centroids, labels = bin_kmeans(target_cluster)

        # Integrate the newly formed sub-clusters back into the lists
        for i in range(2):
            sub_cluster = target_cluster[labels == i]
            
            # Safeguard: only append if the sub-cluster actually contains data points
            if len(sub_cluster) > 0:
                clusters.append(sub_cluster)
                centroids.append(new_centroids[i])
            else:
                # If empty partition occurs, fallback to re-inserting the original target
                clusters.append(target_cluster)
                centroids.append(new_centroids[i])
                
    return np.array(centroids)