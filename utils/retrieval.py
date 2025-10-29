import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

class Retriever:
    def __init__(self, public_features, public_labels, n_clusters=10, random_state=42):
        """
        Args:
            public_features (np.ndarray): Array of shape (N, D), D=feature size.
            public_labels (list or np.ndarray): List/array of length N with labels.
            n_clusters (int): Number of clusters for KMeans.
            random_state (int): Random seed.
        """
        self.public_features = public_features
        self.public_labels = np.asarray(public_labels)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.kmeans.fit(public_features)
        self.cluster_labels = self.kmeans.labels_
        self.cluster_centers = self.kmeans.cluster_centers_

    def query(self, query_features, k=10):
        """
        Retrieve top-k nearest public samples for each query.
        Args:
            query_features (np.ndarray): Array (M, D), M=query samples.
            k (int): Number of samples to retrieve per query.
        Returns:
            indices (list): List of lists, indices of retrieved samples per query.
        """
        distances = cosine_distances(query_features, self.public_features)
        topk_indices = np.argsort(distances, axis=1)[:, :k]
        return topk_indices.tolist()

    def cluster_members(self, cluster_idx):
        """
        Return indices of public samples in the given cluster.
        """
        return np.where(self.cluster_labels == cluster_idx)[0]

    def get_public_labels(self, indices):
        """
        Fetch labels for given indices.
        """
        return self.public_labels[indices]
