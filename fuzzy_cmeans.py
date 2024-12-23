import numpy as np
from dataclasses import dataclass
from skfuzzy.cluster import cmeans


@dataclass
class FuzzyCMeans:
    n_clusters: int

    def fit_predict(self, X):
        centroids, u_membership, _, _, _, _, _ = cmeans(X.T, c=self.n_clusters, m=2, maxiter=1000, error=1e-6)
        clusters = np.argmax(u_membership, axis=0)

        possible_clusters = np.unique(clusters)
        possible_clusters.sort()

        self.cluster_centers_  = np.array([centroids[c] for c in possible_clusters])

        self.n_clusters = len(possible_clusters) # Get the real number of clusters. Empty clusters are ignored.

        # this is used to fix skipping cluster numbers.
        # Example: [0, 2, 3, 5] should be [0, 1, 2, 3]
        cluster_index = 0

        # Fix when a cluster has no samples
        for c in possible_clusters:
            clusters[clusters == c] = cluster_index
            cluster_index += 1

        return clusters
