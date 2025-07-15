import numpy as np
from skfuzzy.cluster import cmeans
from scipy.spatial import distance_matrix
from fcmeans import FCM

class FuzzyCMeans:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        # centroids, u_membership, _, _, _, _, _ = cmeans(
        #         X.T, c=self.n_clusters, m=2, maxiter=1000, error=1e-6)

        self.fcm = FCM(n_clusters=self.n_clusters)
        self.fcm.fit(X)
        clusters = self.fcm.predict(X)
        # clusters = np.argmax(u_membership, axis=0)

        # selected_random_values = np.random.random(len(X))

        # probability_matrix = np.cumsum(u_membership, axis=0)

        # mask_clusters = (selected_random_values <= probability_matrix).T
        # clusters = mask_clusters.shape[1] - mask_clusters.sum(axis=1)

        possible_clusters = np.unique(clusters)
        possible_clusters.sort()

        self.n_clusters = len(possible_clusters) # Get the real number of clusters. Empty clusters are ignored.
        self.fcm.n_clusters = self.n_clusters

        # self.cluster_centers_  = np.array([centroids[c] for c in possible_clusters])
        self.cluster_centers_  = np.array([self.fcm.centers[c] for c in possible_clusters])
        # self.fcm.centers = self.cluster_centers_

        clusters_numbers_dict = {cluster: fixed_cluster
                                 for fixed_cluster, cluster in enumerate(possible_clusters)}

        # this is used to fix skipping cluster numbers.
        # Example: [0, 2, 3, 5] should be [0, 1, 2, 3]
        for c in possible_clusters:
            clusters[clusters == c] = clusters_numbers_dict[c]
        return clusters

    def predict(self, X):
        clusters = self.fcm.predict(X)
        possible_clusters = np.unique(clusters)
        possible_clusters.sort()
        # dist_matrix = distance_matrix(X, self.cluster_centers_)
        # clusters = np.argmin(dist_matrix, axis=1)
        clusters_numbers_dict = {cluster: fixed_cluster
                                 for fixed_cluster, cluster in enumerate(possible_clusters)}
        for c in possible_clusters:
            clusters[clusters == c] = clusters_numbers_dict[c]
        return clusters

    def __repr__(self):
        return f"FuzzyCMeans(n_clusters={self.n_clusters})"
