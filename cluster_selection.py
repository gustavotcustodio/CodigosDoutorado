from ast import Call
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Callable
from sklearn.cluster import DBSCAN, BisectingKMeans, KMeans, SpectralClustering, kmeans_plusplus
from sklearn.metrics.pairwise import cosine_distances


CLUSTERING_ALGORITHMS = {
    #"kmeans": KMeans,
    "kmeans++": KMeans,
    #"spectral_clustering": SpectralClustering,
    #"bisecting_kmeans": BisectingKMeans
}


@dataclass
class ClusteringModule:
    n_clusters: int
    X: NDArray
    y: NDArray
    evaluation_metric = "DBC"

    def create_clusterer(self, algorithm: str):
        if algorithm == "kmeans++":
            kmeans_pp_init = kmeans_plusplus(self.X, n_clusters=self.n_clusters)
            return KMeans(n_clusters=self.n_clusters, init=kmeans_pp_init[0])
        elif algorithm == "kmeans":
            return KMeans(n_clusters=self.n_clusters, init="random")
        else:
            return CLUSTERING_ALGORITHMS[algorithm](n_clusters=self.n_clusters)
         
    def select_evaluation_function(self) -> Callable:
        if self.evaluation_metric == "DBC":
            # Calculate distances between samples
            return self.get_DBC_distance()
        else:
            # TODO add other validation methods
            return self.get_DBC_distance()

    def compare_clusterers_and_select(self):
        """ Compare multiple different clusterers and select the best
        according to some metric.
        """
        evaluation_values = []
        possible_clusterers = []

        # TODO change the number of clusters and parallelize
        for clustering_algorithm in CLUSTERING_ALGORITHMS:
            clusterer = self.create_clusterer(clustering_algorithm)
            clusters = clusterer.fit_predict(self.X)

            evaluation_value = self.evaluation_function(clusters)

            evaluation_values.append(evaluation_value)
            possible_clusterers.append(clusterer)

        # Select the best clusterer according to an evaluation value
        idx_best_clusterer = np.argmax(evaluation_values)
        return possible_clusterers[idx_best_clusterer]

    def cluster_data(self):
        """ Try different clustering algorithms and select the best one
        for the given dataset.
        """
        self.n_labels = len(np.unique(self.y))

        # Calculate the average cosine distance between samples
        self.distances_between_samples = cosine_distances(self.X, self.X)

        # Select the clustering evaluation function
        self.evaluation_function = self.select_evaluation_function()

        self.best_clusterer = self.compare_clusterers_and_select()
        clusters = self.best_clusterer.fit_predict(self.X)

        # Define the centroids for the best clusterer
        # If it's Spectral Clustering, the centroids need to be calculated.
        if isinstance(self.best_clusterer, SpectralClustering):
            self.centroids = np.array([self.X[clusters == c].mean(axis=0)
                                       for c in range(self.n_clusters)])
        else:
            self.centroids = self.best_clusterer.cluster_centers_

        # Split the samples according to the cluster they are assigned
        samples_by_cluster = {}
        labels_by_cluster = {}

        for c in range(self.n_clusters):
            indexes_c = np.where(clusters == c)[0]
            samples_by_cluster[c] = self.X[indexes_c]
            labels_by_cluster[c] = self.y[indexes_c]

        return samples_by_cluster, labels_by_cluster


    def get_DBC_distance(self) -> Callable[[NDArray], float]:
        """ The DBC distance measures the average cosine distance between
        samples of different classes in clusters.
        """
        def wrapper(clusters: NDArray) -> float:
            dists_centroids_cluster = np.empty((self.n_clusters))

            for c in range(self.n_clusters):
                samples_c = np.where(clusters == c)[0]

                avg_distance_cluster = 0
                n_distances = 0

                for lbl1 in range(self.n_labels):
                    for lbl2 in range(lbl1+1, self.n_labels):

                        samples_lbl1 = np.where(self.y == lbl1)[0]
                        samples_lbl1 = np.intersect1d(samples_lbl1, samples_c)

                        samples_lbl2 = np.where(self.y == lbl2)[0]
                        samples_lbl2 = np.intersect1d(samples_lbl2, samples_c)

                        n_distances += len(samples_lbl1) * len(samples_lbl2)
                        avg_distance_cluster += self.distances_between_samples[samples_lbl1, :][:, samples_lbl2].sum()

                if n_distances > 0:
                    dists_centroids_cluster[c] = avg_distance_cluster / n_distances
                else:
                    dists_centroids_cluster[c] = 1.0

            macro_avg_metric = np.mean(dists_centroids_cluster).astype(float)

            return macro_avg_metric
        return wrapper

    def calc_membership_matrix(self, X_samples, centroids):
        # u_ik =          1
        #       -----------------------
        #       sum_j=1^C (d_ik²/d_ij²)
        n_clusters = centroids.shape[0]
        n_samples = X_samples.shape[0]

        # np.linalg.norm(samples-c[0], axis=1)**2
        dists_samples_centroids = [np.linalg.norm(X_samples - centroids[k], axis=1)**2
                                  for k in range(n_clusters)]
        dists_samples_centroids = np.array(dists_samples_centroids).T

        u = np.empty((n_samples, n_clusters))

        for k in range(n_clusters):
            u[:, k] = np.sum([
                dists_samples_centroids[:, k] / (dists_samples_centroids[:, c] + 0.00000001)
                for c in range(n_clusters)
            ], axis=0)
        u = 1 / (u + 0.00000001)
        u[u > 1] = 1
        return u
