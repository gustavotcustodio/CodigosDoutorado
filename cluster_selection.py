import numpy as np
import sys
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Callable, Optional
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from sklearn.metrics.pairwise import cosine_distances
from fuzzy_cmeans import FuzzyCMeans
from scipy.spatial import distance_matrix
from sklearn.metrics import adjusted_rand_score

CLUSTERING_ALGORITHMS = {
    'kmeans': KMeans,
    'kmeans++': KMeans,
    'fcm': FuzzyCMeans,
    'spectral': SpectralClustering,
}


@dataclass
class ClusteringModule:
    X: NDArray
    y: NDArray
    n_clusters: str | int = "compare"
    clustering_algorithm: str = "kmeans++"
    evaluation_metric: str = "dbc"  # Possible values: dbc, silhouette, dbc_ss, dbc_rand
    weights_dbc_external = (0.5, 0.5) # This attribute weights each part of the metric when using DBC combined with the silhoutte score
    allow_fcm: bool = False

    def __post_init__(self):
        self.n_labels = len(np.unique(self.y))
        # Calculate the average cosine distance between samples
        self.distances_between_samples = cosine_distances(self.X, self.X)
        # self.allow_fcm = True

    def get_clusters_by_centroids(self, centroids):
        dist_matrix = distance_matrix(self.X, centroids)
        assigned_clusters = np.argmin(dist_matrix, axis=1)

        clusters = self.fix_cluster_sequence(assigned_clusters)
        return clusters

    def fix_cluster_sequence(self, assigned_clusters):
        unique_clusters = np.unique(assigned_clusters)

        cluster_correspondence = {
            cluster: fixed_cluster
            for fixed_cluster, cluster in enumerate(unique_clusters)
        }

        clusters = [cluster_correspondence[cluster]
                    for cluster in assigned_clusters]
        return np.array(clusters)

    def create_clusterer(self, algorithm: str, n_clusters: int
                         ) -> KMeans | SpectralClustering:

        if algorithm == "kmeans++":
            kmeans_pp_init = kmeans_plusplus(self.X, n_clusters=n_clusters)
            return KMeans(n_clusters, init=kmeans_pp_init[0])
        elif algorithm == "kmeans":
            return KMeans(n_clusters, init="random", random_state=42)
        else:
            return CLUSTERING_ALGORITHMS[algorithm](n_clusters)

    def select_evaluation_function(self) -> Callable:
        if self.evaluation_metric == "dbc":
            # Calculate distances between samples
            return self.get_DBC_distance()

        elif self.evaluation_metric == "ext":
            return self.get_external_score()

        elif self.evaluation_metric == "dbc_ext":
            return self.get_DBC_external(self.get_DBC_distance(), self.get_external_score())

        elif self.evaluation_metric == "silhouette":
            return self.get_silhouette()

        else: # DBC combined with silhouette
            return self.get_DBC_silhouette(self.get_DBC_distance(), self.get_silhouette())

    def add_class_samples_in_cluster(self, clusters):
        classes = np.unique(self.y)
        new_X = self.X
        new_y = self.y

        new_clusters = clusters
        for c in np.unique(clusters):
            idx_clusters = np.where(clusters == c)[0]
            X_cluster = self.X[idx_clusters]
            y_cluster = self.y[idx_clusters]

            for lbl in classes:
                # If a cluster has more than 0, but less than
                # 10 samples of a class, resample it until it gets to 10.
                n_samples_class_cluster = np.sum(y_cluster == lbl)

                if n_samples_class_cluster > 0 and n_samples_class_cluster < 10:
                    n_samples_to_select = 10 - n_samples_class_cluster
                    idx_class_cluster = np.where(y_cluster == lbl)[0]

                    idx_selected_samples = np.random.choice(
                        idx_class_cluster, size=n_samples_to_select, replace=True
                    )
                    X_class = X_cluster[idx_selected_samples]
                    y_class = y_cluster[idx_selected_samples]

                    new_X = np.vstack((new_X, X_class))
                    new_y = np.hstack((new_y, y_class))

                    new_clusters = np.hstack(
                        (new_clusters, np.full(n_samples_to_select, c))
                    )
        return new_clusters, new_X, new_y

    def compare_clusterers_and_select(self) -> tuple["Clusterer", NDArray]:
        """ Compare multiple different clusterers and select the best
        according to some metric.
        """
        evaluation_values = []
        possible_clusterers = []
        candidate_clusters = []

        best_X, best_y = self.X, self.y

        n_samples = self.X.shape[0]
        max_clusters = int(np.sqrt(n_samples) / 2)

        for clustering_algorithm in CLUSTERING_ALGORITHMS:
            if clustering_algorithm == "fcm" and not(self.allow_fcm):
                continue

            for c in range(2, max_clusters):
                # dbscan = DBSCAN(eps=3, min_samples=2)

                # clusters = dbscan.fit_predict(self.X)
                # idx_clusters = np.where(clusters > -1)[0]
                # c = len(np.unique(clusters[idx_clusters]))

                clusterer = self.create_clusterer(clustering_algorithm, c)
                clusters = clusterer.fit_predict(self.X)

                # Merge clusters with very few instances
                clusters = self.merge_small_clusters(clusters)
                clusters = self.fix_cluster_sequence(clusters)

                n_clusters = len(np.unique(clusters))

                # We need to pass n_clusters as a param instead of c, because the number
                # of clusters might change for Fuzzy C-means.
                evaluation_value = self.evaluation_function(clusters, n_clusters)
                #print(clustering_algorithm, c)
                #print(evaluation_value)

                evaluation_values.append(evaluation_value)
                possible_clusterers.append((clustering_algorithm, clusterer))
                candidate_clusters.append(clusters)

        # Select the best clusterer according to an evaluation value
        idx_best_clusterer = np.argmax(evaluation_values)

        self.clustering_algorithm = possible_clusterers[idx_best_clusterer][0]
        self.best_evaluation_value = evaluation_values[idx_best_clusterer]
        best_clusters = candidate_clusters[idx_best_clusterer]

        clusterer = possible_clusterers[idx_best_clusterer][1]

        # Update samples with the new generated ones
        best_clusters, self.X, self.y = \
            self.add_class_samples_in_cluster(best_clusters
        )
        self.distances_between_samples = cosine_distances(self.X, self.X)

        print("Best clusterer:", possible_clusterers[idx_best_clusterer])
        print("Best evaluation:", self.best_evaluation_value)

        return clusterer, best_clusters

    def merge_small_clusters(self, clusters: NDArray):
        # Choose to merge this cluster with the one that increases the clustering metric
        n_clusters = len(np.unique(clusters))
        n_clusters_non_zero = n_clusters

        indexes_clusters = [
            np.where(clusters == c)[0] for c in range(n_clusters)
        ]
        empty_clusters = []

        for current_c in range(n_clusters):

            if len(indexes_clusters[current_c]) < 10:
                best_eval = float("-inf")
                best_c = current_c  # (current_c + 1) % n_clusters
                best_clusters = clusters

                # Try different clusters combinations and select
                # the one with best metric
                # print("Problema no", current_c)

                n_clusters_non_zero -= 1

                for c in range(n_clusters):
                    # print("current_c:", current_c, " - c:", c)
                    if current_c == c or c in empty_clusters:
                        continue
                    copy_clusters = np.copy(clusters)

                    copy_clusters[indexes_clusters[current_c]] = c

                    eval_cluster = self.evaluation_function(
                        copy_clusters, n_clusters_non_zero)
                    # if current_c == (n_clusters - 1):
                    #     print(f"Num. samples {current_c}:", len(indexes_clusters[current_c]))
                    #     print(f"Num. samples {c}:", len(indexes_clusters[c]))
                    #     print("Empty clusters:", empty_clusters)
                    #     print("Eval cluster:", eval_cluster)
                    #     print("Best eval:", best_eval)

                    # print(eval_cluster)
                    if eval_cluster > best_eval:
                        best_clusters = copy_clusters
                        best_eval = eval_cluster
                        best_c = c

                clusters = best_clusters
                indexes_clusters[best_c] = np.hstack(
                    (indexes_clusters[current_c], indexes_clusters[best_c])
                )

                indexes_clusters[current_c] = np.array([])

                empty_clusters.append(current_c)

                # print(best_c)
        # print([len(idx_cl) for idx_cl in indexes_clusters])
        # print(np.unique(clusters))

        # print("*******************************************")
        return clusters


    def cluster_data(self):
        """ Try different clustering algorithms and select the best one
        for the given dataset.
        """
        if self.n_clusters != "compare" and type(self.n_clusters) != int:
            print("Error. Invalid n_clusters value.")
            sys.exit(1)

        # Select the clustering evaluation function
        self.evaluation_function = self.select_evaluation_function()

        # If the number of clusters is 'compare', select the optimal
        # number of clusters and the best clustering algorithm
        if self.n_clusters == "compare":
            self.best_clusterer, clusters = self.compare_clusterers_and_select()

            #if isinstance(self.best_clusterer, SpectralClustering):
            #    clusters = self.best_clusterer.fit_predict(self.X)
            #else:
            #    clusters = self.best_clusterer.predict(self.X)

            #clusters = self.merge_small_clusters(clusters)
            #clusters = self.fix_cluster_sequence(clusters)

            self.n_clusters = len(np.unique(clusters))
            print(self.evaluation_function(clusters, self.n_clusters))

        else:
            # self.clustering_algorithm is kmeans++ as default
            self.best_clusterer = self.create_clusterer(
                self.clustering_algorithm, self.n_clusters)

            clusters = self.best_clusterer.fit_predict(self.X)
            self.best_evaluation_value = self.evaluation_function(
                    clusters, self.best_clusterer.n_clusters)

            # Change the number of clusters to the optimal number found
            self.n_clusters = int(self.best_clusterer.n_clusters)

        # Define the centroids for the best clusterer
        # If it's Spectral Clustering, the centroids need to be calculated.
        if isinstance(self.best_clusterer, SpectralClustering):
            self.calc_centroids(self.X, clusters, self.n_clusters)
        else:
            self.centroids = self.best_clusterer.cluster_centers_

        # Split the samples according to the cluster they are assigned
        return self.create_clusters_dict(clusters)

    def create_clusters_dict(self, clusters):
        # Split the samples according to the cluster they are assigned
        samples_by_cluster = {}
        labels_by_cluster = {}

        for c in range(self.n_clusters):
            indexes_c = np.where(clusters == c)[0]
            samples_by_cluster[c] = self.X[indexes_c]
            labels_by_cluster[c] = self.y[indexes_c]

        return samples_by_cluster, labels_by_cluster

    def calc_centroids(self, X, clusters, n_clusters):
        self.centroids = np.array([X[clusters == c].mean(axis=0)
                                   for c in range(n_clusters)])

    def update_clusters_and_centroids(
            self, X_by_cluster, y_by_cluster):

        for c in X_by_cluster:
            self.centroids[c] = np.mean(X_by_cluster[c], axis=0)

        X_by_cluster = [X_cluster for X_cluster in iter(X_by_cluster.values())]
        y_by_cluster = [y_cluster for y_cluster in iter(y_by_cluster.values())]

        self.X = np.vstack(X_by_cluster)
        self.y = np.hstack(y_by_cluster)

    def get_silhouette(self) -> Callable[[NDArray, int], float]:
        def wrapper(clusters: NDArray[np.int32], _: Optional[int]) -> float:
            if np.all(clusters == clusters[0]):
                return -1
            return silhouette_score(self.X, clusters)
        return wrapper

    def get_external_score(self) -> Callable[[NDArray, int], float]:
        def wrapper(clusters: NDArray[np.int32], _: Optional[int]) -> float:
            if np.all(clusters == clusters[0]):
                return 0
            rand = adjusted_rand_score(self.y, clusters)
            mi =  normalized_mutual_info_score(self.y, clusters)
            vm = v_measure_score(self.y, clusters)
            fm = fowlkes_mallows_score(self.y, clusters)

            avg_external = float(rand + mi + vm + fm) / 4
            return avg_external
        return wrapper

    def get_DBC_external(self, func_DBC: Callable, func_external_score: Callable
                           ) -> Callable[[NDArray, int], float]:
        def wrapper(clusters: NDArray[np.int32], n_clusters: int) -> float:
            w1 = self.weights_dbc_external[0]
            w2 = self.weights_dbc_external[1]

            dbc = func_DBC(clusters, n_clusters)
            external_score_val = func_external_score(clusters, n_clusters)

            # print("DBC:", dbc)
            # print("External:", external_score_val)
            return w1 * dbc + w2 * external_score_val
        return wrapper

    def get_DBC_silhouette(self, func_DBC: Callable, func_silhouette: Callable
                           ) -> Callable[[NDArray, int], float]:

        def wrapper(clusters: NDArray[np.int32], n_clusters: int) -> float:
            w1 = self.weights_dbc_external[0]
            w2 = self.weights_dbc_external[1]

            dbc = func_DBC(clusters, n_clusters)
            silhouette_val = func_silhouette(clusters, n_clusters)
            return w1 * dbc + w2 * silhouette_val
        return wrapper

    def get_DBC_distance(self) -> Callable[[NDArray, int], float]:
        """ The DBC distance measures the average cosine distance between
        samples of different classes in clusters.
        """
        def wrapper(clusters: NDArray[np.int32], n_clusters: int) -> float:
            if np.all(clusters == clusters[0]):
                return 0

            dists_centroids_cluster = np.empty((n_clusters))

            for c in range(n_clusters):
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
                        avg_distance_cluster += \
                            self.distances_between_samples[samples_lbl1, :][:, samples_lbl2].sum()

                if n_distances > 0:
                    dists_centroids_cluster[c] = avg_distance_cluster / n_distances
                else:
                    # If all samples are in the same cluster, it returns 1
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

        u_membership = np.empty((n_samples, n_clusters))

        for k in range(n_clusters):
            u_membership[:, k] = np.sum([
                dists_samples_centroids[:, k] / (dists_samples_centroids[:, c] + 0.00000001)
                for c in range(n_clusters)
            ], axis=0)
        u_membership = 1 / (u_membership + 1e-8)
        u_membership[u_membership > 1] = 1
        return u_membership

if __name__ == "__main__":
    fcm = FuzzyCMeans(n_clusters=3)
    import dataset_loader as dl
    X, y = dl.select_dataset_function('german_credit')()
    clusters = fcm.fit_predict(X)
    print(clusters)
