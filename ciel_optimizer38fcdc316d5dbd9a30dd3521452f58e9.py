from numpy.typing import NDArray
import numpy as np
import math
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch, KMeans, MeanShift, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from sklearn.model_selection import train_test_split
import dataset_loader
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# (eps = 0.3, k = 2, damping = 0.5, bandwidth = 0.1)
# (eps = 0.5, k = 2, damping = 0.5, bandwidth = auto)
# Agglomerative, MS, Birch, WHC, MB, Kmeans, DBScan(eps0.3), SpectralClustering

POSSIBLE_CLUSTERERS = [
    'kmeans',
    'kmeans++',
    'mini_batch_kmeans',
    # 'mean_shift',
    'dbscan',
    'birch',
    'spectral_clustering',
    'agglomerative_clustering',
    'affinity_propagation'
]

external_metrics = {
    'adjusted_rand_score': adjusted_rand_score,
    'normalized_mutual_info_score': normalized_mutual_info_score,
    'v_measure_score': v_measure_score,
    'fowlkes_mallows_score': fowlkes_mallows_score,
}

internal_metrics = {
    'silhouette': silhouette_score,
    'davies_bouldin': davies_bouldin_score, # Min
    'calinski_harabasz_score': calinski_harabasz_score
}

class CielOptimizer:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def generate_clusterer(self, clusterer_name: str):
        if clusterer_name == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, init='random')
        elif clusterer_name == 'mini_batch_kmeans':
            return MiniBatchKMeans(n_clusters=self.n_clusters)
        elif clusterer_name == 'mean_shift':
            return MeanShift()
        elif clusterer_name == 'dbscan':
            return DBSCAN(eps=0.3)
        elif clusterer_name == 'birch':
            return Birch(n_clusters=self.n_clusters)
        elif clusterer_name == 'spectral_clustering':
            return SpectralClustering(n_clusters=self.n_clusters)
        elif clusterer_name == 'agglomerative_clustering':
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        elif clusterer_name == 'affinity_propagation':
            return AffinityPropagation(damping=0.5)
        else:
            return KMeans(self.n_clusters, random_state=42)


    def calc_metrics_clustering(self, clusters_pred: NDArray,
                                X_val: NDArray, y_val: NDArray) -> tuple[dict, dict]:
        # Dictionary with number of victories per metric
        external_metrics_evals = {}
        internal_metrics_evals = {}

        # External metrics
        for metric_name, metric_func in external_metrics.items():
            metric_value = metric_func(y_val, clusters_pred)

            external_metrics_evals[metric_name] = metric_value

        # Internal metrics
        for metric_name, metric_func in internal_metrics.items():
            if np.all(clusters_pred == clusters_pred[0]):
                metric_value = math.inf if metric_name == "davies_bouldin" else -math.inf
            else:
                metric_value = metric_func(X_val, clusters_pred)

            internal_metrics_evals[metric_name] = metric_value

        return external_metrics_evals, internal_metrics_evals

    def update_best_clusterer(self, clustering_metrics, best_clustering_metrics):
        if not best_clustering_metrics:
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()
            return

        n_external_improved = 0

        for metric, value_metric in clustering_metrics["external"].items():
            best_value_metric = best_clustering_metrics["external"][metric]

            if value_metric > best_value_metric:
                n_external_improved += 1

        # Tie break with internal metrics if external metric are a draw
        if n_external_improved >= 3 or (n_external_improved == 2 and self.internal_breaks_tie()):
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()
            return

        n_internal_improved = 0

        for metric, value_metric in clustering_metrics["internal"].items():
            best_value_metric = best_clustering_metrics["internal"][metric]

            if metric == "davies_bouldin" and value_metric < best_value_metric:
                n_internal_improved += 1

            elif metric != "davies_bouldin" and value_metric > best_value_metric:
                n_internal_improved += 1

        if n_internal_improved >= 2: 
            return True
        return False

    def select_optimal_clustering_algorithm(self, X: NDArray, y: NDArray):
        best_clustering_metrics = {}

        # External indicators are the main ones
        for clusterer_name in POSSIBLE_CLUSTERERS:

            clustering_metrics = {}

            clusterer = self.generate_clusterer(clusterer_name)
            clusters = clusterer.fit_predict(X)
            print(clusterer_name)
            external_metrics_evals, internal_metrics_evals = \
                    self.calc_metrics_clustering(clusters, X, y)

            clustering_metrics["external"] = external_metrics_evals
            clustering_metrics["internal"] = internal_metrics_evals
            self.update_best_clusterer(clustering_metrics, best_clustering_metrics)

            print(best_clustering_metrics)

    def select_optimal_classifier(self, X, y):
        pass

    def fit(self, X, y):
        # Traverse all clustering algorithms and select the optimal one
        optimal_clustering_algorithm = self.select_optimal_clustering_algorithm(X, y)

        # Traverse all classification algorithms to select the optimal one
        # optimal_classifier = self.select_optimal_clustering_algorithm(X, y)

        # Use the optimal clustering algorithm  cluster to divide the training set into clusters

        # Generate classifiers

        # Dynamic weighted probability combination strategy for the final classification results;

        # Multi-parameter optimization based on improved swarm intelligence algorithm;

def main():
    X, y = dataset_loader.select_dataset_function("german_credit")()
    ciel_opt = CielOptimizer(n_clusters=3)
    ciel_opt.fit(X, y)

if __name__ == "__main__":
    main()
