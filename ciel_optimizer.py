from numpy.typing import NDArray
import numpy as np
import math
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch, KMeans, MeanShift, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from sklearn.svm import SVC
import dataset_loader
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

N_FOLDS = 10

# (eps = 0.3, k = 2, damping = 0.5, bandwidth = 0.1)
# (eps = 0.5, k = 2, damping = 0.5, bandwidth = auto)
# Agglomerative, MS, Birch, WHC, MB, Kmeans, DBScan(eps0.3), SpectralClustering

POSSIBLE_CLUSTERERS = [
    'kmeans',
    'kmeans++',
    'mini_batch_kmeans',
    # 'mean_shift',
    # 'dbscan',
    'birch',
    'spectral_clustering',
    'agglomerative_clustering',
    # 'affinity_propagation'
]

BASE_CLASSIFIERS = [
    'gb',
    'extra_tree'
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
    def __init__(self, n_clusters: int, svm_params=None,
                 extra_tree_params=None, grad_boost_params=None, weights=None):
        self.n_clusters = n_clusters

        self.svm_params = svm_params
        self.extra_tree_params = extra_tree_params
        self.grad_boost_params = grad_boost_params
        
        # If there are no defined weights, use random weights
        if weights is None:
            self.weights = np.random.random(n_clusters).astype(np.float32)
            self.weights = self.weights / self.weights.sum()

    def create_clusterer(self, clusterer_name: str):
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

    def create_classifier(self, classifier_name, params=None):
        if classifier_name == 'svm':
            if params is None:
                return SVC()
            else:
                return SVC()

        elif classifier_name == 'extra_tree':
            if params is None:
                return ExtraTreesClassifier()
            else:
                return ExtraTreesClassifier()

        else: #classifier_name == 'gb' 
            if params is None:
                return GradientBoostingClassifier()
            else:
                return GradientBoostingClassifier()

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
                metric_value = math.inf if metric_name == 'davies_bouldin' else -math.inf
            else:
                metric_value = metric_func(X_val, clusters_pred)

            internal_metrics_evals[metric_name] = metric_value

        return external_metrics_evals, internal_metrics_evals

    def internal_breaks_tie(self, clustering_metrics, best_clustering_metrics):
        """ If there is a tie in the external metrics, break the tie
        using the internal metrics. """
        # Number of internal metrics improved compared to the best solution
        n_internal_improved = 0

        for metric, value_metric in clustering_metrics['internal'].items():
            # Current best value of this metric
            best_value_metric = best_clustering_metrics['internal'][metric]

            # Davies Bouldin is a special case, because lower values are better values
            if metric == 'davies_bouldin' and value_metric < best_value_metric:
                n_internal_improved += 1

            elif metric != 'davies_bouldin' and value_metric > best_value_metric:
                n_internal_improved += 1

        # if more than half of the internal metrics are improved, the new clusterer
        # is the new best
        if n_internal_improved >= 2: 
            return True
        return False


    def update_best_clusterer(self, clustering_metrics, best_clustering_metrics) -> bool:
        if not best_clustering_metrics:
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()
            return True

        n_external_improved = 0

        for metric, value_metric in clustering_metrics['external'].items():
            best_value_metric = best_clustering_metrics['external'][metric]

            if value_metric > best_value_metric:
                n_external_improved += 1

        # Tie break with internal metrics if external metric are a draw
        if n_external_improved >= 3 or (
                n_external_improved == 2 and self.internal_breaks_tie(clustering_metrics, best_clustering_metrics)
            ):
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()

            return True
        return False


    def select_optimal_clustering_algorithm(self, X: NDArray, y: NDArray):
        best_clusterer = 'kmeans'
        best_clustering_metrics = {}

        # External indicators are the main ones
        for clusterer_name in POSSIBLE_CLUSTERERS:

            clustering_metrics = {}

            clusterer = self.create_clusterer(clusterer_name)
            clusters = clusterer.fit_predict(X)
            external_metrics_evals, internal_metrics_evals = \
                    self.calc_metrics_clustering(clusters, X, y)

            clustering_metrics['external'] = external_metrics_evals
            clustering_metrics['internal'] = internal_metrics_evals

            clusterer_updated = self.update_best_clusterer(clustering_metrics, best_clustering_metrics)

            if clusterer_updated:
                best_clusterer = clusterer_name
        
        print(f'Selected clusterer: {best_clusterer}')
        return best_clusterer

    def crossval_classifiers_scores(self, classifiers: dict, X_train: NDArray, y_train: NDArray):
        if self.n_labels > 2:
            classification_metrics = ['roc_auc_ovo', 'accuracy']
        else:
            classification_metrics = ['roc_auc', 'accuracy']

        auc_by_classifier = {}

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

        for clf_name, classifier in classifiers.items():

            cv_results = cross_validate(classifier, X_train, y_train, cv=cv,
                                        scoring=classification_metrics)
            # Get the mean AUC of the classifier
            if 'test_roc_auc_ovo' in cv_results:
                mean_auc = cv_results['test_roc_auc_ovo'].mean()
            else:
                mean_auc = cv_results['test_roc_auc'].mean()

            auc_by_classifier[clf_name] = mean_auc
        # Return a dict with the format classifier_name -> mean_auc 
        return auc_by_classifier

    def select_optimal_classifier(self, X, y):
        ''' Choose the best classifier according to the average AUC score''' 
        classifiers = {
            clf_name: self.create_classifier(clf_name) for clf_name in BASE_CLASSIFIERS
        }
            
        auc_by_classifier = self.crossval_classifiers_scores(classifiers, X, y)
        selected_classifier = max(auc_by_classifier, key=auc_by_classifier.get)

        print(f'Selected classifier: {selected_classifier}')
        return selected_classifier

    def train_classifiers(self, samples_by_cluster, labels_by_cluster, best_classifier):
        self.classifiers = []

        for c in range(self.n_clusters):
            clf = self.create_classifier(best_classifier)
            clf.fit(samples_by_cluster[c], labels_by_cluster[c])
            self.classifiers.append( clf )

    def cluster_samples(self, X, y, optimal_clusterer):
        # Use the optimal clustering algorithm to divide the training set into clusters
        clusters = optimal_clusterer.fit_predict(X)

        # Split the samples according to the cluster they are assigned
        samples_by_cluster = []
        labels_by_cluster = []

        for c in range(self.n_clusters):
            indexes_c = np.where(clusters == c)[0]
            samples_by_cluster.append( X[indexes_c] )
            labels_by_cluster.append( y[indexes_c] )

        return samples_by_cluster, labels_by_cluster

    def fit(self, X, y):
        # TODO adicionar inputs de parâmetros do PSO
        self.n_labels = len(np.unique(y))

        print('Selecting best clustering algorithm...')
        # Traverse all clustering algorithms and select the optimal one
        name_best_clusterer = self.select_optimal_clustering_algorithm(X, y)

        print('Selecting best classifier...')
        # Traverse all classification algorithms to select the optimal one
        name_best_classifier = self.select_optimal_classifier(X, y)

        optimal_clusterer = self.create_clusterer(name_best_clusterer)
        samples_by_cluster, labels_by_cluster = self.cluster_samples(X, y, optimal_clusterer)

        # Generate and train classifiers
        self.train_classifiers(
                samples_by_cluster, labels_by_cluster, name_best_classifier)


    def predict(self, X):
        self.weights = np.tile(self.weights, (len(X), 1))

        voting_weights = np.zeros((len(X), self.n_labels))
        rows = range(len(X))

        # Dynamic weighted probability combination strategy for the final classification results;
        for c, classifier in enumerate(self.classifiers):
            y_pred_cluster = classifier.predict(X)

            voting_weights[rows, y_pred_cluster] += self.weights[:, c]

        return np.argmax(voting_weights, axis=1)


def main():
    X, y = dataset_loader.select_dataset_function('german_credit')()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ciel_opt = CielOptimizer(n_clusters=7)
    ciel_opt.fit(X_train, y_train)
    y_pred = ciel_opt.predict(X_test)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
