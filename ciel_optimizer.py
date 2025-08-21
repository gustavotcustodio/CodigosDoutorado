import sys
import warnings
from numpy.typing import NDArray
import numpy as np
import math
import time
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch, KMeans, MeanShift, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from sklearn.svm import SVC
import dataset_loader
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from utils.clusters import fix_predict_prob

N_FOLDS = 10

# (eps = 0.3, k = 2, damping = 0.5, bandwidth = 0.1)
# (eps = 0.5, k = 2, damping = 0.5, bandwidth = auto)
# Agglomerative, MS, Birch, WHC, MB, Kmeans, DBScan(eps0.3), SpectralClustering

def create_clusterer(clusterer_name: str, n_clusters: int):
    if clusterer_name == 'kmeans':
        return KMeans(n_clusters=n_clusters, init='random', random_state=42)
    elif clusterer_name == 'mini_batch_kmeans':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    elif clusterer_name == 'mean_shift':
        return MeanShift()
    elif clusterer_name == 'dbscan':
        return DBSCAN(eps=0.3)
    elif clusterer_name == 'birch':
        return Birch(n_clusters=n_clusters, threshold=0.5)
    elif clusterer_name == 'spectral_clustering':
        return SpectralClustering(n_clusters=n_clusters, random_state=42)
    elif clusterer_name == 'agglomerative_clustering':
        return AgglomerativeClustering(n_clusters=n_clusters)
    elif clusterer_name == 'affinity_propagation':
        return AffinityPropagation(damping=0.5, random_state=42)
    else:
        return KMeans(n_clusters, random_state=42)


class CielOptimizer:
    def __init__(self,
                 best_clusterer: str,
                 best_classifier: str,
                 n_clusters: int,
                 classifiers_params=None,
                 weights=None,
                 combination_strategy='dynamic_weighted_prob'):

        self.best_classifier = best_classifier
        self.n_clusters = n_clusters
        self.best_clusterer = best_clusterer
        # warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.classifiers_params = classifiers_params
        self.combination_strategy = combination_strategy

        # If there are no defined weights, use random weights
        if weights is None:
            self.weights = np.random.random(n_clusters).astype(np.float32)
            self.weights = self.weights / self.weights.sum()

        else:
            self.weights = weights

    def create_classifier(self, classifier_name: str, cluster=None):
        # return SVC(probability=True)
        if classifier_name == 'svm':
            if self.classifiers_params is None or cluster is None:
                return SVC(probability=True)
            else:
                return SVC(
                    probability=True,
                    C=self.classifiers_params[cluster]['cost'],
                    gamma=self.classifiers_params[cluster]['gamma'])

        elif classifier_name == 'extra_tree':
            if self.classifiers_params is None or cluster is None:
                return ExtraTreesClassifier()
            else:
                clf_params = self.classifiers_params
                return ExtraTreesClassifier(
                    n_estimators=clf_params[cluster]["n_estimators"],
                    max_depth=clf_params[cluster]['max_depth'],
                    min_samples_split=clf_params[cluster]['min_samples_split'],
                    min_samples_leaf=clf_params[cluster]['min_samples_leaf']
                )
        elif classifier_name == 'gb':
            if self.classifiers_params is None or cluster is None:
                return GradientBoostingClassifier()
            else:
                clf_params = self.classifiers_params
                return GradientBoostingClassifier(
                    n_estimators=clf_params[cluster]["n_estimators"],
                    max_depth=clf_params[cluster]['max_depth'],
                    min_samples_split=clf_params[cluster]['min_samples_split'],
                    min_samples_leaf=clf_params[cluster]['min_samples_leaf'],
                    learning_rate=clf_params[cluster]['learning_rate'])
        else:
            print(f"Error: invalid base classifier: {classifier_name}")
            sys.exit(1)

    def train_classifiers(self, samples_by_cluster, labels_by_cluster):
        self.classifiers = []

        for c in range(self.n_clusters):
            if np.all(labels_by_cluster[c] == labels_by_cluster[c][0]):
                clf = DummyClassifier(strategy="most_frequent")
            else:
                clf = self.create_classifier(self.best_classifier, c)

            y_cluster = labels_by_cluster[c]
            possible_classes = np.unique(y_cluster)
            if len(possible_classes) > 2:
                clf = OneVsRestClassifier(clf)

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

    def split_clusters_in_lists(self, clusters, X, y):
        # Split the samples according to the cluster they are assigned
        samples_by_cluster = []
        labels_by_cluster = []

        for c in range(self.n_clusters):
            indexes_c = np.where(clusters == c)[0]
            samples_by_cluster.append( X[indexes_c] )
            labels_by_cluster.append( y[indexes_c] )

        return samples_by_cluster, labels_by_cluster

    def get_y_uniform_clusters(self, X_test):
        # Perform a previous prediction seeing the distance between
        # samples and centroids. Give 100% prob
        # Cluster the test samples. This is done to check if any sample
        # belongs in a very well defined cluster.
        n_clusters = self.n_clusters
        y_pred_clustering = np.zeros(X_test.shape[0]).astype(int) - 1

        clusters = self.clusterer.predict(X_test)

        get_classes_in_cluster = lambda c: np.unique(self.labels_by_cluster[c])
        unique_classes_in_clusters = [
            get_classes_in_cluster(c) for c in range(n_clusters)
        ]
        for c in range(n_clusters):
            if len(unique_classes_in_clusters[c]) < 2:
                class_cluster = unique_classes_in_clusters[c][0]

                idx_unique_cluster = np.where(clusters == c)[0]
                y_pred_clustering[idx_unique_cluster] = class_cluster

        # y_pred_clustering = [0, 0, 0, 1, -1, -1] # if -1, the
        return y_pred_clustering

    def fit(self, X, y):

        self.n_classes = len(np.unique(y))

        self.clusterer = create_clusterer(self.best_clusterer, self.n_clusters)
        clusters = self.clusterer.fit_predict(X)

        # samples_by_cluster, self.labels_by_cluster = \
        #         cluster_samples(X, y, optimal_clusterer)
        samples_by_cluster, self.labels_by_cluster = \
                self.split_clusters_in_lists(clusters, X, y)
        # print("Num. clusters:", len(samples_by_cluster))

        # Generate and train classifiers
        self.train_classifiers(
            samples_by_cluster, self.labels_by_cluster
        )

    def predict_labels_by_cluster(self, X) -> NDArray:
        y_pred_by_clusters = []

        for c, classifier in enumerate(self.classifiers):
            #offset = min(self.labels_by_cluster[c])
            y_pred_cluster = classifier.predict(X) #+ offset
            y_pred_by_clusters.append(y_pred_cluster)

        if np.any(self.y_clustering >= 0):
            samples_unique_cluster = np.where(self.y_clustering >= 0)[0]
            labels = self.y_clustering[samples_unique_cluster]
        return np.array(y_pred_by_clusters).T

    def predict_proba(self, X):
        probability_by_class = np.zeros((len(X), self.n_classes))

        self.y_clustering = self.get_y_uniform_clusters(X)

        # Dynamic weighted probability combination strategy for the final classification results;
        for c, classifier in enumerate(self.classifiers):
            if isinstance(classifier, DummyClassifier):
                continue

            predicted_probs = classifier.predict_proba(X)

            #print(predicted_probs)
            predicted_probs = fix_predict_prob(
                predicted_probs, self.labels_by_cluster[c], self.n_classes)

            #print(self.labels_by_cluster[c])
            #print(predicted_probs)
            #print("---------------------------")
            # # If the classifier was not trained with instances from some classes, add
            # # columns with zeros in the predicted_probs for the missing classes
            # if len(classifier.classes_) < self.n_classes:
            #     missing_labels = [label for label in range(self.n_classes)
            #                       if label not in classifier.classes_]

            #     for lbl in missing_labels:
            #         col_zeros = np.zeros((X.shape[0], 1))
            #         predicted_probs = np.hstack(
            #             (predicted_probs[:, :lbl], col_zeros, predicted_probs[:, lbl:])
            #         )
            probability_by_class += predicted_probs * self.weights[c]

        if np.any(self.y_clustering >= 0):
            samples_unique_cluster = np.where(self.y_clustering >= 0)[0]
            labels = self.y_clustering[samples_unique_cluster]

            probability_by_class[samples_unique_cluster, :] = 0
            probability_by_class[samples_unique_cluster, labels] = 1.0
        
        p_class = probability_by_class
        probability_by_class = p_class / p_class.sum(axis=1)[:, np.newaxis]
        weights = np.tile(self.weights, (len(X), 1))

        y_pred_by_cluster = self.predict_labels_by_cluster(X)
        return probability_by_class, weights, y_pred_by_cluster

    def predict(self, X):
        probabilities, weights, y_pred_by_cluster = self.predict_proba(X)

        return np.argmax(probabilities, axis=1), weights, y_pred_by_cluster


def main():
    X, y = dataset_loader.select_dataset_function('german_credit')()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TODO colocar o kmeans e outro classificador
    ciel_opt = CielOptimizer(n_clusters=7)
    ciel_opt.fit(X_train, y_train)
    y_pred, _, _ = ciel_opt.predict(X_test)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
