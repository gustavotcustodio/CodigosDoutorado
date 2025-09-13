import sys
import argparse
from logger import Logger
import numpy as np
import dataset_loader
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from numpy.typing import NDArray
from dataset_loader import normalize_data
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from logger import PredictionResults
from sklearn.multiclass import OneVsRestClassifier

N_FOLDS = 10

@dataclass
class SupervisedClustering:
    """Implementation of the Supervised Clustering Method proposed in
    Ensemble classification based on supervised clustering for credit scoring (2016)."""

    base_classifier: str # Base classifier used
    M: int = 10  # Number of closest samples to evaluate the voting weight
    max_threads: int = 4
    verbose: bool = False

    def get_intra_inter_cluster_distance(
        self, X_class: NDArray, clusters_k: NDArray, centroids_k: NDArray, n_clusters: int
    ):
        """ Calculate the intra / inter cluster distance in order to evaluate
            clustering results.
        """
        intra_dists = np.empty(n_clusters)

        for c in range(n_clusters):
            X_cluster = X_class[np.where(clusters_k == c)[0]]
            n_samples = X_cluster.shape[0]

            if len(X_cluster) < 2:
                intra_dists[c] = 1
            else:
                # Distance between all samples and centroids for the current cluster
                dists_samples_center = np.linalg.norm(X_cluster - centroids_k[c], axis=0)**2
                max_dist = np.max(dists_samples_center)

                # Avoid the 0 in denominator
                if round(max_dist, 5) == 0:
                    intra_dists[c] = 0
                else:
                    # Distance between samples in the same cluster
                    intra_dists[c] = np.sum(dists_samples_center) / (max_dist * n_samples)

        intra = np.sum(intra_dists) / n_clusters
        # Average distance between centroids
        avg_dist_centroids = 2 * distance.pdist(centroids_k).sum() / (
            n_clusters * (n_clusters - 1))

        mean_centroids = np.mean(centroids_k, axis=0)
        beta = np.linalg.norm(centroids_k - mean_centroids)**2 / n_clusters

        inter = np.exp(-avg_dist_centroids / beta)
        return intra * inter

    def cluster_data(
        self, X_label: NDArray, n_clusters: int
    ) -> tuple[NDArray, NDArray]:

        clusterer = KMeans(n_clusters, n_init='auto')
        clusters_label = clusterer.fit_predict(X_label)
        return clusters_label, clusterer.cluster_centers_

    def _next_combination(self, selected_clusters_by_class: list) -> None:
        """ Get next permutation of clusters combination.
        """
        for lbl in range(len(selected_clusters_by_class)):
            selected_clusters_by_class[lbl] += 1

            if selected_clusters_by_class[lbl] < self.n_clusters_by_class[lbl]:
                break
            else:
                selected_clusters_by_class[lbl] = 0

    # Combine clusters through pairwise combination
    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        self.n_classes = len(np.unique(y))

        self.samples_by_class = self.divide_samples_by_class(self.n_classes, X, y)
        self.clusters_by_class, self.n_clusters_by_class = self.perform_supervised_clustering(self.n_classes)

        # Perform the permutation combining the clusters from different classes
        self.pairwise_combination_clusters()

        self.train_base_classifiers()


    def pairwise_combination_clusters(self):
        """ Combine single clusters with different labels.
            All the possible clusters with different labels are created.
            Example:

            clusters lbl 1: [1, 2, 3]
            clusters lbl 2: [4, 5]

            Result: [(1,4), (1,5), (2,4), (2,5), (3,4), (3,5)] """
        # Clusters selected for the partition result of each different label
        selected_clusters_by_class = [0] * self.n_classes

        # Total number of permutations for combination of clusters
        total_clusters = np.prod(self.n_clusters_by_class)

        # Stores all the samples that will belong to this permutation of clusters
        final_samples = []
        final_clusters = []
        final_labels = []
        centroids = []

        samples_in_cluster = []
        self.labels_by_cluster = []

        # After the pairwise combination, samples are replicated.
        # THis loop get the new samples, their labels and their clusters numbers
        for current_cluster in range(total_clusters):

            samples_in_cluster = []
            self.labels_by_cluster.append( [] )

            for lbl in range(self.n_classes):
                X_class = self.samples_by_class[lbl]

                # Selected cluster for samples of this class
                selected_cluster = selected_clusters_by_class[lbl]

                # Samples of the current class that have belong to the selected cluster
                indexes_cluster = np.where(self.clusters_by_class[lbl] == selected_cluster)[0]
                samples_in_cluster.append( X_class[indexes_cluster] )

                # Final labels is the new y of the dataset
                self.labels_by_cluster[current_cluster] += [lbl] * len(indexes_cluster)

            samples_in_cluster = np.vstack(samples_in_cluster)

            # Final samples is the new X of the dataset
            final_samples.append( samples_in_cluster )

            # Final clusters are the cluster labels after pairwise combination
            n_samples_in_cluster = len(samples_in_cluster)
            final_clusters += [current_cluster] * n_samples_in_cluster
            final_labels += self.labels_by_cluster[current_cluster]

            centroids.append( np.mean(samples_in_cluster, axis=0) )
            self._next_combination(selected_clusters_by_class)

        self.X = np.vstack(final_samples)
        self.y = np.array(final_labels)
        self.clusters = np.array(final_clusters)
        self.centroids = np.vstack(centroids)
        self.n_clusters = total_clusters

    def set_base_classifiers(self):
        if self.base_classifier == "svm":
            self.classifiers = [SVC() for _ in range(self.n_clusters)]

        elif self.base_classifier == "dt":
            self.classifiers = [DecisionTreeClassifier() for _ in range(self.n_clusters)]

        elif self.base_classifier == "lr":
            self.classifiers = [LogisticRegression() for _ in range(self.n_clusters)]

        else:
            print("Invalid base classifier")
            sys.exit(1)

    def train_base_classifiers(self):
        self.set_base_classifiers()

        for cluster, clf in enumerate(self.classifiers):
            indexes_cluster = np.where(self.clusters == cluster)[0]
            X_cluster = self.X[indexes_cluster]
            y_cluster = self.y[indexes_cluster]

            possible_classes = np.unique(y_cluster)
            if len(possible_classes) > 2:
                clf = OneVsRestClassifier(clf)
                self.classifiers[cluster] = clf

            clf.fit(X_cluster, y_cluster)

    def combine_votes(self, y_pred_by_clusters, classifiers_weights):
        n_samples = y_pred_by_clusters.shape[1]
        vote_sums = np.zeros(( self.n_classes, n_samples ))
        idx_samples = range(n_samples)

        for c, y_pred_cluster in enumerate(y_pred_by_clusters):
            vote_sums[y_pred_cluster, idx_samples] += classifiers_weights[c, idx_samples]

        return vote_sums # , classifiers_weights.T, np.vstack(y_pred_by_clusters).T

    def predict(self, X_test: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        y_score, classifiers_weights, y_pred_by_clusters = self.predict_proba(X_test)
        y_pred = np.argmax(y_score, axis=0)

        return y_pred, classifiers_weights, y_pred_by_clusters

    def predict_proba(self, X_test: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        y_pred_by_clusters = np.zeros(( self.n_clusters, len(X_test)  ), dtype="int")
        # Get distances between samples to predict and training samples
        dists_samples = cdist(X_test, self.X)

        classifiers_weights = np.zeros(( self.n_clusters, len(X_test) ))

        for c, clf in enumerate(self.classifiers):
            y_pred_by_clusters[c] = clf.predict(X_test)

            for i in range(len(X_test)):
                # Get the M closest samples
                indexes_closest = np.argsort(dists_samples[i, :])[:self.M]
                X_selection = self.X[indexes_closest]
                y_selection = self.y[indexes_closest]

                y_pred_selection = clf.predict(X_selection)
                # Calc the accuracy for the M closest samples
                classifiers_weights[c, i] = accuracy_score(
                        y_selection, y_pred_selection)
        # Process votes 
        # y_pred = process_votes(y_pred_clusters, classifiers_weights)
        classifiers_weights += 1e-100
        classifiers_weights  = classifiers_weights / classifiers_weights.sum(axis=0)
        y_score = self.combine_votes(y_pred_by_clusters, classifiers_weights)

        self.y_pred_by_clusters = y_pred_by_clusters.T
        return y_score.T, classifiers_weights.T, self.y_pred_by_clusters

    def divide_samples_by_class(self, n_classes, X, y):
        samples_by_class = [[]] * n_classes

        for lbl in range(n_classes):
            """ Select data with specific label and cluster it. """
            indexes_class = np.where(y == lbl)[0]

            samples_by_class[lbl] = X[indexes_class]
        return samples_by_class

    def perform_supervised_clustering(self, n_classes):

        clusters_by_class = [[]] * n_classes
        n_clusters_by_class = [0] * n_classes

        for lbl in range(n_classes):
            X_class = self.samples_by_class[lbl]
            max_clusters = int(np.sqrt(len(X_class)))

            clusters_current_label = []
            clustering_evals_label = []

            for k in range(2, max_clusters):
                # Cluster data and evaluate it
                clusters_k, centroids_k = self.cluster_data(X_class, k) 

                intra_inter_dist = self.get_intra_inter_cluster_distance(
                    X_class, clusters_k, centroids_k, k
                )
                clusters_current_label.append(clusters_k)
                clustering_evals_label.append(intra_inter_dist)

            idx_best = np.argmin(clustering_evals_label)
            optimal_n_clusters = idx_best + 2

            # Best clustering result
            best_clusters_class, best_intra_inter_dist = \
                    clusters_current_label[idx_best], clustering_evals_label[idx_best]
            clusters_by_class[lbl] = best_clusters_class
            n_clusters_by_class[lbl] = optimal_n_clusters

            self.best_intra_inter_dist = best_intra_inter_dist

        return clusters_by_class, n_clusters_by_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True, help = "Dataset used.")
    parser.add_argument("-b", "--base_classifier", type=str, default='svm', help = "Values: svm, dt, lr")
    parser.add_argument("-m", "--min_mutual_info_percentage", type=float, default=100.0, help = "Mutual information value.")
    parser.add_argument("-M", type=int, default=10,
                        help = "Number of closest neighbors, used do determine the voting weight of each base classifier.")
    args = parser.parse_args()

    X, y = dataset_loader.select_dataset_function(args.dataset)()

    for fold in range(1, N_FOLDS+1):

        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        s_clf = SupervisedClustering(base_classifier=args.base_classifier, M=args.M)
        s_clf.fit(X_train, y_train)

        y_score, voting_weights, y_pred_by_clusters = s_clf.predict_proba(X_val)
        y_pred = np.argmax(y_score, axis=1)

        prediction_results = PredictionResults(
                y_pred, y_val, voting_weights, y_pred_by_clusters, y_score)
        log = Logger(s_clf, args.dataset, prediction_results)
        log.save_data_fold_supervised_clustering(fold)
        print(classification_report(y_val, y_pred))
