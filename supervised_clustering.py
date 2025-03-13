import sys
import numpy as np
from sklearn.cluster import KMeans
from numpy.typing import NDArray

class SupervisedClustering:
    """Implementation of the Supervised Clustering Method proposed in
    Ensemble classification based on supervised clustering for credit scoring (2016)."""

    M: int  # Number of closest samples to evaluate the voting weight
    max_threads: int = 4
    verbose: bool = False

    def cluster_supervised_data(
        self, label:int, X_label: NDArray, y_label: NDArray, n_clusters: int
    ) -> NDArray:
        """ Select data with specific label and cluster it. """
        idxs_label = np.where(y_label == label)[0]

        X_label, y_label = X_label[idxs_label], y_label[idxs_label]

        # self.clusters_by_label[label] = clusters
        clusterer = KMeans(n_clusters, n_init='auto')
        clusters_label = clusterer.fit_predict(X_label)
        return clusters_label

    # Combine clusters through pairwise combination
    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        n_labels = len(np.unique(y))

        self.clusters_by_label = [] * n_labels
        n_clusters_by_label = [] * n_labels

        # For each label...
        # idxs_label = np.where(y_label == label)[0]

        # X_label, y_label = X_label[idxs_label], y_label[idxs_label]

        # Perform clusters pairwise combination
        # samples_by_cluster = self.cluster_data(X)

        # clustered_data_by_label[0]
        pass 


