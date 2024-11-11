import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Mapping, Optional
from numpy.typing import NDArray
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from dataset_loader import read_german_credit_dataset, read_australian_credit_dataset
from collections import Counter

from experimentos import POSSIBLE_CLASSIFIERS

# Default classifier selected for the 
DEFAULT_CLASSIFIER = "nb"

BASE_CLASSIFIERS = {"nb": GaussianNB,
                    "svm": SVC,
                    "knn5": KNeighborsClassifier,
                    "lr": LogisticRegression}

CLASSIFICATION_METRICS = ['accuracy', 'roc_auc']


def create_classifier(classifier_name: str):
    if classifier_name == "knn5":
        return KNeighborsClassifier(n_neighbors=5)
    else:
        return BASE_CLASSIFIERS[classifier_name]()


@dataclass
class CBEG:
    """Framework for ensemble creation."""
    clustering_evaluation_metric: str = "dbc"
    n_clusters: Optional[int] = None
    base_classifier_selection: bool = True
    mutual_info_percent: float  = 0.0
    
    # Attribute selection using mutual information
    def select_attributes(self):
        pass 

    def choose_best_classifier(self, X_cluster: NDArray, y_cluster: NDArray) -> BaseEstimator:
        """ Choose the best classifier according to the average AUC score""" 
        classifiers = {clf_name: create_classifier(clf_name)
                       for clf_name in BASE_CLASSIFIERS}

        auc_by_classifier = self.crossval_classifiers_scores(
                classifiers, X_cluster, y_cluster)

        selected_classifier = max(auc_by_classifier, key=auc_by_classifier.get)

        # Returns the name of the best classifier for this cluster
        return classifiers[selected_classifier]

    def crossval_classifiers_scores(
        self, classifiers: Mapping[str, BaseEstimator], X_train: NDArray, y_train: NDArray
    ) -> dict[str, float]:
        """ Perform a cross val for multiple different classifiers. """
        auc_by_classifier = {}

        for clf_name, classifier in classifiers.items():
            # clf = create_classifier(base_classifier)
            cv_results = cross_validate(classifier, X_train, y_train, cv=10,
                                        scoring=CLASSIFICATION_METRICS)

            # Get the mean AUC of the classifier
            auc_by_classifier[clf_name] = cv_results['test_roc_auc'].mean()

        # Return a dict with the format classifier_name -> mean_auc 
        return auc_by_classifier

    def select_base_classifiers(self, samples_by_cluster: dict[int, NDArray],
                                labels_by_cluster: dict[int, NDArray]):
        """ Select base classifiers according to the results of cross-val.
        """
        n_clusters = len(samples_by_cluster.keys())

        selected_base_classifiers = []

        for c in range(n_clusters):
            # auc_by_classifier = self.crossval_classifiers_scores(
            #         samples_by_cluster[c], labels_by_cluster[c])

            best_classifier = self.choose_best_classifier(
                    samples_by_cluster[c], labels_by_cluster[c])

            selected_base_classifiers.append(best_classifier)

        return selected_base_classifiers

    def predict(self, X_test: NDArray) -> int:
        # Check if clusterer and n_clusters is set TODO

        n_samples = X_test.shape[0]
        # Rows correspond to the sample and columns correspond to the cluster
        vote_count = np.zeros(shape=(X_test.shape[0], self.n_labels)).astype(int)

        for c in range(self.n_clusters):
            # Get the class for the current cluster
            y_pred_cluster = self.base_classifiers[c].predict(
                    X_test).astype(int)

            # tamanho n x 2
            vote_count[range(n_samples), y_pred_cluster] += 1

        # Get the majority class for each sample
        return np.argmax(vote_count, axis=1)


    # TODO: fix one class in one cluster
    # TODO: cross val with less than 10 clusters and a single cluster
    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        # CBEG chooses the optimal number of clusters unless a number is provided.

        self.n_labels = np.unique(y).shape[0]
        # Perform the clustering step in order to split the data to each
        # different classifier
        self.n_clusters = 8
        kmeans_pp_init = kmeans_plusplus(X, n_clusters=self.n_clusters)
        self.clusterer = KMeans(n_clusters=self.n_clusters, init=kmeans_pp_init[0])
        self.clusterer.fit(X)
        clusters = self.clusterer.predict(X)

        # Split the samples according to the cluster they are assigned
        samples_by_cluster = {}
        labels_by_cluster = {}

        for c in range(self.n_clusters):
            indexes_c = np.where(clusters == c)[0]
            samples_by_cluster[c] = X[indexes_c]
            labels_by_cluster[c] = y[indexes_c]

        self.base_classifiers = self.select_base_classifiers(
                samples_by_cluster, labels_by_cluster)

        # Fit the data in each different cluster to the designated classifier.
        for c in range(self.n_clusters):
            X_cluster = samples_by_cluster[c]
            y_cluster = labels_by_cluster[c]
            self.base_classifiers[c].fit(X_cluster, y_cluster)


if __name__ == "__main__":
    X, y = read_german_credit_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    cbeg = CBEG(mutual_info_percent=75)
    cbeg.fit(X_train, y_train)
    y_pred = cbeg.predict(X_test)

    print("CBEG", classification_report(y_pred, y_test))

    print(cbeg.base_classifiers)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Random Forest", classification_report(y_pred, y_test))

