import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Mapping, Optional
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from cluster_selection import ClusteringModule
from dataset_loader import read_german_credit_dataset, read_australian_credit_dataset, read_contraceptive_dataset, read_heart_dataset, read_hepatitis_dataset, read_pima_dataset, read_iris_dataset, read_wine_dataset
from collections import Counter

from experimentos import POSSIBLE_CLASSIFIERS

# Default classifier selected for the 
DEFAULT_CLASSIFIER = "nb"

BASE_CLASSIFIERS = {"nb": GaussianNB,
                    "svm": SVC,
                    "knn7": KNeighborsClassifier,
                    "lr": LogisticRegression,
                    "rf": RandomForestClassifier,
                    "gb": GradientBoostingClassifier
                    }

CLASSIFICATION_METRICS = ['accuracy', 'roc_auc']


def create_classifier(classifier_name: str):
    if classifier_name == "knn7":
        return KNeighborsClassifier(n_neighbors=7)
    elif classifier_name == "svm":
        return SVC(probability=True)
    else:
        return BASE_CLASSIFIERS[classifier_name]()


@dataclass
class CBEG:
    """Framework for ensemble creation."""
    clustering_evaluation_metric: str = "dbc"
    n_clusters: Optional[int] = None
    base_classifier_selection: bool = True
    mutual_info_percent: float  = 0.0
    
    def choose_best_classifier(self, X_cluster: NDArray, y_cluster: NDArray,
                               classification_metrics: list) -> BaseEstimator:
        """ Choose the best classifier according to the average AUC score""" 
        classifiers = {clf_name: create_classifier(clf_name)
                       for clf_name in BASE_CLASSIFIERS}

        # Count the number of sample in the minority class.
        # If the number is lower than 10, the number of folds is reduced
        n_minority_class = self.count_minority_class(y_cluster)

        if n_minority_class < 10:
            n_folds = n_minority_class
        else:
            n_folds = 10

        # If there are instances of a single class in the cluster or
        # there is only a single instance in the minority class,
        # use a dummy classifier (all samples in this cluster
        # belong to a same class)
        if np.all(y_cluster == y_cluster[0]) or n_minority_class == 1:
            return DummyClassifier(strategy="most_frequent")

        auc_by_classifier = self.crossval_classifiers_scores(
            classifiers, X_cluster, y_cluster, classification_metrics,
            n_folds)

        selected_classifier = max(auc_by_classifier, key=auc_by_classifier.get)

        # Returns the name of the best classifier for this cluster
        return classifiers[selected_classifier]

    def count_minority_class(self, y: NDArray) -> int:
        """ Count the number of samples in the minority class.
        """
        class_count = Counter(y)

        return min(class_count.values())

    def crossval_classifiers_scores(
        self, classifiers: Mapping[str, BaseEstimator], X_train: NDArray, 
        y_train: NDArray, classification_metrics: list, n_folds: int  # list of Scorers
    ) -> dict[str, float]:

        """ Perform a cross val for multiple different classifiers. """
        auc_by_classifier = {}

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for clf_name, classifier in classifiers.items():

            cv_results = cross_validate(classifier, X_train, y_train, cv=cv,
                                        scoring=classification_metrics)

            # Get the mean AUC of the classifier

            if 'test_roc_auc_ovo' in cv_results:
                auc_by_classifier[clf_name] = cv_results['test_roc_auc_ovo'].mean()
            else:
                auc_by_classifier[clf_name] = cv_results['test_roc_auc'].mean()

        # Return a dict with the format classifier_name -> mean_auc 
        return auc_by_classifier

    def select_base_classifiers(
        self, samples_by_cluster: dict[int, NDArray],
        labels_by_cluster: dict[int, NDArray], classification_metrics: list
    ) -> list[BaseEstimator]:

        """ Select base classifiers according to the results of cross-val.
        """
        selected_base_classifiers = []

        for c in range(self.n_clusters):

            best_classifier = self.choose_best_classifier(
                samples_by_cluster[c], labels_by_cluster[c], classification_metrics
            )
            selected_base_classifiers.append(best_classifier)

        return selected_base_classifiers

    def majority_vote_outputs(self, y_pred_by_clusters: list[NDArray]) -> NDArray:
        """ Get the predicted class of each different classifier and
        combine their votes into a single prediction.
        """
        n_samples = y_pred_by_clusters[0].shape[0]
        vote_count = np.zeros(shape=(n_samples, self.n_labels)).astype(int)

        for y_pred_cluster in y_pred_by_clusters:
            # size n X 2
            vote_count[range(n_samples), y_pred_cluster] += 1

        # Get the majority class for each sample
        return np.argmax(vote_count, axis=1)

    def weighted_voting_membership(
            self, X: NDArray, y_pred_by_clusters: list[NDArray]
        ) -> NDArray:
        """ Get the predicted classes from the classifiers and combine them
        through weighted voting. The weight is given according to the
        membership value. """
        n_samples = y_pred_by_clusters[0].shape[0]
        vote_sums = np.zeros(shape=(n_samples, self.n_labels))

        centroids = self.cluster_module.centroids
        u = self.cluster_module.calc_membership_matrix(X, centroids)

        idx_samples = range(n_samples)
        for c, y_pred_cluster in enumerate(y_pred_by_clusters):
            vote_sums[idx_samples, y_pred_cluster] += u[idx_samples, c]

        return np.argmax(vote_sums, axis=1)

    def predict(self, X_test: NDArray) -> NDArray:
        y_pred_by_clusters = []

        # Check if clusterer and n_clusters is set TODO
        for c in range(self.n_clusters):
            # Get the class for the current cluster
            y_pred_cluster = self.base_classifiers[c].predict(X_test).astype(int)
            y_pred_by_clusters.append(y_pred_cluster)

        # Rows correspond to the sample and columns correspond to the cluster
        return self.weighted_voting_membership(X_test, y_pred_by_clusters)

    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        # Check if it's a multi-class classification problem and create
        # appropriate metric for it.
        self.n_labels = np.unique(y).shape[0]

        if self.n_labels > 2:
            classification_metrics = ["roc_auc_ovo", "accuracy"]
        else:
            classification_metrics = ["roc_auc", "accuracy"]
        # CBEG chooses the optimal number of clusters unless a number is provided.

        # Perform the clustering step in order to split the data to each
        # different classifier
        self.n_clusters = 2 

        print("Performing pre-clustering...")
        self.cluster_module = ClusteringModule(self.n_clusters, X, y)
        samples_by_cluster, labels_by_cluster = self.cluster_module.cluster_data()

        print("Performing classifier selection...")
        self.base_classifiers = self.select_base_classifiers(
            samples_by_cluster, labels_by_cluster, classification_metrics
        )

        # Fit the data in each different cluster to the designated classifier.
        for c in range(self.n_clusters):
            X_cluster = samples_by_cluster[c]
            y_cluster = labels_by_cluster[c]
            self.base_classifiers[c].fit(X_cluster, y_cluster)


if __name__ == "__main__":
    X, y = read_pima_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    cbeg = CBEG()
    cbeg.fit(X_train, y_train)

    # TODO fix iris problem in predict
    y_pred = cbeg.predict(X_test)

    print("CBEG", classification_report(y_pred, y_test, zero_division=True))
    print("Classificadores base selecionados:", cbeg.base_classifiers)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Random Forest", classification_report(y_pred, y_test))
