import sys
import numpy as np
import threading
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Mapping, Optional
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from cluster_selection import ClusteringModule
from feature_selection import FeatureSelectionModule
from dataset_loader import read_german_credit_dataset, read_australian_credit_dataset, read_contraceptive_dataset, read_heart_dataset, read_hepatitis_dataset, read_pima_dataset, read_iris_dataset, read_wine_dataset, read_wdbc_dataset
from collections import Counter

# TODO informações úteis:
#   Relação entre distribuição por classe e acurácia por classe

# Default classifier selected for the 
DEFAULT_CLASSIFIER = "nb"

BASE_CLASSIFIERS = {"nb": GaussianNB,
                    "svm": SVC,
                    "knn5": KNeighborsClassifier,
                    "knn7": KNeighborsClassifier,
                    "lr": LogisticRegression,
                    # "adaboost": AdaBoostClassifier
                    }

def create_classifier(classifier_name: str) -> BaseEstimator:
    if classifier_name == "knn7":
        return KNeighborsClassifier(n_neighbors=7)
    elif classifier_name == "knn5":
        return KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == "svm":
        return SVC(probability=True)
    elif classifier_name == "adaboost":
        return AdaBoostClassifier(algorithm="SAMME")
    else:
        return BASE_CLASSIFIERS[classifier_name]()


@dataclass
class CBEG:
    """Framework for ensemble creation."""
    n_clusters: str | int = "compare"
    base_classifier_selection: bool = True
    min_mutual_info_percentage: float  = 100.0
    clustering_evaluation_metric: str = "dbc"
    combination_strategy: str = "weighted_membership"
    max_threads: int = 7
    verbose: bool = False
    
    def choose_best_classifier(
        self, X_cluster: NDArray, y_cluster: NDArray,
        classification_metrics: list, selected_base_classifiers: list,
        cluster: int
    ) -> None:
        """ Choose the best classifier according to the average AUC score""" 
        # TODO: Remove Knn 5 or Knn 7 classifiers if the number of samples is
        # lower than that
        possible_base_classifiers = BASE_CLASSIFIERS.copy()

        if len(X_cluster) // 10 < 6:
            del possible_base_classifiers["knn7"]
            del possible_base_classifiers["knn5"]

        elif len(X_cluster) // 10 < 8:
            del possible_base_classifiers["knn7"]

        classifiers = {clf_name: create_classifier(clf_name)
                       for clf_name in possible_base_classifiers}

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
        if np.all(y_cluster == y_cluster[0]):
            dummy_classifier = DummyClassifier(strategy="most_frequent")
            selected_base_classifiers[cluster] = dummy_classifier
            return

        if n_minority_class == 1:
            # Default classifier is the Naive-Bayes
            selected_base_classifiers[cluster] = create_classifier(DEFAULT_CLASSIFIER)
            return
            
        auc_by_classifier = self.crossval_classifiers_scores(
            classifiers, X_cluster, y_cluster, classification_metrics,
            n_folds)

        selected_classifier = max(auc_by_classifier, key=auc_by_classifier.get)

        # Associate the best classifier for this cluster in the list of the
        # selected classifiers
        selected_base_classifiers[cluster] = classifiers[selected_classifier]

    def count_minority_class(self, y: NDArray) -> int:
        """ Count the number of samples in the minority class.
        """
        class_count = Counter(y)

        return min(class_count.values())

    def crossval_classifiers_scores(
        self, classifiers: Mapping[str, BaseEstimator], X_train: NDArray, 
        y_train: NDArray, classification_metrics: list, n_folds: int
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
        n_clusters = int(self.cluster_module.n_clusters)
        selected_base_classifiers = [None] * n_clusters
        threads = []

        for c in range(n_clusters):
            args = (samples_by_cluster[c], labels_by_cluster[c],
                    classification_metrics, selected_base_classifiers, c)

            # If the maximum number of threads is used, wait
            if len(threads) < self.max_threads:
                threads.append(
                    threading.Thread(target=self.choose_best_classifier, args=args)
                )
                threads[-1].start()

            else: # If all threads are occupied, wait
                all_threads_occupied = True
                idx_thread = 0

                # Check if there are any available threads.
                while all_threads_occupied:
                    if not(threads[idx_thread].is_alive()):
                        all_threads_occupied = False

                        threads[idx_thread] = threading.Thread(
                            target=self.choose_best_classifier, args=args)
                        threads[idx_thread].start()

                    idx_thread = (idx_thread + 1) % self.max_threads


        for idx_thread in range(len(threads)):
            threads[idx_thread].join()

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

    def weighted_membership_outputs(
            self, X: NDArray, y_pred_by_clusters: list[NDArray]
        ) -> NDArray:
        """ Get the predicted classes from the classifiers and combine them
        through weighted voting. The weight is given according to the
        membership value. """
        n_samples = y_pred_by_clusters[0].shape[0]
        vote_sums = np.zeros(shape=(n_samples, self.n_labels))

        centroids = self.cluster_module.centroids
        # Rows correspond to the sample and columns correspond to the cluster
        u = self.cluster_module.calc_membership_matrix(X, centroids)

        idx_samples = range(n_samples)
        for c, y_pred_cluster in enumerate(y_pred_by_clusters):
            vote_sums[idx_samples, y_pred_cluster] += u[idx_samples, c]

        return np.argmax(vote_sums, axis=1)

    def predict(self, X_test: NDArray) -> NDArray:
        y_pred_by_clusters = []

        if self.cluster_module.n_clusters is None:
            print("Error: Number of clusters isn't set.")
            sys.exit(1)

        for c in range(self.cluster_module.n_clusters):
            selected_features = self.features_module.attributes_by_cluster[c]
            X_test_cluster = X_test[:, selected_features]

            # Get the class for the current cluster
            y_pred_cluster = self.base_classifiers[c].predict(X_test_cluster).astype(int)
            y_pred_by_clusters.append(y_pred_cluster)

        if self.combination_strategy == "weighted_membership":
            return self.weighted_membership_outputs(X_test, y_pred_by_clusters)
        elif self.combination_strategy == "majority_voting":
            return self.majority_vote_outputs(y_pred_by_clusters)
        else:
            print("Invalid combination_strategy value." )
            sys.exit(1)

    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        # Check if it's a multi-class classification problem and create
        # appropriate metric for it.
        self.n_labels = np.unique(y).shape[0]

        if self.n_labels > 2:
            classification_metrics = ["roc_auc_ovo", "accuracy"]
        else:
            classification_metrics = ["roc_auc", "accuracy"]

        # Perform the pre-clustering step in order to split the data
        # between the different classifiers
        if self.verbose:
            print("Performing pre-clustering...")

        self.cluster_module = ClusteringModule(X, y, n_clusters=self.n_clusters)

        samples_by_cluster, labels_by_cluster = self.cluster_module.cluster_data()

        if self.verbose and self.min_mutual_info_percentage < 100:
            print("Performing attribute selection...")

        self.features_module = FeatureSelectionModule(
            samples_by_cluster, labels_by_cluster, min_mutual_info_percentage=self.min_mutual_info_percentage
        )
        samples_by_cluster = self.features_module.select_attributes_by_cluster()

        if self.base_classifier_selection:
            if self.verbose:
                print("Performing classifier selection...")

            self.base_classifiers = self.select_base_classifiers(
                samples_by_cluster, labels_by_cluster, classification_metrics
            )
        else:
            # If no base classifier is selected the default is GaussianNB
            self.base_classifiers = [GaussianNB()] * int(self.cluster_module.n_clusters)

        # Fit the data in each different cluster to the designated classifier.
        for c in range(int(self.cluster_module.n_clusters)):
            X_cluster = samples_by_cluster[c]
            y_cluster = labels_by_cluster[c]
            self.base_classifiers[c].fit(X_cluster, y_cluster)


if __name__ == "__main__":
    # TODO fix normalization
    X, y = read_german_credit_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cbeg = CBEG(verbose=True, max_threads=7, n_clusters=5)
    cbeg.fit(X_train, y_train)

    y_pred = cbeg.predict(X_test)

    print("CBEG", classification_report(y_pred, y_test, zero_division=True))
    print("Classificadores base selecionados:", cbeg.base_classifiers)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Random Forest", classification_report(y_pred, y_test))

