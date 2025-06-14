import os
import sys
import argparse
import numpy as np
import threading
import dataset_loader
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Mapping, Optional, Callable
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from cluster_selection import ClusteringModule
from feature_selection import FeatureSelectionModule
from dataset_loader import normalize_data, DATASETS_INFO
from collections import Counter
from imblearn.over_sampling import SMOTE
from process_results import filter_cbeg_experiments_configs, experiment_already_performed
from logger import PredictionResults
from xgboost import XGBClassifier
from pso_optimizator import PsoOptimizator

# A seleção por AUC é baseada no "A cluster-based intelligence ensemble learning method for classification problems"

# TODO opções para resolver o problema do FCM:
    # - Diminuir para a quantidade de grupos encontrada realmente
    # - pelo menos uma amostra em cada grupo. Usar o maior

# TODO consolidar bases do Jesus

# TODO tentar adicionar a parte com GA

# TODO melhorar a revisão sistemática

# TODO informações úteis:
#   Relação entre distribuição por classe e acurácia por classe

N_FOLDS = 10

# Default classifier selected
DEFAULT_CLASSIFIER = 'nb'

BASE_CLASSIFIERS = {'nb': GaussianNB,
                    'svm': SVC,
                    'knn5': KNeighborsClassifier,
                    'knn7': KNeighborsClassifier,
                    'lr': LogisticRegression,
                    'dt': DecisionTreeClassifier,
                    'rf': RandomForestClassifier,
                    'gb': GradientBoostingClassifier,
                    # 'xb': XGBClassifier,
                    'adaboost': AdaBoostClassifier,
                    }

def create_classifier(classifier_name: str) -> BaseEstimator:
    if classifier_name == "knn7":
        clf = KNeighborsClassifier(n_neighbors=7)
    elif classifier_name == "knn5":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == "svm":
        clf = SVC(probability=True)
    elif classifier_name == "adaboost":
        clf = AdaBoostClassifier()
    else:
        clf = BASE_CLASSIFIERS[classifier_name]()
    return clf

#@dataclass
#class PredictionResults:
#    y_pred: NDArray
#    voting_weights: NDArray
#    y_pred_by_clusters: NDArray
#    y_val: NDArray
#    y_score: Optional[NDArray] = None

@dataclass
class CBEG:
    """Framework for ensemble creation."""
    n_clusters: str | int = "compare"
    base_classifier_selection: bool = True
    min_mutual_info_percentage: float  = 100.0
    cluster_selection_method: str = "clustering" # clustering | pso
    clustering_evaluation_metric: str = "dbc" # dbc_ss, silhoutte
    weights_dbc_silhouette = (0.5, 0.5) # ver isso depois TODO
    combination_strategy: str = "weighted_membership"
    smote_oversample: bool = False
    max_threads: int = 4
    verbose: bool = False

    def choose_default_classifier(self, y_cluster: NDArray, cluster: int
                                  ) -> None:
        """Return the default classifier if y has samples of
        two or more classes, otherwise, produce a Dummy Classifier
        """
        unique_y = np.unique(y_cluster)
        if len(unique_y) > 1:
            self.base_classifiers[cluster] = create_classifier(DEFAULT_CLASSIFIER)
        else:
            self.base_classifiers[cluster] = DummyClassifier(strategy="most_frequent")
    
    def choose_best_classifier(
        self, X_cluster: NDArray, y_cluster: NDArray,
        classification_metrics: list, selected_base_classifiers: list,
        cluster: int
    ) -> None:
        """ Choose the best classifier according to the average AUC score""" 
        possible_base_classifiers = BASE_CLASSIFIERS.copy()

        if len(X_cluster) // 10 < 6:
            del possible_base_classifiers["knn7"]
            del possible_base_classifiers["knn5"]

        elif len(X_cluster) // 10 < 8:
            del possible_base_classifiers["knn7"]

        assert hasattr(self, 'n_labels')
        classifiers = {clf_name: create_classifier(clf_name)
                       for clf_name in possible_base_classifiers}

        # Count the number of sample in the minority class.
        # If the number is lower than 10, the number of folds is reduced
        _, n_minority_class = self.count_minority_class(y_cluster)

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

        assert hasattr(self, 'n_labels')
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

    def count_minority_class(self, y: NDArray) -> tuple[int, int]:
        """ Count the number of samples in the minority class.
        """
        class_count = dict(Counter(y))

        minority_class, n_minority_class = min(
            class_count.items(), key=lambda x: x[1]
        )
        minority_class = int(minority_class)
        return minority_class, n_minority_class

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
            if 'test_roc_auc_ovr' in cv_results:
                auc_by_classifier[clf_name] = cv_results['test_roc_auc_ovr'].mean()
            else:
                auc_by_classifier[clf_name] = cv_results['test_roc_auc'].mean()

        # Return a dict with the format classifier_name -> mean_auc 
        return auc_by_classifier

    def select_base_classifiers(self, classification_metrics: list) -> list[BaseEstimator]:
        """ Select base classifiers according to the results of cross-val.
        """
        n_clusters = int(self.cluster_module.n_clusters)
        selected_base_classifiers = [None] * n_clusters
        threads = []

        for c in range(n_clusters):
            args = (self.samples_by_cluster[c], self.labels_by_cluster[c],
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

    def majority_vote_outputs(
            self, y_pred_by_clusters: list[NDArray]
        ) -> tuple[NDArray, NDArray]:

        """ Get the predicted class of each different classifier and
        combine their votes into a single prediction.
        """
        n_samples = y_pred_by_clusters[0].shape[0]
        vote_count = np.zeros(shape=(n_samples, self.n_labels)).astype(int)
        n_clusters = len(y_pred_by_clusters)

        for y_pred_cluster in y_pred_by_clusters:
            # size n X 2
            vote_count[range(n_samples), y_pred_cluster] += 1

        # Get the majority class for each sample
        vote_sums = vote_count / vote_count.sum(axis=1)[:, np.newaxis]
        samples_weights = np.full((n_samples, n_clusters), 1 / n_clusters)

        return vote_sums, samples_weights # Voting weights

    def weighted_membership_outputs(
            self, X: NDArray, y_pred_by_clusters: list[NDArray],
            function_to_combine: Optional[Callable]=None
        ) -> tuple[NDArray, NDArray]:

        """ Get the predicted classes from the classifiers and combine them
        through weighted voting. The weight is given according to the
        membership value. """
        n_samples = y_pred_by_clusters[0].shape[0]
        vote_sums = np.zeros(shape=(n_samples, self.n_labels))

        centroids = self.cluster_module.centroids
        # Rows correspond to the sample and columns correspond to the cluster
        u_membership = self.cluster_module.calc_membership_matrix(X, centroids)

        idx_samples = range(n_samples)

        for c, y_pred_cluster in enumerate(y_pred_by_clusters):

            if function_to_combine is not None \
                    and function_to_combine == self.calc_training_entropy:

                entropy_cluster = function_to_combine(self.labels_by_cluster[c]) 
                vote_sums[idx_samples, y_pred_cluster
                          ] += u_membership[idx_samples, c] + entropy_cluster
            else:
                vote_sums[idx_samples, y_pred_cluster] += u_membership[idx_samples, c]

        vote_sums /= vote_sums.sum(axis=1)[:, np.newaxis]
        return vote_sums, u_membership

    def calc_training_entropy(self, y_pred_cluster: NDArray):
        # Number of samples by each label in cluster
        n_samples_by_label = [(y_pred_cluster == lbl).sum()
                              for lbl in range(self.n_labels)]
        n_samples_by_label = np.array(n_samples_by_label)
        pk = (n_samples_by_label + 1e-10) / sum(n_samples_by_label)

        return -np.sum(pk * np.log(pk)) / np.log(self.n_labels)

    def entropy_outputs(self, y_pred_by_clusters: list[NDArray]):
        n_samples = y_pred_by_clusters[0].shape[0]
        n_clusters = len(y_pred_by_clusters)

        vote_sums = np.zeros(shape=(n_samples, self.n_labels))

        idx_samples = range(n_samples)
        weights = [self.calc_training_entropy(self.samples_by_cluster[c])
                   for c in range(n_clusters)]

        for c, y_pred_cluster in enumerate(y_pred_by_clusters):
            # self.samples_by_cluster[c]
            vote_sums[idx_samples, y_pred_cluster] += weights[c]

        vote_sums = vote_sums / vote_sums.sum(axis=1)[..., np.newaxis]
        samples_weights = np.tile(weights, (n_samples, 1))

        return vote_sums, samples_weights

    def cluster_density_output(self, y_pred_by_clusters: list[NDArray]):
        n_samples = y_pred_by_clusters[0].shape[0]
        n_clusters = len(y_pred_by_clusters)

        vote_sums = np.zeros(shape=(n_samples, self.n_labels))

        idx_samples = range(n_samples)
        n_training_samples = sum(
            [len(labels) for labels in self.labels_by_cluster.values()]
        )
        weights = [len(self.labels_by_cluster[c]) / n_training_samples
                   for c in range(n_clusters)]

        for c, y_pred_cluster in enumerate(y_pred_by_clusters):
            vote_sums[idx_samples, y_pred_cluster] += weights[c]

        vote_sums = vote_sums / vote_sums.sum(axis=1)[..., np.newaxis]
        samples_weights = np.tile(weights, (n_samples, 1))
        return vote_sums, samples_weights

    def smote_oversampling(self):
        # Used to map which data points are synthetic
        self.idx_synth_data_by_cluster = {}

        # Number of real samples
        n_original_samples = sum(
            [len(labels_cluster) for _, labels_cluster in self.labels_by_cluster.items()]
        )

        cluster_samples = self.samples_by_cluster.items()

        for c, X_cluster in cluster_samples:

            y_cluster = self.labels_by_cluster[c]

            n_minority_class = self.minority_classes_by_cluster[c][1]

            n_real_cluster = len(y_cluster) # Number of real samples in this cluster
            n_samples = len(y_cluster)

            if n_minority_class >= 2 and n_samples >= 3:
                k_neighbors = min(n_minority_class - 1, 5)

                oversample = SMOTE(k_neighbors=k_neighbors)
                X_cluster, y_cluster = oversample.fit_resample(X_cluster, y_cluster)

                # Number of samples in cluster including synthetic
                n_samples_cluster = len(y_cluster)

                n_synthetic_cluster = n_samples_cluster - n_real_cluster
                # Used to map which data points n this cluster are synthetic
                synthetic_data_map = np.array(
                    [False] * n_real_cluster + [True] * n_synthetic_cluster )

                idx_samples = np.arange(n_samples_cluster)
                # Shuffle the order of samples in each cluster
                np.random.shuffle(idx_samples)

                self.samples_by_cluster[c] = X_cluster[idx_samples]
                self.labels_by_cluster[c] = y_cluster[idx_samples]
                self.idx_synth_data_by_cluster[c] = np.where(synthetic_data_map[idx_samples])[0]
            else:
                self.idx_synth_data_by_cluster[c] = []
        # Number of samples including the synthetic ones 
        n_total_samples = sum([len(labels_cluster) for _, labels_cluster in self.labels_by_cluster.items()])
        print(f"Num. samples before: {n_original_samples}\nNum. samples after: {n_total_samples}")

    def predict_proba(self, X_test: NDArray) -> NDArray:
        y_pred_by_clusters = []
        n_clusters = self.cluster_module.n_clusters

        if self.cluster_module.n_clusters is None:
            print("Error: Number of clusters isn't set.")
            sys.exit(1)

        for c in range(n_clusters):
            selected_features = self.features_module.features_by_cluster[c]
            X_test_cluster = X_test[:, selected_features]

            # Get the class for the current cluster
            y_pred_cluster = self.base_classifiers[c].predict(X_test_cluster).astype(np.int32)
            y_pred_by_clusters.append(y_pred_cluster)

        self.y_pred_by_clusters = np.vstack(y_pred_by_clusters).T

        if self.combination_strategy == "weighted_membership":
            y_prob, clusters_weights = self.weighted_membership_outputs(X_test, y_pred_by_clusters)
            self.cluster_weights_samples = clusters_weights
            return y_prob

        elif self.combination_strategy == "majority_voting":
            # TODO consertar majority_voting par
            y_prob, clusters_weights = self.majority_vote_outputs(y_pred_by_clusters)
            self.cluster_weights_samples = clusters_weights
            return y_prob

        elif self.combination_strategy == "entropy_voting":
            y_prob, clusters_weights = self.entropy_outputs(y_pred_by_clusters)
            self.cluster_weights_samples = clusters_weights
            return y_prob

        elif self.combination_strategy == "weighted_membership_entropy":
            y_prob, clusters_weights = self.weighted_membership_outputs(
                    X_test, y_pred_by_clusters, self.calc_training_entropy)
            self.cluster_weights_samples = clusters_weights
            return y_prob

        elif self.combination_strategy == "cluster_density":
            y_prob, clusters_weights = self.cluster_density_output(y_pred_by_clusters)
            self.cluster_weights_samples = clusters_weights
            return y_prob

        else:
            print("O ERRO:", self.combination_strategy)
            print("Invalid combination_strategy value." )
            sys.exit(1)

    def save_training_data(self, filename: str, folder: str):
        # Possible clusters
        clusters = self.labels_by_cluster.keys()

        fullpath = os.path.join(folder, filename)

        file_output = open(fullpath, "w")

        print(f"Clustering algorithm selected: {self.cluster_module.clustering_algorithm}",
              file=file_output)
        print(f"=====================================\n", file=file_output)

        cluster_eval = self.cluster_module.best_evaluation_value # Cluster evalution values
        print(f"Clustering evaluation metric: {self.clustering_evaluation_metric}",
              file=file_output)
        print(f"Clustering evaluation value: {cluster_eval}\n", file=file_output)

        for c in sorted(clusters):
            base_classifier = self.base_classifiers[c]

            # Check if is a OneVsRestClassifier, if it is, extract the classifier
            # inside the OneVsRestClassifier
            if isinstance(base_classifier, OneVsRestClassifier):
                base_classifier = base_classifier.estimator

            minority_class = self.minority_classes_by_cluster[c][0]
            selected_features = self.features_module.features_by_cluster[c]
            labels_cluster = self.labels_by_cluster[c]

            print(f"========== Cluster {c} ==========\n", file=file_output)

            print(f"Base classifier: {base_classifier}\n", file=file_output)

            print(f"Minority Class: {minority_class}\n", file=file_output)

            print(f"Selected Features: {selected_features}\n", file=file_output)

            print(f"Labels: {labels_cluster}\n", file=file_output)

            if self.smote_oversample and self.idx_synth_data_by_cluster:
                synthetic_samples = self.idx_synth_data_by_cluster[c]
                print(f"Synthetic samples indexes: {synthetic_samples}\n",
                      file=file_output)

        file_output.close()

    def save_test_data(self, prediction_results: PredictionResults, filename: str, folder: str) -> None:

        y_pred = prediction_results.y_pred
        y_val = prediction_results.y_val
        y_score = prediction_results.y_score

        n_samples = y_val.shape[0]
        n_labels = np.unique(y_val).shape[0]

        multiclass = n_labels > 2 # Used to calculate precision, recall and F1 for multiclass problems

        fullpath = os.path.join(folder, filename)

        file_output = open(fullpath, "w")

        print(f"Clustering algorithm selected: {self.cluster_module.clustering_algorithm}",
              file=file_output)
        print(f"=====================================\n", file=file_output)

        print("------------------------------------\n" +
              "------ Classification results ------\n" +
              "------------------------------------\n", file=file_output)

        for c in range(int(self.cluster_module.n_clusters)):
            y_pred_cluster = prediction_results.y_pred_by_clusters[:, c]
            base_classifier = self.base_classifiers[c]
            if isinstance(base_classifier, OneVsRestClassifier):
                base_classifier = base_classifier.estimator

            print(f"====== Cluster {c} ======", file=file_output)
            print(f"Base classifier: {base_classifier}", file=file_output)
            self.print_classification_report(y_pred_cluster, y_val, file_output,
                                             multiclass=multiclass)

        print(f"====== Total ======", file=file_output)

        self.print_classification_report(y_pred, y_val, file_output,
                                         y_score, multiclass=multiclass)

        cluster_eval = self.cluster_module.best_evaluation_value # Cluster evalution values
        print(f"Clustering evaluation metric: {self.clustering_evaluation_metric}",
              file=file_output)
        print(f"Clustering evaluation value: {cluster_eval}\n", file=file_output)

        print('========= Predictions by sample =========\n', file=file_output)
        for i in range(n_samples):
            row = (
             f"Prediction: {prediction_results.y_pred[i]}, " + 
             f"Real label: {prediction_results.y_val[i]}, " +
             f"Votes by cluster: {prediction_results.y_pred_by_clusters[i]}, "
             f"Weights: {np.round(prediction_results.voting_weights[i], 2)}"
             )
            print(row, file=file_output)

    def print_classification_report(
            self, y_pred: NDArray, y_val: NDArray, file_output: 'File',
            y_score: Optional[NDArray] = None, multiclass: bool = False
        ) -> None:
        # If it is a multiclass problem, we use the weighted avg. to calculate metrics.
        avg_type = "weighted avg" if multiclass else "1"

        clf_report = classification_report(
            y_pred, y_val, output_dict=True, zero_division=0.0
        )
        print(f"Accuracy: {clf_report['accuracy']}", file = file_output)
        print(f"Recall: {clf_report[avg_type]['recall']}", file = file_output)
        print(f"Precision: {clf_report[avg_type]['precision']}", file = file_output)
        print(f"F1: {clf_report[avg_type]['f1-score']}\n", file = file_output)

        # Add AUC score
        if y_score is not None:
            y_score = y_score / y_score.sum(axis=1)[:, np.newaxis]
            if multiclass:
                # print(y_val)
                # print(y_score)
                auc_val = roc_auc_score(y_val, y_score, multi_class="ovr")
            else:
                y_score = y_score[:, 1]
                auc_val = roc_auc_score(y_val, y_score)
            print(f"AUC: {auc_val}\n", file = file_output)

    def fit(self, X: NDArray, y: NDArray):
        """ Fit the classifier to the data. """
        self.n_labels = np.unique(y).shape[0]

        # Perform the pre-clustering step in order to split the data
        # between the different classifiers
        self.perform_clustering_step(X, y)

        ############ SMOTE ###############
        n_clusters = int(self.cluster_module.n_clusters)
        self.minority_classes_by_cluster = [
            self.count_minority_class(self.labels_by_cluster[c])
            for c in range(n_clusters) ]

        if self.smote_oversample:
            print("Running SMOTE oversampling...")
            self.smote_oversampling()
            self.cluster_module.update_clusters_and_centroids(
                    self.samples_by_cluster, self.labels_by_cluster)
        else:
            self.idx_synth_data_by_cluster = {}

        ###################################
        if self.verbose and self.min_mutual_info_percentage < 100:
            print("Performing feature selection...")

        self.features_module = FeatureSelectionModule(
            self.samples_by_cluster, self.labels_by_cluster,
            min_mutual_info_percentage=self.min_mutual_info_percentage
        )
        self.samples_by_cluster = self.features_module.select_features_by_cluster()

        n_clusters = int(self.cluster_module.n_clusters)

        # Check if it's a multi-class classification problem and create
        # appropriate metric for it.
        if self.n_labels > 2:
            classification_metrics = ["roc_auc_ovr", "accuracy"]
        else:
            classification_metrics = ["roc_auc", "accuracy"]

        if self.base_classifier_selection:
            if self.verbose:
                print("Performing classifier selection...")

            self.base_classifiers = self.select_base_classifiers( classification_metrics )

        else:
            self.base_classifiers = []

            # If no base classifier is selected the default is GaussianNB
            # If only a sample is present in cluster, the default is the DummyClassifier
            for c in range(n_clusters):
                single_class_in_cluster = len(np.unique(self.labels_by_cluster[c])) == 1

                if single_class_in_cluster:
                    self.base_classifiers.append ( DummyClassifier(strategy="most_frequent") )
                else:
                    self.base_classifiers.append ( create_classifier(DEFAULT_CLASSIFIER) )

        # Fit the data in each different cluster to the designated classifier.
        for c in range(n_clusters):
            X_cluster = self.samples_by_cluster[c]
            y_cluster = self.labels_by_cluster[c]
            
            possible_classes = np.unique(y_cluster)

            if len(possible_classes) > 2:
                self.base_classifiers[c] = OneVsRestClassifier(self.base_classifiers[c])

            self.base_classifiers[c].fit(X_cluster, y_cluster)

    def perform_clustering_step(self, X, y):
        # Perform the pre-clustering step in order to split the data
        # between the different classifiers
        if self.verbose:
            print("Performing pre-clustering...")

        if self.cluster_selection_method == "clustering" \
                and self.n_clusters == "compare":

            self.cluster_module = ClusteringModule(
                    X, y,
                    n_clusters=self.n_clusters,
                    clustering_algorithm="kmeans++",
                    evaluation_metric=self.clustering_evaluation_metric,
            )
            self.samples_by_cluster, self.labels_by_cluster = self.cluster_module.cluster_data()

        else: 
            self.cluster_module = ClusteringModule(
                    X, y,
                    n_clusters='compare',
                    clustering_algorithm="kmeans++",
                    evaluation_metric=self.clustering_evaluation_metric,
            )
            # self.samples_by_cluster, self.labels_by_cluster = self.cluster_module.cluster_data()
            clustering_func = self.cluster_module.select_evaluation_function()

            #if self.cluster_module.evaluation_metric != 'silhouette':
            fitness_func = lambda cl, n: (1 - clustering_func(cl, n))

            n_samples = X.shape[0]
            max_clusters = int(np.sqrt(n_samples))

            min_bounds = [2] + [0] * max_clusters * X.shape[1]
            max_bounds = [max_clusters] + [1] * max_clusters * X.shape[1]

            pso_optim = PsoOptimizator(
                n_iters=30, n_particles=100, dimensions=len(max_bounds),
                fitness_func=fitness_func,
                min_bounds=min_bounds, max_bounds=max_bounds,
                cluster_module=self.cluster_module
            )
            clusters, best_cost = pso_optim.optimize()
            n_clusters = len(np.unique(clusters))

            self.cluster_module.n_clusters = n_clusters
            self.cluster_module.calc_centroids(X, clusters, n_clusters)
            self.cluster_module.best_evaluation_value = \
                    clustering_func(clusters, n_clusters)

            self.samples_by_cluster, self.labels_by_cluster = \
                    self.cluster_module.create_clusters_dict(clusters)


def save_data(args, cbeg: CBEG, prediction_results: PredictionResults, fold: int) -> None:
    """ Save training and test data.
    """
    folder_name = get_folder_name(args)
    filename = f'run_{fold}.txt'

    folder_training = os.path.join(folder_name, 'training_summary')

    # Save the data: clusters, labels, selected features, etc
    os.makedirs(folder_training, exist_ok=True)
    cbeg.save_training_data(filename, folder_training)

    print(f'{folder_training}/{filename} saved successfully.')

    folder_test = os.path.join(folder_name, 'test_summary')
    os.makedirs(folder_test, exist_ok=True)
    cbeg.save_test_data(prediction_results, filename, folder_test)
    
    print(f'{folder_test}/{filename} saved successfully.')
    print(50 * '-',"\n")

def get_folder_name(args):
    folder_name_suffix = 'classifier_selection_' if args.base_classifier_selection else 'naive_bayes_'
    folder_name_suffix += f'{args.n_clusters}_clusters_'

    if args.n_clusters == "compare":
        folder_name_suffix += f'{args.clustering_evaluation_metric}_'

    folder_name_suffix += f'{args.combination_strategy}_fusion'

    if args.smote_oversample:
        folder_name_suffix += '_oversampling'

    folder_name_prefix = os.path.join(
            'results', args.dataset,
            f'mutual_info_{args.min_mutual_info_percentage}', 'cbeg'
        )
    return os.path.join(folder_name_prefix, folder_name_suffix)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True, help = "Dataset used.")
    parser.add_argument("-n", "--n_clusters", default="compare", help = "Number of clusters.")
    parser.add_argument("-b", "--base_classifier_selection", type=bool, default=False,
                        action=argparse.BooleanOptionalAction, help = "")
    parser.add_argument("-m", "--min_mutual_info_percentage", type=float, default=100.0, help = "")
    parser.add_argument("-p", "--cluster_selection_method", default="clustering", help="")
    parser.add_argument("-e", "--clustering_evaluation_metric", default="dbc", help = "")
    parser.add_argument("-c", "--combination_strategy", default="majority_voting", help = "")
    parser.add_argument("-s", "--smote_oversample", default=False,
                        action=argparse.BooleanOptionalAction, help = "")
    # cluster_selection_method: str = "clustering" # clustering | pso

    # Read arguments from command line
    args = parser.parse_args()

    args.n_clusters = int(args.n_clusters) if args.n_clusters != "compare" else "compare"

    folder_name = get_folder_name(args).split("/")[-1]
    n_classes_dataset = DATASETS_INFO[args.dataset]["nlabels"]
    mutual_info = args.min_mutual_info_percentage
    experiment_config = filter_cbeg_experiments_configs(
            folder_name, mutual_info, n_classes_dataset)

    #if experiment_already_performed(args.dataset, folder_name, mutual_info):
    #    print(folder_name)
    #    print("Experiment already exists...")
    #    return

    if not(experiment_config):
        print(folder_name)
        print("Skipping experiment variation...")
        return

    for fold in range(1, N_FOLDS+1):
        X, y = dataset_loader.select_dataset_function(args.dataset)()
        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        cbeg = CBEG(
            args.n_clusters, args.base_classifier_selection,
            args.min_mutual_info_percentage, args.cluster_selection_method,
            args.clustering_evaluation_metric, args.combination_strategy,
            max_threads=7, verbose=True
        )
        cbeg.fit(X_train, y_train)

        y_score = cbeg.predict_proba(X_val)
        y_pred = np.argmax(y_score, axis=1)

        prediction_results = PredictionResults(
            y_pred, y_val, cbeg.cluster_weights_samples,
            cbeg.y_pred_by_clusters, y_score
        )
        print("CBEG", classification_report(y_pred, y_val, zero_division=0.0))
        print("Selected Base Classifiers:", cbeg.base_classifiers)
        print("Selected clustering_algorithm:", cbeg.cluster_module.clustering_algorithm)

        save_data(args, cbeg, prediction_results, fold)


if __name__ == "__main__":
    main()
