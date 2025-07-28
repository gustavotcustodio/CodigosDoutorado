import numpy as np
import sys
import random
from pydantic.type_adapter import P
import pyswarms as ps
from typing import Callable
from deslib.util.diversity import disagreement_measure
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
# from concurrent.futures import ThreadPoolExecutor, as_completed
from dask.base import compute
from dask.delayed import delayed
from feature_selection import FeatureSelectionModule
from typing import Mapping, Optional, Callable

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
                    #'xb': XGBClassifier,
                    'adaboost': AdaBoostClassifier,
                    }

@dataclass
class ClassifierSelector:

    n_labels: int
    n_clusters: int
    samples_by_cluster: dict
    labels_by_cluster: dict
    fusion_function: Callable
    n_iters: int = 10
    n_particles: int = 40
    # options: tuple = (0.729, 1.49445, 1.49445) # w, c1 and c2
    options: tuple = (0.9, 1.5, 2.5) # w, c1 and c2
    feature_selector: Optional[FeatureSelectionModule] = None

    def __post_init__(self):
        max_idx_clf = len(BASE_CLASSIFIERS.keys()) - 1
        # min_samples_split: position 5
        min_bounds = [0,           1e-4, 1e-4, 0.1,   1,  2,  1,  1] * self.n_clusters
        max_bounds = [max_idx_clf, 1000, 1000,   1, 500, 10, 10, 10] * self.n_clusters
        self.bounds = (min_bounds, max_bounds)

        self.dims = len(max_bounds)
        self.options_dict = {
            "w": self.options[0], "c1": self.options[1], "c2": self.options[2]
        }
        print("Num clusters:", self.n_clusters)

        print("Fusion function:", self.fusion_function)

    def create_classifier(self, idx_classifier: int, params: dict):
        classifier_name = list(BASE_CLASSIFIERS.keys())[idx_classifier]

        if classifier_name == "nb":
            return GaussianNB()
        elif classifier_name == 'svm':
            return SVC(C=params['C'], gamma=params['gamma'], probability=True)
        elif classifier_name == "knn5":
            return KNeighborsClassifier(n_neighbors=5)
        elif classifier_name == "knn7":
            return KNeighborsClassifier(n_neighbors=7)
        elif classifier_name == "lr":
            return LogisticRegression()#(C=params["C"])
        elif classifier_name == "dt":
            return DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
            )
        elif classifier_name == "rf":
            return RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
            )
        elif classifier_name == "gb":
            return GradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
            )
        elif classifier_name == "adaboost":
            return AdaBoostClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"])
        elif classifier_name == "xb":
            return XGBClassifier(
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
            )
        else:
            print("Error: invalid classifier.")
            sys.exit(1)

    def calc_diversity_score(self, n_clf, y_true, y_pred_by_clf):
        # Double fault measure diversity score
        total_diversity = 0

        for i in range(n_clf - 1):
            for j in range(i + 1, n_clf):
                y_pred_i = y_pred_by_clf[i]
                y_pred_j = y_pred_by_clf[j]

                disagreement = disagreement_measure(y_true, y_pred_i, y_pred_j)
                total_diversity += disagreement

        return 2 * total_diversity / (n_clf * (n_clf - 1))

    def calc_double_fault_measure(self, y_true, y_pred_i, y_pred_j):
        """
        df =  N00 / (N11 + N00 + N01 + N10)

        Nij i=0 (number of example incorrectly classified by classifier i)
            i=1 (number of example correctly classified by classifier i)
            same for j=0 and j=1.
        Example : N01 = number of examples incorrectly classified by classifier i
                        and correctly classified for classifier j
        """
        # all_idx = np.arange(len(y_true))

        idx_i_error = np.where(y_true != y_pred_i)[0]
        idx_j_error = np.where(y_true != y_pred_j)[0]

        # idx_i_right = np.delete(all_idx, idx_i_error)
        # idx_j_right = np.delete(all_idx, idx_j_error)

        n00 = len(np.intersect1d(idx_i_error, idx_j_error))
        return n00 / len(y_true)

    def decode_candidate_solution(self, pso_solution):
        size_by_cluster = len(pso_solution) // self.n_clusters
        base_classifiers = []

        for i in range(0, len(pso_solution), size_by_cluster):
            params = {}

            idx_classifier = round(pso_solution[i])
            params['C'] = pso_solution[i+1]
            params['gamma'] = pso_solution[i+2]
            params['learning_rate'] = pso_solution[i+3]
            params['n_estimators'] = round(pso_solution[i+4])
            params['min_samples_split'] = round(pso_solution[i+5])
            params['min_samples_leaf'] = round(pso_solution[i+6])
            params['max_depth'] = round(pso_solution[i+7])

            # Get the classifier by the using the index
            classifier = self.create_classifier(idx_classifier, params)
            base_classifiers.append(classifier)

        return base_classifiers

    def split_clusters_in_train_and_val(self):

        n_clusters = self.n_clusters

        skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

        X_cluster_by_fold = [
            [np.empty(1)for _ in range(N_FOLDS)] for _ in range(n_clusters)]
        y_cluster_by_fold = [
            [np.empty(1) for _ in range(N_FOLDS)] for _ in range(n_clusters)]

        X_val_by_fold = [[] for _ in range(N_FOLDS)]
        y_val_by_fold = [[] for _ in range(N_FOLDS)]

        for c in range(n_clusters):
            fold = 0

            X_cluster = self.samples_by_cluster[c]
            y_cluster = self.labels_by_cluster[c]

            # TODO consertar prolema com número de instâncias em cada cluster

            for train_index, val_index in skf.split(X_cluster, y_cluster):

                X_val_by_fold[fold] += X_cluster[val_index].tolist()
                y_val_by_fold[fold] += y_cluster[val_index].tolist()

                X_cluster_by_fold[c][fold] = X_cluster[train_index]
                y_cluster_by_fold[c][fold] = y_cluster[train_index]

                fold += 1

        X_val_by_fold = [np.array(X_val) for X_val in X_val_by_fold]
        y_val_by_fold = [np.array(y_val) for y_val in y_val_by_fold]

        return X_cluster_by_fold, X_val_by_fold, y_cluster_by_fold, y_val_by_fold

    def train_clf_predict_proba(self, clf, cluster, X_train, y_train, X_val):
        # Select the correct features for the classifier
        n_samples_cluster = len(y_train)

        self.set_max_samples_split(clf, cluster, n_samples_cluster)

        if self.feature_selector is not None:
            features = self.feature_selector.features_by_cluster[cluster]

            X_train = X_train[:, features]
            X_val = X_val[:, features]

        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)

    def train_meta_classifier(self, y_prob_by_clf, y_true):
        meta_clf = SVC(probability=True)
        X = np.hstack(y_prob_by_clf)

        meta_clf.fit(X, y_true)
        return meta_clf

    def replace_single_class_classifier(self, classifiers, y_by_classifier):
        """Replace the classifiers in clusters with samples of
        a single class for Dummy Classifiers
        """
        fold_classifiers = []

        for c, clf in enumerate(classifiers):
            if np.all(y_by_classifier[c] == y_by_classifier[c][0]):
                dummy = DummyClassifier(strategy="most_frequent")
                fold_classifiers.append(dummy)
            else:
                fold_classifiers.append(clf)
        return fold_classifiers



    def eval_base_classifiers(self, X_cluster_by_fold, X_val_by_fold,
                              y_cluster_by_fold, y_val_by_fold):
        def wrapper(selected_classifiers):

            eval_by_fold = []
            n_clusters = len(selected_classifiers)

            for fold in range(N_FOLDS):
                y_pred_by_clf = []

                X_val = X_val_by_fold[fold]
                y_val = y_val_by_fold[fold]

                y_by_classifier = [y_cluster_by_fold[c][fold]
                                   for c in range(n_clusters)]
                #print(selected_classifiers)
                fold_classifiers = self.replace_single_class_classifier(
                    selected_classifiers, y_by_classifier
                )
                #print(fold_classifiers)
                #print("======================================")

                y_prob_by_clf = [self.train_clf_predict_proba(
                    clf, c, X_cluster_by_fold[c][fold],
                    y_cluster_by_fold[c][fold], X_val
                ) for c, clf in enumerate(fold_classifiers)]

                y_pred_by_clf = [y_prob.argmax(1) for y_prob in y_prob_by_clf]

                if self.fusion_function.__name__ == 'meta_classifier_predict':
                    X_train = np.vstack([ X_cluster_by_fold[c][fold]
                                         for c in range(n_clusters) ])

                    y_prob_clusters_train = [classifier.predict_proba(X_train)
                                             for classifier in fold_classifiers]
                    # y_prob_train_by_clf = np.array(y_prob_train_by_clf).T
                    y_train = np.hstack([ y_cluster_by_fold[c][fold]
                                         for c in range(n_clusters) ])

                    self.meta_classifier = self.train_meta_classifier(
                            y_prob_clusters_train, y_train)

                    y_prob, _ = self.fusion_function(y_prob_by_clf, self.meta_classifier)

                elif self.fusion_function.__name__ == 'weighted_membership_outputs':
                    y_prob, _ = self.fusion_function(X_val, y_pred_by_clf)

                else:
                    y_prob, _ = self.fusion_function(y_pred_by_clf)

                y_pred = y_prob.argmax(1)

                if self.n_labels > 2:
                    auc_score = roc_auc_score(y_val, y_prob, multi_class='ovr')
                    f1_val = f1_score(y_val, y_pred, average='weighted')
                else:
                    auc_score = roc_auc_score(y_val, y_prob[:,1])
                    f1_val = f1_score(y_val, y_pred)

                # print("fold:", fold)
                # print("AUC score:", auc_score)
                # avg_auc_clusters = sum(auc_score_by_cluster) / n_clusters

                # Calc diversity for this classifier
                #diversity_score = self.calc_diversity_score(
                #        n_clusters, y_val, y_pred_by_clf)
                # Calc the combination of AUC and diversity for this fold
                eval_fold = auc_score * 0.5 + f1_val * 0.5
                #0.75 * f1_val + 0.25 * diversity_score

                # print("Avg. AUC:", avg_auc_clusters)
                # print("Diversity score:", diversity_score)

                eval_by_fold.append(eval_fold)
            # Get the average value for all folds
            return sum(eval_by_fold) / N_FOLDS
        return wrapper

    def calc_fitness_solutions(self, solutions):
        """ Calculate the fitness value for all
        candidate solutions. """

        # Wrap each function call in delayed
        delayed_costs = [delayed(self.calc_cost)(solution) for solution in solutions]

        # Compute in parallel
        costs = compute(*delayed_costs)

        # costs = [self.calc_cost(solution) for solution in solutions]
        print(costs)

        self.update_inertia()
        print("PSO params:", self.pso.options)
        return costs

    def calc_cost(self, solution):
        classifiers = self.decode_candidate_solution(solution)

        fitness_val = self.fitness_function(classifiers)
        cost = 1 - fitness_val

        return cost

    def update_inertia(self):
        self.current_iter += 1

        w = self.pso.options['w']
        w_min = 0.4
        w_max = self.options_dict['w']
        w = w_max - self.current_iter * (w_max - w_min) / self.n_iters
        self.pso.options['w'] = w

    def set_max_samples_split(self, clf, cluster, n_samples_cluster):
        # max_bounds = [max_idx_clf, 1000, 1000,   1, 500, 10, 10, 10] * self.n_clusters
        max_samples_split = self.bounds[1][cluster * 8 + 5]

        if hasattr(clf, "min_samples_split") and n_samples_cluster < max_samples_split:
            self.bounds[1][cluster * 8 + 5] = n_samples_cluster
            clf.min_samples_split = n_samples_cluster

        #self.bounds = (min_bounds, max_bounds)

    def run_pso(self):
        # Create pso
        self.pso = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=self.dims,
            options=self.options_dict, bounds=self.bounds
        )
        self.current_iter = 0
        # Optimize pso
        best_cost, best_solution = self.pso.optimize(
            self.calc_fitness_solutions, iters=self.n_iters)
        return best_solution

    def select_base_classifiers(self):
        X_cluster_by_fold, X_val_by_fold, y_cluster_by_fold, y_val_by_fold = \
            self.split_clusters_in_train_and_val()

        # Fix the maximum number of leaves for each classifier

        fitness_function = self.eval_base_classifiers(
            X_cluster_by_fold, X_val_by_fold, y_cluster_by_fold, y_val_by_fold
        )
        self.fitness_function = fitness_function

        # Run PSO
        best_solution = self.run_pso()
        base_classifiers = self.decode_candidate_solution(best_solution)

        return base_classifiers
