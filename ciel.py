# Tabela 5
# Quantos indivídios tem nesse PSO?
# pág 17
import sys
import time
import math
import pyswarms as ps
from multiprocessing.pool import ThreadPool
from numpy.typing import NDArray
import argparse
import numpy as np
import dataset_loader
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from ciel_optimizer import CielOptimizer, create_clusterer
from ciel_optimizer import N_FOLDS
from logger import PredictionResults
from logger import Logger
from dataset_loader import normalize_data
import dask
from dask.base import compute
from dask.delayed import delayed
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# TODO colocar o restart de solução no PSO


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
    'extra_tree',
    'svm'
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


class Ciel:
    def __init__(self, n_iters=10, n_particles=30, ftol_iter=4):
        self.n_iters = n_iters
        self.n_particles = n_particles
        self.ftol_iter = ftol_iter
        self.max_n_clusters = 10
        self.options = { 'c1': 1.49445, 'c2': 1.49445, 'w': 0.729, }

    def set_bounds_pso(self):
        if self.best_classifier == 'svm':
            self.n_params_clf = 2
            classifier_bounds = ([1e-4, 1e-4], [1000, 1000])

        elif self.best_classifier == 'extra_tree':
            self.n_params_clf = 4
            classifier_bounds = ([1, 1, 2, 1], [500, 10, 10, 10])

        else: # == 'gb':
            self.n_params_clf = 5
            classifier_bounds = ([1, 1, 2, 1, 0.1], [500, 10, 10, 10, 1.0])

        lower_bounds = [2] + (classifier_bounds[0] * self.max_n_clusters
                              ) + ([0.1] * self.max_n_clusters)
        upper_bounds = [10] + (classifier_bounds[1] * self.max_n_clusters
                               ) + ([1.0] * self.max_n_clusters)
        # lower_bounds = [2] + ([1e-4, 1e-4] + [1, 1, 2, 1] + [1, 1, 2, 1, 0.1]
        #                       ) * self.max_n_clusters + [0.1] * self.max_n_clusters
        # upper_bounds = [10] + ([1000, 1000] + [500, 10, 10, 10] + [500, 10, 10, 10, 1.0]
        #                        ) * self.max_n_clusters + [1.0] * self.max_n_clusters

        assert len(lower_bounds) == len(upper_bounds)
        self.bounds = (lower_bounds, upper_bounds)

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

        # Davies Bouldin is a special case, because lower values are better values
        best_davies_bouldin = best_clustering_metrics['internal']['davies_bouldin']
        davies_bouldin_val = clustering_metrics['internal']['davies_bouldin']

        # Subtract davies bouldin from the sum, because it's a minimization technique
        best_internal_sum = sum(best_clustering_metrics["internal"].values()
                                ) - 2 * best_davies_bouldin
        sum_internal = sum(clustering_metrics["internal"].values()
                           ) - 2 * davies_bouldin_val

        # if more than half of the internal metrics are improved, the new clusterer
        # is the new best
        if sum_internal > best_internal_sum:
            return True
        return False

    def update_best_clusterer(self, clustering_metrics, best_clustering_metrics) -> bool:
        if not best_clustering_metrics:
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()
            return True

        # CUrrent best sum of external clustering metrics
        best_sum_external = sum(best_clustering_metrics["external"].values())
        sum_external = sum(clustering_metrics['external'].values())

        if sum_external > best_sum_external:
            best_clustering_metrics['external'] = clustering_metrics['external'].copy()
            best_clustering_metrics['internal'] = clustering_metrics['internal'].copy()

        # n_external_improved = 0

        # for metric, value_metric in clustering_metrics['external'].items():
        #     best_value_metric = best_clustering_metrics['external'][metric]

        #     if value_metric > best_value_metric:
        #         n_external_improved += 1

        # Tie break with internal metrics if external metric are a draw
        if sum_external == best_sum_external and \
                self.internal_breaks_tie(clustering_metrics, best_clustering_metrics):

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

            clusterer = create_clusterer(clusterer_name, self.n_labels)
            clusters = clusterer.fit_predict(X)

            external_metrics_evals, internal_metrics_evals = \
                    self.calc_metrics_clustering(clusters, X, y)

            # if clusterer_name == "birch":
            # print(external_metrics_evals)

            clustering_metrics['external'] = external_metrics_evals
            clustering_metrics['internal'] = internal_metrics_evals

            clusterer_updated = self.update_best_clusterer(
                    clustering_metrics, best_clustering_metrics)

            if clusterer_updated:
                best_clusterer = clusterer_name

        self.best_clustering_metrics = best_clustering_metrics
        # print(f'Selected clusterer: {best_clusterer}')
        return best_clusterer

    def unwrap_solution(self, solution) -> dict[str, list|int]:
        n_clusters = round(solution[0])
        clf_params = []

        weights = np.zeros(n_clusters)

        # The position where the voting weights from classifiers start.
        n_params = self.n_params_clf
        start_idx_weights = 1 + n_params * self.max_n_clusters

        for c in range(n_clusters):
            clf_params.append({})

            if self.best_classifier == 'svm':
                clf_params[c]['cost'] = solution[n_params * c + 1]
                clf_params[c]['gamma'] = solution[n_params * c + 2]

            elif self.best_classifier == 'extra_tree':
                clf_params[c]['n_estimators'] = round(solution[n_params * c + 1])
                clf_params[c]['max_depth'] = round(solution[n_params * c + 2])
                clf_params[c]['min_samples_split'] = round(solution[n_params * c + 3])
                clf_params[c]['min_samples_leaf'] = round(solution[n_params * c + 4])

            else:
                clf_params[c]['n_estimators'] = round(solution[n_params * c + 1])
                clf_params[c]['max_depth'] = round(solution[n_params * c + 2])
                clf_params[c]['min_samples_split'] = round(solution[n_params * c + 3])
                clf_params[c]['min_samples_leaf'] = round(solution[n_params * c + 4])
                clf_params[c]['learning_rate'] = solution[n_params * c + 5]
            
            weights[c] = solution[c + start_idx_weights]

        weights  = weights / weights.sum()

        params = {}
        params['n_clusters'] = int(n_clusters)
        params['clf_params'] = clf_params
        params['weights'] = weights

        return params

    def fitness_eval(self, X, y):
        def wrapper(possible_solutions):
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            # cost_values = []

            # for solution in possible_solutions:
                # cost = self.calc_cost(solution, kf, X, y)
                # cost_values.append(cost)

                # Convert PSO particle to the parameters
            inicio = time.time()
            cost_values = [self.calc_cost(solution, kf, X, y)
                           for solution in possible_solutions]

            # delayed_costs = [delayed(self.calc_cost)(solution, kf, X, y)
            #                  for solution in possible_solutions]

            # with dask.config.set(pool=ThreadPool(4)):
            # Compute in parallel
            # compute(*delayed_costs)  #, scheduler="threads")
            print("Tempo solução:", time.time() - inicio)

            return cost_values

        return wrapper

    # @dask.delayed
    def calc_cost(self, solution, kf, X, y):
        params = self.unwrap_solution(solution)

        auc_values = []
        folds_splits = kf.split(X, y)

        ciel_opt = CielOptimizer(
            self.best_clusterer,
            self.best_classifier,
            params['n_clusters'],
            params['clf_params'],
            params['weights'],
        )

        for fold, (train_indexes, test_indexes) in enumerate(folds_splits):

            X_train, y_train = X[train_indexes], y[train_indexes]
            X_test, y_test = X[test_indexes], y[test_indexes]

            ciel_opt.fit(X_train, y_train)
            # Predict probability
            y_score, _, _ = ciel_opt.predict_proba(X_test)

            if self.n_labels == 2:
                auc_val = roc_auc_score(y_test, y_score[:,1])
            else:
                auc_val = roc_auc_score(y_test, y_score, multi_class="ovr")

            auc_values.append(auc_val)

        cost = 1 - np.mean(auc_values)
        return cost

    def crossval_classifiers_scores(
        self, classifiers: dict, X: NDArray, y: NDArray
    ):
        if self.n_labels > 2:
            classification_metrics = ['roc_auc_ovr', 'accuracy']
        else:
            classification_metrics = ['roc_auc', 'accuracy']

        auc_by_classifier = {}

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for clf_name, classifier in classifiers.items():

            cv_results = cross_validate(classifier, X, y, cv=cv,
                                        scoring=classification_metrics)
            # Get the mean AUC of the classifier
            if 'test_roc_auc_ovr' in cv_results:
                mean_auc = cv_results['test_roc_auc_ovr'].mean()
            else:
                mean_auc = cv_results['test_roc_auc'].mean()

            auc_by_classifier[clf_name] = mean_auc
        # Return a dict with the format classifier_name -> mean_auc
        return auc_by_classifier

    def select_optimal_classifier(self, X, y):
        ''' Choose the best classifier according to the average AUC score'''
        classifiers = {
            clf_name: self.create_classifier(clf_name)
            for clf_name in BASE_CLASSIFIERS
        }
        auc_by_classifier = self.crossval_classifiers_scores(classifiers, X, y)
        selected_classifier = max(auc_by_classifier, key=auc_by_classifier.get)

        # print(f'Selected classifier: {selected_classifier}')
        return selected_classifier

    def create_classifier(self, classifier_name: str):
        # return SVC(probability=True)
        if classifier_name == 'svm':
            return SVC(probability=True)
        elif classifier_name == 'extra_tree':
            return ExtraTreesClassifier()
        elif classifier_name == 'gb':
            return GradientBoostingClassifier()
        else:
            print(f"Error: invalid base classifier: {classifier_name}")
            sys.exit(1)

    def fit(self, X, y):
        self.n_labels = len(np.unique(y))

        print("Identifying best clusterer and classifier...")

        self.best_clusterer = \
                self.select_optimal_clustering_algorithm(X, y)

        self.best_classifier = self.select_optimal_classifier(X, y)

        self.set_bounds_pso()

        dimensions = len(self.bounds[0])

        print("Searching for best ensemble...")
        pso = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=dimensions,
            options=self.options, bounds=self.bounds,
            ftol_iter=self.ftol_iter, ftol=1e-4
        )
        fitness_func = self.fitness_eval(X, y)
        cost, solution = pso.optimize(fitness_func, iters=self.n_iters)

        self.best_solution = solution
        self.best_cost = cost

        params = self.unwrap_solution(self.best_solution)

        self.best_opt = CielOptimizer(
            self.best_clusterer,
            self.best_classifier,
            params['n_clusters'],
            params['clf_params'],
            params['weights'])
        self.best_opt.fit(X, y)

        self.best_opt.base_classifier = self.best_classifier
        self.best_opt.best_clustering_metrics = self.best_clustering_metrics
        self.best_opt.optimal_clusterer = self.best_clusterer

    def predict(self, X):
        y_pred, voting_weights, y_pred_by_cluster = self.best_opt.predict(X)
        return y_pred, voting_weights, y_pred_by_cluster

    def predict_proba(self, X):
        y_score, voting_weights, y_pred_by_cluster = \
                self.best_opt.predict_proba(X)
        return y_score, voting_weights, y_pred_by_cluster

    def random_restart(self, optimizer, particle_indices):
        """Resets given particles to random positions in bounds."""
        lb, ub = optimizer.bounds
        optimizer.swarm.position[particle_indices] = np.random.uniform(
            low=lb, high=ub,
            size=(len(particle_indices),optimizer.dimensions)
        )
        optimizer.swarm.velocity[particle_indices] = np.random.uniform(
            low=-abs(ub - lb), high=abs(ub - lb), 
            size=(len(particle_indices), optimizer.dimensions)
        )
        optimizer.swarm.pbest_pos[
                particle_indices] = optimizer.swarm.position[particle_indices]
        optimizer.swarm.pbest_cost[particle_indices] = np.inf  # force re-evaluation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True, help = "Dataset used.")
    parser.add_argument("-n", "--num_iters", type=int, default=10, help = "Number of PSO iters.")
    parser.add_argument("-p", "--num_particles", type=int, default=30, help = "Number of PSO particles")
    args = parser.parse_args()

    X, y = dataset_loader.select_dataset_function(args.dataset)()

    for fold in range(1, N_FOLDS+1):

        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        ciel = Ciel(n_iters=args.num_iters, n_particles=args.num_particles)
        ciel.fit(X_train, y_train)

        y_score, voting_weights, y_pred_by_cluster = ciel.predict_proba(X_val)
        y_pred = np.argmax(y_score, axis=1)

        prediction_results = PredictionResults(
            y_pred, y_val, voting_weights, y_pred_by_cluster, y_score
        )
        log = Logger(ciel.best_opt, args.dataset, prediction_results)
        log.save_data_fold_ciel(fold)
        print(classification_report(y_val, y_pred, zero_division=0.0))

if __name__ == "__main__":
    main()

