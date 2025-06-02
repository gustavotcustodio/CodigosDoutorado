# Tabela 5
# Quantos indivídios tem nesse PSO?
# pág 17
import pyswarms as ps
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import dataset_loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from ciel_optimizer import CielOptimizer
from ciel_optimizer import N_FOLDS
from sklearn.model_selection import StratifiedKFold
from logger import PredictionResults
from logger import Logger
from dataset_loader import normalize_data

class Ciel:

    def __init__(self, n_iters: int, n_particles: int):
        self.n_iters = n_iters
        self.n_particles = n_particles
        self.max_n_clusters = 10
                        # n_clusters + svm_params + extra_tree_params + grad_boost_params + weights
        lower_bounds = [2] + ([1e-4, 1e-4] + [1, 1, 2, 1] + [1, 1, 2, 1, 0.1]
                              ) * self.max_n_clusters + [0.1] * self.max_n_clusters
        upper_bounds = [10] + ([1000, 1000] + [500, 10, 10, 10] + [500, 10, 10, 10, 1.0]
                               ) * self.max_n_clusters + [1.0] * self.max_n_clusters

        assert len(lower_bounds) == len(upper_bounds)
        self.bounds = (lower_bounds, upper_bounds)
        self.options = { 'c1': 1.49445, 'c2': 1.49445, 'w': 0.729, }

    def unwrap_solution(self, solution) -> dict[str, list|int]:
        n_clusters = round(solution[0])
        svm_params = []
        extra_tree_params = []
        grad_boost_params = []

        weights = np.zeros(n_clusters)

        for c in range(n_clusters):
            svm_params.append({})
            extra_tree_params.append({})
            grad_boost_params.append({})

            svm_params[c]['cost'] = solution[11 * c + 1]
            svm_params[c]['gamma'] = solution[11 * c + 2]

            extra_tree_params[c]['n_estimators'] = round(solution[11 * c + 3])
            extra_tree_params[c]['max_depth'] = round(solution[11 * c + 4])
            extra_tree_params[c]['min_samples_split'] = round(solution[11 * c + 5])
            extra_tree_params[c]['min_samples_leaf'] = round(solution[11 * c + 6])

            grad_boost_params[c]['n_estimators'] = round(solution[11 * c + 7])
            grad_boost_params[c]['max_depth'] = round(solution[11 * c + 8])
            grad_boost_params[c]['min_samples_split'] = round(solution[11 * c + 9])
            grad_boost_params[c]['min_samples_leaf'] = round(solution[11 * c + 10])
            grad_boost_params[c]['learning_rate'] = solution[11 * c + 11]

            weights[c] = solution[c + 110]

        weights  = weights / weights.sum()

        params = {}
        params['n_clusters'] = n_clusters
        params['svm_params'] = svm_params
        params['extra_tree_params'] = extra_tree_params
        params['grad_boost_params'] = grad_boost_params
        params['weights'] = weights

        return params

    def fitness_eval(self, X, y):
        def wrapper(possible_solutions):
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
            # The function receives (n_particles x n_dims)
            cost_values = []

            for solution in possible_solutions:
                # Convert PSO particle to the parameters
                params = self.unwrap_solution(solution)

                ciel_opt = CielOptimizer(
                    params['n_clusters'], params['svm_params'],
                    params['extra_tree_params'], params['grad_boost_params'],
                    params['weights'])

                auc_values = []
                folds_splits = kf.split(X, y)

                for _, (train_indexes, test_indexes) in enumerate(folds_splits):
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
                cost_values.append(cost)

            return cost_values
        return wrapper

    def fit(self, X, y):
        self.n_labels = len(np.unique(y))

        dimensions = len(self.bounds[0])
        pso = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=dimensions,
            options=self.options, bounds=self.bounds
        )
        fitness_func = self.fitness_eval(X, y)
        cost, solution = pso.optimize(fitness_func, iters=self.n_iters)

        self.best_solution = solution
        self.best_cost = cost

        params = self.unwrap_solution(self.best_solution)
        self.best_opt = CielOptimizer(
            params['n_clusters'], params['svm_params'], params['extra_tree_params'],
            params['grad_boost_params'], params['weights'])
        self.best_opt.fit(X_train, y_train)

        # self.base_classifier = self.best_opt.base_classifier
        # self.labels_by_cluster = self.best_opt.labels_by_cluster
        # self.best_clustering_metrics = self.best_opt.best_clustering_metrics
        # self.n_clusters = self.best_opt.best_clustering_metrics

    def predict(self, X):
        y_pred, voting_weights, y_pred_by_cluster = self.best_opt.predict(X)
        return y_pred, voting_weights, y_pred_by_cluster

    def predict_proba(self, X):
        y_score, voting_weights, y_pred_by_cluster  = self.best_opt.predict_proba(X)
        return y_score, voting_weights, y_pred_by_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True, help = "Dataset used.")
    parser.add_argument("-n", "--num_iters", type=int, required=True, help = "Number of PSO iters.")
    parser.add_argument("-p", "--num_particles", type=int, required=True, help = "Number of PSO particles")
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
                y_pred, y_val, voting_weights, y_pred_by_cluster, y_score)
        log = Logger(ciel.best_opt, args.dataset, prediction_results)
        log.save_data_fold_ciel(fold)
        print(classification_report(y_val, y_pred, zero_division=0.0))

