# Tabela 5
# Quantos indivídios tem nesse PSO?
# pág 17
from math import cos
import pyswarms as ps
import numpy as np
from sklearn.model_selection import train_test_split
import dataset_loader
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ciel_optimizer import CielOptimizer
from ciel_optimizer import N_FOLDS
from sklearn.model_selection import StratifiedKFold, cross_validate

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
                # TODO estudar fazer o hash dos parâmetros?

                auc_values = []
                folds_splits = kf.split(X, y)
                for fold, (train_indexes, test_indexes) in enumerate(folds_splits):
                    X_train, y_train = X[train_indexes], y[train_indexes]
                    X_test, y_test = X[test_indexes], y[test_indexes]

                    ciel_opt.fit(X_train, y_train)
                    # Predict probability
                    y_score = ciel_opt.predict_proba(X_test)
                    
                    if self.n_labels == 2:
                        auc_val = roc_auc_score(y_test, y_score[:,1])
                    else:
                        auc_val = roc_auc_score(y_test, y_score, multi_class="ovr")

                    print(auc_val)
                    auc_values.append(auc_val)

                cost = 1 - np.mean(auc_values)
                cost_values.append(cost)

            print(cost_values)
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

    def predict(self, X):
        return self.best_opt.predict(X)


def test_params_optimization(X_train, X_test, y_train, y_test):
    def wrapper(possible_solutions):
        # The function receives (n_particles x n_dims)
        results = []

        for solution in possible_solutions:
            et = ExtraTreesClassifier(
                n_estimators=round(solution[0]), max_depth=round(solution[1]),
                min_samples_split=round(solution[2]), min_samples_leaf=round(solution[3]),
            )
            et.fit(X_train, y_train)
            y_pred = et.predict(X_test)
            results.append(1 - accuracy_score(y_test, y_pred) )
        return results
    return wrapper
         
    # fit function (inside the fit function the best optimizer is saved)

if __name__ == "__main__":
    X, y = dataset_loader.select_dataset_function("german_credit")()

    # Randomize samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    folds_splits = kf.split(X, y)

    for fold, (train_indexes, test_indexes) in enumerate(folds_splits):
        ciel = Ciel(n_iters=7, n_particles=5)
        X_train, y_train = X[train_indexes], y[train_indexes]
        X_test, y_test = X[test_indexes], y[test_indexes]

        ciel.fit(X_train, y_train)
        y_pred = ciel.predict(X_test)
        print(classification_report(y_test, y_pred))
