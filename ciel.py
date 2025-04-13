# Tabela 5
# Quantos indivídios tem nesse PSO?
# pág 17
import pyswarms as ps
from sklearn.model_selection import train_test_split
import dataset_loader
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ciel_optimizer import CielOptimizer

class Ciel:

    def __init__(self):
        pass

    def fitness_eval(self, X_train, X_test, y_train, y_test):
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
        
    def fit(self):
        bounds = ([1, 1, 2, 1], [500, 10, 10, 10])
        pso = ps.single.GlobalBestPSO(
            n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds
        )

        cost, solution = pso.optimize(fitness_func, iters=10)

        n_estimators=round(solution[0])
        max_depth=round(solution[1])
        min_samples_split=round(solution[2])
        min_samples_leaf=round(solution[3])

        pass


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
    X, y = dataset_loader.select_dataset_function("heart")()

    options = { 'c1': 1.49445, 'c2': 1.49445, 'w': 0.729, }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    fitness_func = test_params_optimization(X_train, X_test, y_train, y_test)
    
    dimensions = 4
    n_particles = 20

    # Add constraints
    bounds = ([1, 1, 2, 1], [500, 10, 10, 10])
    pso = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds
    )
    cost, solution = pso.optimize(fitness_func, iters=10)

    et = ExtraTreesClassifier(
        n_estimators=round(solution[0]), max_depth=round(solution[1]),
        min_samples_split=round(solution[2]), min_samples_leaf=round(solution[3]),
    )
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)

    print( accuracy_score(y_test, y_pred) )
