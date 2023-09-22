import pygad
import numpy as np
from clustering_classification import get_distances_between_diff_classes_per_cluster
from loader_and_preprocessor import read_dataset
from sklearn.model_selection import train_test_split


def fitness_dists_centroids(X_train, y_train, n_clusters, n_labels):
    def wrapper_fitness(ga_instance, solution, solution_idx):
        centroids = np.reshape(solution, (n_clusters, X_train.shape[1]))

        distances = [np.linalg.norm(X_train - center, axis=1)
                     for center in centroids]
        distances = np.array(distances)

        clusters = np.argmin(distances, axis=0)
        dists_clusters = get_distances_between_diff_classes_per_cluster(
            X_train, y_train, clusters, n_clusters, n_labels)
        return sum(dists_clusters)
    return wrapper_fitness


def run_ga(fitness_func, n_genes):
    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=n_genes,
                           init_range_low=0,
                           init_range_high=1,
                           parent_selection_type="rws",
                           keep_parents=1,
                           crossover_type="single_point",
                           mutation_type='random',
                           mutation_percent_genes=10,
                           gene_type=float,
                           gene_space=np.arange(0, 1, 0.001))
    ga_instance.run()
    best_solution = ga_instance.best_solution()[0]
    fitness_value = fitness_func(None, best_solution, 0)
    return best_solution, fitness_value


if __name__ == "__main__":
    df_potability = read_dataset("potabilidade.csv")

    X = df_potability.drop(columns="Potability").values
    X = (X - X.min()) / (X.max() - X.min())

    y = df_potability["Potability"].values
    n_labels = np.unique(y).shape[0]
    n_clusters = 3

    n_genes = X.shape[1] * n_clusters

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    solution = np.random.random(size=(X_train.shape[1] * n_clusters))
    ff = fitness_dists_centroids(X_train, y_train, n_clusters, n_labels)

    import time
    start_time = time.time()

    best_solution = run_ga(ff, n_genes)

    print(best_solution)
    print("Demorou", time.time() - start_time, "segundos")
