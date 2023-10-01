import pygad
import numpy as np
from clustering_classification import get_distances_between_diff_classes_per_cluster
from loader_and_preprocessor import read_potability_dataset
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
        return np.sum(dists_clusters)
    return wrapper_fitness

def calc_avg_overlap_attrib(X_attrib, dict_indices, n_labels):
    means_attrib = []
    stds_attrib = []

    for l in range(n_labels):
        # TODO Problema dos rótulos faltando
        indices = dict_indices[l]

        # if np.any(indices):
        mean = np.mean(X_attrib[indices])
        std = np.std(X_attrib[indices])
        means_attrib.append(mean)
        stds_attrib.append(std)
        # else:
        #    means_attrib.append(0)
        #    stds_attrib.append(0)

    overlap_values = []
    for l1 in range(n_labels):
        for l2 in range(l1+1, n_labels):
            # if stds_attrib[l1] != 0 and stds_attrib[l2] != 0:
            overlap = (
                means_attrib[l1] - means_attrib[l2])**2 / (
                stds_attrib[l1] ** 2 + stds_attrib[l2] ** 2)
            overlap_values.append(overlap)
    # if overlap_values:
    return np.mean(overlap_values)
    # else:
    #     return 0


def fitness_overlap(X_train, y_train, n_clusters, n_labels):
    def wrapper_fitness(ga_instance, solution, solution_idx):
        centroids = np.reshape(solution, (n_clusters, X_train.shape[1]))

        distances = [np.linalg.norm(X_train - center, axis=1)
                     for center in centroids]
        distances = np.array(distances)

        clusters = np.argmin(distances, axis=0)

        overlap_values = []

        for c in range(n_clusters):
            indices_cluster = np.where(clusters == c)[0]
            dict_indices = {}

            X_cluster, y_cluster = X_train[indices_cluster], y_train[indices_cluster]
            for l in range(n_labels):
                dict_indices[l] = np.where(y_cluster == l)[0]

            max_overlap_attrib = float('-inf')

            for attrib in range(X_cluster.shape[1]):
                overlap_attrib = calc_avg_overlap_attrib(
                    X_cluster[:, attrib], dict_indices, n_labels)

                if overlap_attrib > max_overlap_attrib:
                    max_overlap_attrib = overlap_attrib

            overlap_values.append(max_overlap_attrib)
        avg_overlap = np.mean(overlap_values)

        # overlap = np.mean(
        #     [(means[i] - means[j])**2 / (deviations[i]**2 + deviations[j]**2)
        #      for i in range(means.shape[0])
        #      for j in range(i + 1, deviations.shape[0])]
        # )
        return 1 / (avg_overlap + 0.00000001)
    return wrapper_fitness


def run_ga(fitness_func, n_genes):
    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=100,
                           num_genes=n_genes,
                           init_range_low=0,
                           init_range_high=1,
                           parent_selection_type="tournament",
                           keep_parents=1,
                           crossover_type="two_points",
                           mutation_type='random',
                           mutation_percent_genes=10,
                           gene_type=float,
                           gene_space=np.arange(0, 1, 0.001))
    ga_instance.run()
    best_solution = ga_instance.best_solution()[0]
    fitness_value = fitness_func(None, best_solution, 0)
    return best_solution, fitness_value


if __name__ == "__main__":
    df_potability = read_potability_dataset("potabilidade.csv")

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
