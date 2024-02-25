import pygad
import numpy as np
from xgboost.sklearn import PredtT
from clustering_classification import get_distances_between_diff_classes_per_cluster
from loader_and_preprocessor import read_potability_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score


def fitness_dists_centroids(X_train, y_train, n_clusters, n_labels, distances_samples):
    def wrapper_fitness(ga_instance, solution, solution_idx):
        centroids = np.reshape(solution, (n_clusters, X_train.shape[1]))

        cluster_distances = [np.linalg.norm(X_train - center, axis=1)
                             for center in centroids]
        cluster_distances = np.array(cluster_distances)
        clusters = np.argmin(cluster_distances, axis=0)

        dists_clusters = get_distances_between_diff_classes_per_cluster(
            y_train, clusters, n_clusters, n_labels, distances_samples)
        print(f"Distância {n_clusters} clusters {n_labels} labels:")
        print(dists_clusters)
        return np.mean(dists_clusters)
    return wrapper_fitness

def calc_avg_overlap_attrib(X_attrib, y, samples_label):
    means_attrib = []
    stds_attrib = []

    possible_labels = np.unique(sorted(y)).astype(int)

    for l in possible_labels:
        indices = samples_label[l]

        mean = np.mean(X_attrib[indices])
        std = np.std(X_attrib[indices])
        means_attrib.append(mean)
        stds_attrib.append(std)

    overlap_values = []

    for l1 in range(possible_labels.shape[0]):
        for l2 in range(l1+1, possible_labels.shape[0]):
            # if stds_attrib[l1] != 0 and stds_attrib[l2] != 0:
            overlap = (
                means_attrib[l1] - means_attrib[l2])**2 / (
                stds_attrib[l1] ** 2 + stds_attrib[l2] ** 2 + 0.000000001)
            overlap_values.append(overlap)
    if overlap_values:
        return np.mean(overlap_values)
    else:
        return 1


def fitness_overlap(X_train, y_train, n_clusters, n_labels):
    def wrapper_fitness(ga_instance, solution, solution_idx):
        centroids = np.reshape(solution, (n_clusters, X_train.shape[1]))

        distances = [np.linalg.norm(X_train - center, axis=1)
                     for center in centroids]
        distances = np.array(distances)

        clusters = np.argmin(distances, axis=0)

        if np.all(clusters == clusters[0]):
            return float('-inf')

        overlap_values = []
        weights_clusters = []

        for c in range(n_clusters):
            indices_cluster = np.where(clusters == c)[0]
            samples_label = {}

            X_cluster, y_cluster = X_train[indices_cluster], y_train[indices_cluster]
            for l in range(n_labels):
                samples_label[l] = np.where(y_cluster == l)[0]

            max_overlap_attrib = float('-inf')

            for attrib in range(X_cluster.shape[1]):
                overlap_attrib = calc_avg_overlap_attrib(
                    X_cluster[:, attrib], y_cluster, samples_label)

                if overlap_attrib > max_overlap_attrib:
                    max_overlap_attrib = overlap_attrib

            overlap_values.append(max_overlap_attrib)

            # Calculate the weight that this cluster has when calculating the overlap
            w_cluster = indices_cluster.shape[0] / X_train.shape[0]
            weights_clusters.append(w_cluster)

        avg_overlap = np.sum(
            [overlap_values[i] * weights_clusters[i]
             for i in range(n_clusters)]
        )
        # overlap = np.mean(
        #     [(means[i] - means[j])**2 / (deviations[i]**2 + deviations[j]**2)
        #      for i in range(means.shape[0])
        #      for j in range(i + 1, deviations.shape[0])]
        # )

        # silh_score = silhouette_score(X_train, clusters)
        return 1 / (avg_overlap + 0.00000001)
        # return silh_score
    return wrapper_fitness


def run_ga(fitness_func, n_genes):
    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=150,
                           num_genes=n_genes,
                           init_range_low=0,
                           init_range_high=1,
                           parent_selection_type="tournament",
                           keep_parents=1,
                           crossover_type="two_points",
                           mutation_type='random',
                           mutation_percent_genes=10,
                           gene_type=float,
                           gene_space=np.arange(0, 1, 0.001),
                           )
    ga_instance.run()
    best_solution = ga_instance.best_solution()[0]
    fitness_value = fitness_func(None, best_solution, 0)
    return best_solution, fitness_value


if __name__ == "__main__":
    df_potability = read_potability_dataset()

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
