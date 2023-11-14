from clustering_classification import ensemble_classification
from loader_and_preprocessor import read_potability_dataset, read_wine_dataset, read_wdbc_dataset, read_german_credit_dataset
import clustering_classification as cc
from genetic_algorithm import fitness_dists_centroids, run_ga, fitness_overlap
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix


PAPER_66 = 0
CLUSTERING_ANALYSIS = 1
CENTROID_OPTIMIZATION = 2


def find_optimal_n_clusters(X_train, y_train, n_labels):
    dist_matrix = distance_matrix(X_train, X_train)

    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_fitness = float("-inf")
    optimal_n_clusters = min_clusters

    for n_clusters in range(2, max_clusters):
        # centroids, final_fitness, clusters = perform_ga_step(X_train, y_train, n_clusters, n_labels)
        # score = silhouette_score(X_train, clusters)
        clusters, centroids = cc.cluster_data(X_train, n_clusters)
        fitness_func = fitness_dists_centroids(
            X_train, y_train, n_clusters, n_labels, dist_matrix)

        score = fitness_func(None, centroids, 0)
        # score = score + silhouette_score(X_train, clusters)

        if score > best_fitness:
            best_fitness = score
            optimal_n_clusters = n_clusters
    return optimal_n_clusters


def plot_clusters(X_train, y_train, clusters, centroids):
    n_centroids = centroids.shape[0]

    X_train = np.vstack((X_train, centroids))
    # tsne_model = TSNE(n_components=2)
    # low_dim_data = tsne_model.fit_transform(X_train)

    # low_dim_centroids = low_dim_data[-n_centroids:]
    # low_dim_data = low_dim_data[:-n_centroids]
    low_dim_centroids = centroids
    low_dim_data = X_train

    for c in np.unique(clusters):
        idx_cluster = np.where(clusters == c)[0]

        for l in np.unique(y_train):
            idx_labels = np.where(y_train == l)[0]
            idx = np.intersect1d(idx_cluster, idx_labels)

            if len(idx) > 0:
                data_clusters = low_dim_data[idx]
                plt.scatter(data_clusters[:, 0], data_clusters[:, 1],
                            alpha=0.7, label="c %d lbl %d" % (c, l))

    plt.scatter(low_dim_centroids[:, 0], low_dim_centroids[:, 1], marker = "x", s=200, color="black")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def perform_ga_step(X_train, y_train, n_labels):
    n_clusters = find_optimal_n_clusters(X_train, y_train, n_labels)
    n_genes = n_clusters * X_train.shape[1]

    dist_matrix = distance_matrix(X_train, X_train)
    ff = fitness_dists_centroids(X_train, y_train, n_clusters,
                                 n_labels, dist_matrix)

    centroids, final_fitness = run_ga(ff, n_genes)
    centroids = centroids.reshape((n_clusters, X_train.shape[1]))

    distances = [np.linalg.norm(X_train - center, axis=1)
                 for center in centroids]
    distances = np.array(distances)

    clusters = np.argmin(distances, axis=0)

    return centroids, final_fitness, clusters


def perform_clustering_analysis(X_train, y_train, n_labels):
    min_clusters = 2
    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_centroids = None
    best_fitness = float("-inf")
    best_labels = None

    dist_matrix = distance_matrix(X_train, X_train)

    for c in range(min_clusters, max_clusters + 1):
        fitness_function = fitness_dists_centroids(
            X_train, y_train, c, n_labels, dist_matrix)

        clusterer = KMeans(n_clusters=c, n_init='auto')
        clusterer.fit(X_train)
        fitness_value = fitness_function(None, clusterer.cluster_centers_, 0)
        if fitness_value > best_fitness:
            best_centroids = clusterer.cluster_centers_
            best_fitness = fitness_value
            best_labels = clusterer.labels_

    return best_centroids, best_fitness, best_labels


def run_pca(X_train, X_test):
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test


def run_experiments_for_dataset(dataset_name, read_function, experiment):
    print(f"================== {dataset_name} =====================")

    X, y = read_function()

    n_labels = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    # X_train, X_test = run_pca(X_train, X_test)

    if experiment == PAPER_66:
        print("Combining clusters of different labels:")
        X_train, y_train, centroids, clusters = cc.construct_training_clusters(X_train, y_train)
    elif experiment == CLUSTERING_ANALYSIS:
        print("Searching for best partitioning:")
        centroids, final_fitness, clusters = perform_clustering_analysis(X_train, y_train, n_labels)
    else:
        print("Searching centroids using GA:")
        centroids, final_fitness, clusters = perform_ga_step(X_train, y_train, n_labels)

    plot_clusters(X_train, y_train, clusters, centroids)

    best_clf = cc.preselect_classifiers(X_train, y_train)
    ensemble_classification(X_train, y_train, X_test, y_test, centroids,
                            clusters, base_classifier = best_clf)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    print("Random Forest:", accuracy)

    classifier = GradientBoostingClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    print("GBClassifier:", accuracy)


if __name__ == "__main__":
    run_experiments_for_dataset("Credit Score", read_german_credit_dataset, CLUSTERING_ANALYSIS)
    # run_experiments_for_dataset("Water", read_potability_dataset, CLUSTERING_ANALYSIS)
    run_experiments_for_dataset("Wine", read_wine_dataset, CLUSTERING_ANALYSIS)
    run_experiments_for_dataset("Cancer", read_wdbc_dataset, CLUSTERING_ANALYSIS)
