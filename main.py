import os
import matplotlib.pyplot as plt
import pandas as pd
import clustering_classification as cc
from collections import Counter
from clustering_classification import ensemble_classification
from loader_and_preprocessor import read_potability_dataset, read_wine_dataset, read_wdbc_dataset, read_german_credit_dataset
from genetic_algorithm import fitness_dists_centroids, run_ga, fitness_overlap
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.model_selection import cross_val_score


EXPERIMENT_FOLDER = "experiments"


PAPER_66 = 0
CLUSTERING_ANALYSIS = 1
CENTROID_OPTIMIZATION = 2


experiment_names = ["Clusters Combination (comparison)", "Clustering Analysis", "Centroid Optimization"]

def plot_confusion_matrix(y_test, y_pred, dataset_name, classifier, filename):
    # print(f"{classifier}:", accuracy)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{classifier} {dataset_name}")
    fullpath = os.path.join(EXPERIMENT_FOLDER, "confusion_matrices", f"{filename}.png")
    plt.savefig(fullpath)
    plt.clf()
    print("Confusion matrix saved successfully")


def find_optimal_n_clusters(X_train, y_train, n_labels):
    dist_matrix = distance_matrix(X_train, X_train)

    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_fitness = float("-inf")
    optimal_n_clusters = 2

    for n_clusters in range(2, max_clusters):
        # centroids, final_fitness, clusters = perform_ga_step(X_train, y_train, n_clusters, n_labels)
        # score = silhouette_score(X_train, clusters)
        clusters, centroids = cc.cluster_data(X_train, n_clusters)
        fitness_func = fitness_dists_centroids(
            X_train, y_train, n_clusters, n_labels, dist_matrix)

        # score = fitness_func(None, centroids, 0)
        score = davies_bouldin_score(X_train, clusters)

        if score > best_fitness:
            best_fitness = score
            optimal_n_clusters = n_clusters
    return optimal_n_clusters


def plot_clusters(X_train, y_train, clusters, centroids, filename):
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

    fullpath = os.path.join(EXPERIMENT_FOLDER, "pca_matrices", f"{filename}.png")
    plt.savefig(fullpath)
    plt.clf()
    print("Cluster map saved successfully.")


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
    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_centroids = None
    best_fitness = float("-inf")
    best_labels = None

    dist_matrix = distance_matrix(X_train, X_train)

    for c in range(2, max_clusters + 1):
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


def run_experiments_for_dataset(dataset_name, read_function, experiment, n_runs=1):
    print(f"================== {dataset_name} =====================")

    X, y = read_function()

    n_labels = np.unique(y).shape[0]

    accuracies = []
    recalls = []
    precisions = []
    f1scores = []
    numbers_clusters = []

    for run in range(n_runs):
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

        n_clusters = len(centroids)
        print("Number of clusters:", n_clusters)

        filename = f"{experiment_names[experiment]}_{dataset_name}_run_{run + 1}"

        plot_clusters(X_train, y_train, clusters, centroids, filename)

        base_classifiers = cc.preselect_base_classifiers(X_train, y_train, n_clusters)
        y_pred = ensemble_classification(X_train, y_train, X_test, y_test, centroids,
                                         clusters, base_classifiers)

        plot_confusion_matrix(y_test, y_pred, dataset_name,
                              experiment_names[experiment], filename)

        accuracy = sum(y_pred == y_test) / len(y_test)
        accuracies.append( accuracy )
        recalls.append( recall_score(y_test, y_pred, average="macro") )
        precisions.append( precision_score(y_test, y_pred, average="macro") )
        f1scores.append( f1_score(y_test, y_pred, average="macro") )
        numbers_clusters.append(n_clusters)

    return {"Accuracy" : accuracies,
            "Recall" : recalls,
            "Precision" : precisions,
            "F1-Score" : f1scores,
            "n_clusters": numbers_clusters
            }
    # classifier = RandomForestClassifier()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # accuracy = sum(y_pred == y_test) / len(y_test)

    # plot_confusion_matrix(accuracy, y_test, y_pred, dataset_name, "Random Forest")

    # classifier = GradientBoostingClassifier()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # accuracy = sum(y_pred == y_test) / len(y_test)
    # 

    # plot_confusion_matrix(accuracy, y_test, y_pred, dataset_name, "GBClassifier")


def calc_avg_std_save_results(results, n_runs, filename):
    df_results = pd.DataFrame(results)

    avg_values = df_results.mean()
    std_values = df_results.std()

    df_results = df_results._append(avg_values, ignore_index=True)
    df_results = df_results._append(std_values, ignore_index=True)
    df_results["run"] = [f"run{i}" for i in range(1, n_runs+1)] + ["Mean", "Std"]

    fullpath = os.path.join(EXPERIMENT_FOLDER, f"{filename}.csv")
    df_results.to_csv(fullpath, index=False)


if __name__ == "__main__":
    n_runs = 2
    # dataset = "Credit Score"
    # results = run_experiments_for_dataset(dataset, read_german_credit_dataset,
    #                                       CLUSTERING_ANALYSIS, n_runs)
    # filename = f"{dataset}_{experiment_names[CLUSTERING_ANALYSIS]}"
    # calc_avg_std_save_results(results, n_runs, filename)

    # dataset = "Water"
    # results = run_experiments_for_dataset(dataset, read_potability_dataset,
    #                                       CLUSTERING_ANALYSIS, n_runs)
    # filename = f"{dataset}_{experiment_names[CLUSTERING_ANALYSIS]}"
    # calc_avg_std_save_results(results, n_runs, filename)

    dataset = "Wine"
    results = run_experiments_for_dataset(
        dataset, read_wine_dataset, CLUSTERING_ANALYSIS, n_runs
    )

    filename = f"{dataset}_{experiment_names[CLUSTERING_ANALYSIS]}"
    calc_avg_std_save_results(results, n_runs, filename)

    dataset = "Cancer"
    results = run_experiments_for_dataset(
        dataset, read_wdbc_dataset, CLUSTERING_ANALYSIS, n_runs
    )

    filename = f"{dataset}_{experiment_names[CLUSTERING_ANALYSIS]}"
    calc_avg_std_save_results(results, n_runs, filename)

    # run_experiments_for_dataset("Credit Score", read_german_credit_dataset, PAPER_66)
    # run_experiments_for_dataset("Water", read_potability_dataset, PAPER_66)
    # run_experiments_for_dataset("Wine", read_wine_dataset, PAPER_66)
    # run_experiments_for_dataset("Cancer", read_wdbc_dataset, PAPER_66)
