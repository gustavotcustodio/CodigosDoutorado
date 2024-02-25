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
from clustering_classification import Experiments


EXPERIMENT_FOLDER = "experiments"

experiment_names = {Experiments.PAPER_66: "Clusters Combination (comparison)",
                    Experiments.CLUSTERING_ANALYSIS: "Clustering Analysis",
                    Experiments.CENTROID_OPTIMIZATION: "Centroid Optimization",
                    Experiments.RANDOM_FOREST: "Random Forest",
                    Experiments.GRADIENT_BOOSTING: "Gradient Boosting"
                    }

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

    min_clusters = 2
    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_fitness = float("-inf")
    optimal_n_clusters = 2

    for n_clusters in range(min_clusters, max_clusters):
        # centroids, final_fitness, clusters = perform_ga_step(X_train, y_train, n_clusters, n_labels)
        # score = silhouette_score(X_train, clusters)
        clusters, centroids = cc.cluster_data(X_train, n_clusters)
        fitness_func = fitness_dists_centroids(
            X_train, y_train, n_clusters, n_labels, dist_matrix)

        # score = fitness_func(None, centroids, 0)
        #TODO fala disso
        score = davies_bouldin_score(X_train, clusters)

        if score > best_fitness:
            best_fitness = score
            optimal_n_clusters = n_clusters
    return optimal_n_clusters


def plot_clusters(X_train, y_train, clusters, centroids, filename):
    n_centroids = centroids.shape[0]

    X_train = np.vstack((X_train, centroids))
    tsne_model = TSNE(n_components=2)
    low_dim_data = tsne_model.fit_transform(X_train)

    low_dim_centroids = low_dim_data[-n_centroids:]
    low_dim_data = low_dim_data[:-n_centroids]
    # low_dim_centroids = centroids
    # low_dim_data = X_train

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

    plt.legend()  # (loc='upper left', bbox_to_anchor=(1, 1))

    fullpath = os.path.join(EXPERIMENT_FOLDER, "pca_matrices", f"{filename}.png")
    plt.savefig(fullpath)
    plt.clf()
    print("Cluster map saved successfully.")


def perform_ga_step(X_train, y_train, n_labels):
    n_clusters = find_optimal_n_clusters(X_train, y_train, n_labels)
    n_genes = n_clusters * X_train.shape[1]

    dist_matrix = distance_matrix(X_train, X_train)
    ff = fitness_dists_centroids(X_train, y_train, n_clusters, n_labels, dist_matrix)

    centroids, final_fitness = run_ga(ff, n_genes)
    centroids = centroids.reshape((n_clusters, X_train.shape[1]))

    distances = [np.linalg.norm(X_train - center, axis=1)
                 for center in centroids]
    distances = np.array(distances)

    clusters = np.argmin(distances, axis=0)

    return clusters, centroids, final_fitness


def perform_clustering_analysis(X_train, y_train, n_labels, n_clusters=None):
    max_clusters = int(np.sqrt(X_train.shape[0]))
    min_clusters = 2
    best_centroids = None
    best_fitness = float("-inf")
    best_labels = None

    dist_matrix = distance_matrix(X_train, X_train)

    if n_clusters:
        fitness_function = fitness_dists_centroids(
            X_train, y_train, n_clusters, n_labels, dist_matrix
        )
        clusters, centroids, fitness_value = cc.cluster_data(X_train, n_clusters, fitness_function)
        return clusters, centroids, fitness_value

    for c in range(min_clusters, max_clusters + 1):
        fitness_function = fitness_dists_centroids(
            X_train, y_train, c, n_labels, dist_matrix
        )
        clusters, centroids, fitness_value = cc.cluster_data(X_train, c, fitness_function)
        if fitness_value > best_fitness:
            best_centroids = centroids
            best_fitness = fitness_value
            best_labels = clusters

    return best_labels, best_centroids, best_fitness


# def run_pca(X_train, X_test):
#     pca = PCA(n_components = 2)
#     X_train = pca.fit_transform(X_train)
#     X_test = pca.transform(X_test)
#     return X_train, X_test
def show_number_labels_by_cluster(labels, clusters, n_labels, filename): 
    n_samples = len(labels)
    n_clusters = np.unique(clusters).shape[0]
    cluster_counting = [[0 for l in range(n_clusters)]
                        for c in range(n_labels)]
    possible_labels = [l for l in range(n_labels)]

    for i in range(n_samples):
        c = int(clusters[i])
        l = int(labels[i])

        cluster_counting[l][c] += 1
    
    # Plot the barplot with the information
    fig, ax = plt.subplots(layout='constrained')
    width = 0.25
    multiplier = 0

    x = np.arange(n_clusters)
    for lbl in range(n_labels):
        offset = width * multiplier
        current_lbl = multiplier % (n_labels + 1)
        rects = ax.bar(x + offset, cluster_counting[lbl], width,
                       label=f"Label {current_lbl}")
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Number of samples')
    ax.set_title('Distribution of labels by cluster')
    ax.legend(loc='best', ncols=3)
    ax.set_xticks(x + width, [f"Cluster {c+1}" for c in range(n_clusters)])
    fullpath = os.path.join(EXPERIMENT_FOLDER, "distributions_per_cluster", f"{filename}.png")

    plt.savefig(fullpath)
    plt.clf()
    print(f"Cluster distribution saved successfully.")


def calc_avg_separability(X_train, y_train, centroids, predicted_labels):
    # TODO
    n_clusters = centroids.shape[0]

    dist_matrix = distance_matrix(X_train, X_train)
    avg_distance = fitness_function(None, centroids, 0)

    fitness_function = fitness_dists_centroids(
        X_train, y_train, n_clusters, n_labels, dist_matrix
    )

    # for d in distances_samples:
    # fitness_dists_centroids(X_train, y_train, n_clusters, n_labels, distances_samples)
    # Calculate the average distance for each cluster


def run_experiments_for_dataset(dataset_name, read_function, experiment, n_runs=1, n_clusters=None):
    print(f"================== {dataset_name} =====================")

    X, y = read_function()

    n_labels = np.unique(y).shape[0]

    accuracies = []
    recalls = []
    precisions = []
    f1scores = []
    numbers_clusters = []
    all_base_classifiers = []

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        # X_train, X_test = run_pca(X_train, X_test)

        # Definir centroides e clusters
        if experiment == Experiments.PAPER_66:
            print("Combining clusters of different labels:")
            X_train, y_train, centroids, clusters = cc.construct_training_clusters(X_train, y_train)
        elif experiment == Experiments.CLUSTERING_ANALYSIS:
            if n_clusters:
                clusters, centroids, final_fitness = perform_clustering_analysis(
                    X_train, y_train, n_labels, n_clusters
                )
            else:
                print("Searching for best partitioning:")
                clusters, centroids, final_fitness = perform_clustering_analysis(
                    X_train, y_train, n_labels
                )
        elif experiment == Experiments.CENTROID_OPTIMIZATION:
            print("Searching centroids using GA:")
            clusters, centroids, final_fitness = perform_ga_step(X_train, y_train, n_labels)

        if n_clusters:
            filename = f"{experiment_names[experiment]}_{dataset_name}_{n_clusters}clusters_run_{run + 1}"
        else:
            filename = f"{experiment_names[experiment]}_{dataset_name}_run_{run + 1}"

        if experiment != Experiments.RANDOM_FOREST and experiment != Experiments.GRADIENT_BOOSTING:
            n_clusters = len(centroids)
            print("Number of clusters:", n_clusters)

            plot_clusters(X_train, y_train, clusters, centroids, filename)

            base_classifiers = cc.preselect_base_classifiers(X_train, y_train, clusters, n_clusters)
            y_pred = ensemble_classification(X_train, y_train, X_test, y_test, centroids,
                                             clusters, base_classifiers)

            show_number_labels_by_cluster(y_pred, clusters, n_labels, filename)
        else:
            base_classifiers = "Simple"
            # Perform classification with classical algorithms (Random Forest, Gradient Boosting)
            y_pred = cc.regular_classification(X_train, y_train, X_test, y_test, experiment)
        # Show the percentage of each samples with each label in each different cluster.


        plot_confusion_matrix(y_test, y_pred, dataset_name,
                              experiment_names[experiment], filename)

        accuracy = sum(y_pred == y_test) / len(y_test)
        accuracies.append( accuracy )
        recalls.append( recall_score(y_test, y_pred, average="macro") )
        precisions.append( precision_score(y_test, y_pred, average="macro", zero_division=0.0) )
        f1scores.append( f1_score(y_test, y_pred, average="macro") )
        numbers_clusters.append(n_clusters)
        all_base_classifiers.append(str(base_classifiers))

    return {"Accuracy" : accuracies,
            "Recall" : recalls,
            "Precision" : precisions,
            "F1-Score" : f1scores,
            "n_clusters": numbers_clusters,
            "base_classifiers": all_base_classifiers
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
    base_classifiers = df_results["base_classifiers"]
    df_results.drop(columns="base_classifiers", inplace=True)

    avg_values = df_results.mean()
    std_values = df_results.std()

    df_results = df_results._append(avg_values, ignore_index=True)
    df_results = df_results._append(std_values, ignore_index=True)
    df_results["run"] = [f"run{i}" for i in range(1, n_runs+1)] + ["Mean", "Std"]
    df_results["base_classifiers"] = base_classifiers.tolist() + ["", ""]

    fullpath = os.path.join(EXPERIMENT_FOLDER, f"{filename}.csv")
    df_results.to_csv(fullpath, index=False)
    print(f"{filename}.csv saved successfully.")


def run_experiment_and_save_results(dataset, experiment_type, n_runs, n_clusters=None):
    if dataset == "Credit Score":
        loader_function = read_german_credit_dataset
    elif dataset == "Water":
        loader_function = read_potability_dataset
    elif dataset == "Wine":
        loader_function = read_wine_dataset
    else:
        loader_function = read_wdbc_dataset

    results = run_experiments_for_dataset(
        dataset, loader_function, experiment_type, n_runs, n_clusters
    )
    n_clusters = results['n_clusters'][0]
    filename = f"{dataset}_{experiment_names[experiment_type]}_nclusters_{n_clusters}"
    calc_avg_std_save_results(results, n_runs, filename)


if __name__ == "__main__":
    n_runs = 7
    #############################
    # results = run_experiments_for_dataset(dataset, read_german_credit_dataset,
    #                                       experiment_type, n_runs)
    # filename = f"{dataset}_{experiment_names[experiment_type]}"
    # calc_avg_std_save_results(results, n_runs, filename)
    #############################
    experiment_type = Experiments.RANDOM_FOREST

    dataset = "Cancer"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Credit Score"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Water"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Wine"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    ################################
    experiment_type = Experiments.GRADIENT_BOOSTING

    dataset = "Cancer"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Credit Score"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Water"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Wine"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    #############################
    experiment_type = Experiments.CLUSTERING_ANALYSIS

    for c in range(2, 6):

        dataset = "Cancer"
        run_experiment_and_save_results(dataset, experiment_type, n_runs, n_clusters = c)

        dataset = "Credit Score"
        run_experiment_and_save_results(dataset, experiment_type, n_runs, n_clusters = c)

        dataset = "Water"
        run_experiment_and_save_results(dataset, experiment_type, n_runs, n_clusters = c)

        dataset = "Wine"
        run_experiment_and_save_results(dataset, experiment_type, n_runs, n_clusters = c)

    ###############################
    experiment_type = Experiments.PAPER_66

    dataset = "Cancer"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Credit Score"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Water"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    dataset = "Wine"
    run_experiment_and_save_results(dataset, experiment_type, n_runs)

    ###############################
