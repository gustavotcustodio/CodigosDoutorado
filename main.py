from clustering_classification import ensemble_classification
from loader_and_preprocessor import read_dataset
import clustering_classification as cc
from genetic_algorithm import run_ga, fitness_dists_centroids
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans


def perform_ga_step():
    n_genes = n_clusters * X.shape[1]

    ff = fitness_dists_centroids(X_train, y_train, n_clusters, n_labels)

    centroids, final_fitness = run_ga(ff, n_genes)
    centroids = centroids.reshape((n_clusters, X.shape[1]))

    return centroids, final_fitness


if __name__ == "__main__":
    df_potability = read_dataset("potabilidade.csv")

    X = df_potability.drop(columns="Potability").values
    X = (X - X.min()) / (X.max() - X.min())

    y = df_potability["Potability"].values

    n_labels = np.unique(y).shape[0]
    n_clusters = 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    centroids, final_fitness = perform_ga_step()

    u = cc.calc_membership_values(X, centroids)
    # ensemble_classification(X, y, u)

    print("GA:", final_fitness)

    # =======================================================
    ff = fitness_dists_centroids(X_train, y_train, n_clusters, n_labels)

    clusterer = KMeans(n_clusters, n_init='auto')
    clusterer.fit(X_train)
    # print(ff(None, clusterer.cluster_centers_, 0))  # , clusterer.cluster_centers_
    fitness_kmeans = ff(None, clusterer.cluster_centers_, 0)

    print("KMeans:", fitness_kmeans)
