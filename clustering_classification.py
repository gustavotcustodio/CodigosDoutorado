from math import dist
import numpy as np
import pandas as pd
from pandas._libs.hashtable import mode
from sklearn.cluster import KMeans
from numpy.typing import NDArray
from scipy.spatial import distance

def read_dataset(dataset_name: str) -> pd.DataFrame:
    df_potability = pd.read_csv(dataset_name)
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    return df_potability


# def calc_distances_diff_classes(data_cluster: NDArray, n_labels: int
#                                 ) -> NDArray:
#     """
#     Calc distance between data samples from different
#     classes in the same cluster.
#
#     data_cluster: data from a specific cluster
#     """
#     distances = np.zeros((n_labels, n_labels))
#     samples_class = {lbl: np.where(data_cluster == lbl)[0]
#                      for lbl in range(n_labels)}
# 
#     # Percorre todos os labels possíveis
#     for l1 in range(n_labels):
#         for l2 in range(l1, n_labels):
#             distances[l1, l2] = 0  # [i + j for i, j in range(10)]
#             # np.sum(
#             #     (data_cluster[l1] - data_cluster[l2]) ** 2
#             # )
#     return distances


def calc_centroids_same_cluster(X_cluster, y_cluster, n_labels):
    """
    Calculate the centroids for samples that belong to the same class
    inside the same cluster.
    """
    centroids = np.empty((n_labels, X_cluster.shape[1]))
    # Select only the labels that are in the current cluster
    labels_in_cluster = []

    for l in range(n_labels):
        class_idxs = np.where(y_cluster == l)[0]

        # If there is none instances with this label in the cluster, skip it
        if len(class_idxs) > 0:
            centroids[l, :] = np.mean(X_cluster[class_idxs], axis=0)
            labels_in_cluster.append(l)
    return centroids[labels_in_cluster]


def cluster_data(data, n_clusters: int) -> NDArray | None:
    clusterer = KMeans(n_clusters, n_init='auto')
    clusterer.fit(data)
    return clusterer.labels_


def get_distances_between_diff_classes_per_cluster(X, y, clusters, n_clusters, n_labels):
    dist_centroids_cluster = np.empty((n_clusters))

    for c in range(n_clusters):
        # Samples belonging to the specific cluster c
        samples_c = np.where(clusters == c)[0]
        centroids_by_class = calc_centroids_same_cluster(
            X[samples_c], y[samples_c], n_labels)

        # Sum of Distances between the centroids from different classes in the same cluster
        dist_centroids_cluster[c] = np.sum(distance.pdist(centroids_by_class))

    return dist_centroids_cluster


if __name__ == "__main__":
    df_potability = read_dataset("./potabilidade.csv")

    X = df_potability.drop(columns="Potability").values
    X = (X - X.min()) / (X.max() - X.min())
    # distance_matrix = distance.pdist(X)

    y = df_potability["Potability"].values

    n_labels = np.unique(y).shape[0]
    n_clusters = 3

    clusters = cluster_data(X, n_clusters)
    dist_centroids_cluster = get_distances_between_diff_classes_per_cluster(
        X, y, clusters, n_clusters, n_labels)
    print(np.sum(dist_centroids_cluster))

    # i, j
    # 0, 1 = 1
    # 0, 2 = 2
    # 0, 3 = 3
    # 0, 4 = 4
    # 0, 5 = 5
    # 1, 2 = 6
    # 1, 3 = 7
    # 1, 4 = 8
    # 1, 5 = 9
    # 2, 3 = 8
    # 2, 4 = 9
    # 2, 5 = 10
    # 3, 4 = 11
    # 3, 5 = 12
    # 4, 5 = 13
    # n = 5
    # i * n - soma_termos_pa_i + j

# i(1 + i) / 2
