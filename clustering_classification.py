import random
import numpy as np
from xgboost import XGBClassifier
from scipy.spatial import distance

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from loader_and_preprocessor import read_potability_dataset
from sklearn.metrics import silhouette_score

# def calc_distances_diff_classes(data_cluster: NDArray, n_labels: int
#                                 ) -> NDArray:
#
#     Calc distance between data samples from different
#     classes in the same cluster.
#
#     data_cluster: data from a specific cluster
#
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


def cluster_data(data, n_clusters: int) -> tuple:
    clusterer = KMeans(n_clusters, n_init='auto')
    clusterer.fit(data)
    return clusterer.labels_, clusterer.cluster_centers_


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


def calc_membership_values(samples, centroids):
    # u_ik =          1
    #       -----------------------
    #       sum_j=1^C (d_ik²/d_ij²)
    n_clusters = centroids.shape[0]
    n_samples = samples.shape[0]

    # np.linalg.norm(samples-c[0], axis=1)**2
    dists_samples_centroids = [np.linalg.norm(samples - centroids[k], axis=1)**2
                              for k in range(n_clusters)]
    dists_samples_centroids = np.array(dists_samples_centroids).T

    u = np.empty((n_samples, n_clusters))

    for k in range(n_clusters):
        u[:, k] = np.sum([
            dists_samples_centroids[:, k] / dists_samples_centroids[:, c]
            for c in range(n_clusters)
        ], axis=0)
    u = 1 / u
    return u


def select_random_classifier():
    rnd = random.random()
    if rnd < 0.25:
        return RandomForestClassifier()
    elif rnd < 0.5:
        return RandomForestClassifier()
    elif rnd < 0.75:
        return GradientBoostingClassifier()
    else:
        return GradientBoostingClassifier()


def split_train_test(X, train_size = 0.8):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)

    n_train = int(train_size * X.shape[0])

    idx_train = indexes[ : n_train]
    idx_test = indexes[n_train : ]
    return idx_train, idx_test


def classification(X, y, model_name):
    idx_train, idx_test = split_train_test(X)

    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    if model_name == "svm":
        model = SVC(C=1.0, kernel='rbf')
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == "dt":
        model = tree.DecisionTreeClassifier()
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Acurácia %s: %f" % (model_name, accuracy))


def ensemble_classification(X_train, y_train, X_test, y_test, centroids, clusters):
    #idx_train, idx_test = split_train_test(X)

    #X_train, y_train = X[idx_train], y[idx_train]
    #X_test, y_test = X[idx_test], y[idx_test]

    ensemble = []
    possible_clusters = np.unique(clusters)

    for c in possible_clusters:
        idx_cluster = np.where(clusters == c)[0]

        if np.all(y_train[idx_cluster] == y_train[idx_cluster[0]]):
            classifier = RandomForestClassifier()
        else:
            classifier = select_random_classifier()  # SVC(kernel='rbf')

        classifier.fit(X_train[idx_cluster], y_train[idx_cluster])
        ensemble.append(classifier)

    u = calc_membership_values(X_test, centroids[possible_clusters])

    predictions = np.array([ensemble[c].predict(X_test)
                            for c in range(possible_clusters.shape[0])]).T
    # print(predictions)
    probabilities = np.sum(predictions * u, axis=1)
    y_pred = np.round(probabilities)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print(ensemble, "\n")
    print("Proposta:", accuracy)


def calc_intra_cluster(X_class, clusters_k, centroids_k, n_clusters):
    #intra = sum^n_samplesk|X-r| / (n_samplesk * max^n_samplesk|X-r|)
    intra_dists = np.empty(n_clusters)

    for c in range(n_clusters):
        X_cluster = X_class[np.where(clusters_k == c)[0]]
        n_samples = X_cluster.shape[0]
        if len(X_cluster) < 2:
            intra_dists[c] = 1
        else:
            dists_samples_center = np.linalg.norm(X_cluster - centroids_k[c], axis=0)**2
            intra_dists[c] = np.sum(dists_samples_center) / (
                             np.max(dists_samples_center) * n_samples)

    intra = np.sum(intra_dists) / n_clusters

    avg_dist_centroids = 2 * distance.pdist(centroids_k).sum() / (
        n_clusters * (n_clusters - 1))

    mean_centroids = np.mean(centroids_k, axis=0)
    beta = np.linalg.norm(centroids_k - mean_centroids)**2 / n_clusters

    inter = np.exp(-avg_dist_centroids / beta)
    return intra * inter


def find_best_partition_per_class(X_train, y_train):
    n_samples = X_train.shape[0]
    classes = np.unique(y_train)
    clusters = np.empty(n_samples)
    best_cluster_labels = None

    for label in classes:
        best_intra_inter = float("inf")
        idxs = np.where(y_train == label)[0]
        X_class = X_train[idxs]
        max_clusters = int(np.sqrt(n_samples))

        for k in range(2, max_clusters+1):
            clusters_k, centroids_k = cluster_data(X_class, k)
            intra_inter = calc_intra_cluster(X_class, clusters_k, centroids_k, k)
            if intra_inter < best_intra_inter:
                best_cluster_labels = clusters_k.astype(int)
                best_intra_inter = intra_inter

        clusters[idxs] = best_cluster_labels
    return clusters


def combine_all_cluster_permutations(clusters, label, k):
    if len(clusters) == label + 1:
        return clusters[label]


def construct_training_clusters(X_train, y_train):
    classes = np.unique(y_train)
    n_classes = classes.shape[0]
    n_labels = np.unique(y_train).shape[0]
    cluster_configs = []
    clusters_by_label = []

    for label in range(n_labels):
        clusters = find_best_partition_per_class(X_train, y_train, label)
        clusters_by_label.append(clusters)

    combine_all_cluster_permutations(clusters_by_label, 0, y_train)


#if __name__ == "__main__":
#    df_potability = read_potability_dataset("potabilidade.csv")
#
#    X = df_potability.drop(columns="Potability").values
#    X = (X - X.min()) / (X.max() - X.min())
#    # distance_matrix = distance.pdist(X)
#
#    y = df_potability["Potability"].values
#
#    n_labels = np.unique(y).shape[0]
#    n_clusters = 3
#
#    clusters, centroids = cluster_data(X, n_clusters)
#    dist_centroids_cluster = get_distances_between_diff_classes_per_cluster(
#        X, y, clusters, n_clusters, n_labels)
#    print("Distância entre centroides por classe para cada cluster:")
#    print(dist_centroids_cluster)
#    print("======================================")
#
#    u = calc_membership_values(X, centroids)
#    print("Valores de pertinência:")
#    print(u)
#    print("======================================")
#
#    ensemble_classification(X, y, u, clusters)
#    classification(X, y, "svm")
#    classification(X, y, "knn")
#    classification(X, y, "dt")
#
#'''
# - ensemble
#'''
