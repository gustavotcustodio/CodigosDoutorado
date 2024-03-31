import random
from re import M
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from scipy.spatial import distance
from sklearn.metrics import brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from loader_and_preprocessor import read_potability_dataset
from enum import Enum
from deslib.des.knora_e import KNORAE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class Experiments(Enum):
    PAPER_66 = 0
    CLUSTERING_ANALYSIS = 1
    CENTROID_OPTIMIZATION = 2
    RANDOM_FOREST = 3
    GRADIENT_BOOSTING = 4
    SVM = 5

class BaseClassifiers(Enum):
    SVM = 0
    DT = 1
    KNN = 2
    NB = 3

POSSIBLE_CLASSIFIERS = [ SVC(C=1.0, kernel='rbf'), SVC(C=1.0, kernel='linear'), GaussianNB(), DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=5), LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]  # , MLPClassifier(max_iter=100),]
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


def cluster_data(data, n_clusters: int, fitness_function=None) -> tuple:
    clusterer = KMeans(n_clusters, n_init='auto')
    clusterer.fit(data)

    if fitness_function:
        fitness_value = fitness_function(None, clusterer.cluster_centers_, 0)
        return clusterer.labels_, clusterer.cluster_centers_, fitness_value
    return clusterer.labels_, clusterer.cluster_centers_


def get_distances_between_diff_classes_per_cluster(y, clusters, n_clusters, n_labels, distances):
    dist_centroids_cluster = np.empty((n_clusters))

    for c in range(n_clusters):
        possible_labels = np.unique(y)
        samples_c = np.where(clusters == c)[0]

        avg_distance_cluster = 0
        n_distances = 0
        for lbl1 in possible_labels:
            possible_labels = possible_labels[1:]
            for lbl2 in possible_labels:

                samples_lbl1 = np.where(y == lbl1)[0]
                samples_lbl1 = np.intersect1d(samples_lbl1, samples_c)

                samples_lbl2 = np.where(y == lbl2)[0]
                samples_lbl2 = np.intersect1d(samples_lbl2, samples_c)

                n_distances += len(samples_lbl1) * len(samples_lbl2)
                avg_distance_cluster += distances[samples_lbl1, :][:, samples_lbl2].sum()
        if n_distances > 0:
            dist_centroids_cluster[c] = avg_distance_cluster / n_distances
        else:
            dist_centroids_cluster[c] = 0.0
    return dist_centroids_cluster

# def get_distances_between_diff_classes_per_cluster(X, y, clusters, n_clusters, n_labels):
#     dist_centroids_cluster = np.empty((n_clusters))
# 
#     for c in range(n_clusters):
#         # Samples belonging to the specific cluster c
#         samples_c = np.where(clusters == c)[0]
#         centroids_by_class = calc_centroids_same_cluster(
#             X[samples_c], y[samples_c], n_labels)
# 
#         # Sum of Distances between the centroids from different classes in the same cluster
#         dist_centroids_cluster[c] = np.sum(distance.pdist(centroids_by_class))
# 
#     return dist_centroids_cluster


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


def select_classifier(base_classifier):
    # rnd = random.random()
    if base_classifier == BaseClassifiers.DT:
        return DecisionTreeClassifier()
    elif base_classifier == BaseClassifiers.KNN:
        return KNeighborsClassifier(n_neighbors = 5)
    elif base_classifier == BaseClassifiers.NB:
        return GaussianNB()
    else:
        return SVC(kernel='rbf')


def split_train_test(X, train_size = 0.8):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)

    n_train = int(train_size * X.shape[0])

    idx_train = indexes[ : n_train]
    idx_test = indexes[n_train : ]
    return idx_train, idx_test


def regular_classification(X_train, y_train, X_test, y_test, experiment):
    if experiment == Experiments.RANDOM_FOREST:
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def ensemble_classification(X_train, y_train, X_test, y_test, centroids,
                            clusters, base_classifiers):
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
            classifier = base_classifiers[c]

        classifier.fit(X_train[idx_cluster], y_train[idx_cluster])
        ensemble.append(classifier)

    u = calc_membership_values(X_test, centroids[possible_clusters])

    predictions = np.array([ensemble[c].predict(X_test)
                            for c in range(possible_clusters.shape[0])]).T
    # print(predictions)
    probabilities = np.sum(predictions * u, axis=1)

    y_pred = np.round(probabilities)

    return y_pred
    # accuracy = sum(y_pred == y_test) / len(y_test)
    # print(ensemble, "\n")
    # print(f"{experiment_name}:", accuracy)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    # disp.plot()
    # plt.title(f"Proposta")
    # plt.show()

# def preselect_base_classifiers(X_train, y_train, n_to_select):
#     pool_classifiers = [GaussianNB(), SVC(C=1.0, kernel='rbf'),
#                         SVC(C=0.5, kernel='rbf'), SVC(C=1.0, kernel='linear'),
#                         DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=5),
#                         KNeighborsClassifier(n_neighbors=7), LogisticRegression(),
#                         MLPClassifier(max_iter=100),]
#     classifiers_evaluations = []
# 
#     for classifier in pool_classifiers:
#         scores = cross_val_score(classifier, X_train, y_train, cv=5)
#         avg_score = scores.mean()
# 
#         classifiers_evaluations.append((classifier, avg_score))
#         # Perform cross validation of each classifier individually
#         # Get classifiers with highest accuracy
# 
#     classifiers_evaluations.sort(key = lambda x: x[1], reverse=True)
# 
#     sorted_classifiers = [clf for clf, _ in classifiers_evaluations]
# 
#     return sorted_classifiers[0:n_to_select]



def select_best_clf_for_cluster(X_cluster, y_cluster):
    best_clf = POSSIBLE_CLASSIFIERS[0]
    best_accuracy = float("-inf")

    # Check if the minority class has less samples than folds
    n_folds = 5

    labels, n_samples_per_label = np.unique(y_cluster, return_counts=True)

    if min(n_samples_per_label) < n_folds:
        n_folds = min(n_samples_per_label)

    for clf in POSSIBLE_CLASSIFIERS:
        # If the cross validation is not possible, select a SVM as base classifier
        if n_folds < 2:
            return best_clf
        accuracy = cross_val_score(clf, X_cluster, y_cluster, cv=n_folds).mean()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf
    return best_clf


def preselect_base_classifiers(X, y, clusters, n_clusters):
    base_classifiers = []

    X_cluster = [[] for _ in range(n_clusters)]
    y_cluster = [[] for _ in range(n_clusters)]

    for idx, c in zip(range(X.shape[0]), clusters):
        X_cluster[c].append(X[idx])
        y_cluster[c].append(y[idx])

    for c in np.unique(clusters):
        if np.all(np.array(y_cluster[c]) == y_cluster[c][0]):
            best_classifier = KNeighborsClassifier(n_neighbors=5)
        else:
            best_classifier = select_best_clf_for_cluster(
                np.array(X_cluster[c]), np.array(y_cluster[c])
            )
        # best_classifier = SVC()
        base_classifiers.append(best_classifier)
    return base_classifiers


# def preselect_classifiers(X_train, y_train):
#     # SVM = 0
#     # DT = 1
#     # KNN = 2
#     # NB = 3
#     best_accuracy = 0.0
#     best_clf = 0
# 
#     classifiers = {BaseClassifiers.SVM: SVC(kernel='rbf'),
#                    BaseClassifiers.DT: DecisionTreeClassifier(),
#                    BaseClassifiers.KNN: KNeighborsClassifier(n_neighbors=5),
#                    BaseClassifiers.NB: GaussianNB()}
#     for name_clf in BaseClassifiers:
#         clf = classifiers[name_clf]
# 
#         scores = cross_val_score(clf, X_train, y_train, cv=5)
#         avg_accuracy = scores.mean()
# 
#         if avg_accuracy > best_accuracy:
#             best_accuracy = avg_accuracy
#             best_clf = name_clf
#     return best_clf


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

        min_clusters = 2
        max_clusters = int(np.sqrt(n_samples))

        for k in range(min_clusters, max_clusters+1):
            clusters_k, centroids_k = cluster_data(X_class, k)
            intra_inter = calc_intra_cluster(X_class, clusters_k, centroids_k, k)
            if intra_inter < best_intra_inter:
                best_cluster_labels = clusters_k.astype(int)
                best_intra_inter = intra_inter

        clusters[idxs] = best_cluster_labels
    return clusters


def create_dict_labels_cluster(y, clusters):
    S = {}
    for lbl in np.unique(y):
        S[lbl] = {}
        for k in np.unique(clusters):
            idxs_cluster = np.where(clusters == k)[0]
            idxs_label = np.where(y == lbl)[0]
            idxs = np.intersect1d(idxs_cluster, idxs_label)
            if len(idxs) > 0:
                S[lbl][k] = idxs
    return S


def combine_clusters(S, cluster_configs, idxs_cluster, lbl, k):
    # Ver se k < 0 ou k > limite lbl > limite
    if (lbl>=0 and lbl not in S) or (lbl>=0 and k not in S[lbl]):
        return

    if lbl >= 0:
        idxs_cluster = np.concatenate((idxs_cluster.astype(int), S[lbl][k]))
        if lbl == len(S.keys())-1:
            cluster_configs.append(idxs_cluster)

    if lbl < len(S.keys())-1:
        clusters = sorted(S[lbl+1].keys())

        for k in clusters:
            combine_clusters(S, cluster_configs, idxs_cluster, lbl+1, k)

            # Remover o último elemento
            idxs_cluster = idxs_cluster[:-1]


def construct_training_clusters(X_train, y_train):
    classes = np.unique(y_train)
    n_classes = classes.shape[0]
    n_labels = np.unique(y_train).shape[0]
    cluster_configs = []
    clusters_by_label = []

    clusters = find_best_partition_per_class(X_train, y_train)

    S = create_dict_labels_cluster(y_train, clusters)

    cluster_configs = []
    combine_clusters(S, cluster_configs, np.array([]), -1, 0)

    X_new_train = []
    y_new_train = []
    new_clusters = []
    centroids = []

    for c, idxs in enumerate(cluster_configs):
        X_new_train.append(X_train[idxs])
        y_new_train.append(y_train[idxs])
        new_clusters += [c] * len(idxs)
        centroids.append(np.mean(X_train[idxs], axis=0))

    X_new_train = np.vstack(X_new_train)
    y_new_train = np.concatenate(y_new_train)
    new_clusters = np.array(new_clusters)
    centroids = np.array(centroids)
    return X_new_train, y_new_train, centroids, new_clusters

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
