from math import dist
import numpy as np
from sklearn.svm import SVC
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from loader_and_preprocessor import read_potability_dataset, read_wine_dataset, read_wdbc_dataset, read_german_credit_dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from deslib.des.knora_e import KNORAE


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


def cluster_data(data, n_clusters: int) -> tuple:
    clusterer = KMeans(n_clusters, n_init='auto')
    clusterer.fit(data)
    return clusterer.labels_, clusterer.cluster_centers_


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

        for k in range(5, max_clusters+1):
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


#def combine_clusters2(S):
#    cluster_configs = []
#    k1 = len(S[0].keys())
#    for i in range(k1):
#        k2 = len(S[1].keys())
#        for j in range(k2):
#            combined_idxs = np.concatenate((S[0][i], S[1][j]))
#            cluster_configs.append(combined_idxs)
#
#    return cluster_configs


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


def get_distances_between_diff_classes_per_cluster(y, clusters, n_clusters, n_labels, distances):
    dist_centroids_cluster = np.empty((n_clusters))

    for c in range(n_clusters):
        # Samples belonging to the specific cluster c
        samples_c = np.where(clusters == c)[0]
        # centroids_by_class = calc_centroids_same_cluster(
        #     X[samples_c], y[samples_c], n_labels)

        avg_distance_cluster = 0
        n_distances = 0
        # # Sum of Distances between the centroids from different classes in the same cluster
        # dist_centroids_cluster[c] = np.sum(distance.pdist(centroids_by_class))
        for lbl1 in range(n_labels):
            for lbl2 in range(lbl1+1, n_labels):

                samples_lbl1 = np.where(y == lbl1)[0]
                samples_lbl1 = np.intersect1d(samples_lbl1, samples_c)

                samples_lbl2 = np.where(y == lbl2)[0]
                samples_lbl2 = np.intersect1d(samples_lbl2, samples_c)

                n_distances += len(samples_lbl1) * len(samples_lbl2)
                avg_distance_cluster += distances[samples_lbl1, :][:, samples_lbl2].sum()
        dist_centroids_cluster[c] = avg_distance_cluster / n_distances
    return dist_centroids_cluster


def preselect_knora(X_train, y_train, n_to_select):
    pool_classifiers = [GaussianNB(), SVC(C=1.0, kernel='rbf'), SVC(C=0.5, kernel='rbf'), SVC(C=1.0, kernel='linear'), SVC(C=0.5, kernel='linear'), DecisionTreeClassifier(), MLPClassifier(), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=7)]

    for classifier in pool_classifiers:
        classifier.fit(X_train, y_train)

    knora_e = KNORAE(pool_classifiers, k=7)
    knora_e.fit(X_train, y_train)
    knora_e.select()
    return 0

def plot_clusters(X_train, y_train, dataset):
    tsne_model = TSNE(n_components=2)
    low_dim_data = tsne_model.fit_transform(X_train)

    for l in np.unique(y_train):
        idx_labels = np.where(y_train == l)[0]

        plt.scatter(low_dim_data[idx_labels, 0], low_dim_data[idx_labels, 1],
                    alpha=0.7, label="Label %d" % l)

    plt.legend()  # (loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(dataset + ".png")
    plt.clf()


if __name__ == "__main__":
    dataset = "Breast Cancer"
    X, y = read_wdbc_dataset()
    plot_clusters(X, y, dataset)

    dataset = "Wine"
    X, y = read_wine_dataset()
    plot_clusters(X, y, dataset)

    dataset = "Water Potability"
    X, y = read_potability_dataset()
    plot_clusters(X, y, dataset)

    dataset = "German Credit Score"
    X, y = read_german_credit_dataset()
    plot_clusters(X, y, dataset)
