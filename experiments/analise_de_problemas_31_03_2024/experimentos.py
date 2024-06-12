import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial import distance
from sklearn.metrics import recall_score, precision_score, f1_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from enum import Enum
from deslib.des.knora_e import KNORAE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib.colors import ListedColormap
from yellowbrick.cluster import silhouette_visualizer, SilhouetteVisualizer
import dataset_loader

POSSIBLE_CLASSIFIERS = [SVC()] * 100
# POSSIBLE_CLASSIFIERS = [OneVsRestClassifier(SVC())] * 20


def create_PCA_reducer(X):
    """
    :X: 2d array
    :returns: pca

    """
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    return pca


def plotar_clusters(X_train, y_train, clusters, centroids, filename, pca):
    # Rosa, Verde, Azul, Amarelo
    COLORS_CLUSTER = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700', "#5E93A8", "#1D502A"]
    COLORS_CLASS = ["red", "blue", "green", "#7C006C", "#0027B4", "#F1E447"]

    n_clusters = len(np.unique(clusters))
    n_classes = len(np.unique(y_train))
    reduced_data = pca.transform(np.vstack((X_train, centroids)))
    # reduced_data = (reduced_data - reduced_data.min(axis=0)) / (
    #                 reduced_data.max(axis=0) - reduced_data.min(axis=0))
    reduced_centroids = reduced_data[-n_clusters:]
    reduced_data = reduced_data[:-n_clusters]

    plt.figure()

    h = 0.02
    x1_min, x1_max = reduced_data[:, 0].min() - 1e-15, reduced_data[:, 0].max() + 1e-15
    x2_min, x2_max = reduced_data[:, 1].min() - 1e-15, reduced_data[:, 1].max() + 1e-15
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    Z = []
    for v in np.c_[xx.ravel(), yy.ravel()]:
        clust = np.argmin(pairwise_distances(v.reshape(1,-1), reduced_centroids)[0])
        Z.append(clust)

    # Put the result into a color plot
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(COLORS_CLUSTER))

    for l in range(n_classes):
        indexes_lbl = np.where(y_train == l)[0]
        plt.scatter(reduced_data[indexes_lbl,0], reduced_data[indexes_lbl,1], color=COLORS_CLASS[l])

    max_X = reduced_data[:, 0].max()
    min_X = reduced_data[:, 0].min()

    max_Y = reduced_data[:, 1].max()
    min_Y = reduced_data[:, 1].min()

    for c in range(n_clusters):
        # indexes_c = np.where(clusters == c)[0]
        # centroid = np.mean(reduced_data[indexes_c], axis=0)
        if (reduced_centroids[c, 0] <= max_X and reduced_centroids[c, 1] <= max_Y and
            reduced_centroids[c, 0] >= min_X and reduced_centroids[c, 1] >= min_Y):

            plt.scatter(reduced_centroids[c, 0], reduced_centroids[c, 1],
                        c='black', s=200, alpha=0.8, marker='X')
        # plt.scatter(reduced_data[indexes_c,0], reduced_data[indexes_c,1], color=COLORS_CLASS[c])

        # dists = pairwise_distances(centroid.reshape(1,-1), reduced_data[indexes_c])[0]
        # radius = np.max(dists)
        # circle = plt.Circle(centroid, radius, fill=False, color=COLORS_CLUSTER[c])
        # plt.pcolormesh(reduced_data[:,0], reduced_data[:,1], c, cmap=COLORS_CLUSTER[c])
        # ax.add_patch(circle)

    # circle = plt.Circle((0,0), 1, fill=False, color='green')
    plt.savefig(filename)
    plt.clf()


def calc_membership_values(X_samples, centroids):
    # u_ik =          1
    #       -----------------------
    #       sum_j=1^C (d_ik²/d_ij²)
    n_clusters = centroids.shape[0]
    n_samples = X_samples.shape[0]

    # np.linalg.norm(samples-c[0], axis=1)**2
    dists_samples_centroids = [np.linalg.norm(X_samples - centroids[k], axis=1)**2
                              for k in range(n_clusters)]
    dists_samples_centroids = np.array(dists_samples_centroids).T

    u = np.empty((n_samples, n_clusters))


    for k in range(n_clusters):
        u[:, k] = np.sum([
            dists_samples_centroids[:, k] / (dists_samples_centroids[:, c] + 0.00000001)
            for c in range(n_clusters)
        ], axis=0)
    u = 1 / (u + 0.00000001)
    u[u > 1] = 1
    return u


def ensemble_prediction(X_test, centroids, possible_clusters, attribs_by_cluster, ensemble):
    n_clusters = centroids.shape[0]
    u = calc_membership_values(X_test, centroids[possible_clusters])
    # predictions = np.array([ensemble[c].predict(X_test)
    #                         for c in range(possible_clusters.shape[0])]).T
    predictions = []

    for c in range(possible_clusters.shape[0]):
        X_cluster = X_test[:, attribs_by_cluster[c]]
        y_pred = ensemble[c].predict(X_cluster)
        predictions.append(y_pred)

    predictions = np.array(predictions).astype(int)

    if len(predictions.shape) > 2:
        predictions = predictions.argmax(axis=2)

    predictions = predictions.T

    n_labels = int(predictions.max()) + 1
    # probabilities = np.sum(predictions * u, axis=1)
    voting_labels = np.zeros((u.shape[0], n_labels))

    for c in range(n_clusters):
        labels = predictions[:, c]
        for i, lbl in enumerate(labels):
            voting_labels[i, lbl] += u[i, c]

    y_pred_votation = np.argmax(voting_labels, axis=1)

    return y_pred_votation, predictions, u


def get_attribs_by_mutual_info(X_cluster, y_cluster, information):
    return [i for i in range(X_cluster.shape[1])]
    if len(X_cluster) == 1:
        return [i for i in range(X_cluster.shape[1])]
    mutual_info = mutual_info_classif(X_cluster, y_cluster)

    if np.all(mutual_info == 0):
        return [i for i in range(X_cluster.shape[1])]

    norm_mutual_info = mutual_info / np.sum(mutual_info)

    sorted_attrs = mutual_info.argsort()[::-1]

    cumsum_info = norm_mutual_info.cumsum()

    max_attr = np.where(cumsum_info >= information)[0][0]
    selected_attrs = sorted_attrs[0 : max_attr]
    return selected_attrs


def ensemble_training(X_train, y_train, X_test, y_test, centroids,
                      clusters, attribs_by_cluster, base_classifiers=None):

    n_clusters = centroids.shape[0]
    if base_classifiers is None:
        base_classifiers = POSSIBLE_CLASSIFIERS[:n_clusters]

    ensemble = []
    possible_clusters = np.unique(clusters)

    for c in possible_clusters:
        idx_cluster = np.where(clusters == c)[0]
        X_cluster, y_cluster = X_train[:, attribs_by_cluster[c]][idx_cluster], y_train[idx_cluster]

        if np.all(y_cluster == y_cluster[0]):
            classifier = RandomForestClassifier()
        else:
            classifier = base_classifiers[c]

        classifier.fit(X_cluster, y_cluster)
        ensemble.append(classifier)
    return ensemble

    # u = calc_membership_values(X_test, centroids[possible_clusters])

    # predictions = np.array([ensemble[c].predict(X_test)
    #                         for c in range(possible_clusters.shape[0])]).T
    # predictions = predictions.astype(int)

    # n_labels = int(predictions.max()) + 1
    # # probabilities = np.sum(predictions * u, axis=1)
    # voting_labels = np.zeros((u.shape[0], n_labels))

    # for c in range(n_clusters):
    #     labels = predictions[:, c]
    #     for i, lbl in enumerate(labels):
    #         voting_labels[i, lbl] += u[i, c]

    # y_pred_votation = np.argmax(voting_labels, axis=1)

    # return y_pred_votation, predictions, u


def calcular_acuracia_por_cluster(y_pred, y_real, clusters, tipo="teste"):
    print("=" * 10 + f" Análise do {tipo} " + "=" * 10)

    for c in np.unique(clusters):
        indexes_c = np.where(clusters == c)[0]
        acc = accuracy_score(y_real[indexes_c], y_pred[indexes_c])
        # acc = sum(y_pred[indexes_c] == y_real[indexes_c]) / len(y_real[indexes_c])
        recall_value = recall_score(y_real[indexes_c], y_pred[indexes_c], average="weighted", zero_division=0.0) 
        precision_value = precision_score(y_real[indexes_c], y_pred[indexes_c], average="weighted", zero_division=0.0 )
        f1_value = f1_score(y_real[indexes_c], y_pred[indexes_c], average="weighted" )
        print(f"Acurácia cluster {c}: {acc}")
        print(f"Recall cluster {c}: {recall_value}")
        print(f"Precisão cluster {c}: {precision_value}")
        print(f"F1-Score cluster {c}: {f1_value}")

        print(f"Número de amostras no cluster: {len(indexes_c)}")

        print("Classificador Base do cluster:", POSSIBLE_CLASSIFIERS[c])
        print("-------------------------------")

    # acc = sum(y_pred == y_real) / len(y_real)
    acc = accuracy_score(y_real, y_pred)
    recall_value = recall_score(y_real, y_pred, average="weighted") 
    precision_value = precision_score(y_real, y_pred, average="weighted", zero_division=0.0 )
    f1_value = f1_score(y_real, y_pred, average="weighted" )
    print("Acurácia total:", acc)
    print("Recall total:", recall_value)
    print("Precisão total:", precision_value)
    print("F1-Score:", f1_value)
    print()


def pesos_por_amostra(y_pred_cluster, y_pred, y_test, clusters, u):
    for c in np.unique(clusters):
        indexes_c = np.where(clusters == c)[0]
        print(f"---------- Cluster {c} ----------")
        for i in indexes_c:
            print(f"Real: {int(y_test[i])}, Predito: {int(y_pred[i])}, Preditos por cluster: {y_pred_cluster[i]}, Pesos: {u[i]}")
    print()


def mostrar_distribuicao_de_amostras_treino(y_train, clusters, n_clusters):
    n_labels = np.max(y_train) + 1
    print("=" * 10 + " Distribuição das amostras de treino " + "=" * 10)
    for c in range(n_clusters):
        print(f"---------- Cluster {c} ----------")
        indexes_c = np.where(clusters == c)[0]
        if indexes_c.size > 0:
            y_cluster = y_train[indexes_c]
            distribuicao_cluster = Counter(y_cluster)
            for lbl in range(n_labels):
                if lbl in distribuicao_cluster:
                    print(f"Label {lbl}:", distribuicao_cluster[lbl])
                else:
                    print(f"Label {lbl}: 0" )
        else:
            print(f"0 amostras no cluster {c}")
    print()


def calc_centroids(n_clusters, clusters_test, X_test):
    centroids_test = []
    for c in range(n_clusters):
        idx_cluster = np.where(clusters_test == c)[0]
        if len(idx_cluster) > 0:
            X_cluster = X_test[idx_cluster]
            centroids_test.append(np.mean(X_cluster, axis=0))
        else:
            centroids_test.append(np.ones(X_test.shape[1]))
    centroids_test = np.array(centroids_test)
    return centroids_test


def plotar_silhueta(clusterer, X, filename):
    print(f"---------- Silhouette ----------")
    print(silhouette_score(X, clusterer.predict(X)))
    print()
    visualizer = SilhouetteVisualizer(clusterer, is_fitted=True)
    visualizer.fit(X)
    plt.savefig(filename)


def create_co_assoc_matrix(n_samples, votation_clusters):
    co_association_matrix = np.zeros((n_samples, n_samples))

    for partition in votation_clusters:
        for i in range(0, len(partition)):
            for j in range(i, len(partition)):
                if partition[i] == partition[j]:
                    co_association_matrix[i, j] +=1
    # co_association_matrix = co_association_matrix + co_association_matrix.T

    #for i in range(n_samples):
    #    co_association_matrix[i, i] = 0
    return co_association_matrix


def combine_clusters(co_association_matrix):
    co_association_matrix /= co_association_matrix.max()

    n_samples = co_association_matrix.shape[0]
    # max_matching = co_association_matrix.max().astype(int)
    min_matching = 0.5
    clusters = np.empty(n_samples)
    clusters.fill(-1)

    clusters[0] = 0
    current_cluster = 1

    for i in range(n_samples):
        matches = np.where(co_association_matrix[i] >= min_matching)[0]
        if len(matches) > 0:
            if np.any(clusters[matches] > -1):
                clusters[matches] = min(
                    np.delete(clusters[matches], clusters[matches] == -1)
                )
            else:
                clusters[matches] = current_cluster
                current_cluster += 1
    return clusters


def consensus_clustering(X_train):
    n_clusterings = 3
    votation_clusters = []
    max_clusters = int(np.sqrt(len(X_train)))
    n_samples = X_train.shape[0]

    for n_clusters in range(2, max_clusters):
        for _ in range(n_clusterings):
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
            clusterer.fit(X_train)
            # centroids = clusterer.cluster_centers_
            clusters = clusterer.labels_

            votation_clusters.append(clusters)

    # n_samples = 10
    # votation_clusters = []
    # votation_clusters.append([0,0,2,2,2,0,0,1,0,2]) 
    # votation_clusters.append([1,1,2,2,2,1,0,1,1,2])
    # votation_clusters.append([0,0,2,2,2,1,0,1,1,2])
    co_association_matrix = create_co_assoc_matrix(n_samples, votation_clusters)
    clusters = combine_clusters(co_association_matrix)

    return clusters.astype(int)


def get_attribs_by_cluster(X_train, y_train, clusters, n_clusters):
    selected_attribs = []

    for c in range(n_clusters):
        idx_cluster = np.where(clusters == c)[0]
        if len(idx_cluster) > 0:
            X_cluster, y_cluster = X_train[idx_cluster], y_train[idx_cluster]
            attribs_cluster = get_attribs_by_mutual_info(X_cluster, y_cluster, information=0.5)

            selected_attribs.append(attribs_cluster)
    return selected_attribs


def get_distances_between_diff_classes_per_cluster(y, clusters, n_clusters, distances):
    dists_centroids_cluster = np.empty((n_clusters))

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
            dists_centroids_cluster[c] = avg_distance_cluster / n_distances
        else:
            dists_centroids_cluster[c] = 1.0

    #max_dist = np.max(dist_centroids_cluster)
    #idx_zeros = np.where(dist_centroids_cluster <= 0)[0]

    #dist_centroids_cluster[idx_zeros] = max_dist
    return dists_centroids_cluster


def select_optimal_partition(X_train, y_train):
    """Select the best partition according to
        the distance between samples from different classes.

    :arg1: X_train
        2d-numpy array.
    :arg2: y_train
        1d-numpy array.
    :returns: best_clusterer
        Best KMeans found.

    """
    best_clusterer = None
    # Calculate distances between samples
    distances = cosine_distances(X_train, X_train)
    best_macro_avg_distance = 0

    max_clusters = int(np.sqrt(X_train.shape[0]))

    for n_clusters in range(2, max_clusters + 1):
        for _ in range(3):
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
            clusterer.fit(X_train)
            # centroids = clusterer.cluster_centers_
            clusters = clusterer.labels_

            # Calculate distances betweeen samples from different classes
            dists_centroids_cluster = get_distances_between_diff_classes_per_cluster(
                y_train, clusters, n_clusters, distances
            )
            macro_avg_dist = np.mean(dists_centroids_cluster)

            if macro_avg_dist > best_macro_avg_distance:
                best_macro_avg_distance = macro_avg_dist
                best_clusterer = clusterer
    return best_clusterer


def rodar_programa(n_clusters, run, dataset):
    dataset_loader_function = dataset_loader.select_dataset_function(dataset)

    filename = f"./resultados/{dataset}_{n_clusters}_clusters_experimento_{run}"

    print(dataset)
    X, y = dataset_loader_function()
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)  #, random_state=42)
    clusterer = select_optimal_partition(X_train, y_train)

    # clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
    # clusterer.fit(X_train)
    centroids = clusterer.cluster_centers_
    clusters = clusterer.labels_

    # Consensus ##########################################
    # clusters = consensus_clustering(X_train)
    n_clusters = len(np.unique(clusters))
    # centroids = calc_centroids(n_clusters, clusters, X_train)

    # clusterer = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    # clusterer.fit(centroids)
    # Consensus ##########################################

    ####### Attribs by cluster #############
    attribs_by_cluster = get_attribs_by_cluster(X_train, y_train, clusters, n_clusters)

    print("*********** Atributos selecionados **********")
    for i, attribs_c in enumerate(attribs_by_cluster):
        print(f"Cluster {i}:", attribs_c)
        print()

    ########################################
    ensemble = ensemble_training(
        X_train, y_train, X_test, y_test, centroids, clusters, attribs_by_cluster, base_classifiers=None
    )
    y_pred, predictions, u = ensemble_prediction(
        X_test, centroids, np.unique(clusters), attribs_by_cluster, ensemble
    )

    clusters_test = np.argmax(u, axis=1)
    centroids_test = calc_centroids(n_clusters, clusters_test, X_test)

    pca = create_PCA_reducer(np.vstack((X, centroids, centroids_test)))

    if len(y_test.shape) > 1:
        y_test = y_test.argmax(axis=1)

    calcular_acuracia_por_cluster(y_pred, y_test, clusters_test)
    pesos_por_amostra(predictions, y_pred, y_test, clusters_test, u)
    plotar_clusters(X_test, y_test, clusters_test, centroids_test, f"{filename}_test.png", pca)
    plotar_silhueta(clusterer, X_test, f"{filename}_silh_test.png")

    y_pred_train, predictions_train, u_train = ensemble_prediction(
        X_train, centroids, np.unique(clusters), attribs_by_cluster, ensemble
    )

    if len(y_train.shape) > 1:
        y_train = y_train.argmax(axis=1)

    mostrar_distribuicao_de_amostras_treino(y_train, clusters, n_clusters)
    calcular_acuracia_por_cluster(y_pred_train, y_train, clusters, tipo="treino")
    pesos_por_amostra(predictions_train, y_pred_train, y_train, clusters, u_train)
    plotar_clusters(X_train, y_train, clusters, centroids, f"{filename}_train.png", pca)
    plotar_silhueta(clusterer, X_train, f"{filename}_silh_train.png")


if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    run = int(sys.argv[2])
    dataset = sys.argv[3]

    rodar_programa(n_clusters, run, dataset)
