import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import M
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy.spatial import distance
from sklearn.metrics import brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from enum import Enum
from deslib.des.knora_e import KNORAE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap

POSSIBLE_CLASSIFIERS = [SVC(), SVC(), RandomForestClassifier(), SVC(), GradientBoostingClassifier()]

def plotar_clusters(X_train, y_train, clusters, centroids, filename):
    # Rosa, Verde, Azul, Amarelo
    COLORS_CLUSTER = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700']
    COLORS_CLASS = ["red", "blue", "green"]

    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack((X_train, centroids)))
    reduced_data = pca.transform(X_train)
    # reduced_data = (reduced_data - reduced_data.min(axis=0)) / (
    #                 reduced_data.max(axis=0) - reduced_data.min(axis=0))
    reduced_centroids = pca.transform(centroids)

    n_clusters = len(np.unique(clusters))
    n_classes = len(np.unique(y_train))

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

    for c in range(n_clusters):
        # indexes_c = np.where(clusters == c)[0]
        # centroid = np.mean(reduced_data[indexes_c], axis=0)
        plt.scatter(reduced_centroids[c, 0], reduced_centroids[c, 1], c='black', s=200, alpha=0.8, marker='X')
        # plt.scatter(reduced_data[indexes_c,0], reduced_data[indexes_c,1], color=COLORS_CLASS[c])

    for l in range(n_classes):
        indexes_lbl = np.where(y_train == l)[0]
        plt.scatter(reduced_data[indexes_lbl,0], reduced_data[indexes_lbl,1], color=COLORS_CLASS[l])

        # dists = pairwise_distances(centroid.reshape(1,-1), reduced_data[indexes_c])[0]
        # radius = np.max(dists)
        # circle = plt.Circle(centroid, radius, fill=False, color=COLORS_CLUSTER[c])
        # plt.pcolormesh(reduced_data[:,0], reduced_data[:,1], c, cmap=COLORS_CLUSTER[c])
        # ax.add_patch(circle)

    # circle = plt.Circle((0,0), 1, fill=False, color='green')
    # plt.show()
    plt.savefig(filename)


def read_potability_dataset():
    df_potability = pd.read_csv("potabilidade.csv")
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    y = df_potability["Potability"].values
    X = df_potability.drop(columns="Potability").values
    return X, y


def read_german_credit_dataset():
    X = np.loadtxt("./german.data-numeric", delimiter=" ")
    np.random.shuffle(X)
    y = X[:, -1] - 1
    X = X[:, :-1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X, y


def read_wine_dataset():
    X = np.loadtxt("./wine.data", delimiter=",")
    np.random.shuffle(X)
    y = X[:, 0] - 1
    X = X[:, 1:]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X, y


def read_wdbc_dataset():
    X = np.loadtxt("./wdbc.data", delimiter=",")
    np.random.shuffle(X)

    y = X[:, 1]
    X = np.hstack((X[:, [0]], X[:, 2:]))
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X, y


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


def ensemble_classification(X_train, y_train, X_test, y_test, centroids,
                            clusters, base_classifiers=None):

    n_clusters = centroids.shape[0]
    if base_classifiers is None:
        base_classifiers = POSSIBLE_CLASSIFIERS[:n_clusters]

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
    predictions = predictions.astype(int)

    n_labels = int(predictions.max()) + 1
    # probabilities = np.sum(predictions * u, axis=1)
    voting_labels = np.zeros((u.shape[0], n_labels))

    for c in range(n_clusters):
        labels = predictions[:, c]
        for i, lbl in enumerate(labels):
            voting_labels[i, lbl] += u[i, c]

    y_pred_votation = np.argmax(voting_labels, axis=1)

    return y_pred_votation, predictions, u


def calcular_acuracia_por_cluster(y_pred, y_test, clusters):
    print("=" * 10 + " Análise do teste " + "=" * 10)
    for c in np.unique(clusters):
        indexes_c = np.where(clusters == c)[0]
        acc = sum(y_pred[indexes_c] == y_test[indexes_c]) / len(y_test[indexes_c])
        print(f"Acurácia cluster {c}: {acc}")

        print(f"Número de amostras no cluster: {len(indexes_c)}")

        print("Classificador Base do cluster:", POSSIBLE_CLASSIFIERS[c])
        print("-------------------------------")

    acc = sum(y_pred == y_test) / len(y_test)
    print("Acurácia total:", acc)
    print()


def pesos_por_amostra(y_pred_cluster, y_pred, y_test, clusters, u):
    for c in np.unique(clusters):
        indexes_c = np.where(clusters == c)[0]
        print(f"---------- Cluster {c} ----------")
        for i in indexes_c:
            print(f"Real: {int(y_test[i])}, Predito: {int(y_pred[i])}, Preditos por cluster: {y_pred_cluster[i]}, Pesos: {u[i]}")


def mostrar_distribuicao_de_amostras_treino(y_train, clusters, n_clusters):
    n_labels = np.max(y_train) + 1
    print("=" * 10 + " Análise do treino " + "=" * 10)
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


def get_distances_by_cluster(X_lbl1, X_lbl2):
    for c in range(3):
        for c in range(3):
            print("")
    pass


def combine_clusterings(X_by_label, y_by_label, clusters_by_label, centroids_by_label):
    n_labels = len(clusters_by_label.keys())
    avg_dists = np.zeros((n_labels, n_labels))

    # Calcula a distância média por cada cluster de cada label
    for lbl_cl1 in range(n_labels):
        for lbl_cl2 in range(lbl_cl1+1, n_labels):
            # avg_dist_clusters = get_distances_by_cluster(
            #     X_by_label[lbl_cl1], X_by_label[lbl_cl2]
            # )

            # dist_matrix = pairwise_distances(X_by_label[lbl_cl1], X_by_label[lbl_cl2])
            # avg_dists[lbl_cl1, lbl_cl2] = dist_matrix.mean()
            # avg_dists[lbl_cl2, lbl_cl1] = avg_dists[lbl_cl1, lbl_cl2]

    breakpoint()


def rodar_programa(n_clusters, run):
    filename = f"./resultados/{n_clusters}_clusters_experimento_{run}"

    X, y = read_german_credit_dataset()
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    X_by_label, y_by_label = {}, {}
    centroids_by_label = {}
    clusters_by_label = {}

    # mudar aqui agrupar
    for c in range(n_clusters):
        indexes_label = np.where(y_train == c)[0]
        if indexes_label.size > 0:
            X_by_label[c] = X_train[indexes_label]
            y_by_label[c] = y_train[indexes_label]

            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
            clusterer.fit(X_by_label[c])
            centroids_by_label[c] = clusterer.cluster_centers_
            clusters_by_label[c] = clusterer.labels_

    X_train, y_train, clusters, centroids = combine_clusterings(
        X_by_label, y_by_label, clusters_by_label, centroids_by_label
    )

    y_pred, predictions, u = ensemble_classification(
        X_train, y_train, X_test, y_test, centroids, clusters,
        base_classifiers=None
    )

    clusters_test = np.argmax(u, axis=1)
    calcular_acuracia_por_cluster(y_pred, y_test, clusters_test)
    pesos_por_amostra(predictions, y_pred, y_test, clusters_test, u)

    mostrar_distribuicao_de_amostras_treino(y_train, clusters, n_clusters)
    plotar_clusters(X_train, y_train, clusters, centroids, f"{filename}_train.png")


if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    run = int(sys.argv[2])

    rodar_programa(n_clusters, run)
