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

    for l in range(n_classes):
        indexes_lbl = np.where(y_train == l)[0]
        plt.scatter(reduced_data[indexes_lbl,0], reduced_data[indexes_lbl,1], color=COLORS_CLASS[l])

    for c in range(n_clusters):
        # indexes_c = np.where(clusters == c)[0]
        # centroid = np.mean(reduced_data[indexes_c], axis=0)
        plt.scatter(reduced_centroids[c, 0], reduced_centroids[c, 1], c='black', s=200, alpha=0.8, marker='X')
        # plt.scatter(reduced_data[indexes_c,0], reduced_data[indexes_c,1], color=COLORS_CLASS[c])

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
            dists_samples_centroids[:, k] / dists_samples_centroids[:, c]
            for c in range(n_clusters)
        ], axis=0)
    u = 1 / u
    return u


def get_new_samples_cluster(X_out_cluster, y_out_cluster,
                            centroids, cluster, label, n_samples_to_add):
    u_cluster = calc_membership_values(X_out_cluster, centroids)
    samples_label = np.where(y_out_cluster == label)[0]
    sorted_samples = np.argsort(u_cluster[:, cluster])[::-1]
    closest_samples_label = [idx for idx in sorted_samples if idx in samples_label]
    closest_samples_label = np.array(closest_samples_label)[:n_samples_to_add]
    return closest_samples_label


def fix_class_imbalance(X_cluster, y_cluster, X_out_cluster, y_out_cluster, centroids, cluster):
    n_labels = max(np.hstack((y_out_cluster, y_cluster))) + 1
    class_count = Counter(y_cluster)
    num_majority = max(class_count.values())

    for lbl in range(n_labels):
        idx_class = np.where(y_cluster == lbl)[0]
        n_samples_class = idx_class.size

        if n_samples_class < (num_majority ):
            n_samples_to_add = (num_majority ) - n_samples_class
            new_samples = get_new_samples_cluster(
                X_out_cluster, y_out_cluster, centroids, cluster, lbl, n_samples_to_add
            )
            X_cluster_add = X_out_cluster[new_samples]
            y_cluster_add = y_out_cluster[new_samples]

            X_cluster = np.vstack((X_cluster, X_cluster_add))
            y_cluster = np.concatenate((y_cluster, y_cluster_add))

    class_count = Counter(y_cluster)
    return X_cluster, y_cluster


def ensemble_prediction(X_test, centroids, possible_clusters, ensemble):
    n_clusters = centroids.shape[0]
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


def ensemble_training(X_train, y_train, X_test, y_test, centroids,
                            clusters, base_classifiers=None):

    n_clusters = centroids.shape[0]
    if base_classifiers is None:
        base_classifiers = POSSIBLE_CLASSIFIERS[:n_clusters]

    ensemble = []
    possible_clusters = np.unique(clusters)

    for c in possible_clusters:
        idx_cluster = np.where(clusters == c)[0]
        X_cluster = X_train[idx_cluster]
        y_cluster = y_train[idx_cluster]

        if np.all(y_cluster == y_train[idx_cluster[0]]):
            classifier = KNeighborsClassifier(n_neighbors=1)
        else:
            classifier = base_classifiers[c]

        X_out_cluster = np.delete(X_train, idx_cluster, axis=0)
        y_out_cluster = np.delete(y_train, idx_cluster)
        X_cluster, y_cluster = fix_class_imbalance(
            X_cluster, y_cluster, X_out_cluster, y_out_cluster, centroids, c
        )

        classifier.fit(X_cluster, y_cluster)
        ensemble.append(classifier)
    return ensemble


def calcular_acuracia_por_cluster(y_pred, y_real, clusters, tipo="teste"):
    print("=" * 10 + f" Análise do {tipo} " + "=" * 10)
    for c in np.unique(clusters):
        indexes_c = np.where(clusters == c)[0]
        acc = sum(y_pred[indexes_c] == y_real[indexes_c]) / len(y_real[indexes_c])
        print(f"Acurácia cluster {c}: {acc}")

        print(f"Número de amostras no cluster: {len(indexes_c)}")

        print("Classificador Base do cluster:", POSSIBLE_CLASSIFIERS[c])
        print("-------------------------------")

    acc = sum(y_pred == y_real) / len(y_real)
    print("Acurácia total:", acc)
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
        X_cluster = X_test[idx_cluster]
        centroids_test.append(np.mean(X_cluster, axis=0))

    centroids_test = np.array(centroids_test)
    return centroids_test


def rodar_programa(n_clusters, run):
    filename = f"./resultados/{n_clusters}_clusters_experimento_{run}"

    X, y = read_german_credit_dataset()
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
    clusterer.fit(X_train)
    centroids = clusterer.cluster_centers_
    clusters = clusterer.labels_

    ensemble = ensemble_training(
        X_train, y_train, X_test, y_test, centroids, clusters, base_classifiers=None)
    y_pred, predictions, u = ensemble_prediction(
        X_test, centroids, np.unique(clusters), ensemble
    )

    clusters_test = np.argmax(u, axis=1)
    centroids_test = calc_centroids(n_clusters, clusters_test, X_test)

    calcular_acuracia_por_cluster(y_pred, y_test, clusters_test)
    pesos_por_amostra(predictions, y_pred, y_test, clusters_test, u)
    plotar_clusters(X_test, y_test, clusters_test, centroids_test, f"{filename}_test.png")
    
    y_pred_train, predictions_train, u_train = ensemble_prediction(
        X_train, centroids, np.unique(clusters), ensemble
    )

    mostrar_distribuicao_de_amostras_treino(y_train, clusters, n_clusters)
    calcular_acuracia_por_cluster(y_pred_train, y_train, clusters, tipo="treino")
    pesos_por_amostra(predictions_train, y_pred_train, y_train, clusters, u_train)
    plotar_clusters(X_train, y_train, clusters, centroids, f"{filename}_train.png")


if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    run = int(sys.argv[2])

    rodar_programa(n_clusters, run)
