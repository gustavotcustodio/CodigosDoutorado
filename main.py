from clustering_classification import ensemble_classification
from loader_and_preprocessor import read_potability_dataset, read_wine_dataset, read_wdbc_dataset
import clustering_classification as cc
from genetic_algorithm import run_ga, fitness_overlap
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def perform_ga_step(X_train, y_train, n_clusters, n_labels):
    n_genes = n_clusters * X_train.shape[1]

    ff = fitness_overlap(X_train, y_train, n_clusters, n_labels)

    centroids, final_fitness = run_ga(ff, n_genes)
    centroids = centroids.reshape((n_clusters, X_train.shape[1]))

    return centroids, final_fitness


def perform_clustering_analysis(X_train, y_train, n_labels):
    max_clusters = int(np.sqrt(X_train.shape[0]))
    best_centroids = None
    best_fitness = float("-inf")

    for c in range(5, max_clusters + 1):
        fitness_function = fitness_overlap(X_train, y_train, c, n_labels)

        clusterer = KMeans(c, n_init='auto')
        clusterer.fit(X_train)
        fitness_value = fitness_function(None, clusterer.cluster_centers_, 0)

        if fitness_value > best_fitness:
            best_centroids = clusterer.cluster_centers_
            best_fitness = fitness_value
            # best_c = c

    # print(best_c)
    return best_centroids, best_fitness


if __name__ == "__main__":
    df_potability = read_potability_dataset("potabilidade.csv")

    df_potability.to_csv("lalala.csv", index=False)
    X = df_potability.drop(columns="Potability").values
    X = (X - X.min()) / (X.max() - X.min())

    y = df_potability["Potability"].values

    n_labels = np.unique(y).shape[0]
    n_clusters = 5

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # centroids, final_fitness = perform_ga_step(X_train, y_train, n_clusters, n_labels)
    centroids, final_fitness = perform_clustering_analysis(X_train, y_train, n_labels)

    u = cc.calc_membership_values(X_train, centroids)
    ensemble_classification(X_train, y_train, u)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)

    print("Random Forest:", accuracy)
    ## -----------------------------------------------------
    #clusters = np.argmax(u, axis=1)

    #print("GA:", final_fitness)
    #print("Silhouette GA:", silhouette_score(X_train, clusters))

    ## =======================================================
    # ff = fitness_dists_centroids(X_train, y_train, n_clusters, n_labels)

    # clusterer = KMeans(n_clusters, n_init='auto')
    # clusterer.fit(X_train)
    # # print(ff(None, clusterer.cluster_centers_, 0))  # , clusterer.cluster_centers_
    # fitness_kmeans = ff(None, clusterer.cluster_centers_, 0)

    # print("KMeans:", fitness_kmeans)
    # print("Silhouette KMeans:", silhouette_score(X_train, clusterer.labels_))

    # from sklearn.svm import SVC
    # X = np.loadtxt("./glass/glass.data", delimiter=',')

    # np.random.shuffle(X)
    # y = X[:, -1]

    # X = (X - X.min()) / (X.max() - X.min())
    # X = X[:, :-1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # n_labels = np.unique(y).shape[0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # centroids, final_fitness = perform_clustering_analysis(X_train, y_train, n_labels)

    # u = cc.calc_membership_values(X_train, centroids)
    # ensemble_classification(X_train, y_train, u)

    print("================= Wine =====================")
    X, y = read_wine_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    n_labels = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Random Forest:", accuracy)


    print("================= Cancer =====================")
    X, y = read_wdbc_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    n_labels = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Random Forest:", accuracy)

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("XGBClassifier:", accuracy)
# Overlap de cada atributo para cada cluster diferente
