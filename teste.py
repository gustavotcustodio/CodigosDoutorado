import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from loader_and_preprocessor import read_potability_dataset, read_wine_dataset, read_wdbc_dataset

def plot_clusters(X_train):
    clusterer = KMeans(n_clusters=3, n_init='auto')
    clusterer.fit(X_train)
    clusters = clusterer.labels_
    tsne_model = TSNE(n_components=2, random_state=0)
    low_dim_data = tsne_model.fit_transform(X_train)

    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        data_clusters = low_dim_data[idx]
        plt.scatter(data_clusters[:, 0], data_clusters[:, 1], alpha=0.7)
    plt.show()


if __name__ == "__main__":
    df_potability = read_potability_dataset()
    X = df_potability.drop(columns="Potability").values
    X = (X - X.min()) / (X.max() - X.min())

    y = df_potability["Potability"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    n_labels = np.unique(y).shape[0]
    plot_clusters(X_train)



    X, y = read_wine_dataset()

    n_labels = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    plot_clusters(X_train)



    X, y = read_wdbc_dataset()

    n_labels = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    plot_clusters(X_train)
