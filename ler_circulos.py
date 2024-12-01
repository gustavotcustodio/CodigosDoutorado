from types import CellType
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.common import random_state
import seaborn as sns
from sklearn.datasets import make_biclusters, make_blobs, make_moons, make_circles
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from cbeg import CBEG

def visualize_data(dados: NDArray, labels: NDArray) -> None:
    # Conta a quantidade de labels diferentes
    possible_labels = np.unique(labels)
    colors = ['red', 'blue', 'orange', 'green']

    for lbl in possible_labels:
        # Encontra todos os dados com esse rótulo
        idx_lbl = np.where(labels==lbl)[0]

        dict_data = {
         "x1": dados[idx_lbl, 0],
         "x2": dados[idx_lbl, 1]}

        sns.scatterplot(dict_data, x="x1", y="x2", color=colors[lbl], alpha=(1.5-lbl) / 1.5)
    
    plt.show()
    

def main():
    X = np.loadtxt("./datasets/circles.txt", delimiter=",")
    # # Médias
    # mean = [1, 0]
    # # Cov matriz de covariância
    # cov = [[1, 0], [0 ,1]]
    # rng = np.random.default_rng()
    # X1 = rng.multivariate_normal(mean, cov, size=500)
    y1 = np.zeros(X.shape[0]//2, dtype=int)

    # mean = [-1, 0]
    # X2 = rng.multivariate_normal(mean, cov, size=500)
    y2 = np.ones(X.shape[0]//2, dtype=int)
    # # circles = make_circles()
    # # X = circles[0]
    # # y = circles[1]

    # X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    np.random.seed(42)
    np.random.shuffle(X)
    # y = X[:,-1].astype(int)
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # Save synthetic data
    clf = RandomForestClassifier()    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_pred, y_test))

    visualize_data(X, y)


if __name__ == "__main__":
    main()


# Como eu preciso que meus dados sintéticos sejam?

# - Overlap
# - desbalanceamento
# - classificação fácil
