import pandas as pd
import numpy as np


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

if __name__ == "__main__":
    print(read_german_credit_dataset())
