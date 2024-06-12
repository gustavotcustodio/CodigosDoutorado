import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def read_australian_credit_dataset():
    X = np.loadtxt("./datasets/contraceptive.dat", delimiter=",")
    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_pima_dataset():
    X = np.loadtxt("./datasets/pima.dat", delimiter=",")
    y = X[:,-1].astype(int)
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_iris_dataset():
    X = np.loadtxt("./datasets/iris.data", delimiter=",")
    y = X[:,-1].astype(int)
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_heart_dataset():
    X = np.loadtxt("./datasets/heart.dat", delimiter=",")
    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_contraceptive_dataset():
    X = np.loadtxt("./datasets/contraceptive.dat", delimiter=",")
    y = X[:,-1].astype(int) - 1

    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    enc = MultiLabelBinarizer(sparse_output=False)
    y_1_hot = enc.fit_transform(y.reshape(-1, 1))
    return X, y


def read_hepatitis_dataset():
    X = np.loadtxt("./datasets/hepatitis.dat", delimiter=",", dtype="str")
    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]

    for col in range(X.shape[1]):

        idx = np.argwhere(X[:,col]=="?")
        valid_values = np.delete(X[:,col], idx)
        mean_col = valid_values.astype(np.float32).mean()

        X[idx, col] = mean_col

    X = X.astype(np.float32)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_potability_dataset():
    df_potability = pd.read_csv("datasets/potabilidade.csv")
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    y = df_potability["Potability"].values
    X = df_potability.drop(columns="Potability").values
    return X, y


def read_german_credit_dataset():
    X = np.loadtxt("./datasets/german.data-numeric", delimiter=" ")
    np.random.shuffle(X)
    y = X[:, -1] - 1
    X = X[:, :-1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X, y


def read_wine_dataset():
    X = np.loadtxt("./datasets/wine.data", delimiter=",")
    np.random.shuffle(X)
    y = X[:, 0].astype(int) - 1
    X = X[:, 1:]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    enc = MultiLabelBinarizer(sparse_output=False)
    y_1_hot = enc.fit_transform(y.reshape(-1, 1))

    return X, y


def read_wdbc_dataset():
    X = np.loadtxt("./datasets/wdbc.data", delimiter=",")
    np.random.shuffle(X)

    y = X[:, 1]
    X = np.hstack((X[:, [0]], X[:, 2:]))
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    return X, y


def read_vehicle_dataset():
    X = np.loadtxt("./datasets/vehicle.dat", delimiter=",")
    np.random.shuffle(X)

    y = X[:, -1]
    X = X[:, :-1]

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    return X, y


def select_dataset_function(dataset):
    if dataset == "wine":
        read_function = read_wine_dataset

    elif dataset == "german_credit":
        read_function = read_german_credit_dataset

    elif dataset == "wdbc":
        read_function = read_wdbc_dataset

    elif dataset == "water":
        read_function = read_potability_dataset

    elif dataset == "contraceptive":
        read_function = read_contraceptive_dataset

    elif dataset == "hepatitis":
        read_function = read_hepatitis_dataset

    elif dataset == "vehicle":
        read_function = read_vehicle_dataset

    elif dataset == "australian_credit":
        read_function = read_australian_credit_dataset

    elif dataset == "pima":
        read_function = read_pima_dataset

    elif dataset == "heart":
        read_function = read_heart_dataset

    elif dataset == "iris":
        read_function = read_iris_dataset

    else:
        print("Erro ao selecionar dataset.")
        sys.exit()
    return read_function


if __name__ == "__main__":
    X, y = read_pima_dataset()

    for i in range(X.shape[1]):
        print(f"Máximo e mínimo da coluna {i}")
        print(X[:, i].max())
        print(X[:, i].min())
    print("Valores y:",y)
