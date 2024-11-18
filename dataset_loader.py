import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold


def split_training_test(X, y, fold, n_runs=10):
    # n_samples = X.shape[0]
    # fold_size = n_samples // n_runs + int(n_samples % n_runs > 1)

    # X_test = X[fold * fold_size: (fold + 1) * fold_size]
    # X_train = np.vstack((X[0: fold * fold_size], X[(fold + 1) * fold_size : ]))

    # y_test = y[fold * fold_size: (fold + 1) * fold_size]
    # y_train = np.concatenate((y[0: fold * fold_size], y[(fold + 1) * fold_size : ]))

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    splits = skf.split(X, y)

    idx = [split for split in splits]

    X_train, y_train = X[idx[fold][0]], y[idx[fold][0]]
    X_test, y_test = X[idx[fold][1]], y[idx[fold][1]]

    return X_train, X_test, y_train, y_test


def read_australian_credit_dataset():
    X = np.loadtxt("./datasets/australian.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]

    ct = ColumnTransformer(
            transformers = [('one_hot_encoder', OneHotEncoder(categories = 'auto', sparse_output=False),
                             [3, 4, 5, 11])],
            remainder = 'passthrough')
    X = ct.fit_transform(X)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_pima_dataset():
    X = np.loadtxt("./datasets/pima.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_iris_dataset():
    X = np.loadtxt("./datasets/iris.data", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_heart_dataset():
    X = np.loadtxt("./datasets/heart.dat", delimiter=",")

    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_contraceptive_dataset():
    X = np.loadtxt("./datasets/contraceptive.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int) - 1

    X = X[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # enc = MultiLabelBinarizer(sparse_output=False)
    # y_1_hot = enc.fit_transform(y.reshape(-1, 1))
    return X, y


def read_hepatitis_dataset():
    X = np.loadtxt("./datasets/hepatitis.dat", delimiter=",", dtype="str")
    np.random.seed(42)
    np.random.shuffle(X)

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
    # np.random.seed(42)
    # np.random.shuffle(X)

    df_potability = pd.read_csv("datasets/potabilidade.csv")
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    y = df_potability["Potability"].values
    X = df_potability.drop(columns="Potability").values
    return X, y


def read_german_credit_dataset():
    X = np.loadtxt("./datasets/german.data-numeric", delimiter=" ")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:, -1].astype(int) - 1
    X = X[:, :-1]
    ct = ColumnTransformer(
            transformers = [('one_hot_encoder', OneHotEncoder(categories = 'auto', sparse_output=False),
                             [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19 ])],
            remainder = 'passthrough')
    X = ct.fit_transform(X)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def read_wine_dataset():
    X = np.loadtxt("./datasets/wine.data", delimiter=",")
    y = X[:, 0].astype(int) - 1
    X = X[:, 1:]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    # enc = MultiLabelBinarizer(sparse_output=False)
    # y_1_hot = enc.fit_transform(y.reshape(-1, 1))

    return X, y


def read_wdbc_dataset():
    X = np.loadtxt("./datasets/wdbc.data", delimiter=",")

    y = X[:, 1].astype(int)
    X = np.hstack((X[:, [0]], X[:, 2:]))
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    return X, y


def read_vehicle_dataset():
    X = np.loadtxt("./datasets/vehicle.dat", delimiter=",")

    y = X[:, -1].astype(int)
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
        sys.exit(1)
    return read_function


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = read_iris_dataset(0)

    datasets = [ "wine", "german_credit", "wdbc", "contraceptive",
                  "australian_credit", "pima", "heart", "iris"]

    for dataset in datasets:
        func = select_dataset_function(dataset)
        print(f"--------- {dataset} ----------")
        for run in range(4, 10):
            print(f"Fold {run}")
            X, y = func()
            X_train, X_test, y_train, y_test = split_training_test(X, y, run, n_runs=10)
            print(X_train.shape)
            print(X_test.shape)
