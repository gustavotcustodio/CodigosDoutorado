import sys
import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold


def split_training_test(X, y, fold, n_runs=10):
    fold -= 1

    skf = StratifiedKFold(n_splits=n_runs, shuffle=False)
    splits = skf.split(X, y)

    idx = [split for split in splits]

    X_train, y_train = X[idx[fold][0]], y[idx[fold][0]]
    X_test, y_test = X[idx[fold][1]], y[idx[fold][1]]

    return X_train, X_test, y_train, y_test

def read_electricity_dataset():
    df = pd.read_csv("./datasets/electricity.csv").astype(np.float32)
    X = df.values
    np.random.seed(42)
    np.random.shuffle(X)

    ct = ColumnTransformer(
            transformers = [(
                'one_hot_encoder',
                OneHotEncoder(categories = 'auto', sparse_output=False),
                [1]
            )],
            remainder = 'passthrough')
    y = X[:, -1].astype(int)
    X = X[:, :-1]

    X = ct.fit_transform(X)
    return X, y


def read_blood_dataset():
    X = np.loadtxt("./datasets/blood.csv", delimiter=",", skiprows=1)
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]

    return X, y


def read_australian_credit_dataset():
    X = np.loadtxt("./datasets/australian.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]

    ct = ColumnTransformer(
            transformers = [
                ('one_hot_encoder',
                 OneHotEncoder(categories = 'auto', sparse_output=False),
                 [3, 4, 5, 11])
            ],
            remainder = 'passthrough')
    X = ct.fit_transform(X)
    return X, y

def read_elipses_dataset():
    X = np.loadtxt("./datasets/elipses.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)
    y = X[:,-1].astype(int)
    X = X[:, :-1]
    return X, y

def read_rectangles_dataset():
    X = np.loadtxt("./datasets/rectangles.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)
    y = X[:,-1].astype(int)
    X = X[:, :-1]
    return X, y

def read_pima_dataset():
    X = np.loadtxt("./datasets/pima.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]
    return X, y


def read_iris_dataset():
    X = np.loadtxt("./datasets/iris.data", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int)
    X = X[:, :-1]
    return X, y


def read_heart_dataset():
    X = np.loadtxt("./datasets/heart.dat", delimiter=",")

    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int) - 1
    X = X[:, :-1]
    return X, y


def read_contraceptive_dataset():
    X = np.loadtxt("./datasets/contraceptive.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:,-1].astype(int) - 1

    X = X[:, :-1]
    ct = ColumnTransformer(
            transformers = [
                ('one_hot_encoder',
                 OneHotEncoder(categories = 'auto', sparse_output=False),
                 [1, 2, 6, 7])],
            remainder = 'passthrough')
    X = ct.fit_transform(X)

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
    return X, y


def read_wine_dataset():
    X = np.loadtxt("./datasets/wine.data", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:, 0].astype(int) - 1
    X = X[:, 1:]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    # enc = MultiLabelBinarizer(sparse_output=False)
    # y_1_hot = enc.fit_transform(y.reshape(-1, 1))

    return X, y


def read_wdbc_dataset():
    X = np.loadtxt("./datasets/wdbc.data", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:, 1].astype(int)
    X = np.hstack((X[:, [0]], X[:, 2:]))
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())

    return X, y


def read_vehicle_dataset():
    X = np.loadtxt("./datasets/vehicle.dat", delimiter=",")
    np.random.seed(42)
    np.random.shuffle(X)

    y = X[:, -1].astype(int)
    X = X[:, :-1]

    return X, y


DATASETS_INFO = {
    "wine": {"function": read_wine_dataset, "nlabels": 3},
    "blood": {"function": read_blood_dataset, "nlabels": 2},
    "electricity": {"function": read_electricity_dataset, "nlabels": 2},
    "german_credit": {"function": read_german_credit_dataset, "nlabels": 2},
    "wdbc": {"function": read_wdbc_dataset, "nlabels": 2},
    "water": {"function": read_potability_dataset, "nlabels": 2},
    "contraceptive": {"function": read_contraceptive_dataset, "nlabels": 3},
    "hepatitis": {"function": read_hepatitis_dataset, "nlabels": 2},
    "vehicle": {"function": read_vehicle_dataset, "nlabels": 3},
    "australian_credit": {"function": read_australian_credit_dataset, "nlabels": 2},
    "pima": {"function": read_pima_dataset, "nlabels": 2},
    "heart": {"function": read_heart_dataset, "nlabels": 2},
    "iris": {"function": read_iris_dataset, "nlabels": 3},
    "elipses": {"function": read_elipses_dataset, "nlabels": 3},
    "rectangles": {"function": read_rectangles_dataset, "nlabels": 2},
}


def select_dataset_function(dataset):
    if dataset in DATASETS_INFO:
        return DATASETS_INFO[dataset]["function"]
    else:
        print("Error: invalid dataset name.")
        sys.exit(1)


def normalize_data(X_train, X_test):
    min_max_scaler = MinMaxScaler()

    min_max_scaler.fit(X_train)

    X_train = min_max_scaler.transform(X_train)

    X_test = min_max_scaler.transform(X_test)
    return X_train, X_test


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = read_iris_dataset(0)

    datasets = [ "wine", "blood", "german_credit", "wdbc", "contraceptive",
                  "australian_credit", "pima", "heart", "iris"]

    for dataset in datasets:
        func = select_dataset_function(dataset)
        print(f"--------- {dataset} ----------")
        for run in range( 10):
            print(f"Fold {run}")
            X, y = func()
            X_train, X_test, y_train, y_test = split_training_test(X, y, run, n_runs=10)
            print(X_train.shape)
            print(X_test.shape)
