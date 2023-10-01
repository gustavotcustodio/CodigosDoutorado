import pandas as pd
import numpy as np

def read_potability_dataset(dataset_name: str) -> pd.DataFrame:
    df_potability = pd.read_csv(dataset_name)
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    return df_potability

def read_wine_dataset():
    X = np.loadtxt("./wine.data", delimiter=",")
    np.random.shuffle(X)
    y = X[:, 0] - 1
    X = X[:, 1:]
    X = (X - X.min()) / (X.max() - X.min())
    return X, y


def read_wdbc_dataset():
    X = np.loadtxt("./wdbc.data", delimiter=",")
    np.random.shuffle(X)

    y = X[:, 1]
    X = np.hstack((X[:, [0]], X[:, 2:]))
    X = (X - X.min()) / (X.max() - X.min())
    return X, y
