import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV


def filter_files(filelist, dataset):
    files_dataset =  [fname for fname in filelist
                      if "Clustering Analysis" in fname and dataset in fname]
    return sorted(files_dataset)


def count_base_classifiers(filelist, metric="Accuracy"):

    df_base_classifiers = pd.DataFrame()
    target_values = []

    for fname in filelist:
        df_experiments = pd.read_csv(fname)[:-2]
        target_values += list(df_experiments[metric].values)
        group_base_classifiers = [row[1:-1].split(", ")
                                  for row in df_experiments["base_classifiers"]]

        for base_classifiers in group_base_classifiers:
            dict_base_classifiers = Counter(base_classifiers)
            cols = list(dict_base_classifiers.keys())

            df_exp = pd.DataFrame(columns=cols)
            df_exp = df_exp._append(dict_base_classifiers, ignore_index=True)
            df_base_classifiers = pd.concat((df_base_classifiers, df_exp))
            # dict_base_classifiers["Accuracy"] = accuracy_values[i]
    df_base_classifiers.fillna(0, inplace=True)
    df_base_classifiers = df_base_classifiers.apply(lambda row: row / row.sum(), axis=1)

    df_base_classifiers[metric] = target_values
    return df_base_classifiers


def save_tree(df_base_classifiers, filename, metric="Accuracy"):
    y = df_base_classifiers[metric]
    X = df_base_classifiers.drop(columns=metric).values
     
    regressor = DecisionTreeRegressor(max_depth=3)
    regressor.fit(X, y)

    colunas = list(df_base_classifiers.drop(columns=metric).columns)

    _, ax = plt.subplots(figsize=(20,8)) # Resize figure
    plot_tree(regressor, filled=True, ax=ax, feature_names=colunas)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Arquivo {filename} salvo com sucesso.")


def cross_val_tree(df_base_classifiers, filepath_tree, metric="Accuracy"):
    params =  [{
        'criterion': ['mse', 'friedman_mse', 'absolute_error'],
        'max_depth': [3, 5, 7, None],
        'ccp_alpha': [0.0, 0.02, 0.1, 0.3, 0.5]
    }]

    y = df_base_classifiers[metric]
    X = df_base_classifiers.drop(columns=metric).values

    dt = DecisionTreeRegressor()
    gs_tree = GridSearchCV(dt,
                           param_grid=params,
                           scoring='r2',
                           cv=10)
    gs_tree.fit(X, y)
    print("-"*70)
    print(filepath_tree)
    combination_parameters = list(
        zip(gs_tree.cv_results_["params"],
            gs_tree.cv_results_['mean_test_score'])
    )
    print(combination_parameters)

    print(f"\nBest estimator: {gs_tree.best_estimator_}")
    print(f"Best R2: {gs_tree.best_score_}")


relevant_files = [f for f in os.listdir(".") if ".csv" in f]
datasets = ["Water", "Cancer", "Credit", "Wine"]

for dataset in datasets:
    filtered_files = filter_files(relevant_files, dataset)
    df_base_classifiers = count_base_classifiers(filtered_files)

    base_dir = "arvores_regressao"
    filepath_tree = os.path.join(base_dir, f"{dataset}.png")
    os.makedirs(base_dir, exist_ok=True)
    cross_val_tree(df_base_classifiers, filepath_tree)
    # save_tree(df_base_classifiers, filepath_tree)
