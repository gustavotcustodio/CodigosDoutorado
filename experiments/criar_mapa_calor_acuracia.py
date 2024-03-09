import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pandas.core import base
from pandas.io.html import pprint_thing
from sklearn.tree import DecisionTreeRegressor, plot_tree
import seaborn as sns


def filter_files(filelist, dataset):
    files_dataset =  [fname for fname in filelist
                      if "Clustering Analysis" in fname and dataset in fname]
    return sorted(files_dataset)


def create_classification_matrix(filelist, filename, metric="Accuracy"):
    df_heatmap = pd.DataFrame()

    for fname in filelist:
        df_experiments = pd.read_csv(fname)[:-2]
        n_clusters = int(df_experiments['n_clusters'][0])

        df_heatmap[n_clusters] = df_experiments[metric]

    df_heatmap.index = df_heatmap.index + 1
    sns.heatmap(df_heatmap, cmap="Reds", annot=True)
    plt.xlabel("Number of clusters")
    plt.ylabel("Run")
    plt.savefig(filename)
    plt.clf()
    print(f"Arquivo {filename} salvo com sucesso.")


relevant_files = [f for f in os.listdir(".") if ".csv" in f]
datasets = ["Water", "Cancer", "Credit", "Wine"]

for dataset in datasets:
    filtered_files = filter_files(relevant_files, dataset)

    base_dir = "mapas_de_calor_acuracia"
    filepath_heatmap = os.path.join(base_dir, f"{dataset}.png")
    os.makedirs(base_dir, exist_ok=True)

    create_classification_matrix(filtered_files, filepath_heatmap)
