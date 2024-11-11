import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset_loader import select_dataset_function


def plot_clusters(X_train, y_train, dataset):
    tsne_model = TSNE(n_components=2)
    low_dim_data = tsne_model.fit_transform(X_train)

    for l in np.unique(y_train):
        idx_labels = np.where(y_train == l)[0]

        plt.scatter(low_dim_data[idx_labels, 0], low_dim_data[idx_labels, 1],
                    alpha=0.7, label="Label %d" % l)

    plt.legend()

    plt.savefig(f"{dataset}")
    plt.clf()
    print(f"{dataset} salvo com sucesso.")


if __name__ == "__main__":
    PASTA_TSNE = "tsne"

    datasets = [
        "wine", "german_credit", "wdbc", "contraceptive",
        "australian_credit", "pima", "heart", "iris"
    ]

    os.makedirs(PASTA_TSNE, exist_ok=True)

    for dataset in datasets:
        func = select_dataset_function(f"{dataset}")
        X, y = func()
        plot_clusters(X, y, f"{PASTA_TSNE}/{dataset}.png")
