import os
import numpy as np
import matplotlib.pyplot as plt
import dataset_loader
from sklearn.manifold import TSNE


def plot_clusters(X, y, dataset, use_tsne=True):
    if use_tsne:
        tsne_model = TSNE(n_components=2)
        low_dim_data = tsne_model.fit_transform(X)
    else:
        low_dim_data = X

    for l in np.unique(y):
        idx_labels = np.where(y == l)[0]

        plt.scatter(low_dim_data[idx_labels, 0], low_dim_data[idx_labels, 1],
                    alpha=0.5, label="Label %d" % l)

    plt.legend()
    plt.savefig(f"{dataset}")
    plt.clf()
    print(f"{dataset} salvo com sucesso.")


if __name__ == "__main__":
    PASTA_TSNE = "tsne"

    datasets = [
        "australian_credit", "german_credit", "heart", "pima", "wdbc",
        "blood", "electricity", "iris", "wine", "contraceptive",
        "rectangles", "elipses", "normal_2_class", "normal_3_class",
    ]

    os.makedirs(PASTA_TSNE, exist_ok=True)

    for dataset in datasets:
        X, y = dataset_loader.select_dataset_function(dataset)()
        if dataset == 'elipses' or dataset == 'rectangles':
            plot_clusters(X, y, f"{PASTA_TSNE}/{dataset}.png", use_tsne=False)
        else:
            plot_clusters(X, y, f"{PASTA_TSNE}/{dataset}.png", use_tsne=True)


