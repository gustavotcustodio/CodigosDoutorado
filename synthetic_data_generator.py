import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from sklearn.manifold import TSNE

def save_information_random_generation(
    output_file: str, means_by_class: list[NDArray], cov_by_class: list[NDArray],
):
    n_classes = len(means_by_class)
    with open(output_file, 'w') as f:

        for cls in range(n_classes):
            print(f"Means class {cls}:", file=f) 
            print(means_by_class[cls], file=f) 
            print(f"\nCovariance class {cls}:", file=f) 
            print(f"{cov_by_class[cls]}\n", file=f) 

    print(output_file, "saved.")


def visualize_data(dados: NDArray, labels: NDArray) -> None:
    # Conta a quantidade de labels diferentes
    possible_labels = np.unique(labels).astype(int)
    colors = ['red', 'blue', 'orange', 'green']

    if dados.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        # Check if data has less than 3 dims. If it
        # does, use t-SNE to reduce to 2 dims.
        dados_2d = tsne.fit_transform(dados)
    else:
        dados_2d = dados

    for lbl in possible_labels:
        # Encontra todos os dados com esse rótulo
        idx_lbl = np.where(labels==lbl)[0]

        dict_data = {
         "x1": dados_2d[idx_lbl, 0],
         "x2": dados_2d[idx_lbl, 1]}

        sns.scatterplot(dict_data, x="x1", y="x2", color=colors[lbl],
                        alpha=(0.75 / 1.5))
    
    plt.show()
    

def generate_normal_distributed_data(
    output_file: str, n_features: int=2,
    n_classes: int=2, n_samples_class: int=500
) -> None:
    # Covariance matrix
    # [1.0, 0.5 ...]
    # [0.5, 1.0,...]
    # [...]

    features_by_class = []
    labels_by_class = []
    means_by_class = []
    cov_by_class = []

    rng = np.random.default_rng()

    for cls in range(n_classes):
        means = np.random.random(n_features)
        # Cov matriz de covariância
        cov = np.random.random((n_features, n_features))
        cov = np.dot(cov, cov.T)

        means_by_class.append(means)
        cov_by_class.append(cov)

        X_class = rng.multivariate_normal(means, cov, size=n_samples_class)
        y_class = np.full(X_class.shape[0], cls, dtype=int)

        features_by_class.append(X_class)
        labels_by_class.append(y_class)

    X = np.vstack(features_by_class)
    y = np.hstack(labels_by_class)

    resulting_data = np.hstack((X, y[:, np.newaxis]))
    # Save synthetic data
    np.savetxt(output_file, resulting_data, delimiter=",")
    print(f"Dataset {output_file} salvo com sucesso")

    visualize_data(X, y)

    info_file_no_ext, _ = os.path.splitext( output_file.split('/')[-1] )
    info_file_name =  f"info_{info_file_no_ext}.txt"
    path_info_file = "/".join(output_file.split('/')[:-1])
    full_info_file_path = f"{path_info_file}/{info_file_name}"
    save_information_random_generation(full_info_file_path, means_by_class, cov_by_class)


def create_elipse(width=1, height=1, shift_x=0.0, shift_y=0.0, label=0,
                  n_samples_class: int=500):
    t = np.random.random(size=n_samples_class)
    u = np.random.random(size=n_samples_class)

    x1 = width  * np.sqrt(t) * np.cos(2 * math.pi * u) + shift_x
    x2 = height *  np.sqrt(t) * np.sin(2 * math.pi * u) + shift_y

    X = np.vstack((x1, x2)).T
    y = np.array([label] * n_samples_class)

    return X, y

def create_multiple_elipses(
    widths:list, heights:list, shift_vals_x:list, shift_vals_y:list,
    n_samples_class: int=500,
):
    X = []
    y = []

    for i in range(len(widths)):
        X_el, y_el = create_elipse(
            widths[i], heights[i], shift_vals_x[i], shift_vals_y[i], label=i,
            n_samples_class=n_samples_class
        )
        X.append(X_el)
        y.append(y_el)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def create_rectangle(width=1, height=1, shift_x=0.0, shift_y=0.0, label=0,
                     n_samples_class=500):
    x1 = 2 * np.random.random(size=n_samples_class) * width + shift_x - 1
    x2 = 2 * np.random.random(size=n_samples_class) * height + shift_y - 1

    X = np.vstack((x1, x2)).T
    y = np.array([label] * n_samples_class)

    return X, y


def create_multiple_rectangles(
    widths:list, heights:list, shift_vals_x:list, shift_vals_y:list,
    n_samples_class:int=500
):
    X = []
    y = []

    for i in range(len(widths)):
        X_el, y_el = create_rectangle(
            widths[i], heights[i], shift_vals_x[i], shift_vals_y[i], label=i,
            n_samples_class=n_samples_class
        )
        X.append(X_el)
        y.append(y_el)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def generate_elipses(
    output_file: str, n_features: int=2,
    n_classes: int=2, n_samples_class: int=500
) -> None:

    X, y = create_multiple_elipses(
            widths=[1, 1, 1], heights=[1, 1, 1],
            shift_vals_x=[0, 1, 0.5], shift_vals_y=[0, 0, -1],
            n_samples_class=n_samples_class
            )
    visualize_data(X, y)

    dataset_elipses = np.hstack((X, y[:, np.newaxis]))

    np.savetxt(output_file, dataset_elipses, delimiter=",")
    print(f"Dataset {output_file} salvo com sucesso")

def generate_rectangles(
    output_file: str, n_features: int=2,
    n_classes: int=2, n_samples_class: int=500
) -> None:
    X, y = create_multiple_rectangles(
            widths=[1, 1], heights=[1, 1], shift_vals_x=[0, 1],
            shift_vals_y=[0, 0], n_samples_class=n_samples_class)
    visualize_data(X, y)

    dataset_rectangles = np.hstack((X, y[:, np.newaxis]))

    np.savetxt(output_file, dataset_rectangles, delimiter=",")
    print(f"Dataset {output_file} salvo com sucesso")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g", "--generation_function", type=str, required=True,
        help = "Dataset to be generated (elipses/rectangles/normal).")
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help = "Output file name.")
    parser.add_argument(
        "-f", "--n_features", type=int, default=2,
        help = "Number of features for normal distrib. multivariate.")
    parser.add_argument(
        "-c", "--n_classes", type=int, default=2,
        help = "Number of classes.")
    parser.add_argument(
        "-s", "--n_samples_class", type=int, default=500,
        help = "Number of samples for each class.")

    # Read arguments from command line
    args = parser.parse_args()
    gen_function = args.generation_function.lower()

    assert gen_function in ['elipses', 'rectangles', 'normal'], \
            f"Error. Unknown generation function {gen_function}"

    GENERATION_FUNCTIONS[gen_function](
        args.output_file,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_samples_class=args.n_samples_class
    )


if __name__ == "__main__":
    GENERATION_FUNCTIONS = {
        "elipses": generate_elipses,
        "rectangles": generate_rectangles,
        "normal": generate_normal_distributed_data
    }

    main()
