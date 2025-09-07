import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from sklearn.manifold import TSNE

def save_information_random_generation(
    output_file: str, means1: NDArray, means2: NDArray,
    cov1: NDArray, cov2: NDArray
):
    with open(output_file, 'w') as f:
        print("Means class 1:", file=f) 
        print(means1, file=f) 
        print("\nCovariance class 1:", file=f) 
        print(cov1, file=f) 

        print("\nMeans class 2:", file=f) 
        print(means2, file=f) 
        print("\nCovariance class 2:", file=f) 
        print(cov2, file=f) 
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
    output_file: str, n_features: int=2, n_samples_class: int=500
) -> None:
    # Covariance matrix
    # [1.0, 0.5 ...]
    # [0.5, 1.0,...]
    # [...]

    means1 = np.random.random(n_features)
    # Cov matriz de covariância
    cov1 = np.random.random((n_features, n_features))
    cov1 = np.dot(cov1, cov1.T)

    rng = np.random.default_rng()
    X1 = rng.multivariate_normal(means1, cov1, size=n_samples_class)

    y1 = np.zeros(X1.shape[0], dtype=int)

    means2 = np.random.random(n_features)
    # Cov matriz de covariância
    cov2 = np.random.random((n_features, n_features))
    cov2 = np.dot(cov2, cov2.T)

    X2 = rng.multivariate_normal(means2, cov2, size=n_samples_class)
    y2 = np.ones(X2.shape[0], dtype=int)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    resulting_data = np.hstack((X, y[:, np.newaxis]))
    # Save synthetic data
    np.savetxt(output_file, resulting_data, delimiter=",")
    print(f"Dataset {output_file} salvo com sucesso")

    visualize_data(X, y)

    info_file_name =  f"info_{output_file.split('/')[-1]}".strip(".csv")
    path_info_file = "/".join(output_file.split('/')[:-1])
    full_info_file_path = f"{path_info_file}/{info_file_name}.txt"
    save_information_random_generation(full_info_file_path, means1, means2, cov1, cov2)


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

def generate_elipses(output_file: str, n_features: int=2,
                     n_samples_class: int=500) -> None:
    X, y = create_multiple_elipses(
            widths=[1, 1, 1], heights=[1, 1, 1],
            shift_vals_x=[0, 1, 0.5], shift_vals_y=[0, 0, -1],
            n_samples_class=n_samples_class
            )
    visualize_data(X, y)

    dataset_elipses = np.hstack((X, y[:, np.newaxis]))

    np.savetxt(output_file, dataset_elipses, delimiter=",")
    print(f"Dataset {output_file} salvo com sucesso")

def generate_rectangles(output_file: str, n_features: int=2,
                        n_samples_class: int=500) -> None:
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
        "-s", "--n_samples_class", type=int, default=500,
        help = "Number of samples for each class.")

    # Read arguments from command line
    args = parser.parse_args()
    gen_function = args.generation_function.lower()

    assert gen_function in ['elipses', 'rectangles', 'normal'], \
            f"Error. Unknown generation function {gen_function}"

    GENERATION_FUNCTIONS[gen_function](
        args.output_file, n_features=args.n_features,
        n_samples_class=args.n_samples_class
    )


if __name__ == "__main__":
    GENERATION_FUNCTIONS = {
        "elipses": generate_elipses,
        "rectangles": generate_rectangles,
        "normal": generate_normal_distributed_data
    }

    main()
