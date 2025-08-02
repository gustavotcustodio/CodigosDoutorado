import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def visualize_data(dados: NDArray, labels: NDArray) -> None:
    # Conta a quantidade de labels diferentes
    possible_labels = np.unique(labels).astype(int)
    colors = ['red', 'blue', 'orange', 'green']

    for lbl in possible_labels:
        # Encontra todos os dados com esse rótulo
        idx_lbl = np.where(labels==lbl)[0]

        dict_data = {
         "x1": dados[idx_lbl, 0],
         "x2": dados[idx_lbl, 1]}

        sns.scatterplot(dict_data, x="x1", y="x2", color=colors[lbl], alpha=(0.75 / 1.5))
    
    plt.show()
    

def generate_normal_distributed_data():
    # Médias
    mean = [1, 0]
    # Cov matriz de covariância
    cov = [[1, 0], [0 ,1]]
    rng = np.random.default_rng()
    X1 = rng.multivariate_normal(mean, cov, size=5000)

    y1 = np.zeros(X1.shape[0], dtype=int)

    mean = [-1, 0]
    X2 = rng.multivariate_normal(mean, cov, size=5000)
    y2 = np.ones(X2.shape[0], dtype=int)
    # circles = make_circles()
    # X = circles[0]
    # y = circles[1]

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # Save synthetic data
    clf = RandomForestClassifier()    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    visualize_data(X, y)

    # dados = np.hstack((X,y[:, np.newaxis]))
    # np.savetxt("./datasets/circles.dat", dados, delimiter=",")

def create_elipse(width=1, height=1, shift_x=0.0, shift_y=0.0, label=0):
    n_samples_class = 150

    t = np.random.random(size=n_samples_class)
    u = np.random.random(size=n_samples_class)

    x1 = width  * np.sqrt(t) * np.cos(2 * math.pi * u) + shift_x
    x2 = height *  np.sqrt(t) * np.sin(2 * math.pi * u) + shift_y

    X = np.vstack((x1, x2)).T
    y = np.array([label] * n_samples_class)

    return X, y

def create_multiple_elipses(widths:list, heights:list,
                            shift_vals_x:list, shift_vals_y:list):
    X = []
    y = []

    for i in range(len(widths)):
        X_el, y_el = create_elipse(widths[i], heights[i],
                                   shift_vals_x[i], shift_vals_y[i], label=i)
        X.append(X_el)
        y.append(y_el)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def create_rectangle(width=1, height=1, shift_x=0.0, shift_y=0.0, label=0):
    n_samples_class = 500
    x1 = 2 * np.random.random(size=n_samples_class) * width + shift_x - 1
    x2 = 2 * np.random.random(size=n_samples_class) * height + shift_y - 1

    X = np.vstack((x1, x2)).T
    y = np.array([label] * n_samples_class)

    return X, y


def create_multiple_rectangles(widths:list, heights:list,
                               shift_vals_x:list, shift_vals_y:list):
    X = []
    y = []

    for i in range(len(widths)):
        X_el, y_el = create_rectangle(
            widths[i], heights[i], shift_vals_x[i], shift_vals_y[i], label=i
        )
        X.append(X_el)
        y.append(y_el)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def main():
    name_dataset_elipses = 'elipses.dat'

    X, y = create_multiple_elipses(
            widths=[1, 1, 1], heights=[1, 1, 1],
            shift_vals_x=[0, 1, 0.5], shift_vals_y=[0, 0, -1]
            )
    visualize_data(X, y)

    dataset_elipses = np.hstack((X, y[:, np.newaxis]))

    np.savetxt(f"./datasets/{name_dataset_elipses}", dataset_elipses, delimiter=",")
    print(f"Dataset {name_dataset_elipses} salvo com sucesso")

    #########################################################
    name_dataset_rectangles = 'rectangles.dat'

    X, y = create_multiple_rectangles(
            widths=[1, 1], heights=[1, 1],
            shift_vals_x=[0, 1], shift_vals_y=[0, 0]
            )
    visualize_data(X, y)

    dataset_rectangles = np.hstack((X, y[:, np.newaxis]))

    np.savetxt(f"./datasets/{name_dataset_rectangles}", dataset_rectangles, delimiter=",")
    print(f"Dataset {name_dataset_rectangles} salvo com sucesso")

if __name__ == "__main__":
    main()


# Como eu preciso que meus dados sintéticos sejam?

# - Overlap
# - desbalanceamento
# - classificação fácil
