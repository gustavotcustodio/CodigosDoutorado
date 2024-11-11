import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_biclusters, make_blobs, make_moons, make_circles
from numpy.typing import NDArray

def visualizar_dados(dados: NDArray, labels: NDArray) -> None:
    # Conta a quantidade de labels diferentes
    possible_labels = np.unique(labels)
    colors = ['red', 'blue', 'orange', 'green']

    for lbl in possible_labels:
        # Encontra todos os dados com esse rótulo
        idx_lbl = np.where(labels==lbl)[0]
    
        plt.scatter(dados[idx_lbl, 0], dados[idx_lbl, 1], color=colors[lbl])
    plt.show()
    

def main():
    # Médias
    mean = [1, 2]
    # Cov matriz de covariância
    cov = [[1, 0], [0 ,1]]
    rng = np.random.default_rng()
    X = rng.multivariate_normal( mean, cov, size=3000)
    y = np.zeros(X.shape[0], dtype=int)
    # circles = make_circles()
    # X = circles[0]
    # y = circles[1]
    visualizar_dados(X, y)


if __name__ == "__main__":
    main()


# Como eu preciso que meus dados sintéticos sejam?

# - Overlap
# - desbalanceamento
# - classificação fácil
