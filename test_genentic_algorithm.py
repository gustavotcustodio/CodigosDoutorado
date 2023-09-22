from math import exp
from numpy.testing import assert_almost_equal, assert_approx_equal
from genetic_algorithm import fitness_dists_centroids
import numpy as np


def test_fitness_func():
    X = np.array([[1, 4], [4, 4], [5,4], [1, 3], [2, 3], [4, 3], [5, 3] ])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    n_clusters = 2
    n_labels = 2

    centroids = np.array([[1.5, 3.5], [4.5, 3.5]])
    expected = 2.118
    predicted = fitness_dists_centroids(X, y, n_clusters, n_labels)(None, centroids, 0)
    predicted = round(predicted, 3)

    assert_approx_equal(predicted, expected)

test_fitness_func()
