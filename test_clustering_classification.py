import numpy as np
import clustering_classification as cc


def test_calc_centroids_same_cluster():
    X = np.array([[1, 2, 3],
                  [1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [3, 2, 0.5]])
    y = np.array([0, 0, 1, 1, 1])
    n_labels = 2
    # centroid 1 = 1, 1, 1.5
    # centroid 2 = 1, 1, 0.5
    expected = np.array([[1, 1, 1.5],
                         [1, 1, 0.5]])
    predicted = cc.calc_centroids_same_cluster(X, y, n_labels)

    np.testing.assert_equal(expected, predicted)


def test_get_distances_between_diff_classes_per_cluster():
    X = np.array([[1, 2], [1, 0], [0, 0], [0, 1], [3, 2], [2, 2]])
    y = np.array([0, 0, 1, 0, 1, 1])
    clusters = np.array([1, 2, 1, 0, 2, 2])
    n_clusters = 3
    n_labels = 2

    expected = np.array([0, 2.236068, 2.5])
    predicted = cc.get_distances_between_diff_classes_per_cluster(
        X, y, clusters, n_clusters, n_labels
    )
    np.testing.assert_almost_equal(expected, predicted)


if __name__ == "__main__":
    test_calc_centroids_same_cluster()
    test_get_distances_between_diff_classes_per_cluster()
