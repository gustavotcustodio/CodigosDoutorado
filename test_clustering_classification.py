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


def test_calc_intra_cluster():
    X = np.array([[-2.0000000001, -1.0000000001], [2.0000000001, 0.0000000001], [2, 0], [-2, -1], [8.0000000001, 8.0000000001], [8, 8]])
    clusters = np.array([0, 1, 1, 0, 2, 2])
    n_clusters = 3
    centroids = np.empty((n_clusters, X.shape[1]))

    for c in range(n_clusters):
        c_samples = np.where(clusters == c)[0]
        centroids[c] = np.mean(X[c_samples], axis=0)
    predicted = cc.calc_intra_cluster(X, clusters, centroids, n_clusters)
    print(predicted)


def test_find_best_partition_per_class():
    # X = np.array([[1, 2], [1, 0], [0, 0], [0, 1], [3, 2], [2, 2], [7, 1], [5, 5], [2, 9], [8, 7], [5,5], [5,2], [9,0], [3,1]])
    n_samples = 2000
    X = np.random.random(size=(n_samples, 10))
    y = np.random.randint(2, size=(n_samples))
    print(cc.find_best_partition_per_class(X, y))


if __name__ == "__main__":
    # test_calc_centroids_same_cluster()
    # test_get_distances_between_diff_classes_per_cluster()
    # test_calc_intra_cluster()
    # test_find_best_partition_per_class()
