import numpy as np

def fix_predict_prob(y_prob_cluster, labels_in_cluster, total_labels):
    y_prob_cluster[:, 0] = np.nan_to_num(y_prob_cluster[:, 0], nan=1.0)
    y_prob_cluster = np.nan_to_num(y_prob_cluster, nan=0.0)

    possible_labels = np.unique(labels_in_cluster)

    y_prob_cluster_allclasses = np.zeros(
            (y_prob_cluster.shape[0], total_labels))

    for i, lbl in enumerate(possible_labels):
        y_prob_cluster_allclasses[:, lbl] = y_prob_cluster[:, i]

    return y_prob_cluster_allclasses
