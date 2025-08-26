import numpy as np
from collections import Counter

def fix_predict_prob(y_prob_cluster, labels_in_cluster, total_labels):
    # y_prob_cluster[:, 0] = np.nan_to_num(y_prob_cluster[:, 0], nan=1.0)
    # y_prob_cluster = np.nan_to_num(y_prob_cluster, nan=1/3)

    possible_labels = np.unique(labels_in_cluster)

    y_prob_cluster_allclasses = np.zeros(
            (y_prob_cluster.shape[0], total_labels))

    for i, lbl in enumerate(possible_labels):
        y_prob_cluster_allclasses[:, lbl] = y_prob_cluster[:, i]
        # print(y_prob_cluster)
        # print(possible_labels)

    return y_prob_cluster_allclasses

def show_samples_by_cluster(labels_by_cluster):
    for cluster, labels in labels_by_cluster.items():
        print(f"Cluster {cluster} counting:", Counter(labels))


def replace_nan_probs_by_predictions(predicted_probs, classifier, X):
    if np.isnan(predicted_probs).any():
        idx_nans = np.where(np.isnan(predicted_probs))[0]
        idx_nans = np.unique(idx_nans)

        predictions = classifier.predict(X[idx_nans])
        predicted_probs[idx_nans] = 0.0
        predicted_probs[idx_nans, predictions] = 1.0
