import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif
from dataclasses import dataclass

@dataclass
class FeatureSelectionModule:
    samples_by_cluster: dict[int, NDArray]
    labels_by_cluster: dict[int, NDArray]
    min_mutual_info_percentage: float = 100.0
        
    def get_attribs_by_mutual_info(self, X_cluster, y_cluster) -> NDArray:
        n_features = X_cluster.shape[1]

        # Check if there is no need to select attributes
        if self.min_mutual_info_percentage >= 100 or len(X_cluster) == 1:
            return np.arange(n_features)

        diff_y_values = np.diff(sorted(y_cluster)) == 0

        # If no y values appear more than once, you can't calculate the mutual_info
        if np.any(diff_y_values):
            mutual_info = mutual_info_classif(X_cluster, y_cluster)
        else:
            return np.arange(n_features)

        min_mutual_info = self.min_mutual_info_percentage / 100

        if np.all(mutual_info == 0):
            return np.arange(n_features)

        # Mutual info values normalized between 0 and 1. The sum is 1.
        normalized_mutual_info = mutual_info / np.sum(mutual_info)
        
        # features sorted by mutual information.
        sorted_features = mutual_info.argsort()[::-1]

        cumsum_info = normalized_mutual_info.cumsum()

        max_attr = np.where(cumsum_info >= min_mutual_info)[0][0]
        selected_attrs = sorted_features[0 : (max_attr + 1)]
        return selected_attrs

    def select_features_by_cluster(self) -> dict[int, NDArray]:
        self.features_by_cluster = {}
        self.new_samples_by_cluster = {}

        for c in self.samples_by_cluster.keys():
            X_cluster = self.samples_by_cluster[c]
            y_cluster = self.labels_by_cluster[c]

            selected_features = self.get_attribs_by_mutual_info(X_cluster, y_cluster)

            self.new_samples_by_cluster[c] = X_cluster[:, selected_features]
            self.features_by_cluster[c] = selected_features

        return self.new_samples_by_cluster
