import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif
from dataclasses import dataclass

@dataclass
class FeatureSelectionModule:
    samples_by_cluster: dict[int, NDArray]
    labels_by_cluster: dict[int, NDArray]
    min_mutual_info_percentage: float = 100.0
        
    def get_attribs_by_mutual_info(self, X_cluster, y_cluster):
        # Check
        if self.min_mutual_info_percentage >= 100 or len(X_cluster) == 1:
            return [i for i in range(X_cluster.shape[1])]

        diff_y_values = np.diff(sorted(y_cluster)) == 0

        # If no y values appear more than once, you can't calculate the mutual_info
        if np.any(diff_y_values):
            mutual_info = mutual_info_classif(X_cluster, y_cluster)
        else:
            return [i for i in range(X_cluster.shape[1])]

        min_mutual_info = self.min_mutual_info_percentage / 100

        if np.all(mutual_info == 0):
            return [i for i in range(X_cluster.shape[1])]

        # Mutual info values normalized between 0 and 1. The sum is 1.
        normalized_mutual_info = mutual_info / np.sum(mutual_info)
        
        # Attributes sorted by mutual information.
        sorted_attributes = mutual_info.argsort()[::-1]

        cumsum_info = normalized_mutual_info.cumsum()

        max_attr = np.where(cumsum_info >= min_mutual_info)[0][0]
        selected_attrs = sorted_attributes[0 : (max_attr + 1)]
        return selected_attrs

    def select_attributes_by_cluster(self) -> dict[int, NDArray]:
        self.attributes_by_cluster = {}
        self.new_samples_by_cluster = {}

        for c in self.samples_by_cluster.keys():
            X_cluster = self.samples_by_cluster[c]
            y_cluster = self.labels_by_cluster[c]

            selected_attributes = self.get_attribs_by_mutual_info(X_cluster, y_cluster)

            self.new_samples_by_cluster[c] = X_cluster[:, selected_attributes]
            self.attributes_by_cluster[c] = selected_attributes

        return self.new_samples_by_cluster