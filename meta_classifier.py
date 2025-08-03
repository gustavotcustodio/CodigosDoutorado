import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC

class MetaClassifier:
    def __init__(self, y_prob_by_clusters, y):
        self.X = np.hstack(y_prob_by_clusters)
        self.X = np.round(self.X, 10)
        self.y = y
        self.meta_clf = SVC(probability=True)

    def train(self) -> None:
        X, self.cols_kept = self.remove_single_val_cols()
        self.meta_clf.fit(X, self.y)

    def predict_proba(self, y_prob_by_clusters: list) -> NDArray:
        X_test = np.hstack(y_prob_by_clusters)
        X = X_test[:, self.cols_kept]
        y_prob = self.meta_clf.predict_proba(X)
        return y_prob

    def remove_single_val_cols(self):
        mask_0 = np.all(self.X == 0, axis=0)
        mask_1 = np.all(self.X == 1, axis=0)
        mask = ~(mask_0 + mask_1)

        X_no_zero_cols = self.X[:, mask]
        cols_kept = np.where(mask == 1)[0]

        return X_no_zero_cols, cols_kept
