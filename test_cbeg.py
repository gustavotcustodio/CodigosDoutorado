import unittest
from deslib.base import KNeighborsClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from cbeg import CBEG


class Testing(unittest.TestCase):
    def setUp(self):
        self.cbeg = CBEG()

    def test_choose_best_classifier(self):
        X_cluster = np.array([[0.2, 0.9],
                              [0.5, 0.4],
                              [0.1, 0.1]])
        y_cluster = np.array([1, 1, 1])
        best_classifier = self.cbeg.choose_best_classifier(
                X_cluster, y_cluster, ["roc_auc"])
        assert isinstance(best_classifier, DummyClassifier)

    def test_dummy_classifier(self):
        X_train = np.array([[0.2, 0.9], [0.5, 0.4], [0.1, 0.1]])
        y_train = np.array([1, 0, 0])

        X_test = np.array([[0, 3], [5, 6], [9, 9], [1, 1]])
        dc = DummyClassifier(strategy="most_frequent")
        dc.fit(X_train, y_train)
        y_pred = dc.predict(X_test).astype(int)
        np.testing.assert_array_equal(y_pred, [0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()
