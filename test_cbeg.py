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
                X_cluster, y_cluster, classification_metrics)
        assert isinstance(best_classifier, DummyClassifier)

    # TODO test crossval

if __name__ == '__main__':
    unittest.main()
