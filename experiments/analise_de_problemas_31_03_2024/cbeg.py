# import kmeans

class Cbeg:
    def __init__(self, n_clusters=None, method_n_clusters_selection=None, base_classifier=None):
        self.n_clusters = n_clusters
        self.base_classifier = base_classifier
        self.method_n_clusters_selection = method_n_clusters_selection

    def fit(self, X_train, y_train):
        pass

    def predict(self):
        pass
