# import kmeans

class Cbeg:
    def __init__(self, n_clusters=None, method_n_clusters_selection=None, base_classifier=None):
        self.n_clusters = n_clusters
        self.base_classifier = base_classifier
        self.method_n_clusters_selection = method_n_clusters_selection

    def fit(self, X_train, y_train):
        # Perform clustering
        pass
        # Select the base classifier for each cluster
        

    def predict(self, X_test):
        n_clusters = self.centroids.shape[0]
        u = calc_membership_values(X_test, self.centroids[possible_clusters])
        # predictions = np.array([ensemble[c].predict(X_test)
        #                         for c in range(possible_clusters.shape[0])]).T
        predictions = []

        for c in range(possible_clusters.shape[0]):
            X_cluster = X_test[:, attribs_by_cluster[c]]
            y_pred = ensemble[c].predict(X_cluster)
            predictions.append(y_pred)

        predictions = np.array(predictions).astype(int)

        if len(predictions.shape) > 2:
            predictions = predictions.argmax(axis=2)

        predictions = predictions.T

        n_labels = int(predictions.max()) + 1
        # probabilities = np.sum(predictions * u, axis=1)
        voting_labels = np.zeros((u.shape[0], n_labels))

        for c in range(n_clusters):
            labels = predictions[:, c]
            for i, lbl in enumerate(labels):
                voting_labels[i, lbl] += u[i, c]

        y_pred_votation = np.argmax(voting_labels, axis=1)

        return y_pred_votation, predictions, u
