import dataset_loader as dl
from fuzzy_cmeans import FuzzyCMeans

if __name__ == "__main__":
    fcm = FuzzyCMeans(n_clusters=3)
    import dataset_loader as dl
    X, y = dl.select_dataset_function('german_credit')()
    clusters = fcm.fit_predict(X)
    print(clusters)
