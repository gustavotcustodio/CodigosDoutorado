import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    X = np.loadtxt("./glass/glass.data", delimiter=',')

    np.random.shuffle(X)
    y = X[:, -1]

    X = (X - X.min()) / (X.max() - X.min())
    X = X[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    # distance_matrix = distance.pdist(X)

    model = SVC(C=1.0, kernel='rbf')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(sum(y_pred == y_test) / len(y_test))

