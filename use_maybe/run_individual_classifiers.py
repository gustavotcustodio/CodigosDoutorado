from sklearn.naive_bayes import GaussianNB
from dataset_loader import DATASETS_INFO, select_dataset_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

func = select_dataset_function("german_credit")
X, y = func()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

gnb1 = GaussianNB()
gnb1.fit(X_train, y_train)
y_pred1 = gnb1.predict(X_test)

print(classification_report(y_test, y_pred1, zero_division=0.0))
