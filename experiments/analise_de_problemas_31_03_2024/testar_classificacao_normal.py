from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import dataset_loader


X, y = dataset_loader.read_pima_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

clf = XGBClassifier()
# clf = SVC()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
