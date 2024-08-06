import re
import sys
import numpy as np
from scipy.sparse import data
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import dataset_loader
from feature_selector import get_attribs_by_mutual_info


def select_classifier(classifer_name):
    if classifer_name == 'xgboost':
        return XGBClassifier()
    elif classifer_name == 'gradient_boosting':
        return GradientBoostingClassifier()
    elif classifer_name == 'random_forest':
        return RandomForestClassifier()
    else:
        return SVC()


def calcular_e_printar_metricas(classifier_name, run, dataset, mutual_info_percent):
    model = select_classifier(classifier_name)
    read_function = dataset_loader.select_dataset_function(dataset)
    X, y = read_function()

    X_train, X_test, y_train, y_test = dataset_loader.split_training_test(X, y, run)

    if mutual_info_percent < 100:
        selected_attrs = get_attribs_by_mutual_info(X_train, y_train, mutual_info_percent)
        X_train = X_train[:, selected_attrs]
        X_test = X_test[:, selected_attrs]

        print(len(selected_attrs), "atributos selecionados.")
    else:
        print("Todos os", X.shape[1], "atributos selecionados.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_value = accuracy_score(y_test, y_pred)

    n_classes = len(np.unique(y_train))

    if n_classes > 2:
        avg_type = "weighted"
    else:
        avg_type = "binary"

    recall_value = recall_score(y_test, y_pred, average=avg_type, zero_division=0.0)
    precision_value = precision_score(y_test, y_pred, average=avg_type, zero_division=0.0 )
    f1_value = f1_score(y_test, y_pred, average=avg_type )

    print("Acurácia total:", accuracy_value)
    print("Recall total:", recall_value)
    print("Precisão total:", precision_value)
    print("F1-Score:", f1_value)


def main():
    # ["10_runs"]
    # ["xgboost", "gradient_boosting", "random_forest", "svm"]
    # ["german_credit", "australian_credit", "heart", "iris", "pima", "wdbc"]

    classifier_name = sys.argv[1]
    dataset = sys.argv[2]
    run = int(sys.argv[3])

    if len(sys.argv) > 4:
        mutual_info_percent = float(sys.argv[4])
    else:
        mutual_info_percent = 100

    calcular_e_printar_metricas(classifier_name, run, dataset, mutual_info_percent)


if __name__ == "__main__":
    main()
