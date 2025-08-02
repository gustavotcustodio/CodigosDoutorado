import os
import numpy as np
import argparse
from sklearn.metrics import classification_report
import dataset_loader
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from dataset_loader import normalize_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from feature_selection import FeatureSelectionModule
from logger import Logger, PredictionResults
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

N_FOLDS = 10
RESULTS_FOLDER = "results"

CLASSIFIERS = {'nb': GaussianNB,
               'svm': SVC,
               'lr': LogisticRegression,
               'knn5': KNeighborsClassifier,
               'knn7': KNeighborsClassifier,
               'dt': DecisionTreeClassifier,
               'rf': RandomForestClassifier,
               'gb': GradientBoostingClassifier,
               'xb': XGBClassifier,
               #'adaboost': AdaBoostClassifier,
               }

# def save_data(y_pred, y_true, args, fold):
#     """ Save the results """
#     
#     n_labels = np.unique(y_true).shape[0]
#     multiclass = n_labels > 2
# 
#     folder_name = os.path.join(
#         RESULTS_FOLDER, args.dataset, f'mutual_info_{args.min_mutual_info_percentage}',
#         "baselines", args.classifier, "test_summary"
#     )
#     os.makedirs(folder_name, exist_ok=True)
# 
#     fullpath = os.path.join(folder_name, f"run_{fold}.txt")
#     file_output = open(fullpath, "w")
# 
#     # If it is a multiclass problem, we use the weighted avg. to calculate metrics.
#     avg_type = "weighted avg" if multiclass else "1"
# 
#     clf_report = classification_report(y_pred, y_true, output_dict=True, zero_division=0.0)
#     print(f"Accuracy: {clf_report['accuracy']}", file = file_output)
#     print(f"Recall: {clf_report[avg_type]['recall']}", file = file_output)
#     print(f"Precision: {clf_report[avg_type]['precision']}", file = file_output)
#     print(f"F1: {clf_report[avg_type]['f1-score']}\n", file = file_output)
# 
#     file_output.close()
# 
#     print(fullpath, "saved successfully.")


def select_classifier(name_classifier: str) -> "Classifier":
    if name_classifier ==  'svm':
        base_classifier = CLASSIFIERS[name_classifier](probability=True)
    else:
        base_classifier = CLASSIFIERS[name_classifier]()

    return base_classifier


def perform_feature_selection(X_train, y_train, X_val, min_mutual_info_percentage):
    samples_by_cluster = {0: X_train}
    labels_by_cluster = {0: y_train}

    feat_selection_module = FeatureSelectionModule(
        samples_by_cluster, labels_by_cluster, min_mutual_info_percentage
    )
    feat_selection_module.select_features_by_cluster()[0]

    selected_features = feat_selection_module.features_by_cluster[0]
    return X_train[:, selected_features], X_val[:, selected_features]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str,
                        help="Dataset used.", required=True)
    parser.add_argument("-c", "--classifier", type=str,
                        help="Selected classifier.", required=True)
    parser.add_argument("-m", "--min_mutual_info_percentage",
                        type=float, default=100.0, help = "")

    # Read arguments from command line
    args = parser.parse_args()

    accuracy_values, f1_values = [], []

    for fold in range(1, N_FOLDS+1):
        X, y = dataset_loader.select_dataset_function(args.dataset)()
        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = \
                dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        X_train, X_val = perform_feature_selection(
                X_train, y_train, X_val, args.min_mutual_info_percentage)

        clf = select_classifier(args.classifier)

        n_classes = len(np.unique(y))

        if n_classes > 2 and not isinstance(clf, XGBClassifier) \
                and not isinstance(clf, DecisionTreeClassifier):
            clf = OneVsRestClassifier(clf)
        clf.fit(X_train, y_train)

        y_score = clf.predict_proba(X_val)
        y_pred = np.argmax(y_score, axis=1)

        prediction_results = PredictionResults(
                y_pred, y_val, y_score = y_score)

        print(f"{args.classifier.capitalize()}",
              classification_report(y_pred, y_val, zero_division=0.0))

        # save_data(y_pred, y_val, args, fold)
        log = Logger(clf, args.dataset, prediction_results)
        log.save_data_fold_baseline(fold, args.classifier,
                                    args.min_mutual_info_percentage)

        average = "weighted" if len(np.unique(y_val)) > 2 else "binary"

        f1_values.append(f1_score(y_val, y_pred, average=average))
        accuracy_values.append(accuracy_score(y_val, y_pred))

    print("Accuracy:", np.mean(accuracy_values))
    print("F1:", np.mean(f1_values))


if __name__ == "__main__":
    main()
