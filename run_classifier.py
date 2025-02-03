import argparse
from sklearn.base import BaseEstimator
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
from feature_selection import FeatureSelectionModule

N_FOLDS = 10

CLASSIFIERS = {'nb': GaussianNB,
               'svm': SVC,
               'lr': LogisticRegression,
               'dt': DecisionTreeClassifier,
               'rf': RandomForestClassifier,
               'gb': GradientBoostingClassifier,
               'xb': XGBClassifier,
               #'adaboost': AdaBoostClassifier,
               }

def select_classifier(name_classifier: str) -> "Classifier":
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

    parser.add_argument("-d", "--dataset", help = "Dataset used.", required=True)
    parser.add_argument("-c", "--classifier", help = "Selected classifier.", required=True)
    parser.add_argument("-m", "--min_mutual_info_percentage", help = "")

    # Read arguments from command line
    args = parser.parse_args()

    if not args.min_mutual_info_percentage:
        args.min_mutual_info_percentage = 100.0
    else:
        args.min_mutual_info_percentage = float(args.min_mutual_info_percentage)

    for fold in range(1, N_FOLDS+1):
        X, y = dataset_loader.select_dataset_function(args.dataset)()
        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        X_train, X_val = perform_feature_selection(
                X_train, y_train, X_val, args.min_mutual_info_percentage)

        clf = select_classifier(args.classifier)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        # prediction_results = PredictionResults(y_pred, voting_weights, y_pred_by_clusters, y_val)

        print(f"{args.classifier.capitalize()}", classification_report(y_pred, y_val, zero_division=0.0))

        # save_data(args, cbeg, prediction_results, fold)


if __name__ == "__main__":
    main()
