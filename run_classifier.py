import os
import sys
import argparse
from numpy import savez_compressed
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from cbeg import N_FOLDS
import dataset_loader
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from dataset_loader import normalize_data

N_FOLDS = 10

CLASSIFIERS = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "gboost": GradientBoostingClassifier,
    "svm": SVC,
}


def select_classifier(name_classifier: str) -> BaseEstimator:
    base_classifier = CLASSIFIERS[name_classifier]()

    return base_classifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", help = "Dataset used.", required=True)
    parser.add_argument("-c", "--classifier", help = "Selected classifier.", required=True)

    # Read arguments from command line
    args = parser.parse_args()

    for fold in range(1, N_FOLDS+1):
        X, y = dataset_loader.select_dataset_function(args.dataset)()
        # Break dataset in training and validation
        X_train, X_val, y_train, y_val = dataset_loader.split_training_test(X, y, fold)
        X_train, X_val = normalize_data(X_train, X_val)

        clf = select_classifier(args.classifier)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        # prediction_results = PredictionResults(y_pred, voting_weights, y_pred_by_clusters, y_val)

        print(f"{args.classifier.capitalize()}", classification_report(y_pred, y_val, zero_division=0.0))

        # save_data(args, cbeg, prediction_results, fold)


if __name__ == "__main__":
    main()
