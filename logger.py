import os
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from cbeg import CBEG
from sklearn.metrics import classification_report
from ciel_optimizer import CielOptimizer

@dataclass
class PredictionResults:
    y_pred: NDArray
    y_val: NDArray
    # y_score: NDArray
    voting_weights: NDArray
    y_pred_by_clusters: NDArray


class Logger:

    def __init__(self, classifier, dataset: str, prediction_results: PredictionResults):
        self.classifier = classifier
        self.dataset = dataset
        self.prediction_results = prediction_results

    def print_classification_report(self, y_pred: NDArray, y_val: NDArray, file_output: 'File',
                                    multiclass: bool = False) -> None:
        # If it is a multiclass problem, we use the weighted avg. to calculate metrics.
        avg_type = "weighted avg" if multiclass else "1"

        clf_report = classification_report(y_pred, y_val, output_dict=True, zero_division=0.0)
        print(f"Accuracy: {clf_report['accuracy']}", file = file_output)
        print(f"Recall: {clf_report[avg_type]['recall']}", file = file_output)
        print(f"Precision: {clf_report[avg_type]['precision']}", file = file_output)
        print(f"F1: {clf_report[avg_type]['f1-score']}\n", file = file_output)

    def save_training_data(self, filename: str, folder: str):
        fullpath = os.path.join(folder, filename)

        with open(fullpath, "w") as file_output:

            if isinstance(self.classifier, CielOptimizer):
                print("============== Classifiers Parameters ==============", file=file_output)
                print(f"{self.classifier.classifiers_params}\n", file=file_output)

            # Save clustering information 
            self.save_clustering_metric(file_output)

            print(f"Base classifier: {self.classifier.base_classifier}", file=file_output)

            for c in range(self.classifier.n_clusters):
                labels_cluster = self.classifier.labels_by_cluster[c]

                print(f"========== Cluster {c} ==========\n", file=file_output)

                print(f"Labels: {labels_cluster}\n", file=file_output)

    def save_clustering_metric(self, file_output):
        # Check if there is an attribute best intra inter clustering distance
        # If there is this is the supervised clustering algorithm
        if hasattr(self.classifier, 'best_intra_inter_dist'):
            intra_inter_dist = self.classifier.best_intra_inter_dist
            print(f"Clustering evaluation metric: intra-inter cluster distance", file=file_output)
            print(f"Clustering evaluation value: {intra_inter_dist}\n", file=file_output)

        # This is for the ciel algorithm
        else:
            # Save clustering metrics for the CIEL classifier
            clustering_metrics = self.classifier.best_clustering_metrics

            # Selected clustering algorithm
            print(f"Optimal clusterer: {self.classifier.optimal_clusterer}", file=file_output)

            # Run through all metrics and print them on a file
            print("\nExternal clustering metrics:", file=file_output)
            for metric, value in clustering_metrics["external"].items():
                print(f"{metric}: {value}", file=file_output)

            print("\nInternal clustering metrics:", file=file_output)
            for metric, value in clustering_metrics["internal"].items():
                print(f"{metric}: {value}", file=file_output)
            print("", file=file_output)


    def save_test_data(self, filename: str, folder: str):
        y_pred = self.prediction_results.y_pred
        y_val = self.prediction_results.y_val

        n_samples = y_val.shape[0]
        n_labels = self.classifier.n_classes

        multiclass = n_labels > 2 # Used to calculate precision, recall and F1 for multiclass problems

        fullpath = os.path.join(folder, filename)

        file_output = open(fullpath, "w")

        print("------------------------------------\n" +
              "------ Classification results ------\n" +
              "------------------------------------\n", file=file_output)
        print(f"Base classifier: {self.classifier.base_classifier}", file=file_output)

        if hasattr(self.classifier, 'M'):
            print(f"M (closest neighbors): {self.classifier.M}", file=file_output)

        for c in range(int(self.classifier.n_clusters)):
            y_pred_cluster = self.prediction_results.y_pred_by_clusters[:, c]
            print(f"====== Cluster {c} ======", file=file_output)
            self.print_classification_report(y_pred_cluster, y_val, file_output, multiclass)

        print(f"====== Total ======", file=file_output)
        self.print_classification_report(y_pred, y_val, file_output, multiclass)

        self.save_clustering_metric(file_output)

        print('========= Predictions by sample =========\n', file=file_output)
        for i in range(n_samples):
            row = (
             f"Prediction: {self.prediction_results.y_pred[i]}, " +
             f"Real label: {self.prediction_results.y_val[i]}, " +
             f"Votes by cluster: {self.prediction_results.y_pred_by_clusters[i]}, "
             f"Weights: {np.round(self.prediction_results.voting_weights[i], 2)}"
             )
            print(row, file=file_output)

    def save_data_fold_supervised_clustering(self, fold: int) -> None:
        """ Save training and test data.
        """
        specific_experiment = f'supervised_clustering_base_classifier_{self.classifier.base_classifier}'

        for mutual_info in [50.0, 75.0, 100.0]:
            output_folder_prefix = os.path.join('results', self.dataset,
                                                f'mutual_info_{mutual_info}', 'supervised_clustering')
            output_folder = os.path.join(output_folder_prefix, specific_experiment)
            self.save_data_fold(fold, output_folder)


    def save_data_fold_ciel(self, fold: int) -> None:
        """ Save training and test data.
        """
        for mutual_info in [50.0, 75.0, 100.0]:
            output_folder = os.path.join('results', self.dataset,
                                         f'mutual_info_{mutual_info}', 'ciel')
            self.save_data_fold(fold, output_folder)


    def save_data_fold(self, fold: int, output_folder: str):

        folder_training = os.path.join(output_folder, 'training_summary')

        filename = f'run_{fold}.txt'

        # Save the data: clusters, labels, selected features, etc
        os.makedirs(folder_training, exist_ok=True)
        self.save_training_data(filename, folder_training)

        print('Training data saved successfully.')

        folder_test = os.path.join(output_folder, 'test_summary')
        os.makedirs(folder_test, exist_ok=True)
        self.save_test_data(filename, folder_test)

        print('Test data saved successfully.')
        print(50 * '-',"\n")
