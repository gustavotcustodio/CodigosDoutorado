import os
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Mapping, Optional, Callable
from cbeg import CBEG
from sklearn.metrics import classification_report

@dataclass
class PredictionResults:
    y_pred: NDArray
    y_val: NDArray
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
        # Possible clusters
        clusters = self.classifier.labels_by_cluster.keys()

        fullpath = os.path.join(folder, filename)

        file_output = open(fullpath, "w")

        intra_inter_dist = self.classifier.best_intra_inter_dist
        print(f"Clustering evaluation metric: intra-inter cluster distance", file=file_output)
        print(f"Clustering evaluation value: {intra_inter_dist}\n", file=file_output)

        print(f"Base classifier: {self.classifier.base_classifier}", file=file_output)

        for c in sorted(clusters):
            labels_cluster = self.classifier.labels_by_cluster[c]

            print(f"========== Cluster {c} ==========\n", file=file_output)

            print(f"Labels: {labels_cluster}\n", file=file_output)

        file_output.close()

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
              "-----------------------------------o-\n", file=file_output)
        print(f"Base classifier: {self.classifier.base_classifier}", file=file_output)
        print(f"M (closest neighbors): {self.classifier.M}", file=file_output)

        for c in range(int(self.classifier.n_clusters)):
            y_pred_cluster = self.prediction_results.y_pred_by_clusters[:, c]
            print(f"====== Cluster {c} ======", file=file_output)
            self.print_classification_report(y_pred_cluster, y_val, file_output, multiclass)

        print(f"====== Total ======", file=file_output)
        self.print_classification_report(y_pred, y_val, file_output, multiclass)

        intra_inter_dist = self.classifier.best_intra_inter_dist
        print(f"Clustering evaluation metric: intra-inter cluster distance", file=file_output)
        print(f"Clustering evaluation value: {intra_inter_dist}\n", file=file_output)

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
        folder_name_suffix = f'supervided_clustering_base_classifier_{self.classifier.base_classifier}'

        folder_name_prefix = os.path.join('results', self.dataset, 'mutual_info_100.0', 'supervised_clustering')
        filename = f'run_{fold}.txt'

        folder_training = os.path.join(folder_name_prefix, folder_name_suffix, 'training_summary')
        
        # Save the data: clusters, labels, selected features, etc
        os.makedirs(folder_training, exist_ok=True)
        self.save_training_data(filename, folder_training)

        print('Training data saved successfully.')

        folder_test = os.path.join(folder_name_prefix, folder_name_suffix, 'test_summary')
        os.makedirs(folder_test, exist_ok=True)
        self.save_test_data(filename, folder_test)
        
        print('Test data saved successfully.')
        print(50 * '-',"\n")
