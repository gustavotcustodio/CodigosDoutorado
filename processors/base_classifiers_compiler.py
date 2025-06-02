import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from processors.data_reader import CLASSIFICATION_METRICS, CLASSIFIERS_FULLNAMES


class BaseClassifierResult:
    def __init__(self, test_info_folds: list[str], classifier_name: str,
                 mutual_info_percentage: float=100.0):
        self.classifier_name = classifier_name
        self.mutual_info_percentage = mutual_info_percentage

        self.classification_results_folds = []

        for content_test in test_info_folds:
            classification_results_fold = self.get_classification_metrics(content_test)

            self.classification_results_folds.append(
                self.get_general_classification_results(classification_results_fold)
            )
        self.calc_mean_and_std_metrics()

    def get_general_classification_results(
            self, classification_results: dict[str, list[float]]
        ) -> dict[str, float]:

        classification_metrics = {}

        for metric in CLASSIFICATION_METRICS:
            classification_metrics[metric] = classification_results[metric][-1]

        return classification_metrics

    def get_metric_value(self, metric):
        if metric == "Accuracy":
            return self.mean_accuracy

        elif metric == "Recall":
            return self.mean_recall

        elif metric == "Precision":
            return self.mean_precision

        elif metric == "F1":
            return self.mean_f1
        
        else: # metric == "AUC":
            return self.mean_auc

    def calc_mean_and_std_metrics(self):
        self.mean_accuracy  = np.mean([metrics["Accuracy"] for metrics in self.classification_results_folds])
        self.mean_recall    = np.mean([metrics["Recall"] for metrics in self.classification_results_folds])
        self.mean_precision = np.mean([metrics["Precision"] for metrics in self.classification_results_folds])
        self.mean_f1        = np.mean([metrics["F1"] for metrics in self.classification_results_folds])
        self.mean_auc       = np.mean([metrics["AUC"] for metrics in self.classification_results_folds])

        self.std_accuracy   = np.std([metrics["Accuracy"] for metrics in self.classification_results_folds])
        self.std_recall     = np.std([metrics["Recall"] for metrics in self.classification_results_folds])
        self.std_precision  = np.std([metrics["Precision"] for metrics in self.classification_results_folds])
        self.std_f1         = np.std([metrics["F1"] for metrics in self.classification_results_folds])
        self.std_auc        = np.std([metrics["AUC"] for metrics in self.classification_results_folds])

    def get_classification_metrics(self, content_test: str):
        # Dictionary where the values of accuracy, recall, precision F1 and AUC are stored
        dict_classification_results = {}

        for metric in CLASSIFICATION_METRICS:
            # All patterns found in text corresponding to the searched metric
            found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", content_test)

            dict_classification_results[metric] = [
                float(pattern.split(": ")[1]) for pattern in found_metric_patterns]

        return dict_classification_results

    def __repr__(self):
        return f"""
            Accuracy: {self.mean_accuracy} +- {self.std_accuracy}
            Recall: {self.mean_recall} +- {self.std_recall}
            Precision: {self.mean_precision} +- {self.std_precision}
            F1 Score: {self.mean_f1} +- {self.std_f1}
            AUC Score: {self.mean_auc} +- {self.std_auc}
        """

@dataclass
class BaseClassifiersCompiler:

    baseline_results: list[BaseClassifierResult]
    dataset: str

    def plot_classification_heatmap(self):
        """ Save the confusion matrix for the ablation study.
        """
        heatmaps_folder = f"results/{self.dataset}/ablation_results"

        os.makedirs(heatmaps_folder, exist_ok=True)

        mutual_info_columns = {100.0: 0, 75.0: 1, 50.0: 2}
            
        classifiers_rows = {
            name_classifier: row
            for row, name_classifier in enumerate(list(CLASSIFIERS_FULLNAMES.values()))
        }
        num_rows = len(CLASSIFIERS_FULLNAMES.keys())
        num_cols = len(mutual_info_columns.keys())

        for metric in CLASSIFICATION_METRICS:
            data_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)

            for baseline_result in self.baseline_results:
                mutual_info = baseline_result.mutual_info_percentage
                base_classifier = baseline_result.classifier_name

                row = classifiers_rows[base_classifier]
                col = mutual_info_columns[mutual_info]

                data_matrix[row, col] = baseline_result.get_metric_value(metric)
            
            indexes = [classifier for classifier in classifiers_rows]
            columns = [f"{mutual_info} %" for mutual_info in mutual_info_columns]

            filename = os.path.join( heatmaps_folder, f"{metric}_heatmap_baselines.png")
            self.save_heatmap(data_matrix, columns, indexes, filename)


    def save_heatmap(self, data, columns, indexes, filename):
        sns.heatmap(data, annot=True, cmap='Blues',
                    xticklabels=columns, yticklabels=indexes, vmin=0, vmax=1)

        plt.xlabel("Mutual Information Percentage", fontdict={'weight': 'bold'})
        plt.ylabel("Base Classifier", fontdict={'weight': 'bold'})
        # Save the heat map
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()

        print(f"{filename} saved successfully.")
