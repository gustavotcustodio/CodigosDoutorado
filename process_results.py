import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay#, plot_confusion_matrix
from cbeg import N_FOLDS
from dataclasses import dataclass
from dataset_loader import DATASETS_INFO
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# results/iris/mutual_info_100.0/cbeg/naive_bayes_2_clusters_dbc_weighted_membership_fusion/test_summary/run_1.txt

BASE_PATH_FOLDER = "results/{dataset}/mutual_info_{mutual_info_percentage}/{algorithm}/{experiment_folder}"

CLASSIFICATION_METRICS = ["Accuracy", "Recall", "Precision", "F1"]

BASE_CLASSIFIERS = ['nb', 'svm', 'lr', 'dt', 'rf', 'gb', 'xb']

RESULTS_FILENAMES = {"cbeg": "results.csv" , "baseline": "results_baseline.csv"}


class TrainingInformation:
    def __init__(self, text_training_result):
        self.text_training_result = text_training_result
        self.labels_by_cluster = self._get_labels_by_cluster()

    def _get_labels_by_cluster(self) -> dict[int, list[int]]:
        # Labels: [0 0 0  0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0
        #  0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0]
        found_strings = re.findall(r"Labels:\s\[[0-9\s\n]+\]", self.text_training_result)
        labels_by_clusters = {}

        n_clusters = len(found_strings)

        for c in range(n_clusters):
            str_labels = found_strings[c].replace("Labels: [", "").replace("]", "")
            labels = re.split(r"[\n\s]+", str_labels)
            labels_by_clusters[c] = [int(lbl) for lbl in labels]
        return labels_by_clusters            


class ClustersClassificationResults:

    def __init__(self, classification_by_cluster: dict[str, list[float]]):
        # cluster number starts at zero
        self.n_clusters = len(classification_by_cluster["Accuracy"])
        self.extract_classification_results_by_cluster(classification_by_cluster)

    def extract_classification_results_by_cluster(self, classification_by_cluster):
        # Extract information from 
        self._classification_results = []

        for cluster in range(self.n_clusters):
            self._classification_results.append( {} )

            for metric in CLASSIFICATION_METRICS:
                self._classification_results[cluster][metric] = classification_by_cluster[metric][cluster]

    def __getitem__(self, cluster: int):
        return self._classification_results[cluster]

    def __str__(self):
        metrics_info = ""

        for cluster in range(self.n_clusters):
            metrics_info += f"""
            Cluster {cluster+1}:
                Accuracy: {self._classification_results[cluster]['Accuracy']}
                Recall: {self._classification_results[cluster]['Recall']}
                Precision: {self._classification_results[cluster]['Precision']}
                F1 Score: {self._classification_results[cluster]['F1']}
            """
        return metrics_info


class CbegClassificationRow:
    def __init__(self, experiment_variation: int,
                 experiment_params: str,
                 mutual_info_percentage: float,
                 classification_results_folds: list[dict],
                 training_information_folds: list[TrainingInformation]
                 ):

        self.experiment_variation = experiment_variation
        self.experiment_params = experiment_params
        self.mutual_info_percentage = mutual_info_percentage
        self.classification_results_fold = classification_results_folds
        self.training_information_folds = training_information_folds

        self.calc_mean_and_std_deviation()

    def set_classification_values_by_fold(self):
        self.accuracy_values = [self.classification_results_fold[fold]["Accuracy"][-1]
                                for fold in range(N_FOLDS)]
        self.recall_values = [self.classification_results_fold[fold]["Recall"][-1]
                              for fold in range(N_FOLDS)]
        self.precision_values = [self.classification_results_fold[fold]["Precision"][-1]
                                 for fold in range(N_FOLDS)]
        self.f1_values = [self.classification_results_fold[fold]["F1"][-1]
                          for fold in range(N_FOLDS)]

        if "Number clusters" in self.classification_results_fold[1]:
            self.n_clusters = [self.classification_results_fold[fold]["Number clusters"]
                               for fold in range(N_FOLDS)]

    def set_classification_results_clusters(self):
        classification_by_cluster = {}

        self.clusters_results_folds = []

        for fold in range(N_FOLDS):
            for metric in CLASSIFICATION_METRICS:
                classification_by_cluster[metric] = self.classification_results_fold[fold][metric][0:-1]

            self.clusters_results_folds.append(
                ClustersClassificationResults(classification_by_cluster)
            )

    def calc_mean_and_std_deviation(self):
        if not(hasattr(self, 'accuracy_values')):
            self.set_classification_values_by_fold()

        self.mean_accuracy = round(np.mean(self.accuracy_values), 3)
        self.mean_recall = round(np.mean(self.recall_values), 3)
        self.mean_precision = round(np.mean(self.precision_values), 3)
        self.mean_f1 = round(np.mean(self.f1_values), 3)

        self.std_accuracy = round(np.std(self.accuracy_values), 3)
        self.std_recall = round(np.std(self.recall_values), 3)
        self.std_precision = round(np.std(self.precision_values), 3)
        self.std_f1 = round(np.std(self.f1_values), 3)

        if hasattr(self, 'n_clusters'):
            self.mean_n_clusters = round(np.mean(self.n_clusters), 3)
            self.std_n_clusters = round(np.std(self.n_clusters), 3)

    def plot_accuracy_by_clusters(self, dataset: str):
        # Check if self.clusters_results_folds is set
        if not(hasattr(self, "clusters_results_folds")):
            self.set_classification_results_clusters()

        x_values, y_values = [], []
        hue_values = []

        # Get all cluster accuracy values and label distribution
        for fold in range(N_FOLDS):
            labels_by_cluster = self.training_information_folds[fold].labels_by_cluster

            for c, cluster_labels in labels_by_cluster.items():
                # Count the number of labels by cluster
                label_count = Counter(cluster_labels)
                
                count_lbls = [(count, lbl) for lbl, count in label_count.items()]
                n_majority, lbl_majority = max(count_lbls)
                n_minority, lbl_minority = min(count_lbls)

                accuracy = self.clusters_results_folds[fold][c]["Accuracy"]

                # x axis: number of majority class X minority class
                # y axis: accuracy
                x_values.append(n_majority / n_minority)
                y_values.append(accuracy)
                hue_values.append(lbl_majority)

        data = pd.DataFrame({"Majority Class / Minority Class": x_values,
                             "Accuracy": y_values, "Majority Class": hue_values})

        _, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=data, x="Majority Class / Minority Class",
                        y="Accuracy", hue="Majority Class", palette="deep")
        plt.title(dataset)
        ax.set_ylim(0, 1)
        ax.grid()

        folder_scatter = f"results/{dataset}/accuracy_by_clusters"
        figname = f"mi_{self.mutual_info_percentage}_{self.experiment_params}.png"
        os.makedirs(folder_scatter, exist_ok=True)
        full_filename = os.path.join(folder_scatter, figname)

        plt.savefig(full_filename)
        print(full_filename, "salvo com sucesso.")
        plt.clf()
        plt.close()

    def row_to_latex(self):
        return (f"{self.experiment_params.replace("_", " ").title()}" +
                f" & ${self.mean_accuracy} \\pm {self.std_accuracy}$" +
                f" & ${self.mean_recall} \\pm {self.std_recall}$" +
                f" & ${self.mean_precision} \\pm {self.std_precision}$" +
                f" & ${self.mean_f1} \\pm {self.std_f1}$" +
                f" & ${self.mean_n_clusters} \\pm {self.std_n_clusters}$" +
                f" & ${self.mutual_info_percentage}$ \\\\ \\midrule\n"
                )

    def __str__(self):
        classification_row = f"""
            Accuracy: {self.mean_accuracy} +- {self.std_accuracy}
            Recall: {self.mean_recall} +- {self.std_recall}
            Precision: {self.mean_precision} +- {self.std_precision}
            F1 Score: {self.mean_f1} +- {self.std_f1}
        """

        if hasattr(self, 'mean_n_clusters'):
            classification_row += f"""
            Number clusters: {self.mean_n_clusters} +- {self.std_n_clusters}"""

        return classification_row


@dataclass
class CbegClassificationResultsTable:
    rows_table: list[CbegClassificationRow]
    folder_experiments: str
    dataset: str

    def save_results_table_in_latex(self):
        latex_folder = f"{self.folder_experiments}/table_latex"

        os.makedirs(latex_folder, exist_ok=True)

        content = """
            \\documentclass[12pt,a4paper]{standalone}
            \\usepackage{booktabs}
            \\usepackage{caption}

            \\begin{document}
            \\begin{tabular}{lllllll}
                \\toprule
                \\textbf{Method} & \\textbf{Accuracy} & \\textbf{Recall}  & \\textbf{Precision} & \\textbf{F1-Score} & \\textbf{Clusters} & \\textbf{Mutual Information} \\\\ \\midrule\n
                """
        for row in self.rows_table:
            content += row.row_to_latex()

        content += """
            \\end{tabular}
            \\end{document}"""

        latex_filename = f"{latex_folder}/results.tex"

        with open(latex_filename, "w") as file_output:
            print(content, file=file_output)
            print(latex_filename, "saved successfully.")


    def save_results_table_in_csv(self):
        csv_folder = f"{self.folder_experiments}/table_csv"

        os.makedirs(csv_folder, exist_ok=True)

        table_results = {}

        # Create the table to be converted to csv
        table_results["Experiment Params"] = [row.experiment_params for row in self.rows_table]
        table_results["Variation"] = [row.experiment_variation for row in self.rows_table]
        table_results["Accuracy"] = [f"{row.mean_accuracy} +- {row.std_accuracy}" for row in self.rows_table]
        table_results["Recall"] = [f"{row.mean_recall} +- {row.std_recall}" for row in self.rows_table]
        table_results["Precision"] = [f"{row.mean_precision} +- {row.std_precision}" for row in self.rows_table]
        table_results["F1"] = [f"{row.mean_f1} +- {row.std_f1}" for row in self.rows_table]

        if hasattr(self.rows_table[0], "mean_n_clusters"):
            table_results["Clusters"] = [f"{row.mean_n_clusters} +- {row.std_n_clusters}" for row in self.rows_table]
            csv_filename = f"{csv_folder}/{RESULTS_FILENAMES['cbeg']}"
        else:
            table_results["Clusters"] = [f"-" for _ in self.rows_table]
            csv_filename = f"{csv_folder}/{RESULTS_FILENAMES['baseline']}"

        table_results["Mutual information"] = [f"{row.mutual_info_percentage}" for row in self.rows_table]

        df_results = pd.DataFrame.from_dict(table_results)
        df_results.to_csv(csv_filename, index=False)

        print(f"{csv_filename} saved successfully.")

    def experiment_is_part_of_ablation(self, experiment_params: str, cluster_selection_type: str) -> bool:
        # get the correct number of labels
        n_labels = int(DATASETS_INFO[self.dataset]["nlabels"])

        if str(n_labels) in experiment_params:
            return True

        if ("compare" in experiment_params and
            (f"{cluster_selection_type}_weighted" in experiment_params or
             f"{cluster_selection_type}_majority" in experiment_params)):
            return True

        return False

    def create_heatmap_ablation_study(self):
        """ Save the confusion matrix for the ablation study.
        """
        cluster_selection_variations = ["dbc", "dbc_ss", "silhouette"]

        for cluster_selection_metric in cluster_selection_variations:

            heatmaps_folder = f"{self.folder_experiments}/ablation_results"
            os.makedirs(heatmaps_folder, exist_ok=True)
            filename = os.path.join(heatmaps_folder, f"heatmap_ablation_{cluster_selection_metric}.png")

            valid_ablation_rows = [
                row for row in self.rows_table
                if self.experiment_is_part_of_ablation(row.experiment_params, cluster_selection_metric)
            ]
            # Shows the experiment variation and the percentage of mutual information
            present_info = lambda r: f"{r.experiment_variation} ({r.mutual_info_percentage} %)" \
                    if r.mutual_info_percentage < 100 else r.experiment_variation

            data = np.vstack((
                [row.mean_accuracy for row in valid_ablation_rows],
                [row.mean_recall for row in valid_ablation_rows],
                [row.mean_precision for row in valid_ablation_rows],
                [row.mean_f1 for row in valid_ablation_rows],
            )).T
            
            indexes = [present_info(row) for row in valid_ablation_rows]

            sns.heatmap(data, annot=True, cmap='Blues',
                        xticklabels=["Accuracy", "Recall", "Precision", "F1"], yticklabels=indexes)

            plt.xlabel("Classification Metric", fontdict={'weight': 'bold'})
            plt.ylabel("Experiment variation", fontdict={'weight': 'bold'})
            # Save the heat map
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()
            plt.close()
            print(filename, "salvo com sucesso.")


def get_all_classification_metrics(text_experiment_data: str) -> dict[str, list[float]]:

    # Dictionary where the values of accuracy, recall, precision and F1 are stored
    dict_classification_results = {}

    for metric in CLASSIFICATION_METRICS:
        # All patterns found in text corresponding to the searched metric
        found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", text_experiment_data)
        dict_classification_results[metric] = [float(pattern.split(": ")[1]) for pattern in found_metric_patterns]
        # Average value of the metric
        # dict_classification_metrics[f"Total {metric}"] = float(found_metric_patterns[-1].split(": ")[1])

    if "Cluster" in text_experiment_data:
        pattern_n_clusters = re.findall(r"Cluster [0-9]", text_experiment_data)[-1]
        n_clusters = int(pattern_n_clusters.split(" ")[1]) + 1

        dict_classification_results["Number clusters"] = n_clusters

    return dict_classification_results


def get_labels_and_predictions(text_file: str) -> tuple[list[int], list[int]]:
    # Extract the true labels and predicted labels
    pattern_prediction = r"Prediction: [0-9], Real label: [0-9]"

    predicted_labels = []
    true_labels = []
    
    label_prediction_patterns = re.findall(pattern_prediction, text_file)

    for found_predictions in label_prediction_patterns:
        predicted_label_str, true_label_str = found_predictions.split(", ")
        y_pred = int(predicted_label_str.split(": ")[1])
        y_true = int(true_label_str.split(": ")[1])

        predicted_labels.append(y_pred)
        true_labels.append(y_true)

    return predicted_labels, true_labels


def generate_row_test_results(results_files_contents: list[str], experiment_variation: int,
                              experiment_params: str, mutual_info_percentage: float,
                              training_information_folds: list[TrainingInformation]):

    classification_results_folds = [get_all_classification_metrics(text_experiment_data)
                                    for text_experiment_data in results_files_contents]
    # Cluster classification results
    return CbegClassificationRow(
        experiment_variation, experiment_params, mutual_info_percentage,
        classification_results_folds, training_information_folds
    )


def read_files_results(experiment_result_folder: str, stage: str = "test") -> list[str]:

    text_file_folds = []

    for fold in range(1, N_FOLDS+1):

        full_filename = f"{experiment_result_folder}/{stage}_summary/run_{fold}.txt"
        text_file = open(full_filename).read()

        text_file_folds.append(text_file)

    return text_file_folds


def compile_results(algorithm, datasets, experiments_parameters, ablation=False, latex=False):
    for dataset in datasets:

        rows_table_results = []

        print(f"\n========== {dataset} dataset ==========\n".title())

        for parameters in experiments_parameters:
            mutual_info_percentage = parameters["mutual_info"]
            experiment_folder = parameters["folder"]
            experiment_variation = parameters["variation"]

            print(f"Analyzing experiment mutual_info_{mutual_info_percentage}/{algorithm}/{experiment_folder}...")

            # Create the folders for saving the output of experiments
            experiment_result_folder = BASE_PATH_FOLDER.format(
                dataset=dataset,
                mutual_info_percentage=mutual_info_percentage,
                algorithm=algorithm,
                experiment_folder=experiment_folder,
            ) 

            filecontent_folds_test = read_files_results(experiment_result_folder, "test")
            filecontent_folds_training = read_files_results(experiment_result_folder, "training")

            training_information_by_fold = [TrainingInformation(fold_content)
                                            for fold_content in filecontent_folds_training]
            row_test_table = generate_row_test_results(
                filecontent_folds_test, experiment_variation, experiment_folder,
                mutual_info_percentage, training_information_by_fold
            )
            rows_table_results.append( row_test_table )

            print(row_test_table)

        results_folder = f"./results/{dataset}"
        results_table = CbegClassificationResultsTable(rows_table_results, results_folder, dataset)

        results_table.save_results_table_in_csv()

        if algorithm == "cbeg":
            for row in results_table.rows_table:
                row.plot_accuracy_by_clusters(dataset)

        if ablation:
            results_table.create_heatmap_ablation_study()

        if latex:
            results_table.save_results_table_in_latex()


def combine_best_cbeg_baselines(dataset: str):
    combination_results = []

    folder_results = f"results/{dataset}/table_csv"
    output_path = f"{folder_results}/combination_results.csv" 

    for algorithm, filename in RESULTS_FILENAMES.items():
        input_files_path = f"{folder_results}/{filename}"

        df_results = pd.read_csv(input_files_path)

        if algorithm == "cbeg":
            max_accuracy = df_results["Accuracy"].max()
            idx_best = np.where(max_accuracy == df_results["Accuracy"])

            combination_results.append(df_results.iloc[idx_best])
        else:
            combination_results.append(df_results)             

    df_combination = pd.concat(combination_results)
    df_combination.to_csv(output_path, index=False)
    print(f"{output_path} saved successfully.")

def main():

    datasets = ["australian_credit", "contraceptive", "german_credit", "heart", "iris", "pima", "wdbc", "wine"]

    baseline_experiments_parameters = []

    for clf in BASE_CLASSIFIERS:
        for mutual_info in [100.0, 75.0, 50.0]:
            baseline_experiments_parameters.append(
                {"mutual_info": mutual_info, "folder": clf, "variation": 0},
            )
        
        # compile_results("baselines", datasets, baseline_experiments_parameters)

    cbeg_experiments_parameters = [
        {"mutual_info": 100.0, "folder": "naive_bayes_2_clusters_silhouette_majority_voting_fusion", "variation": 0},
        {"mutual_info": 100.0, "folder": "naive_bayes_3_clusters_silhouette_majority_voting_fusion", "variation": 0},
        {"mutual_info": 100.0, "folder": "naive_bayes_compare_clusters_silhouette_majority_voting_fusion", "variation": 1},
        {"mutual_info": 100.0, "folder": "naive_bayes_compare_clusters_dbc_majority_voting_fusion", "variation": 1},
        {"mutual_info": 100.0, "folder": "naive_bayes_compare_clusters_dbc_ss_majority_voting_fusion", "variation": 1},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_silhouette_majority_voting_fusion", "variation": 2},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_dbc_majority_voting_fusion", "variation": 2},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_dbc_ss_majority_voting_fusion", "variation": 2},
        {"mutual_info": 75.0, "folder": "naive_bayes_2_clusters_silhouette_majority_voting_fusion", "variation": 3},
        {"mutual_info": 75.0, "folder": "naive_bayes_3_clusters_silhouette_majority_voting_fusion", "variation": 3},
        {"mutual_info": 50.0, "folder": "naive_bayes_2_clusters_silhouette_majority_voting_fusion", "variation": 3},
        {"mutual_info": 50.0, "folder": "naive_bayes_3_clusters_silhouette_majority_voting_fusion", "variation": 3},
        {"mutual_info": 100.0, "folder": "naive_bayes_2_clusters_silhouette_weighted_membership_fusion", "variation": 4},
        {"mutual_info": 100.0, "folder": "naive_bayes_3_clusters_silhouette_weighted_membership_fusion", "variation": 4},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_dbc_ss_weighted_membership_fusion", "variation": 124},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_dbc_weighted_membership_fusion", "variation": 124},
        {"mutual_info": 100.0, "folder": "classifier_selection_compare_clusters_silhouette_weighted_membership_fusion", "variation": 124},
        {"mutual_info": 75.0, "folder": "classifier_selection_compare_clusters_dbc_ss_weighted_membership_fusion", "variation": 1234},
        {"mutual_info": 75.0, "folder": "classifier_selection_compare_clusters_dbc_weighted_membership_fusion", "variation": 1234},
        {"mutual_info": 75.0, "folder": "classifier_selection_compare_clusters_silhouette_weighted_membership_fusion", "variation": 1234},
        {"mutual_info": 50.0, "folder": "classifier_selection_compare_clusters_dbc_ss_weighted_membership_fusion", "variation": 1234},
        {"mutual_info": 50.0, "folder": "classifier_selection_compare_clusters_dbc_weighted_membership_fusion", "variation": 1234},
        {"mutual_info": 50.0, "folder": "classifier_selection_compare_clusters_silhouette_weighted_membership_fusion", "variation": 1234},
    ]
    compile_results("cbeg", datasets, cbeg_experiments_parameters, ablation=True, latex=True)
    
    for dataset in datasets:
        combine_best_cbeg_baselines(dataset)

if __name__ == "__main__":
    main()
