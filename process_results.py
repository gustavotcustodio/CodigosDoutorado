import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay#, plot_confusion_matrix
from cbeg import N_FOLDS
from dataclasses import dataclass
import matplotlib.pyplot as plt

# results/iris/mutual_info_100.0/cbeg/naive_bayes_2_clusters_dbc_weighted_membership_fusion/test_summary/run_1.txt

BASE_PATH_FOLDER = "results/{dataset}/mutual_info_{mutual_info_percentage}/{algorithm}/{experiment_folder}"

CLASSIFICATION_METRICS = ["Accuracy", "Recall", "Precision", "F1"]

class ResultsRow:

    def __init__(self, experiment_variation: int,
                 experiment_params: str,
                 mutual_info_percentage: float,
                 classification_results_fold: list[dict]):
        # A different dict containing the classification results for each fold.

        self.experiment_variation = experiment_variation
        self.experiment_params = experiment_params
        self.mutual_info_percentage = mutual_info_percentage
        self.classification_results_fold = classification_results_fold

        self.calc_mean_and_std_deviation()

    def get_classification_values(self):
        
        self.accuracy_values = [self.classification_results_fold[fold]["Accuracy"][-1]
                                for fold in range(N_FOLDS)]
        self.recall_values = [self.classification_results_fold[fold]["Recall"][-1]
                              for fold in range(N_FOLDS)]
        self.precision_values = [self.classification_results_fold[fold]["Precision"][-1]
                                 for fold in range(N_FOLDS)]
        self.f1_values = [self.classification_results_fold[fold]["F1"][-1]
                          for fold in range(N_FOLDS)]
        self.n_clusters = [self.classification_results_fold[fold]["Number clusters"]
                          for fold in range(N_FOLDS)]

    def calc_mean_and_std_deviation(self):
        self.get_classification_values()

        self.mean_accuracy = round(np.mean(self.accuracy_values), 3)
        self.mean_recall = round(np.mean(self.recall_values), 3)
        self.mean_precision = round(np.mean(self.precision_values), 3)
        self.mean_f1 = round(np.mean(self.f1_values), 3)
        self.mean_n_clusters = round(np.mean(self.n_clusters), 3)

        self.std_accuracy = round(np.std(self.accuracy_values), 3)
        self.std_recall = round(np.std(self.recall_values), 3)
        self.std_precision = round(np.std(self.precision_values), 3)
        self.std_f1 = round(np.std(self.f1_values), 3)
        self.std_n_clusters = round(np.std(self.n_clusters), 3)

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
        return f"""
            Accuracy: {self.mean_accuracy} +- {self.std_accuracy}
            Recall: {self.mean_recall} +- {self.std_recall}
            Precision: {self.mean_precision} +- {self.std_precision}
            F1 Score: {self.mean_f1} +- {self.std_f1}
            Number clusters: {self.mean_n_clusters} +- {self.std_n_clusters}
        """

@dataclass
class ResultsTable:
    rows_table: list[ResultsRow] 
    folder_experiments: str

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
        table_results["Clusters"] = [f"{row.mean_n_clusters} +- {row.std_n_clusters}" for row in self.rows_table]
        table_results["Mutual information"] = [f"{row.mutual_info_percentage}" for row in self.rows_table]

        csv_filename = f"{csv_folder}/results.csv"

        df_results = pd.DataFrame.from_dict(table_results)

        df_results.to_csv(csv_filename, index=False)

        print(f"{csv_filename} saved successfully.")


def get_all_classification_metrics(text_file: str) -> dict[str, list[float]]:

    # Dictionary where the values of accuracy, recall, precision and F1 are stored
    dict_classification_results = {}

    for metric in CLASSIFICATION_METRICS:
        # All patterns found in text corresponding to the searched metric
        found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", text_file)
        dict_classification_results[metric] = [float(pattern.split(": ")[1]) for pattern in found_metric_patterns]
        # Average value of the metric
        # dict_classification_metrics[f"Total {metric}"] = float(found_metric_patterns[-1].split(": ")[1])

    pattern_n_clusters = re.findall(r"Cluster [0-9]", text_file)[-1]
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


def save_confusion_matrix(y_true, y_pred, filename, show=False):
    # Save the confusion matrix for the fold
    cm = confusion_matrix(y_true, y_pred)
    cm_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(y_true)
    )

    cm_disp.plot(cmap=plt.cm.Blues)
    if show:
        plt.show()

    plt.savefig(filename)
    plt.clf()
    print(filename, "salvo com sucesso.")


def create_confusion_matrices(text_file_folds: list[str], experiment_result_folder: str):
    # Prediction: 1, Real label: 1, Votes by cluster: [1 1], Weights: [0.5 0.5]

    all_folds_ypred = []
    all_folds_ytrue = []

    # Create a confusion matrix for each fold and a general one
    for fold, text_file in enumerate(text_file_folds):
        y_pred_fold, y_true_fold = get_labels_and_predictions(text_file)

        all_folds_ypred += y_pred_fold
        all_folds_ytrue += y_true_fold

        filename = os.path.join(experiment_result_folder, f"cm_fold_{fold+1}.png")
        save_confusion_matrix(y_true_fold, y_pred_fold, filename)
    
    # TODO mudar para plotar e salvar e colocar um mapa de cores melhor
    # Save the general confusion matrix
    filename = os.path.join(experiment_result_folder, "cm_all_folds.png")
    save_confusion_matrix(all_folds_ytrue, all_folds_ypred, filename)
    

def generate_row_results(text_file_folds: list[str], experiment_variation: int,
                         experiment_params: str, mutual_info_percentage: float):

    list_classification_results = [get_all_classification_metrics(text_file)
                                   for text_file in text_file_folds]
    return ResultsRow(experiment_variation, experiment_params,
                      mutual_info_percentage, list_classification_results)


def read_files_test_results(experiment_result_folder: str) -> list[str]:

    text_file_folds = []

    for fold in range(1, N_FOLDS+1):

        full_filename = f"{experiment_result_folder}/test_summary/run_{fold}.txt"
        text_file = open(full_filename).read()

        text_file_folds.append(text_file)

    return text_file_folds


def main():

    datasets = ["australian_credit", "contraceptive", "german_credit",
                "heart", "iris", "pima", "wdbc", "wine"]

    experiments_parameters = [
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

    algorithm = "cbeg"

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

            text_file_folds = read_files_test_results(experiment_result_folder)

            row_table = generate_row_results(
                text_file_folds, experiment_variation, experiment_folder, mutual_info_percentage
            )
            rows_table_results.append( row_table )

            print(row_table)

            os.makedirs(f"{experiment_result_folder}/confusion_matrix", exist_ok=True)  # Confusion matrix folder
            create_confusion_matrices(text_file_folds, f"{experiment_result_folder}/confusion_matrix")

        results_folder = f"./results/{dataset}"
        # TODO create results table here
        results_table = ResultsTable(rows_table_results, results_folder)

        results_table.save_results_table_in_latex()
        results_table.save_results_table_in_csv()


if __name__ == "__main__":
    main()
