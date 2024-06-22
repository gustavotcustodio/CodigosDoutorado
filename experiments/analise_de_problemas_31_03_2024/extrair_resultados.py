import os
import sys
import re
import numpy as np
from pandas.core import algorithms
import dataset_loader

PASTA_RESULTADOS = "resultados"


class ResultsAverage:

    def __init__(self, results_runs, algorithm):
        self.algorithm = algorithm

        self.accuracy_values = []
        self.recall_values = []
        self.precision_values = []
        self.f_score_values = []
        self.n_clusters_list = []

        self.mean_accuracy = 0.0
        self.std_accuracy = 0.0

        self.mean_recall = 0.0
        self.std_recall = 0.0

        self.mean_precision = 0.0
        self.std_precision = 0.0

        self.mean_f_score = 0.0
        self.std_f_score = 0.0

        self.mean_n_clusters = 0.0
        self.std_n_clusters = 0.0

        for result in results_runs:
            self.accuracy_values.append(result.accuracy)
            self.recall_values.append(result.recall)
            self.precision_values.append(result.precision)
            self.f_score_values.append(result.f_score)
            self.n_clusters_list.append(result.n_clusters)

        self.mean_accuracy = np.mean(self.accuracy_values)
        self.std_accuracy = np.std(self.accuracy_values)

        self.mean_recall = np.mean(self.recall_values)
        self.std_recall = np.std(self.recall_values)

        self.mean_precision = np.mean(self.precision_values)
        self.std_precision = np.std(self.precision_values)

        self.mean_f_score = np.mean(self.f_score_values)
        self.std_f_score = np.std(self.f_score_values)

        self.mean_n_clusters = np.mean(self.n_clusters_list)
        self.std_n_clusters = np.std(self.n_clusters_list)

    # Rewrite the __str__ method to print in latex format
    def __str__(self):
        mean_accuracy = np.round(self.mean_accuracy, 3)
        mean_recall = np.round(self.mean_recall, 3)
        mean_precision = np.round(self.mean_precision, 3)
        mean_f_score = np.round(self.mean_f_score, 3)
        mean_n_clusters = np.round(self.mean_n_clusters, 3)

        std_accuracy = np.round(self.std_accuracy, 3)
        std_recall = np.round(self.std_recall, 3)
        std_precision = np.round(self.std_precision, 3)
        std_f_score = np.round(self.std_f_score, 3)
        std_n_clusters = np.round(self.std_n_clusters, 3)

        return (f"{self.algorithm.replace("_", " ").title()}" +
                f" & ${mean_accuracy} \\pm {std_accuracy}$" +
                f" & ${mean_recall} \\pm {std_recall}$" +
                f" & ${mean_precision} \\pm {std_precision}$" +
                f" & ${mean_f_score} \\pm {std_f_score}$" +
                f" & ${mean_n_clusters} \\pm {std_n_clusters}$ \\\\ \\midrule\n"
                )


class ResultsRun:

    def __init__(self, accuracy, recall, precision, f_score, n_clusters):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f_score = f_score
        self.n_clusters = n_clusters


def count_clusters(results_text):
    cluster_patterns = re.findall(r"Cluster [0-9]+: \[", results_text)

    return len(cluster_patterns)


def get_classification_metrics(results_text):
    pattern_accuracy = re.findall(r"Acurácia total: [0-9]\.[0-9]+", results_text)[0]
    accuracy = float(pattern_accuracy.split(": ")[1])

    pattern_recall = re.findall(r"Recall total: [0-9]\.[0-9]+", results_text)[0]
    recall = float(pattern_recall.split(": ")[1])

    pattern_precision = re.findall(r"Precisão total: [0-9]\.[0-9]+", results_text)[0]
    precision = float(pattern_precision.split(": ")[1])

    pattern_f1 = re.findall(r"F1-Score: [0-9]\.[0-9]+", results_text)[0]
    f1score = float(pattern_f1.split(": ")[1])

    return accuracy, recall, precision, f1score


def extract_information(filelist, algorithm):
    results_runs = []

    for f in filelist:
        results_text = open(f).read()
        n_clusters = count_clusters(results_text)
        print(f)
        accuracy, recall, precision, f1score = get_classification_metrics(results_text)

        result = ResultsRun(accuracy, recall, precision, f1score, n_clusters)
        results_runs.append(result)
    return ResultsAverage(results_runs, algorithm)


def get_filelist(fullpath_dataset):
    os.makedirs(fullpath_dataset, exist_ok=True)
    filelist = os.listdir(fullpath_dataset)
    filelist = [f for f in filelist if ".png" not in f]
    filelist = [f"{fullpath_dataset}/{f}" for f in filelist]

    return sorted(filelist)


def get_compilation_results(dataset, experiment, algorithm):
    results_compilation = []

    fullpath_dataset = os.path.join(
        PASTA_RESULTADOS, dataset, experiment, algorithm
    )
    filelist = get_filelist(fullpath_dataset)
    results_compilation.append(extract_information(filelist, algorithm))

    return results_compilation


def create_results_table(results_compilation, filename):
    content = """
        \\documentclass[12pt,a4paper]{standalone}
        \\usepackage{booktabs}
        \\usepackage{caption}

        \\begin{document}
        \\begin{tabular}{llllll}
            \\toprule
            \\textbf{Method} & \\textbf{Accuracy} & \\textbf{Recall}  & \\textbf{Precision} & \\textbf{F1-Score}  & \\textbf{Clusters} \\\\ \\midrule\n
            """
    for results_mean in results_compilation:
        content += results_mean[0].__str__()

    content += """
        \\end{tabular}
        \\end{document}"""

    print(content)
    f = open(filename, "w")
    f.write(content)
    f.close()
    print(filename + " saved successfully.")


def main():
    datasets = ["german_credit", "australian_credit", "heart", "iris", "pima", "wdbc", "wine"]
    experiments = ["mutual_info_75", "mutual_info_50", "10_runs", ]
    algorithms = ["CBEG_distances", "CBEG_silhouette", "CBEG_distances_silhouette", "CBEG_2_clusters", "CBEG_3_clusters", "CBEG_4_clusters", "CBEG_5_clusters", "xgboost", "gradient_boosting", "random_forest", "svm"]

    output_latex_dir = "resultados/latex"
    os.makedirs(output_latex_dir, exist_ok=True)

    for dataset in datasets:
        for experiment in experiments:
            results_compilations = []
            for algorithm in algorithms:
                results_compilations.append( get_compilation_results(dataset, experiment, algorithm) )

            output_latex_file = f"{output_latex_dir}/result_{dataset}_{experiment}.tex"
            create_results_table(results_compilations, output_latex_file)


if __name__ == '__main__':
    main()
