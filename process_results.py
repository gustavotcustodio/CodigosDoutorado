from multiprocessing import process
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from collections import Counter
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cbeg import N_FOLDS
from dataclasses import dataclass
from dataset_loader import DATASETS_INFO

BASE_PATH_FOLDER = "results/{dataset}/mutual_info_{mutual_info_percentage}/{algorithm}/{experiment_folder}"

CLASSIFICATION_METRICS = ["Accuracy", "Recall", "Precision", "F1"]

CLASSIFIERS_FULLNAMES = {
    'nb': "Naive Bayes", 'svm': "SVM", 'lr': "Logistic Reg", 'dt': "Decision Tree",
    'rf': "Random Forest", 'gb': "Grad. Boosting", 'xb': "XGBoost" }

RESULTS_FILENAMES = {"cbeg": "results.csv" , "baseline": "results_baseline.csv"}

@dataclass
class DataReader:
    path: str 
    training: bool = True

    def read_data(self):
        self.data = {}

        self.data["test"] = self.read_training_or_test_data("test")
        if self.training:
            self.data["training"] = self.read_training_or_test_data("training")

    def read_training_or_test_data(self, stage: str) -> list[str]:
        text_folds = []
        for fold in range(1, N_FOLDS+1):
            full_filename = f"{self.path}/{stage}_summary/run_{fold}.txt"
            text_file = open(full_filename).read()
            text_folds.append(text_file)

        return text_folds

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

        else: # F1
            return self.mean_f1

    def calc_mean_and_std_metrics(self):
        self.mean_accuracy  = np.mean([metrics["Accuracy"] for metrics in self.classification_results_folds])
        self.mean_recall    = np.mean([metrics["Recall"] for metrics in self.classification_results_folds])
        self.mean_precision = np.mean([metrics["Precision"] for metrics in self.classification_results_folds])
        self.mean_f1        = np.mean([metrics["F1"] for metrics in self.classification_results_folds])

        self.std_accuracy   = np.std([metrics["Accuracy"] for metrics in self.classification_results_folds])
        self.std_recall     = np.std([metrics["Recall"] for metrics in self.classification_results_folds])
        self.std_precision  = np.std([metrics["Precision"] for metrics in self.classification_results_folds])
        self.std_f1         = np.std([metrics["F1"] for metrics in self.classification_results_folds])

    def get_classification_metrics(self, content_test: str):
        # Dictionary where the values of accuracy, recall, precision and F1 are stored
        dict_classification_results = {}

        for metric in CLASSIFICATION_METRICS:
            # All patterns found in text corresponding to the searched metric
            found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", content_test)
            dict_classification_results[metric] = [
                float(pattern.split(": ")[1]) for pattern in found_metric_patterns]

        return dict_classification_results

    def __str__(self):
        return f"""
            Accuracy: {self.mean_accuracy} +- {self.std_accuracy}
            Recall: {self.mean_recall} +- {self.std_recall}
            Precision: {self.mean_precision} +- {self.std_precision}
            F1 Score: {self.mean_f1} +- {self.std_f1}
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

class SingleCbegResult(BaseClassifierResult):

    def __init__(self, training_info_folds: list[str], test_info_folds: list[str],
                 folder_name: str, experiment_variation: int,
                 mutual_info_percentage: float=100.0):
        self.n_folds = len(test_info_folds)
        self.experiment_variation = experiment_variation
        self.mutual_info_percentage = mutual_info_percentage

        self._get_experiments_params(folder_name)

        self.classification_results_folds =  []
        self.cluster_classification_folds = []
        self.labels_by_cluster_folds = []

        for fold in range(self.n_folds):
            self.extract_training_information(training_info_folds[fold], fold)
            self.extract_test_information(test_info_folds[fold], fold)

        self.calc_mean_and_std_metrics()

    def _get_experiments_params(self, folder_name):

        if 'dbc_ss' in folder_name:
            self.cluster_selection_strategy = "DBC + SS"
        elif 'silhouette' in folder_name:
            self.cluster_selection_strategy = "SS"
        elif 'dbc' in folder_name:
            self.cluster_selection_strategy = "DBC"
        else:
            self.cluster_selection_strategy = ""

        if "cluster_density_fusion" in folder_name:
            self.fusion_strategy = "Cluster Density"
        elif "entropy_voting_fusion" in folder_name:
            self.fusion_strategy = "Entropy Voting"
        elif "weighted_membership_fusion" in folder_name:
            self.fusion_strategy = "Weighted Membership"
        elif "weighted_membership_entropy_fusion" in folder_name:
            self.fusion_strategy = "Weighted Membership Entropy"
        else:
            self.fusion_strategy = "Majority Voting"


    def extract_training_information(self, content_training, fold):
        self.labels_by_cluster_folds.append( self._get_labels_by_cluster(content_training) )
        # Get base classifiers TODO
        # Get selected features TODO

    def _get_labels_by_cluster(self, content_training: str) -> list[list[int]]:
        # Labels: [0 0 0  0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0
        #  0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0]
        found_strings = re.findall(r"Labels:\s\[[0-9\s\n]+\]", content_training)
        self.n_clusters = len(found_strings)

        labels_by_cluster = []

        for c in range(self.n_clusters):
            str_labels = found_strings[c].replace("Labels: [", "").replace("]", "")
            labels = re.split(r"[\n\s]+", str_labels)
            labels_by_cluster.append( [int(lbl) for lbl in labels] )
        return labels_by_cluster

    def extract_test_information(self, content_test, fold):
        classification_results_fold = self.get_classification_metrics(content_test)

        self.cluster_classification_folds.append(
            self._get_classification_results_by_cluster(classification_results_fold)
        )
        self.classification_results_folds.append(
            self.get_general_classification_results(classification_results_fold)
        )

    def _get_classification_results_by_cluster(
            self, classification_results: dict[str, list[float]]
        ) -> dict[str, list[float]]:

        classification_by_cluster = {}

        for metric in CLASSIFICATION_METRICS:
            classification_by_cluster[metric] = classification_results[metric][0:-1]

        return classification_by_cluster

    def get_classification_metrics(self, content_test: str):
        # Dictionary where the values of accuracy, recall, precision and F1 are stored
        dict_classification_results = {}

        for metric in CLASSIFICATION_METRICS:
            # All patterns found in text corresponding to the searched metric
            found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", content_test)
            dict_classification_results[metric] = [float(pattern.split(": ")[1]) for pattern in found_metric_patterns]

        if "Cluster" in content_test:
            pattern_n_clusters = re.findall(r"Cluster [0-9]", content_test)[-1]
            n_clusters = int(pattern_n_clusters.split(" ")[1]) + 1

            dict_classification_results["Number clusters"] = n_clusters

        return dict_classification_results

@dataclass
class CbegResultsCompiler:

    cbeg_results: list[SingleCbegResult]
    dataset: str

    def _get_experiment_params(self, row_cbeg):
        variation = row_cbeg.experiment_variation
        selection_strategy = row_cbeg.cluster_selection_strategy
        mutual_info = row_cbeg.mutual_info_percentage

        output = str(variation)
        if row_cbeg.cluster_selection_strategy:
            output += f" - {selection_strategy}"

        if row_cbeg.mutual_info_percentage < 100.0:
            output += f" ({mutual_info} %)"
        return output

    def plot_classification_heatmap(self):
        """ Save the confusion matrix for the ablation study.
        """
        heatmaps_folder = f"results/{self.dataset}/ablation_results"
        os.makedirs(heatmaps_folder, exist_ok=True)

        # All possible methods for
        vote_fusion_strategies = set([row_cbeg.fusion_strategy for row_cbeg in self.cbeg_results])
        vote_fusion_strategies = sorted(list(vote_fusion_strategies))
        self.cbeg_results = sorted(
                self.cbeg_results,
                key=lambda x: (x.experiment_variation, x.cluster_selection_strategy))

        for metric in CLASSIFICATION_METRICS:
            filename = os.path.join( heatmaps_folder, f"{metric}_heatmap_ablation_cbeg.png")
            dict_results = self._fill_heatmap_dict(metric, vote_fusion_strategies)
            data = []
            # Convert dict of dictionaries to 2D matrix
            for experiment_label in dict_results.keys():
                data_row = [dict_results[experiment_label][fusion_strategy]
                            for fusion_strategy in vote_fusion_strategies]
                data.append(data_row)
            
            data = np.vstack(data)
            indexes = list(dict_results.keys())
            columns = vote_fusion_strategies
            
            self.save_heatmap(data, columns, indexes, filename)

    def save_heatmap(self, data, columns, indexes, filename):
        sns.heatmap(data, annot=True, cmap='Blues',
                    xticklabels=columns, yticklabels=indexes, vmin=0, vmax=1)

        plt.xlabel("Fusion Strategy", fontdict={'weight': 'bold'})
        plt.ylabel("Experiment variation", fontdict={'weight': 'bold'})
        # Save the heat map
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()

        print(f"{filename} saved successfully.")

    def _fill_heatmap_dict(self, metric, vote_fusion_strategies):
        results_dict_heatmap = {}

        for row_cbeg in self.cbeg_results:
            clf_metric_value = row_cbeg.get_metric_value(metric)
            experiment_label = self._get_experiment_params(row_cbeg)
            fusion_strategy = row_cbeg.fusion_strategy

            if experiment_label not in results_dict_heatmap:
                results_dict_heatmap[experiment_label] = {}

            results_dict_heatmap[experiment_label][fusion_strategy] = float(clf_metric_value)

        # Complete the empty values in the matrix with zeroes
        for experiment_label in results_dict_heatmap.keys():
            for fusion_strategy in vote_fusion_strategies:
                if fusion_strategy not in results_dict_heatmap[experiment_label]:
                    results_dict_heatmap[experiment_label][fusion_strategy] = 0
        return results_dict_heatmap

def filter_cbeg_experiments_configs(experiment_variation: str,mutual_info_percentage: float,
                                    n_clusters: int) -> dict:
    # - Versão básica - Naive bayes, fusão por votação, numero de grupos = numero de classes, etc.
    # (algoritmo de agrupamento default kmeans++)
    # - compare_clusters: Versão com seleção de número de grupos e algoritmo de agrupamento: (1)
    # - classifier_selection: Versão com seleção de classificadores base: (2)
    # - mutual_info < 100.0: Versão com redução de dimensionalidade: (3)
    # - diferente de majority_voting_fusion: Versão com fusão (votação) proposta: (4)
    # - Versão completa: (1 + 2 + 3 + 4)
    # Invalid case when we have a fixed number of clusters and cluster selection simultaneously

    found_numbers_clusters = re.findall(r"[0-9]+_clusters", experiment_variation)

    if found_numbers_clusters:
        if ("dbc" in experiment_variation or "silhouette" in experiment_variation):
            return {}

        found_n_clusters = int(found_numbers_clusters[0].split("_")[0])
        if found_n_clusters != n_clusters:
            return {}

    has_cluster_selection = True if "compare_clusters" in experiment_variation else False
    has_classifier_selection = True if "classifier_selection" in experiment_variation else False
    has_feature_selection = True if mutual_info_percentage < 100 else False
    has_weighted_voting_fusion = True if "majority_voting" not in experiment_variation else False

    if (not(has_cluster_selection) and not(has_classifier_selection) and
          not(has_feature_selection) and not(has_weighted_voting_fusion)):
        return {"folder": experiment_variation, "mutual_info": 100.0, "variation": 0}

    elif (has_cluster_selection and not(has_classifier_selection) and
          not(has_feature_selection) and not(has_weighted_voting_fusion)):
        return {"folder": experiment_variation, "mutual_info": 100.0, "variation": 1}
        
    elif (not(has_cluster_selection) and has_classifier_selection and
          not(has_feature_selection) and not(has_weighted_voting_fusion)):
        return {"folder": experiment_variation, "mutual_info": 100.0, "variation": 2}

    elif (not(has_cluster_selection) and not(has_classifier_selection) and
          has_feature_selection and not(has_weighted_voting_fusion)):
        return {"folder": experiment_variation, "mutual_info": mutual_info_percentage, "variation": 3}

    elif (not(has_cluster_selection) and not(has_classifier_selection) and
          not(has_feature_selection) and has_weighted_voting_fusion):
        return {"folder": experiment_variation, "mutual_info": 100.0, "variation": 4}

    elif (has_cluster_selection and has_classifier_selection and
          not(has_feature_selection) and has_weighted_voting_fusion):
        return {"folder": experiment_variation, "mutual_info": 100.0, "variation": 124}

    elif (has_cluster_selection and has_classifier_selection and
          has_feature_selection and has_weighted_voting_fusion):
        return {"folder": experiment_variation, "mutual_info": mutual_info_percentage, "variation": 1234}
    else:
        return {}

def process_cbeg_results(datasets, mutual_info_percentages):
    for dataset in datasets:
        experiments_configs = []
        # The number of classes in the dataset is the default number of clusters
        n_classes_dataset = DATASETS_INFO[dataset]["nlabels"]

        for mutual_info in mutual_info_percentages:
            possible_experiments = os.listdir(f'./results/{dataset}/mutual_info_{mutual_info}/cbeg')

            experiments_configs += [
                filter_cbeg_experiments_configs(experiment_variation, mutual_info, n_classes_dataset)
                for experiment_variation in possible_experiments
            ]

        experiments_configs = [config for config in experiments_configs if config]
        experiments_configs = sorted(
            experiments_configs, key=lambda x: (x['variation'], x['folder'], x["mutual_info"])
        )
        cbeg_results = []

        for config in experiments_configs:
            path = f"./results/{dataset}/mutual_info_{config['mutual_info']}/cbeg/{config['folder']}"
            loader = DataReader(path, training=True)
            loader.read_data()

            cbeg_single_result = SingleCbegResult(
                loader.data["training"], loader.data["test"], config['folder'],
                config["variation"], config['mutual_info']
            )
            cbeg_results.append( cbeg_single_result )
        breakpoint()

        cbeg_compilation = CbegResultsCompiler(cbeg_results, dataset)
        cbeg_compilation.plot_classification_heatmap()


def process_base_results(datasets: list[str], mutual_info_percentages: list[float]):

    for dataset in datasets:
        classifiers_results = []

        for mutual_info in mutual_info_percentages:

            for abbrev_classifier, classifier_name in CLASSIFIERS_FULLNAMES.items():
                path = f'results/{dataset}/mutual_info_{mutual_info}/baselines/{abbrev_classifier}'

                loader = DataReader(path, training=False)
                loader.read_data()
                classifier = BaseClassifierResult(loader.data["test"], classifier_name, mutual_info)

                classifiers_results.append(classifier)

        # Plot the heatmap with classification metrics from base classifiers
        classifier_compiler = BaseClassifiersCompiler(classifiers_results, dataset)

        classifier_compiler.plot_classification_heatmap()

def filter_no_exper_datasets(datasets: list[str]) -> list[str]:
    valid_datasets = []
    results_list = os.listdir("./results")

    for dataset in datasets:
        if dataset in results_list:
            valid_datasets.append(dataset)
    return valid_datasets

def main():
    datasets = ["australian_credit", "contraceptive", "german_credit", "heart", "iris", "pima", "wdbc", "wine"]
    datasets = filter_no_exper_datasets(datasets)

    mutual_info_percentages = [100.0, 75.0, 50.0]

    process_cbeg_results(datasets, mutual_info_percentages)
    # process_base_results(datasets, mutual_info_percentages) 

if __name__ == "__main__":
    main()
