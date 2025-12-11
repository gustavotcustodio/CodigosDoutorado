import os
import copy
import sys
import re
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import Counter
from cluster_selection import CLUSTERING_ALGORITHMS
from processors.data_reader import CLASSIFICATION_METRICS, N_FOLDS
from processors.base_classifiers_compiler import BaseClassifierResult
from processors.latex_processor import create_latex_table


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)


class SingleCbegResult(BaseClassifierResult):

    def __init__(self, training_info_folds: list[str], test_info_folds: list[str],
                 folder_name: str, experiment_variation: int,
                 mutual_info_percentage: float=100.0):
        self.folder_name = folder_name
        self.experiment_variation = experiment_variation
        self.mutual_info_percentage = mutual_info_percentage

        self._get_experiments_params(folder_name)

        self.n_clusters_by_fold = []
        self.classification_results_folds =  []
        self.cluster_classification_folds = []
        self.labels_by_cluster_folds = []
        self.idx_synthetic_by_cluster_folds = []
        self.classifier_by_cluster_folds = []
        self.minority_class_by_cluster_folds = []
        self.clustering_algorithms_folds = []

        self.n_folds = len(test_info_folds)
        for fold in range(self.n_folds):
            self.extract_training_information(training_info_folds[fold])
            #print(folder_name)
            #print(self.experiment_variation)
            self.extract_test_information(test_info_folds[fold])

        self.calc_mean_and_std_metrics()
        self._calc_mean_std_clusters()

    def _calc_mean_std_clusters(self):

        self.mean_n_clusters = np.mean(self.n_clusters_by_fold)
        self.std_n_clusters = np.std(self.n_clusters_by_fold)

        if int(self.mean_n_clusters) == self.mean_n_clusters:
            self.mean_n_clusters = int(self.mean_n_clusters)


    def _get_experiments_params(self, folder_name):

        if 'dbc_ss' in folder_name:
            self.cluster_selection_strategy = "DBC + SS"
        elif 'silhouette' in folder_name:
            self.cluster_selection_strategy = "SS"
        elif 'dbc_rand' in folder_name:
            self.cluster_selection_strategy = "DBC + Rand"
        elif 'rand' in folder_name:
            self.cluster_selection_strategy = "Rand Score"
        elif 'dbc_ext' in folder_name:
            self.cluster_selection_strategy = "DBC + Ext."
        elif '_ext' in folder_name:
            self.cluster_selection_strategy = "External"
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
            self.fusion_strategy = "W. Membership + Entropy"
        elif "meta_classifier" in folder_name:
            self.fusion_strategy = "Meta Classifier"
        else:
            self.fusion_strategy = "Majority Voting"

    def extract_training_information(self, content_training: str):
        """ Extract labels by cluster, base classifiers selected and more
        information related to the training stage. """

        labels_by_cluster = self._get_labels_by_cluster(content_training)

        self.labels_by_cluster_folds.append( labels_by_cluster )

        self.idx_synthetic_by_cluster_folds.append(
                self._get_synthetic_samples_by_cluster(content_training))
        self.minority_class_by_cluster_folds.append(
                self._get_minority_class_by_cluster(content_training))
        self.classifier_by_cluster_folds.append(
                self._get_base_classifiers_by_cluster(content_training))
        # Get selected clustering algorithm
        self.clustering_algorithms_folds.append(
                self._get_selected_clustering_algorithms(content_training))

    def _get_selected_clustering_algorithms(self, content_training: str) -> str:
        found_strings = re.findall(r"Clustering algorithm selected: .*", content_training)
        
        if found_strings:
            return found_strings[0].split(": ")[1]
        else:
            return ""

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

    def _get_minority_class_by_cluster(self, content_training: str):
        found_strings = re.findall(r"Minority Class:\s[0-9]+", content_training)
        lbl_minority_by_cluster = []

        for c in range(self.n_clusters):
            lbl_minority = int(found_strings[c].split(": ")[1])
            lbl_minority_by_cluster.append(lbl_minority)

        return lbl_minority_by_cluster

    def _get_base_classifiers_by_cluster(self, content_training: str):
        found_strings = re.findall(r"Base classifier:\s.+", content_training)
        base_classifiers = []

        for c in range(self.n_clusters):
            base_classifier = found_strings[c].split(": ")[1].strip()

            base_classifiers.append(base_classifier)
        return base_classifiers

    def _get_synthetic_samples_by_cluster(self, content_training: str) -> list:
        found_strings = re.findall(r"Synthetic samples indexes:\s\[[0-9\s\n]*\]",
                                   content_training)
        if not(found_strings):
            return [[] for _ in range(self.n_clusters)]

        synthetic_samples_by_cluster = []

        for c in range(self.n_clusters):
            str_labels = found_strings[c].replace(
                    "Synthetic samples indexes: [", "").replace("]", "").strip()
            idx_synthetic = re.split(r"[\n\s]+", str_labels)
            if idx_synthetic[0] != "":
                synthetic_samples_by_cluster.append( [int(idx) for idx in idx_synthetic] )
            else:
                synthetic_samples_by_cluster.append( [] )
        return synthetic_samples_by_cluster

    def extract_test_information(self, content_test):
        #print("COMEÇA AQUI")
        classification_results_fold = self.get_classification_metrics(content_test)
        #print(classification_results_fold)
        #print(self.mutual_info_percentage)
        #print("TERMINA AQUI")

        self.cluster_classification_folds.append(
            self._get_classification_results_by_cluster(classification_results_fold)
        )
        self.classification_results_folds.append(
            self.get_general_classification_results(classification_results_fold)
        )
        n_clusters_fold = classification_results_fold["Number clusters"]
        self.n_clusters_by_fold.append(n_clusters_fold)


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
            # print(self.folder_name)
            # print(found_metric_patterns)
            dict_classification_results[metric] = [
                    float(pattern.split(": ")[1]) for pattern in found_metric_patterns]

        if "Cluster" in content_test:
            pattern_n_clusters = re.findall(r"Cluster [0-9]", content_test)[-1]
            n_clusters = int(pattern_n_clusters.split(" ")[1]) + 1

            dict_classification_results["Number clusters"] = n_clusters

        return dict_classification_results

    def plot_clusters_scatterplot(self, dataset: str, classification_metric: str):
        x_values, y_values = [], []
        hue_values = []
        n_samples_clusters = []
        base_classifiers = []

        folder_scatter = f"results/{dataset}/results_by_clusters/{classification_metric}"
        # Get all cluster metric values and label distribution
        for fold in range(N_FOLDS):
            # labels_by_cluster = self.training_information_folds[fold].labels_by_cluster
            labels_by_cluster = self.labels_by_cluster_folds[fold]
            synth_samples_by_cluster = self.idx_synthetic_by_cluster_folds[fold]

            for c, cluster_labels in enumerate(labels_by_cluster):
                # Count the number of labels by cluster
                label_count = Counter(cluster_labels)
                n_synth_samples = len(synth_samples_by_cluster[c])

                # Count the number of samples per cluster.
                # If it's the minority class, remove the synthetic samples from the ounting
                minority_class = self.minority_class_by_cluster_folds[fold][c]

                count_lbls = [
                    (count - n_synth_samples, lbl) if lbl == minority_class else (count, lbl)
                    for lbl, count in label_count.items()
                ]

                n_majority, lbl_majority = max(count_lbls)
                n_minority, _ = min(count_lbls)
                # if there's only one class in the cluster, we say that the minority class
                # contains only a single samples from minority class.
                if len(count_lbls) == 1 or n_minority == 0:
                    n_minority = 1
                
                # print(len(self.cluster_classification_folds[fold][classification_metric]))
                metric_value = self.cluster_classification_folds[fold][classification_metric][c]
                base_classifier = self.classifier_by_cluster_folds[fold][c]
                base_classifier = base_classifier.split("(")[0]
                # x axis: number of majority class X minority class
                # y axis: metric value
                x_values.append(n_majority / n_minority)
                # if (n_majority / n_minority) < 0:
                #     print(n_majority / n_minority)
                #     print(n_majority, n_minority)
                #     print("------------")
                y_values.append(metric_value)
                hue_values.append(lbl_majority)
                n_samples_clusters.append(len(cluster_labels))
                base_classifiers.append(base_classifier)

        cols_names = {"imbalance": "Cluster Imbalance (# Majority Class / # Minority Class)",
                      classification_metric: f"Test {classification_metric}",
                      "majority": "Label of Majority Class",
                      "n_samples": "# of Samples in Cluster",
                      "base_classifier": "Base Classifier"
                      }

        data = pd.DataFrame({cols_names["imbalance"]: x_values,
                             cols_names[classification_metric]: y_values,
                             cols_names["majority"]: hue_values,
                             cols_names["n_samples"]: n_samples_clusters,
                             cols_names["base_classifier"]: base_classifiers})

        _, ax = plt.subplots(figsize=(9, 6))

        sp = sns.scatterplot(
            data=data, x=cols_names["imbalance"], y=cols_names[classification_metric],
            hue=cols_names["majority"], size=cols_names["n_samples"],
            style=cols_names["base_classifier"], palette="deep",
        )
        sp.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)
        ax.grid()
        plt.title(dataset.replace("_", " ").title())
        plt.tight_layout()

        folder_scatter = f"results/{dataset}/results_by_clusters/{classification_metric}"
        figname = f"mi_{self.mutual_info_percentage}_{self.folder_name}.png"
        full_filename = os.path.join(folder_scatter, figname)

        os.makedirs(folder_scatter, exist_ok=True)
        plt.savefig(full_filename)
        print(full_filename, "saved successfully.")
        plt.clf()
        plt.close()

    def __repr__(self):
        return f"""
            Accuracy: {self.mean_accuracy} +- {self.std_accuracy}
            Recall: {self.mean_recall} +- {self.std_recall}
            Precision: {self.mean_precision} +- {self.std_precision}
            F1 Score: {self.mean_f1} +- {self.std_f1}

            Experiment variation: {self.folder_name}
        """


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

        if hasattr(row_cbeg, "base_classifier_selection"):
            output += f" - {row_cbeg.base_classifier_selection}"
        return output

    def generate_latex_table(self):
        """ Generate a latex table corresponding
        to the heatmap of the ablation study.
        """
        create_latex_table(self.cbeg_results, self.dataset) 

    def calc_increase_decrease_matrix(
        self, base_result: SingleCbegResult, step_results: list[SingleCbegResult]
    ) -> NDArray:
        """Calculate the matrix showing increase/decrease
        of each metric when an experiment step is added.

        Args:
            base_result: The result with experiment_variation = 0.
            step_results: Experiments with only a single improvement
                experiment_variation is 1, 2, 3 or 4.

        Returns:
            variation_matrix: Matrix showing increase/decrease of metrics.
                It has size 9x5 (number of variations x number of metrics).
        """

        assert base_result.experiment_variation == 0, \
                f"Invalid experiment variation for variation_heatmap {base_result.experiment_variation}."

        for result in step_results:
            assert 0 < result.experiment_variation <= 4, \
                f"Invalid experiment variation for variation_heatmap {result.experiment_variation}."

        variation_matrix = np.empty((9, 5))
        for i, result in enumerate(step_results):
            variation_matrix[i, 0] = result.mean_accuracy - base_result.mean_accuracy
            variation_matrix[i, 1] = result.mean_recall - base_result.mean_recall
            variation_matrix[i, 2] = result.mean_precision - base_result.mean_precision
            variation_matrix[i, 3] = result.mean_f1 - base_result.mean_f1
            variation_matrix[i, 4] = result.mean_auc - base_result.mean_auc

        return variation_matrix


    def create_variation_heatmap(self):
        """Create a heatmap showing how much each metric
           improved or reduced with the inclusion of each
           step.
        """
        cbeg_results = copy.deepcopy(self.cbeg_results)

        for i in range(len(cbeg_results)):
            if cbeg_results[i].experiment_variation == 2:
                cbeg_results[i].base_classifier_selection = "Crossval"
            elif cbeg_results[i].experiment_variation == 5:
                cbeg_results[i].experiment_variation = 2
                cbeg_results[i].base_classifier_selection = "PSO"

        # Sort results by experiment variation
        sorted_results = sorted(
            cbeg_results,
            key=lambda result: result.experiment_variation
        )
        # Get only experiments with a single step added
        # or the base version (0, 1, 2, 3 and 4)
        base_result = sorted_results[0]
        single_step_results = sorted_results[1:10]
         
        # Calculate the difference between each experiment
        # variation and the base result (0)
        variation_matrix = self.calc_increase_decrease_matrix(
            base_result, single_step_results)
        
        # Transform the results in a matrix and create the
        # heatmap
        xlabels = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
        ylabels = []
        for result in single_step_results:
            if result.experiment_variation == 4 and "Weighted" in result.fusion_strategy:
                ylabels.append(f"4 - W. Memb.")
            elif result.experiment_variation == 4 and "Meta" in result.fusion_strategy:
                ylabels.append(f"4 - Meta-Clf.")
            else:
                ylabels.append(self._get_experiment_params(result))

        # Maximum variance ofr heatmap
        max_var = max(variation_matrix.max(), abs(variation_matrix.min()))
        _, ax = plt.subplots(figsize=(9.4, 5.0)) 
        sns.heatmap(variation_matrix, annot=True, cmap="RdBu",
                    vmin=-max_var, vmax=max_var, ax=ax, annot_kws={"size": 13},
                    fmt='.2f', xticklabels=xlabels, yticklabels=ylabels)
        #title = " ".join(self.dataset.capitalize().split("_"))
        #ax.set_title(title)
        plt.xticks(fontsize=12)

        output_file = f"results/{self.dataset}/step_ablation_{self.dataset}.png"
        os.makedirs(f"results/{self.dataset}", exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        print(f"{output_file} saved.")


    def plot_classification_heatmap(self):
        """ Save the heatmap for the ablation study.
        """
        heatmaps_folder = f"results/{self.dataset}/ablation_results"
        os.makedirs(heatmaps_folder, exist_ok=True)

        # All possible methods for
        vote_fusion_strategies = set(
                [row_cbeg.fusion_strategy for row_cbeg in self.cbeg_results
                 if row_cbeg.fusion_strategy != "Majority Voting"])
        vote_fusion_strategies = sorted(list(vote_fusion_strategies))

        # Order the results by the number of experiment variation and
        # the clustering selection strategy
        self.cbeg_results = sorted(
                self.cbeg_results,
                key=lambda x: (x.experiment_variation, x.cluster_selection_strategy))

        for metric in CLASSIFICATION_METRICS:
            filename = os.path.join( heatmaps_folder, f"{metric}_heatmap_ablation_cbeg.png")

            # We use two different dictionaries in order to split the majority_voting
            # and the other experiments.
            dict_results_weighted = self._fill_heatmap_dict(metric, vote_fusion_strategies)
            dict_results_majority_vote = self._fill_heatmap_dict(metric, ["Majority Voting"])

            data_weighted_fusion = []
            data_majority_voting = []
            # Convert dict of dictionaries to 2D matrix
            for experiment_label in dict_results_weighted.keys():
                # Row of the heatmap.
                # Each row represents the experiment variation (1, 2, 3, ..., 1234) and
                # each column a fusion strategy.
                data_row = [dict_results_weighted[experiment_label][fusion_strategy]
                            for fusion_strategy in vote_fusion_strategies]
                data_weighted_fusion.append(data_row)

            for experiment_label in dict_results_majority_vote.keys():
                data_row = [dict_results_majority_vote[experiment_label]["Majority Voting"]]
                data_majority_voting.append(data_row)
            
            data_weighted_fusion = np.vstack(data_weighted_fusion)

            _, ax = plt.subplots(1, 2, figsize=(20, 9), width_ratios=[1, 2])

            if data_majority_voting:
                data_majority_voting = np.vstack(data_majority_voting)

                # plt.margins(x=5,y=5)
                plt.subplots_adjust( wspace=0.38)
                indexes_majority = list(dict_results_majority_vote.keys())
                columns_majority = ["Majority Voting"]
                self.add_heatmap(data_majority_voting, columns_majority, indexes_majority, ax[0], cbar=False)

            indexes_weighted = list(dict_results_weighted.keys())
            columns_weighted = vote_fusion_strategies
            self.add_heatmap(data_weighted_fusion, columns_weighted, indexes_weighted, ax[1], cbar=True)

            plt.savefig(filename)
            plt.close()
            print(f"{filename} saved successfully.")

    def plot_clusterers_heatmap(self, metric='Accuracy'):
        """Plot the heatmaps showing how many times each
        clustering algorithm was selected.
        """
        #data_count = np.zeros(
        #    (len(self.cbeg_results), len(CLUSTERING_ALGORITHMS))).astype(np.int32)
        selected_clusterers_folder = f"results/{self.dataset}/selected_clusterers"
        os.makedirs(selected_clusterers_folder, exist_ok=True)

        xlabels = list(CLUSTERING_ALGORITHMS.keys())
        ylabels = []

        classif_per_clusterer = self.get_classif_values_per_clusterer(
                xlabels, ylabels, metric)
        
        _, ax = plt.subplots(figsize=(15, 12))
        # selections_by_classifier = np.sum(data_count, axis=0)
        # sns.histplot(data=aaa)
        g = sns.heatmap(classif_per_clusterer, annot=True,
                        cmap='Blues', ax=ax, annot_kws={"size": 14},
                        xticklabels=xlabels, yticklabels=ylabels, vmin=0, vmax=1)

        g.set_yticklabels(g.get_yticklabels(), size=10)
        g.set_xticklabels(g.get_xticklabels(), size=11)

        filename = os.path.join(selected_clusterers_folder,
                                f'clusterers_comparison_{metric}.png')
        plt.savefig(filename)

        print(f'{filename} saved successfully.')
        plt.close()

    def get_classif_values_per_clusterer(
            self, clustering_methods, experiments, metric):
        classification_values_per_clusterer = []

        for _, result in enumerate(self.cbeg_results):
            if (str(result.experiment_variation)[0] != '1' or
                result.mutual_info_percentage < 100 or
                result.fusion_strategy == "Majority Voting"
            ):
                continue

            if metric == "Accuracy":
                metric_value = result.mean_accuracy
            elif metric == "Precision":
                metric_value = result.mean_precision
            elif metric == "Recall":
                metric_value = result.mean_recall
            elif metric == "F1":
                metric_value = result.mean_f1
            else:  # metric == "AUC"
                metric_value = result.mean_auc

            # Get the classification results for all metrics
            experim_variation = f'{result.experiment_variation} -' + \
                    f'{result.cluster_selection_strategy}\n{result.fusion_strategy}'
            experiments.append(experim_variation)

            classification_metrics = len(clustering_methods) * [0.0]

            for clustering_method in result.clustering_algorithms_folds:

                k = clustering_methods.index(clustering_method)
                classification_metrics[k] = metric_value

            classification_values_per_clusterer.append(classification_metrics)
        return np.array(classification_values_per_clusterer)

    def add_heatmap(self, data, columns, indexes, ax, cbar=True):
        g = sns.heatmap(data, ax=ax, annot=True, cmap='Blues',
                    cbar=cbar, annot_kws={"size": 14},
                    xticklabels=columns, yticklabels=indexes, vmin=0, vmax=1)
        g.set_yticklabels(g.get_yticklabels(), size=11)
        g.set_xticklabels(g.get_xticklabels(), size=13)

        ax.set_xlabel("Fusion Strategy", fontdict={'weight': 'bold', 'size': 13})
        if not(cbar):
            ax.set_ylabel("Experiment variation", fontdict={'weight': 'bold', 'size': 13})
        # Save the heat map
        ax.tick_params(axis='x', labelrotation=0)
        ax.tick_params(axis='y', labelrotation=0)

    def _fill_heatmap_dict(self, metric, vote_fusion_strategies):
        results_dict_heatmap = {}

        for row_cbeg in self.cbeg_results:
            # Get value from classification metric
            clf_metric_value = row_cbeg.get_metric_value(metric)
            # Get group of experiment parameters (what shows in y axis)
            experiment_label = self._get_experiment_params(row_cbeg)
            fusion_strategy = row_cbeg.fusion_strategy

            if fusion_strategy in vote_fusion_strategies:
                if experiment_label not in results_dict_heatmap:
                    results_dict_heatmap[experiment_label] = {}

                results_dict_heatmap[experiment_label][fusion_strategy] = float(clf_metric_value)

        return results_dict_heatmap

    def plot_clusters_scatterplot(self):
        val_classif_metrics = [metric for metric in CLASSIFICATION_METRICS if metric != "AUC"]

        for cbeg_result in self.cbeg_results:
            for metric in val_classif_metrics:
                cbeg_result.plot_clusters_scatterplot(self.dataset, metric)

