import os
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import DATASETS_INFO
from processors.base_classifiers_compiler import BaseClassifiersCompiler, BaseClassifierResult
from processors.cbeg_compiler import SingleCbegResult, CbegResultsCompiler
from processors.ciel_compiler import SingleCielResult, CielCompiler
from processors.data_reader import DataReader, CLASSIFIERS_FULLNAMES, CLASSIFICATION_METRICS


BEST_CBEG_CONFIG_BY_DATASET = {
    'australian_credit': { # 1-2-4 Rand PSO 100 MC
        'folder': 'pso_compare_clusters_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
    'german_credit': { # 1-2-3-4 DBC CV 75 MC
        'folder': 'classifier_selection_clusters_dbc_meta_classifier_fusion',
        'mutual_info': 75
    },
    'heart': { # 1-2-4 DBC CV 100 MC
        'folder': 'classifier_selection_clusters_dbc_meta_classifier_fusion',
        'mutual_info': 100
    },
    'pima': { # 1-2-4 DBC+R PSO 100 WM
        'folder': 'pso_compare_clusters_dbc_rand_weighted_membership_fusion',
        'mutual_info': 100
    },
    'wdbc': { # 1-2-4 DBC+R PSO 100 MC
        'folder': 'pso_compare_clusters_dbc_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
    'blood': { # 1 Rand - 100 MV
        'folder': 'naive_bayes_compare_clusters_rand_majority_voting_fusion',
        'mutual_info': 100
    },
    'electricity': { # 2 - PSO 100 MV 2
        'folder': 'pso_2_clusters_majority_voting_fusion',
        'mutual_info': 100
    },
    'iris': { # 1 Rand - 100 MV
        'folder': 'naive_bayes_compare_clusters_rand_majority_voting_fusion',
        'mutual_info': 100
    },
    'wine': { # 1-2-4 Rand PSO 100 MC
        'folder': 'pso_compare_clusters_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
    'contraceptive': { # 1-2-4 DBC PSO 100 MC
        'folder': 'pso_compare_clusters_dbc_meta_classifier_fusion',
        'mutual_info': 100
    },
    'rectangles': {  # 1-2-4 Rand PSO 100 MC
        'folder': 'pso_compare_clusters_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
    'elipses': { # 1-2-3-4 DBC+R CV 75 WM
        'folder': 'classifier_selection_compare_clusters_dbc_rand_weighted_membership_fusion',
        'mutual_info': 75
    },
    'normal_2_class': { # 2 - CV 100 MV 2
        'folder': 'classifier_selection_2_clusters_majority_voting_fusion',
        'mutual_info': 100
    },
    'normal_3_class': {  # 1-2-4 Rand CV 100 MC
        'folder': 'classifier_selection_compare_clusters_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
    'universal_config': {
        'folder': 'classifier_selection_compare_clusters_rand_meta_classifier_fusion',
        'mutual_info': 100
    },
}


def experiment_already_performed(dataset, experiment_folder, mutual_info):
    folder_path = f'./results/{dataset}/mutual_info_{mutual_info}/cbeg/{experiment_folder}'
    
    if os.path.exists(folder_path):
        return True
    else:
        return False

def filter_cbeg_experiments_configs(experiment_variation: str, mutual_info_percentage: float,
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
        if ("dbc_" in experiment_variation or "ext_" in experiment_variation or
            'rand_' in experiment_variation):
            return {}

        found_n_clusters = int(found_numbers_clusters[0].split("_")[0])
        if found_n_clusters != n_clusters:
            print(f"{n_clusters} - {found_n_clusters}")
            return {}

    variation_number = "0"

    # 0 não para todos
    # 1, 2, 3, 4, 5
    if "compare_clusters" in experiment_variation:  # has_cluster_selection
        variation_number += "1"

    if "classifier_selection" in experiment_variation: # has_classifier_selection
        variation_number += "2"

    if mutual_info_percentage < 100: # has_feature_selection
        variation_number += "3"

    if "majority_voting" not in experiment_variation: # has_weighted_voting_fusion
        variation_number += "4"

    if "pso_" in experiment_variation:
        variation_number += "5"

    if "_oversampling" in experiment_variation:  # has_oversampling
        variation_number += "6"

    accepted_variations = [0, 1, 2, 3, 4, 5, 123, 124, 1234, 145]
    variation_number = int(variation_number)

    # print(f"Variation {variation_number}...")
    if variation_number in accepted_variations:
        return {"folder": experiment_variation, "mutual_info": mutual_info_percentage,
                "variation": variation_number}
    return {}


def extract_cbeg_results(dataset: str, mutual_info_percentages):
    experiments_configs = []
    # The number of classes in the dataset is the default number of clusters
    n_classes_dataset = DATASETS_INFO[dataset]["nlabels"]

    for mutual_info in mutual_info_percentages:
        possible_experiments = os.listdir(f'./results/{dataset}/mutual_info_{mutual_info}/cbeg')

        experiments_configs += [
            filter_cbeg_experiments_configs(
                experiment_variation, mutual_info, n_classes_dataset)
            for experiment_variation in possible_experiments
        ]

    # Remove empty dictionaries (invalid experiments configs)
    experiments_configs = [config for config in experiments_configs if config]
    experiments_configs = sorted(
        experiments_configs, key=lambda x: (x['variation'], x['folder'], x["mutual_info"])
    )
    cbeg_results = []

    print("Processing results for", dataset)
    for config in experiments_configs:
        path = f"./results/{dataset}/mutual_info_{config['mutual_info']}/cbeg/{config['folder']}"
        loader = DataReader(path, training=True)
        loader.read_data()
        cbeg_single_result = SingleCbegResult(
            loader.data["training"], loader.data["test"], config['folder'],
            config["variation"], config['mutual_info']
        )
        cbeg_results.append( cbeg_single_result )

    cbeg_compilation = CbegResultsCompiler(cbeg_results, dataset)
    return cbeg_compilation


def extract_base_results(dataset: str, mutual_info_percentages: list[float]):
    classifiers_results = []

    for mutual_info in mutual_info_percentages:

        for abbrev_classifier, classifier_name in CLASSIFIERS_FULLNAMES.items():
            if "sc" in abbrev_classifier:
                base_clf = abbrev_classifier.split("_")[1]
                path = (f'results/{dataset}/mutual_info_{mutual_info}/' +
                        f'supervised_clustering/supervised_clustering_base_classifier_{base_clf}')
            else:
                path = f'results/{dataset}/mutual_info_{mutual_info}/baselines/{abbrev_classifier}'

            loader = DataReader(path, training=False)
            loader.read_data()
            classifier = BaseClassifierResult(loader.data["test"], classifier_name, mutual_info)

            classifiers_results.append(classifier)

    # Plot the heatmap with classification metrics from base classifiers
    classifier_compiler = BaseClassifiersCompiler(classifiers_results, dataset)
    return classifier_compiler


def extract_ciel_results(dataset: str, mutual_info_percentages: list[float]):
    ciel_results = []

    for mutual_info in mutual_info_percentages:

        path = f"./results/{dataset}/mutual_info_{mutual_info}/ciel"
        loader = DataReader(path, training=True)
        loader.read_data()

        ciel_result = SingleCielResult(loader.data["training"], loader.data["test"],
                                       mutual_info_percentage=mutual_info)

        ciel_results.append(ciel_result)

    ciel_compiler = CielCompiler(ciel_results, dataset)
    return ciel_compiler


# def plot_comparison_graphs(cbeg_compiler, ciel_compiler, baseline_compiler):
#     dataset = cbeg_compiler.dataset
#     path_results = os.path.join(
#         'results', dataset, f'comparison_classifiers_{dataset}.png')
# 
#     universal_config = 'classifier_selection_compare_clusters_rand_meta_classifier_fusion'
# 
#     valid_cbeg_results = [result for result in cbeg_compiler.cbeg_results
#                           if result.folder_name == universal_config
#                           and result.mutual_info_percentage == 100]
#     baseline_results = [result for result in baseline_compiler.baseline_results
#                         if result.mutual_info_percentage == 100]
#     ciel_results = [result for result in ciel_compiler.ciel_results if
#                    result.mutual_info_percentage == 100.0]
# 
#     all_results = valid_cbeg_results + baseline_results + ciel_results
# 
#     baselines_names = [result.classifier_name for result in baseline_results]
# 
#     classification_metrics = CLASSIFICATION_METRICS.copy()
#     color_palette = ['blue', 'red', 'orange', 'green', 'purple']
# 
#     if DATASETS_INFO[dataset]['nlabels'] >= 3:
#         classification_metrics[0] = "Accuracy / Recall"
#         classification_metrics.remove("Recall")
#         color_palette.remove('blue')
# 
#     classifiers = ['CBEG (general)'] + baselines_names + ['CIEL']
#     classifiers_ylabel = []
#     metric_names = []
#     metric_values = []
# 
#     for metric_name in classification_metrics:
#         classifiers_ylabel += classifiers
#         metric_names += [metric_name] * len(all_results)
# 
#     if DATASETS_INFO[dataset]['nlabels'] < 3:
#         metric_values += [result.mean_accuracy for result in all_results]
#     metric_values += [result.mean_recall for result in all_results]
#     metric_values += [result.mean_precision for result in all_results]
#     metric_values += [result.mean_f1 for result in all_results]
#     metric_values += [result.mean_auc for result in all_results]
# 
#     dict_values = {
#         'Value': metric_values, 'Metric name': metric_names,
#         'Classifier': classifiers_ylabel
#     }
#     sns.set_style("darkgrid")
#     _, ax = plt.subplots(figsize=(13, 6.5))
# 
#     sns.scatterplot(
#         data=dict_values, x="Value", y="Classifier",
#         alpha=0.5, hue="Metric name", s=120, ax=ax, palette=color_palette,
#     )
#     plt.savefig(path_results)
#     print(path_results, 'saved.')
#     plt.clf()


def is_cbeg_variation_valid(result, dataset):
    best_config_dataset = BEST_CBEG_CONFIG_BY_DATASET[dataset]
    universal_config = BEST_CBEG_CONFIG_BY_DATASET['universal_config']

    if (result.folder_name == universal_config['folder']
            and result.mutual_info_percentage == universal_config['mutual_info']):
        return True
    if (result.folder_name == best_config_dataset['folder']
            and result.mutual_info_percentage == best_config_dataset['mutual_info']):
        return True
    return False


def filter_classif_results(cbeg_compiler, baseline_compiler, ciel_compiler):
    dataset = cbeg_compiler.dataset
    universal_config = BEST_CBEG_CONFIG_BY_DATASET['universal_config']

    cbeg_results = cbeg_compiler.cbeg_results

    valid_cbeg_results = list(filter(
        lambda x: is_cbeg_variation_valid(x, dataset), cbeg_results
    ))

    # This only happens when the general config and the best one aren't the same
    if len(valid_cbeg_results) > 1:
        folder_name =  valid_cbeg_results[0].folder_name
        mutual_info = valid_cbeg_results[0].mutual_info_percentage

        # Check if the best config is the first one
        if (folder_name == universal_config['folder']
            and mutual_info == universal_config['mutual_info']
        ):
            aux = valid_cbeg_results[0]
            valid_cbeg_results[0] = valid_cbeg_results[1]
            valid_cbeg_results[1] = aux
        cbeg_variations = ['CBEG (best)', 'CBEG (general)']
    else:
        cbeg_variations = ['CBEG (best)']

    baseline_results = [result for result in baseline_compiler.baseline_results
                        if result.mutual_info_percentage == 100]
    ciel_results = [result for result in ciel_compiler.ciel_results
                    if result.mutual_info_percentage == 100]
    all_results = valid_cbeg_results + baseline_results + ciel_results

    baselines_names = [result.classifier_name
                       for result in baseline_results]
    classifiers = cbeg_variations + baselines_names + ['CIEL']
    return all_results, classifiers


def plot_comparison_graphs(cbeg_compiler, ciel_compiler, baseline_compiler):
    all_results, classifiers = filter_classif_results(
        cbeg_compiler, baseline_compiler, ciel_compiler
    )
    dataset = cbeg_compiler.dataset

    if DATASETS_INFO[dataset]['nlabels'] >= 3:
        metric_values = {
            'Accuracy/Recall': [
                result.mean_recall for result in all_results],
        }
    else:
        metric_values = {
            'Accuracy': [result.mean_accuracy for result in all_results],
            'Recall': [result.mean_recall for result in all_results],
        }
    metric_values['Precision'] = [result.mean_precision for result in all_results]
    metric_values['F1'] = [result.mean_f1 for result in all_results]
    metric_values['AUC'] = [result.mean_auc for result in all_results]

    data_classification = pd.DataFrame(data=metric_values)

    sns.set_style("darkgrid")
    _, ax = plt.subplots(figsize=(7.5, 5))

    sns.heatmap(data=data_classification, annot=True, cmap="Blues",
                annot_kws={"size": 11}, ax=ax, fmt='.2f',
                yticklabels=classifiers)
    plt.tight_layout()
    path_results = os.path.join(
        'results', dataset, f'comparison_classifiers_{dataset}.png')
    plt.savefig(path_results)
    print(path_results, 'saved.')
    plt.clf()


def generate_graphs_and_tables(
    cbeg_compiler: CbegResultsCompiler, ciel_compiler: CielCompiler,
    baseline_compiler: BaseClassifiersCompiler
):
    # for metric in CLASSIFICATION_METRICS:
    #     cbeg_compilation.plot_clusterers_heatmap(metric)

    # cbeg_compilation.plot_classification_heatmap()
    # cbeg_compilation.plot_clusters_scatterplot()
    # ciel_compiler.plot_classification_heatmap()
    # classifier_compiler.plot_classification_heatmap()
    plot_comparison_graphs(
        cbeg_compiler, ciel_compiler, baseline_compiler)

    cbeg_compiler.generate_barplot_base_clf()
    cbeg_compiler.create_variation_heatmap()
    cbeg_compiler.generate_latex_table()


def filter_no_experim_datasets(datasets: list[str]) -> list[str]:
    valid_datasets = []
    results_list = os.listdir("./results")

    for dataset in datasets:
        if dataset in results_list:
            valid_datasets.append(dataset)
    return valid_datasets


def main():
    datasets = ["australian_credit",
                "german_credit",
                "heart",
                "pima",
                "wdbc",
                "blood",
                "electricity",
                "iris",
                "wine",
                "contraceptive",
                "rectangles",
                "elipses",
                "normal_2_class",
                "normal_3_class",
                ]
    datasets = filter_no_experim_datasets(datasets)

    mutual_info_percentages = [100.0, 75.0, 50.0]

    for dataset in datasets:
        cbeg_compiler = extract_cbeg_results(dataset, mutual_info_percentages)
        ciel_compiler = extract_ciel_results(dataset, mutual_info_percentages)
        baseline_compiler = extract_base_results(dataset, mutual_info_percentages)

        generate_graphs_and_tables(
            cbeg_compiler, ciel_compiler, baseline_compiler
        )


if __name__ == "__main__":
    main()
