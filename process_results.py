import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_loader import DATASETS_INFO
from processors.base_classifiers_compiler import BaseClassifiersCompiler, BaseClassifierResult
from processors.cbeg_compiler import SingleCbegResult, CbegResultsCompiler
from processors.ciel_compiler import SingleCielResult, CielCompiler
from processors.data_reader import DataReader, CLASSIFIERS_FULLNAMES, CLASSIFICATION_METRICS

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


def plot_comparison_graphs(cbeg_compiler, ciel_compiler, baseline_compiler):
    dataset = cbeg_compiler.dataset
    path_results = os.path.join(
        'results', dataset, f'comparison_classifiers_{dataset}.png')

    universal_config = 'classifier_selection_compare_clusters_rand_meta_classifier_fusion'

    valid_cbeg_results = [result for result in cbeg_compiler.cbeg_results
                          if result.folder_name == universal_config
                          and result.mutual_info_percentage == 100]
    baseline_results = [result for result in baseline_compiler.baseline_results
                        if result.mutual_info_percentage == 100]
    ciel_results = [result for result in ciel_compiler.ciel_results if
                   result.mutual_info_percentage == 100.0]

    all_results = valid_cbeg_results + baseline_results + ciel_results

    baselines_names = [result.classifier_name for result in baseline_results]

    classification_metrics = CLASSIFICATION_METRICS.copy()
    color_palette = ['blue', 'red', 'orange', 'green', 'purple']

    if DATASETS_INFO[dataset]['nlabels'] >= 3:
        classification_metrics[0] = "Accuracy / Recall"
        classification_metrics.remove("Recall")
        color_palette.remove('blue')

    classifiers = ['CBEG (general)'] + baselines_names + ['CIEL']
    classifiers_ylabel = []
    metric_names = []
    metric_values = []

    for metric_name in classification_metrics:
        classifiers_ylabel += classifiers
        metric_names += [metric_name] * len(all_results)

    if DATASETS_INFO[dataset]['nlabels'] < 3:
        metric_values += [result.mean_accuracy for result in all_results]
    metric_values += [result.mean_recall for result in all_results]
    metric_values += [result.mean_precision for result in all_results]
    metric_values += [result.mean_f1 for result in all_results]
    metric_values += [result.mean_auc for result in all_results]

    dict_values = {
        'Value': metric_values, 'Metric name': metric_names,
        'Classifier': classifiers_ylabel
    }
    sns.set_style("darkgrid")
    _, ax = plt.subplots(figsize=(13, 6.5))

    sns.scatterplot(
        data=dict_values, x="Value", y="Classifier",
        alpha=0.5, hue="Metric name", s=120, ax=ax, palette=color_palette,
    )
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
                "blood",
                "normal_2_class",
                "normal_3_class",
                "electricity",
                "elipses",
                "rectangles",
                "german_credit",
                "contraceptive",
                "wine",
                "wdbc",
                "pima",
                "iris",
                "heart", ]
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
