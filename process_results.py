import os
import re
from dataset_loader import DATASETS_INFO
from processors.base_classifiers_compiler import BaseClassifiersCompiler, BaseClassifierResult
from processors.cbeg_compiler import SingleCbegResult, CbegResultsCompiler
from processors.ciel_compiler import SingleCielResult, CielCompiler
from processors.data_reader import DataReader, CLASSIFIERS_FULLNAMES


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

        print(dataset)
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
        cbeg_compilation.plot_clusterers_heatmap()
        cbeg_compilation.plot_classification_heatmap()
        cbeg_compilation.plot_clusters_scatterplot()


def process_base_results(datasets: list[str], mutual_info_percentages: list[float]):
    for dataset in datasets:
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
        classifier_compiler.plot_classification_heatmap()


def process_ciel_results(datasets: list[str], mutual_info_percentages: list[float]):
    for dataset in datasets:
        ciel_results = []

        for mutual_info in mutual_info_percentages:

            path = f"./results/{dataset}/mutual_info_{mutual_info}/ciel"
            loader = DataReader(path, training=True)
            loader.read_data()

            ciel_result = SingleCielResult(loader.data["training"], loader.data["test"],
                                           mutual_info_percentage=mutual_info)

            ciel_results.append(ciel_result)

        ciel_compiler = CielCompiler(ciel_results, dataset)
        ciel_compiler.plot_classification_heatmap()


def filter_no_experim_datasets(datasets: list[str]) -> list[str]:
    valid_datasets = []
    results_list = os.listdir("./results")

    for dataset in datasets:
        if dataset in results_list:
            valid_datasets.append(dataset)
    return valid_datasets

def main():
    datasets = ["australian_credit", "german_credit", "contraceptive", 
                "heart", "iris", "pima", "wdbc", "wine"]
    datasets = filter_no_experim_datasets(datasets)

    mutual_info_percentages = [100.0, 75.0, 50.0]

    process_ciel_results(datasets, mutual_info_percentages)
    # process_cbeg_results(datasets, mutual_info_percentages)
    # process_base_results(datasets, mutual_info_percentages) 

if __name__ == "__main__":
    main()
