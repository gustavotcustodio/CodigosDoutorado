import sys
import os
import math
# import re
from processors.data_reader import CLASSIFICATION_METRICS
from dataset_loader import DATASETS_INFO

PATH_LATEX_TEMPLATE = "templates/modelo_tabela_completa.tex"

dict_abbrevs = {
    'Weighted Membership': 'WM',
    'Meta Classifier': 'MC',
    'Majority Voting': 'MV',
    "DBC + Rand": "DBC+R",
    "Rand Score": "Rand",
    "DBC": "DBC",
    "": "-",
}

def format_mean_std(mean, std) -> str:
    n_valid_digits = 0
    msd = std

    # 0.0043
    if std == 0:
        return str(mean)

    while msd < 1:
        n_valid_digits += 1
        msd *= 10

    msd = math.floor(msd)
    mean_sd = round(mean, n_valid_digits)
    # std_sd = round(std, n_valid_digits)
    return f'{mean_sd:.{n_valid_digits}f}({msd})'


def get_mean_std_metric(result, metric: str) -> str:
    attr_mean = f"mean_{metric.strip().lower()}"
    attr_std = f"std_{metric.strip().lower()}"

    if not hasattr(result, attr_mean) or not hasattr(result, attr_std):
        raise ValueError(f"{metric} not found.")

    mean = getattr(result, attr_mean)
    std = getattr(result, attr_std)

    mean_std = format_mean_std(mean, std)

    return mean_std


def complete_row_table(result, latex_table):

    variation = result.experiment_variation

    if '2' in str(variation):
        selection_clf = "CV"

    elif '5' in str(variation):
        variation = str(variation).replace("5", "2")
        variation = "".join(sorted(variation))
        variation = int(str(variation))
        selection_clf = "PSO"
    else:
        selection_clf = "-"

    cluster_selection = dict_abbrevs[result.cluster_selection_strategy]
    mutual_info = int(result.mutual_info_percentage)
    fusion_strategy = dict_abbrevs[result.fusion_strategy]

    metric_values = []

    mean_std_n_clusters = get_mean_std_metric(result, 'n_clusters')
    metric_values.append(mean_std_n_clusters)

    for metric in CLASSIFICATION_METRICS:

        mean_std_metric = get_mean_std_metric(result, metric)
        metric_values.append(mean_std_metric)

    experiment_name = f'{variation} & {cluster_selection} &' + \
            f' {selection_clf} & {mutual_info} & {fusion_strategy}'
    experiment_name = experiment_name.replace("  ", " ")

    metrics_row = " & ".join(metric_values)

    complete_row = f'{experiment_name} & {metrics_row} \\\\'
    pattern_to_replace = \
        f'[{variation}_{cluster_selection}_{selection_clf}_{mutual_info}_{fusion_strategy}]'
    latex_table = latex_table.replace(pattern_to_replace, complete_row)

    return latex_table


def create_latex_table(cbeg_results, dataset: str):
    try:
        initial_template = open(PATH_LATEX_TEMPLATE).read()
    except:
        print("Error. Latex template not found.")
        sys.exit(1)

    latex_table = initial_template

    for result in cbeg_results:   
        latex_table = complete_row_table(result, latex_table)

    latex_table = latex_table.replace("[dataset-label]", dataset)
    latex_table = latex_table.replace(
        "[dataset-caption]", DATASETS_INFO[dataset]['full_name'])

    folder_latex = os.path.join('results', 'latex')
    os.makedirs(folder_latex, exist_ok=True)

    with open(os.path.join(folder_latex, f'{dataset}.tex'), 'w') as f:
        # content_file = "\n\n".join(tables_dataset)
        f.write(latex_table)
        print(f'{dataset}.tex saved.')
