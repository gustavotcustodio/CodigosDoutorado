import sys
import os
# import re
from processors.data_reader import CLASSIFICATION_METRICS

dict_abbrevs = {
    'Weighted Membership': 'Weighted',
    'Meta Classifier': 'Meta',
    'Majority Voting': 'Voting',
    "DBC + Rand": "DBC+R",
    "Rand Score": "Rand",
    "DBC": "DBC",
    "": "-",
}

def format_significant_digits(mean, std) -> tuple[float, float]:
    n_valid_digits = 0
    msd = std
    # 0.0043

    while msd < 1 and msd > 0:
        n_valid_digits += 1
        msd *= 10

    mean_sd = round(mean, n_valid_digits)
    std_sd = round(std, n_valid_digits)
    return mean_sd, std_sd


def get_metric(result, metric: str):
    attr_mean = f"mean_{metric.strip().lower()}"
    attr_std = f"std_{metric.strip().lower()}"

    if not hasattr(result, attr_mean) or not hasattr(result, attr_std):
        raise ValueError(f"{metric} not found.")

    mean = getattr(result, attr_mean)
    std = getattr(result, attr_std)

    mean_sd, std_sd = format_significant_digits(mean, std)

    return mean_sd, std_sd


def complete_row_table(result, latex_table):

    variation = result.experiment_variation

    if '2' in str(variation):
        selection_clf = "Crossval"

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

    for metric in CLASSIFICATION_METRICS:
        mean, std = get_metric(result, metric)
        mean_std_metric = f'{mean} $\\pm$ {std}'

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
    path_template = "templates/modelo_tabela_completa.tex"

    try:
        initial_template = open(path_template).read()
    except:
        print("Error. Latex template not found.")
        sys.exit(1)

    # tables_dataset = []

    latex_table = initial_template

    for result in cbeg_results:   
        latex_table = complete_row_table(result, latex_table)

    folder_latex = os.path.join('results', 'latex')
    os.makedirs(folder_latex, exist_ok=True)

    with open(os.path.join(folder_latex, f'{dataset}.tex'), 'w') as f:
        # content_file = "\n\n".join(tables_dataset)
        f.write(latex_table)
        print(f'{dataset}.tex saved.')
