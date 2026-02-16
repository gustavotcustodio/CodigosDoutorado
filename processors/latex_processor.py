import sys
import re
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

def get_best_mean_std(latex_table, pattern):
    matches = pattern.findall(latex_table)
    matches = [tuple(match[:-1].split('(')) for match in matches]
    mean_values = [float(mean) for mean, _ in matches]
    max_mean = max(mean_values)

    min_std = min(int(std) for mean, std in matches
                  if float(mean) == max_mean)
    return max_mean, min_std


def update_best_mean_and_std(column_val, max_mean, min_std):
    if "(" not in column_val:
        return 0, 0
    mean, std = tuple(column_val[:-1].split('('))
    meanf, stdf = float(mean), float(std)

    max_meanf, min_stdf = float(max_mean), float(min_std)

    if meanf > max_meanf:
        max_meanf = float(meanf)
        max_mean = mean
        min_stdf = int(stdf)
        min_std = std
    elif meanf == max_meanf:
        if stdf < min_stdf:
            min_stdf = int(stdf)
            min_std = std
    return max_mean, min_std


def highlight_best(latex_table):
    rows_table = latex_table.strip().split('\n')
    max_mean_acc, max_mean_rec, max_mean_prec = 0, 0, 0
    max_mean_f1, max_mean_auc = 0, 0
    min_std_acc, min_std_f1, min_std_auc = float('inf'), float('inf'), float('inf')
    min_std_rec, min_std_prec = float('inf'), float('inf')
    # First valid row = 9
    # Last valid rpw = 48
    n_columns = 0

    for row in rows_table[9:49]:
        row = re.sub(r' \\.*$', '', row)
        columns = re.split(r"\s+&\s+", row)
        assert len(columns) == 11 or len(columns) == 10, "Error. Invalid row size."
        n_columns = len(columns)

        first_col = 6
        max_mean_acc, min_std_acc = update_best_mean_and_std(
            columns[first_col], max_mean_acc, min_std_acc)

        max_mean_rec, min_std_rec = update_best_mean_and_std(
            columns[first_col+1], max_mean_rec, min_std_rec)
        max_mean_prec, min_std_prec = update_best_mean_and_std(
            columns[first_col+2], max_mean_prec, min_std_prec)
        max_mean_f1, min_std_f1 = update_best_mean_and_std(
            columns[first_col+3], max_mean_f1, min_std_f1)
        max_mean_auc, min_std_auc = update_best_mean_and_std(
            columns[first_col+4], max_mean_auc, min_std_auc)

    best_clf_vals = [f'{max_mean_acc}({min_std_acc})',
                     f'{max_mean_rec}({min_std_rec})',
                     f'{max_mean_prec}({min_std_prec})',
                     f'{max_mean_f1}({min_std_f1})',
                     f'{max_mean_auc}({min_std_auc})']

    for idx_row in range(9,49):
        row = rows_table[idx_row]
        columns = re.split(r"\s+&\s+", row)

        for i in range(6, n_columns):
            best = best_clf_vals[i-6]
            if f'{best_clf_vals[i-6]}' in columns[i]:
                columns[i] = columns[i].replace(best, f'\\cellcolor{{blue!25}}\\textbf{{{best}}}')
        rows_table[idx_row] = ' & '.join(columns)
    return '\n'.join(rows_table)


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

    variation = str(result.experiment_variation)

    if '2' in str(variation):
        selection_clf = "CV"

    elif '5' in str(variation):
        variation = variation.replace("5", "2")
        variation = "".join(sorted(variation))
        selection_clf = "PSO"
    else:
        selection_clf = "-"
    variation = '-'.join(variation)

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

    # Highlight best results in latex table
    latex_table = highlight_best(latex_table)

    folder_latex = os.path.join('results', 'latex')
    os.makedirs(folder_latex, exist_ok=True)

    with open(os.path.join(folder_latex, f'{dataset}.tex'), 'w') as f:
        # content_file = "\n\n".join(tables_dataset)
        f.write(latex_table)
        print(f'{dataset}.tex saved.')
