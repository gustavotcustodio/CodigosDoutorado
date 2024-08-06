import re
import os
import sys
import numpy as np
import pandas as pd
from extrair_resultados import get_classification_metrics
import matplotlib.pyplot as plt

def count_attribs_no_clusters(file_content):
    pattern = r"[0-9]+ atributos selecionados"
    information_n_attribs = re.findall(pattern, file_content)[0]
    n_attribs = information_n_attribs.split(" ")[0]
    n_attribs = int(n_attribs)
    return n_attribs


def count_attribs_by_cluster(file_content):
    pattern = re.compile(r"^Cluster [0-9]+: \[[0-9\ \n]+\]", re.MULTILINE)
    found_clusters_features = re.findall(pattern, file_content)

    n_attribs_clusters = []

    for features_text in found_clusters_features:
        features_text = features_text.split(": ")[1][1:-1]
        features = features_text.split(" ")
        features = [int(feature) for feature in features if feature.strip() != ""]

        n_attribs_clusters.append(len(features))
    return n_attribs_clusters

# df = pd.DataFrame([['A', 10, 20, 10, 30, 40, 50], ['B', 20, 25, 15, 25, 20, 40]],
#                   columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4']) 

def plot_comparison_and_save(results_mutual_info, baseline_name, filename):
    """ metrics_algorithm contains 3 keys, which are averages for
        f1, accuracy and n_attribs"""
    fig, ax = plt.subplots(layout='constrained')

    offset = 0

    x_ticklabels = []

    max_features = 0

    mutual_info_variations = ["all features", "mutual_info_75", "mutual_info_50"]

    for k in mutual_info_variations:
        metrics_baseline = results_mutual_info[k][0]
        metrics_cbeg = results_mutual_info[k][1]
        max_features = max(metrics_baseline["n_attribs"], metrics_cbeg["n_attribs"], max_features)

    for k in mutual_info_variations:
        metrics_baseline = results_mutual_info[k][0]
        metrics_cbeg = results_mutual_info[k][1]

        # Check if is a nan
        if metrics_cbeg["n_attribs"] != metrics_cbeg["n_attribs"]:
            metrics_cbeg["n_attribs"] = metrics_baseline["n_attribs"]

        if "50" in k:
            mutual_info_label = "\n\n\n50% of\nmutual\ninformation"
        elif "75" in k:
            mutual_info_label = "\n\n\n75% of\nmutual\ninformation"
        else:
            mutual_info_label = "\n\n\nall\nfeatures"

        x_ticklabels += [
                         "",
                         f"{round(metrics_baseline['n_attribs'], 2)}\nfeatures",
                         "",
                         mutual_info_label,
                         "",
                         f"{round(metrics_cbeg['n_attribs'], 2)}\nfeatures",
                         "", "", "", "", ""
                        ]
        label1 = f"Accuracy {baseline_name.replace("_", " ").upper()}"
        label2 = f"F1 {baseline_name.replace("_", " ").upper()}"
        label3 = f"N. features {baseline_name.replace("_", " ").upper()}"
        label4 = "Accuracy CBEG"
        label5 = "F1 CBEG"
        label6 = "N. features CBEG"

        b1 = ax.bar(0 + offset, metrics_baseline["accuracy"] , width=1, color="slateblue", label=label1)
        b2 = ax.bar(1 + offset, metrics_baseline["f1"] , width=1, color="peru", label=label2)
        b3 = ax.bar(2 + offset, metrics_baseline["n_attribs"] / max_features, width=1, color="springgreen", label=label3)

        offset += 4

        b4 = ax.bar(0 + offset, metrics_cbeg["accuracy"] , width=1, color="cadetblue", label=label4)
        b5 = ax.bar(1 + offset, metrics_cbeg["f1"] , width=1, color="darksalmon", label=label5)
        b6 = ax.bar(2 + offset, metrics_cbeg["n_attribs"] / max_features, width=1, color="darkseagreen", label=label6)

        offset += 7

    ax.grid(linestyle='-', linewidth=0.5)

    ax.set_xticks([i for i in range(len(x_ticklabels))])
    ax.set_xticklabels(x_ticklabels)
    ax.set_ylabel("Metric values normalized between 0 and 1")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),  loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True)

    plt.savefig(output_filename)
    print(output_filename, "salvo com sucesso.")


def get_metrics_algorithm(count_attribs_function, path_files):
    acc_scores, f1_scores, n_attribs_all_folds  = [], [], []

    metrics_algorithm = {}

    filelist = os.listdir(path_files)
    filelist = [f for f in filelist if ".png" not in f]

    for filename in filelist:
        full_filename = f"{path_files}/{filename}"
        file_content = open(full_filename).read()

        # Calculate the average number of attributes by cluster
        accuracy, _, _, f1 = get_classification_metrics(file_content)

        n_attribs = count_attribs_function(file_content)

        acc_scores.append(accuracy)
        f1_scores.append(f1)

        if type(n_attribs) == list:
            n_attribs_all_folds += n_attribs
        else:
            n_attribs_all_folds.append(n_attribs)

    # Save the mean values from relevant metrics to plot the graphs
    metrics_algorithm["accuracy"] = np.mean(acc_scores)
    metrics_algorithm["f1"] = np.mean(f1_scores)
    metrics_algorithm["n_attribs"] = np.mean(n_attribs_all_folds)

    return metrics_algorithm


if __name__ == "__main__":
    # dataset = "australian_credit"
    # algorithms_compared = ["xgboost", "CBEG_silhouette"]

    results_mutual_info = {}

    dataset = sys.argv[1]
    algorithms_compared = sys.argv[2].split(",")

    for mutual_info in ["10_runs", "mutual_info_75", "mutual_info_50"]:

        metrics_info = []

        for algorithm in algorithms_compared:

            path_files = f"./resultados/{dataset}/{mutual_info}/{algorithm}"

            if algorithm in ["gradient_boosting", "random_forest", "svm", "xgboost"]:
                count_attribs_function = count_attribs_no_clusters
            else:
                count_attribs_function = count_attribs_by_cluster

            metrics_info.append(get_metrics_algorithm(count_attribs_function, path_files))

        results_mutual_info[mutual_info] = metrics_info

    results_mutual_info["all features"] = results_mutual_info.pop("10_runs")

    output_filename = f"./resultados/comparacao_features/{dataset}_features.png"
    plot_comparison_and_save(results_mutual_info, algorithms_compared[0], output_filename)
