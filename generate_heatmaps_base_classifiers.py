import os
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import base

BASE_CLASSIFIERS = ['xb', 'gb', 'rf', 'nb', 'svm', 'lr', 'dt']
DATASETS = [ "german_credit", "australian_credit", "contraceptive", "heart", "iris", "pima", "wdbc", "wine"]
CLASSIFICATION_METRICS = ["Accuracy", "Recall", "Precision", "F1"]

def get_all_classification_metrics(text_experiment_data: str) -> dict[str, list[float]]:

    # Dictionary where the values of accuracy, recall, precision and F1 are stored
    dict_classification_results = {}

    for metric in CLASSIFICATION_METRICS:
        # All patterns found in text corresponding to the searched metric
        found_metric_patterns = re.findall(fr"{metric}: [0-9]\.[0-9]+", text_experiment_data)
        dict_classification_results[metric] = [float(pattern.split(": ")[1]) for pattern in found_metric_patterns]
        # Average value of the metric
        # dict_classification_metrics[f"Total {metric}"] = float(found_metric_patterns[-1].split(": ")[1])
    return dict_classification_results


for dataset in DATASETS:
    indexes = []
    classification_results = {}
    classification_results["Accuracy"] = []
    classification_results["Precision"] = []
    classification_results["Recall"] = []
    classification_results["F1"] = []

    for base_classifier in BASE_CLASSIFIERS:
        for mutual_info in [100.0, 75.0, 50.0]:

            acc_values = []
            precision_values = []
            recall_values = []
            f1_values = []

            for i in range(1, 11):
                full_filename = f"./results/{dataset}/mutual_info_{mutual_info}/baselines/{base_classifier}/test_summary/run_{i}.txt"

                text_file = open(full_filename).read()

                dict_classification = get_all_classification_metrics(text_file)

                acc_values.append(dict_classification["Accuracy"][0])
                precision_values.append(dict_classification["Precision"][0])
                recall_values.append(dict_classification["Recall"][0])
                f1_values.append(dict_classification["F1"][0])

            indexes.append(f"{base_classifier} ({mutual_info} %)")
            classification_results["Accuracy"].append( np.mean(acc_values) )
            classification_results["Precision"].append( np.mean(precision_values) )
            classification_results["Recall"].append( np.mean(recall_values) )
            classification_results["F1"].append( np.mean(f1_values) )

    data = np.vstack((
        classification_results["Accuracy"],
        classification_results["Precision"],
        classification_results["Recall"],
        classification_results["F1"])
    ).T

    sns.heatmap(
        data, annot=True, cmap='Blues', vmin=0, vmax=1,
        xticklabels=["Accuracy", "Recall", "Precision", "F1"], yticklabels=indexes
    )

    heatmaps_folder = f"./results/{dataset}/ablation_results"
    os.makedirs(heatmaps_folder, exist_ok=True)

    filename = os.path.join(
        heatmaps_folder, f"heatmap_ablation_baselines.png"
    )

    plt.xlabel("Classification Metric", fontdict={'weight': 'bold'})
    plt.ylabel("Experiment variation", fontdict={'weight': 'bold'})
    # Save the heat map
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()
    print(filename, "salvo com sucesso.")
