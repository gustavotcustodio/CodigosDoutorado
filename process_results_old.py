
"""

def save_confusion_matrix(y_true, y_pred, filename, show=False):
    # Save the confusion matrix for the fold
    cm = confusion_matrix(y_true, y_pred)
    cm_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(y_true)
    )

    cm_disp.plot(cmap=plt.cm.Blues)
    if show:
        plt.show()
    plt.savefig(filename)
    plt.clf()
    plt.close()
    print(filename, "salvo com sucesso.")


def create_confusion_matrices(experiments_results: CbegExperimentData) -> None:
    # Prediction: 1, Real label: 1, Votes by cluster: [1 1], Weights: [0.5 0.5]
    cm_folder = os.path.join(experiments_results.experiment_folder, "confusion_matrix")
    os.makedirs(cm_folder, exist_ok=True)  # Confusion matrix folder

    all_y_pred, all_y_true = experiments_results.get_labels_and_predictions_folds()

    # Create a confusion matrix for each fold and a general one
    for fold, fold_data in enumerate(experiments_results):
        filename = os.path.join(cm_folder, f"cm_fold_{fold+1}.png")
        save_confusion_matrix(fold_data.y_true, fold_data.y_pred, filename)

    # Save the general confusion matrix
    filename = os.path.join(cm_folder, "cm_all_folds.png")
    save_confusion_matrix( all_y_pred, all_y_true, filename)
    
def plot_clusters_and_labels(text_file_folds: list[str], experiment_result_folder: str):

    for fold, text_file in enumerate(text_file_folds):

        clusters = get_clusters_predictions()
        base_classifiers = get_base_classifiers()

        data_clusters_labels = {
            "cluster": clusters,
            "label": y_pred,
            "base_classifier": base_classifiers,
        }
        # Variáveis hue=label y=cluster, x=base classifier
        df_clusters_labels = pd.DataFrame(data_clusters_labels) 

        sns.catplot(data=df_clusters_labels, x="base_classifier", y="cluster", hue="label", kind="swarm")
        plt.show()
"""


class CbegExperimentData:

    def __init__(self, content_file_folds_training: list[str],
                 content_file_folds_test: list[str], experiment_folder: str
                 ):
        self.content_file_folds_training = content_file_folds_training
        self.content_file_folds_test = content_file_folds_test
        self.experiment_folder = experiment_folder

        self.idx = 0

        self.experiments_folds = []

        self.y_true = []
        self.y_pred = []

        for fold in range(len(content_file_folds_test)):
            content_fold_training = content_file_folds_training[fold]
            content_fold_test = content_file_folds_test[fold]

            self.experiments_folds.append(
                CbegFoldData(content_fold_training, content_fold_test, experiment_folder, fold+1)
            )

    def plot_instance_distribution_and_accuracy_clusters(self):
        """ Plot the distribution of training instances per cluster and the
            validation accuracy per cluster.
        """
        from collections import Counter

        for exp_fold in self.experiments_folds:
            for c, cluster_labels in enumerate(exp_fold.labels_by_cluster):
                # Count the number of labels by cluster
                label_count = Counter(cluster_labels)

                n_majority = max(label_count)
                n_minority = min(label_count)

                # x axis: number of majority class X minority class
                # y axis: accuracy
                f"{n_majority}/{n_minority}"


    def get_labels_and_predictions_folds(self) -> tuple[list[int], list[int]]:
        # Create a confusion matrix for each fold and a general one
        if self.y_true  and self.y_pred:
            return self.y_pred, self.y_true

        for _, experiment_fold in enumerate(self.experiments_folds):
            y_pred_fold, y_true_fold = experiment_fold.get_labels_and_predictions()

            self.y_pred += y_pred_fold
            self.y_true += y_true_fold

        return self.y_pred, self.y_true

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.experiments_folds):
            raise StopIteration
        idx = self.idx
        self.idx += 1
        return self.experiments_folds[idx]

    def __str__(self) -> str:
        str_content = ""

        for fold, data_fold in enumerate(self.experiments_folds):

            str_content += f"Fold {fold+1}: \n {data_fold.__str__()}"

        return str_content



class CbegFoldData:
    def __init__(self, content_fold_training: str, content_fold_test: str,
                 experiment_folder: str, fold: int
                 ):
        self.content_fold_test = content_fold_test
        self.content_fold_training = content_fold_training
        self.experiment_folder = experiment_folder
        self.fold = fold
        self.y_true, self.y_pred = [], []

        self.n_clusters = self.get_n_clusters()
        self.labels_by_cluster = self.get_labels_by_cluster_training()
        self.base_classifiers_by_cluster = self.get_base_classifiers_by_cluster()
        # self.plot_clusters_and_labels(fold)

    def get_n_clusters(self) -> int:
        clusters_pattern = re.findall(r"Cluster [0-9]", self.content_fold_test)[-1]

        n_clusters = int(clusters_pattern.split(" ")[1]) + 1
         
        return n_clusters

    def get_labels_by_cluster_training(self) -> dict[int, list[int]]:
        # Labels: [0 0 0  0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0
        #  0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0]
        found_strings = re.findall(r"Labels:\s\[[0-9\s\n]+\]", self.content_fold_training)
        labels_by_clusters = {}

        for c in range(self.n_clusters):
            str_labels = found_strings[c].replace("Labels: [", "").replace("]", "")
            labels = re.split(r"[\n\s]+", str_labels)
            labels_by_clusters[c] = [int(lbl) for lbl in labels]
        return labels_by_clusters            


    def get_base_classifiers_by_cluster(self) -> dict[int, str]:
        found_strings = re.findall(r"Base classifier:\s.+\n", self.content_fold_training)

        base_classifiers_by_cluster = {}
        for c in range(self.n_clusters):
            base_classifiers_by_cluster[c] = found_strings[c].split(": ")[1].strip()

        return base_classifiers_by_cluster

    def plot_clusters_and_labels(self, fold: int) -> None:
        clusters = [] 
        labels = []
        base_classifiers = []

        os.makedirs(f"{self.experiment_folder}/catplot", exist_ok=True)

        for c in range(self.n_clusters):

            clusters += [c+1] * len(self.labels_by_cluster[c])
            labels += self.labels_by_cluster[c] 
            base_classifiers += [self.base_classifiers_by_cluster[c]] * len(self.labels_by_cluster[c])
        
        data_clusters_labels = {
            "Cluster": np.array(clusters, dtype="str"),
            "Label": labels,
            "Base Classifier": base_classifiers,
        }
        # Variáveis hue=label y=cluster, x=base classifier
        df_clusters_labels = pd.DataFrame(data_clusters_labels) 
        fig_catplot = sns.catplot(data=df_clusters_labels, x="Base Classifier", y="Cluster",
                                  hue="Label", kind="swarm", s=20)

        fig_catplot.figure.savefig(f"{self.experiment_folder}/catplot/catplot_{fold}.png")

    def get_labels_and_predictions(self) -> tuple[list[int], list[int]]:
        # Extract the true labels and predicted labels

        if self.y_true and self.y_pred:
            return self.y_pred, self.y_true

        pattern_prediction = r"Prediction: [0-9], Real label: [0-9]"

        predicted_labels = []
        true_labels = []
        
        label_prediction_patterns = re.findall(pattern_prediction, self.content_fold_test)

        for found_predictions in label_prediction_patterns:
            predicted_label_str, true_label_str = found_predictions.split(", ")
            y_pred = int(predicted_label_str.split(": ")[1])
            y_true = int(true_label_str.split(": ")[1])

            predicted_labels.append(y_pred)
            true_labels.append(y_true)

        self.y_pred = predicted_labels
        self.y_true = true_labels

        return self.y_pred, self.y_true

    def __str__(self):
        return f"y_pred: {self.y_pred}\n"


