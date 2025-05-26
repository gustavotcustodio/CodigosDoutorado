from dataclasses import dataclass

N_FOLDS = 10

BASE_PATH_FOLDER = "results/{dataset}/mutual_info_{mutual_info_percentage}/{algorithm}/{experiment_folder}"

CLASSIFICATION_METRICS = ["Accuracy", "Recall", "Precision", "F1", "AUC"]

CLASSIFIERS_FULLNAMES = {
    'nb': "Naive Bayes", 'svm': "SVM", 'lr': "Logistic Reg", 'dt': "Decision Tree",
    'rf': "Random Forest", 'gb': "Grad. Boosting", 'xb': "XGBoost",
    'sc_dt': 'S. Clustering (DT)', 'sc_svm': 'S. Clustering (SVM)', 'sc_lr': 'S. Clustering (LR)'}

LABELS_CLASSIFIERS = ['GaussianNB', 'SVC', 'KNeighborsClassifier',
                      'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', ]

RESULTS_FILENAMES = {"cbeg": "results.csv" , "baseline": "results_baseline.csv"}

@dataclass
class DataReader:
    path: str 
    training: bool = True

    def read_data(self):
        self.data = {}

        self.data["test"] = self.read_training_or_test_data("test")
        if self.training:
            self.data["training"] = self.read_training_or_test_data("training")

    def read_training_or_test_data(self, stage: str) -> list[str]:
        text_folds = []
        for fold in range(1, N_FOLDS+1):
            full_filename = f"{self.path}/{stage}_summary/run_{fold}.txt"
            text_file = open(full_filename).read()
            text_folds.append(text_file)

        return text_folds
