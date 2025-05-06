import re
from typing import Optional
from processors.base_classifiers_compiler import BaseClassifierResult
from processors.data_reader import CLASSIFICATION_METRICS, N_FOLDS
#
class SingleCielResult(BaseClassifierResult):

    def __init__(self, training_info_folds: list[str], test_info_folds: list[str]):
        self.n_folds = N_FOLDS

        self.cluster_classification_folds = []
        self.classification_results_folds = []

        for fold in range(self.n_folds):
            self.extract_training_information(training_info_folds[fold])
            self.extract_test_information(test_info_folds[fold])

        self.calc_mean_and_std_metrics()

    def extract_training_information(self, content_training: str):
        # The training results include the external clustering metrics,
        # the base classifier, the optimal clusterer, the parameters and
        # the number of iterations and particles.

        # Get base classifier
        self.base_classifier = self._get_base_classifier(content_training)

        # Get the external metrics
        external_metrics = ['adjusted_rand_score', 'normalized_mutual_info_score',
                            'v_measure_score', 'fowlkes_mallows_score']
        self.external_clustering_values = self._get_clustering_metrics(
                content_training, external_metrics)

        # Get the internal metrics
        internal_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz_score']
        self.internal_clustering_values = self._get_clustering_metrics(
                content_training, internal_metrics)

        # Get the labels by cluster

        # Get classifier params TODO

    def _get_clustering_metrics(
        self, content_training: str, metrics_list: list[str]) -> dict[str, float]:

        # adjusted_rand_score: 0.007122438111881982
        # normalized_mutual_info_score: 0.021796835872478413
        # v_measure_score: 0.021796835872478413
        # fowlkes_mallows_score: 0.47747867571420666
        metric_values = {}

        for metric_name in metrics_list:
            found_metric_values = re.findall(rf"{metric_name}:\s.+", content_training)

            if found_metric_values:
                value = float(found_metric_values[0].split(": ")[1])
                metric_values[metric_name] = value
            else:
                metric_values[metric_name] = 0.0

        return metric_values

    def _get_base_classifier(self, content_training: str) -> Optional[str]:
        found_strings = re.findall(r"Base classifier:\s.+", content_training)

        if found_strings:
            base_classifier = found_strings[0].split(": ")[1].strip()
            return base_classifier
        else:
            return None

    def extract_test_information(self, content_test: str) -> None:
        # The test results extraction is similar to the base classifiers.
        classification_results_fold = self.get_classification_metrics(content_test)

        self.cluster_classification_folds.append(
            self._get_classification_results_by_cluster(classification_results_fold)
        )
        self.classification_results_folds.append(
            self.get_general_classification_results(classification_results_fold)
        )

    def _get_classification_results_by_cluster(
            self, classification_results: dict[str, list[float]]
        ) -> dict[str, list[float]]:

        classification_by_cluster = {}

        for metric in CLASSIFICATION_METRICS:
            classification_by_cluster[metric] = classification_results[metric][0:-1]

        return classification_by_cluster
