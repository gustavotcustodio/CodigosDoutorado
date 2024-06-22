import numpy as np
from sklearn.feature_selection import mutual_info_classif


def get_attribs_by_mutual_info(X, y, mutual_info_percent):
    if mutual_info_percent >= 100 or len(X) == 1:
        return [i for i in range(X.shape[1])]

    repeated_y = np.diff(sorted(y)) == 0
    if np.any(repeated_y):
        mutual_info = mutual_info_classif(X, y)
    else:
        return [i for i in range(X.shape[1])]

    mutual_info_percent /= 100

    if np.all(mutual_info == 0):
        return [i for i in range(X.shape[1])]

    norm_mutual_info = mutual_info / np.sum(mutual_info)

    sorted_attrs = mutual_info.argsort()[::-1]

    cumsum_info = norm_mutual_info.cumsum()

    max_attr = np.where(cumsum_info >= mutual_info_percent)[0][0]
    selected_attrs = sorted_attrs[0 : (max_attr + 1)]
    return selected_attrs
