import numpy as np
from scipy.stats import pearsonr


def correlation_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def cluster_perception_diff(labels, ratings):
    """
    Measure perceptual separation across clusters.
    """

    results = {}

    for c in np.unique(labels):
        if c == -1:
            continue

        results[int(c)] = float(np.mean(ratings[labels == c]))

    return results
