"""AUC with a guard for single-class samples, and its percentile bootstrap confidence interval."""
import numpy as np
from sklearn.metrics import roc_auc_score

from hnc_popgraph import config


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=None, level=None, seed=None):
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    level = level or config.CONFIDENCE_LEVEL
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    aucs = []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) == 2:
            aucs.append(roc_auc_score(y_true[i], y_prob[i]))
    if not aucs:
        return float('nan'), float('nan')
    a = (1 - level) / 2
    return float(np.percentile(aucs, 100 * a)), float(np.percentile(aucs, 100 * (1 - a)))
