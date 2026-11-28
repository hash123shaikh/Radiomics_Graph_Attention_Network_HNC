"""
Feature selectors, always fitted on the training patients of a split.

  mrmr   minimum-redundancy maximum-relevance (FCQ: ANOVA F / mean |r|)
  lasso  L1 logistic regression, C chosen by cross-validated AUC; ranking by |coefficient|, then by the
         order in which features enter along the regularisation path
The univariate filter (AUC ranking with |r| >= 0.8 redundancy removal) is implemented in the scripts.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from hnc_popgraph import config


@dataclass
class SelectionResult:
    method: str
    selected: list
    ranking: list
    native_size: int = None
    info: dict = field(default_factory=dict)

    def to_dict(self):
        return {'method': self.method, 'selected': list(self.selected), 'ranking': list(self.ranking),
                'native_size': self.native_size, 'info': self.info}


# ─── Matrix preparation ──────────────────────────────────────────────────────

def prepare(X):
    """Prefilter + transform used by every data-driven selector.
    Drops (near-)constant features, applies a signed log1p to heavy-tailed features,
    then z-scores. Returns a DataFrame with the same row index."""
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > config.FS_VARIANCE_MIN]
    if config.FS_LOG_TRANSFORM:
        skew = X.skew().abs()
        heavy = skew.index[skew > 2.0]
        X[heavy] = np.sign(X[heavy]) * np.log1p(np.abs(X[heavy]))
    Z = (X - X.mean()) / X.std().replace(0, 1)
    return Z.loc[:, Z.std() > 0]


def _k(k, native, n_features):
    """Final number of features: fixed k, or the native size clipped to FS_K_RANGE."""
    if k is None:
        lo, hi = config.FS_K_RANGE
        k = int(np.clip(native if native else lo, lo, hi))
    return int(min(k, n_features))


def mrmr_ranking(Z, y):
    """mRMR, FCQ scheme (the mrmr-selection default): relevance = ANOVA F statistic,
    redundancy = mean |Pearson r| with the already selected features, score = relevance / redundancy.
    Vectorised re-implementation (identical criterion, ~100x faster for bootstrapping)."""
    from sklearn.feature_selection import f_classif
    F, _ = f_classif(Z.values, y)
    F = np.nan_to_num(F, nan=0.0)
    C = np.abs(np.nan_to_num(np.corrcoef(Z.values, rowvar=False)))
    names = list(Z.columns)
    chosen = [int(np.argmax(F))]
    red_sum = C[:, chosen[0]].copy()
    left = np.ones(len(names), bool); left[chosen[0]] = False
    while left.any():
        score = np.where(left, F / np.maximum(red_sum / len(chosen), 1e-6), -np.inf)
        j = int(np.argmax(score))
        chosen.append(j); left[j] = False
        red_sum += C[:, j]
    return [names[i] for i in chosen]


def sel_mrmr(X, y, k, seed=0, **_):
    Z = prepare(X)
    rank = mrmr_ranking(Z, y)
    kk = _k(k, 10, len(rank))
    return SelectionResult('mrmr', rank[:kk], rank, None)


def sel_lasso(X, y, k, seed=0, **_):
    Z = prepare(X)
    Cs = np.logspace(-2.5, 1, 25)
    n_splits = int(max(2, min(5, np.bincount(y).min())))
    cv = LogisticRegressionCV(Cs=Cs, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                              penalty='l1', solver='liblinear', scoring='roc_auc',
                              class_weight='balanced', max_iter=5000, n_jobs=1).fit(Z.values, y)
    best_C = float(cv.C_[0])
    coef = pd.Series(np.abs(cv.coef_.ravel()), index=Z.columns)
    # entry point along the regularisation path (earlier entry = stronger feature)
    entry = pd.Series(np.inf, index=Z.columns)
    for C in Cs:
        m = LogisticRegression(C=C, penalty='l1', solver='liblinear', class_weight='balanced',
                               max_iter=5000).fit(Z.values, y)
        nz = Z.columns[np.abs(m.coef_.ravel()) > 1e-10]
        entry[nz] = np.minimum(entry[nz], C)
    nonzero = coef[coef > 1e-10].sort_values(ascending=False)
    rest = entry.drop(nonzero.index).sort_values(kind='stable')
    rank = list(nonzero.index) + list(rest.index)
    kk = _k(k, len(nonzero), len(rank))
    return SelectionResult('lasso', rank[:kk], rank, int(len(nonzero)),
                           {'best_C': best_C, 'n_nonzero': int(len(nonzero)),
                            'coef': {f: float(v) for f, v in nonzero.items()}})


SELECTORS = {'mrmr': sel_mrmr, 'lasso': sel_lasso}


def run_selector(method, X, y, k, seed=0, **_):
    """X: DataFrame of training features, y: 0/1 array -> SelectionResult (ranking = all features, best first)."""
    if method not in SELECTORS:
        raise ValueError(f"Unknown selector '{method}'. Use one of {list(SELECTORS)}")
    return SELECTORS[method](X, np.asarray(y).astype(int), k, seed=seed)
