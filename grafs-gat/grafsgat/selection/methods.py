"""
Feature-selection methods compared in the study.

Every selector takes the GTV radiomic features of TRAINING patients (DataFrame X,
labels y) and returns a SelectionResult with
    selected : the features used as GAT node features (length k, or native size)
    ranking  : ALL candidate features, best first (used for Kendall's W stability)
    info     : method-specific diagnostics (feature graph, clusters, coefficients ...)

Methods
  table_s3          fixed Table S3 features of Bae et al. (2025) (reference, not data driven)
  all               every non-constant feature (no selection; control)
  mrmr              minimum-redundancy maximum-relevance (FCQ: F-test / mean |r|)
  lasso             L1 logistic regression, C chosen by 5-fold CV AUC
  boruta            Boruta all-relevant selection (RF + shadow features, binomial tests)
  rfe_rf            recursive feature elimination with a random forest
  graph_fs_cc       Graph-FS, connected components + degree centrality   (label free)
  graph_fs_spectral Graph-FS, spectral clustering of the feature graph    (label free)
  graph_fs_hybrid   graph-guided relevance (proposed): Graph-FS feature graph -> >=2k spectral
                    communities (label free) -> most relevant feature per community -> top k

Graph-FS follows Moradmand et al., Sci Rep 2025;15:27995:
  feature graph: edge if |Pearson r| > mean + alpha*std of all pairwise |r| (alpha = 0.3);
  isolated features are linked to their most similar feature;
  CC: one representative per connected component = highest degree centrality
      (alpha raised from 0.3 until >= k components exist; see sel_graph_fs_cc);
  spectral: Laplacian L = D - A, eigenvectors of the smallest eigenvalues, k-means,
            representative = feature closest to its cluster centroid in the embedding.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from grafsgat import config

LABEL_FREE = {'table_s3', 'all', 'graph_fs_cc', 'graph_fs_spectral'}
FIXED = {'table_s3', 'all'}            # not data-driven -> no stability analysis

PRETTY = {
    'table_s3': 'Table S3 (paper)', 'all': 'All features', 'mrmr': 'mRMR', 'lasso': 'Lasso',
    'boruta': 'Boruta', 'rfe_rf': 'RFE-RF', 'graph_fs_cc': 'Graph-FS (CC)',
    'graph_fs_spectral': 'Graph-FS (spectral)', 'graph_fs_hybrid': 'Graph-FS hybrid',
}


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


def _rf(seed, n_jobs=-1, n_estimators=None):
    return RandomForestClassifier(n_estimators=n_estimators or config.RF_FS_TREES, max_depth=6,
                                  class_weight='balanced_subsample', random_state=seed, n_jobs=n_jobs)


# ─── Reference / control ─────────────────────────────────────────────────────

def sel_table_s3(X, y, k, task, **_):
    feats = config.SELECTED_FEATURES[task]
    rest = [c for c in X.columns if c not in feats]
    return SelectionResult('table_s3', list(feats), list(feats) + rest, len(feats))


def sel_all(X, y, k, task, **_):
    cols = list(X.loc[:, X.astype(float).std() > config.FS_VARIANCE_MIN].columns)
    return SelectionResult('all', cols, cols, len(cols))


# ─── Supervised filters / wrappers / embedded ────────────────────────────────

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


def boruta(Z, y, seed=0, max_iter=None, alpha=None, n_jobs=-1):
    """Boruta (Kursa & Rudnicki 2010) with two-sided binomial tests and Bonferroni correction."""
    max_iter, alpha = max_iter or config.BORUTA_MAX_ITER, alpha or config.BORUTA_ALPHA
    rng = np.random.default_rng(seed)
    X = Z.values
    p = X.shape[1]
    status = np.zeros(p, dtype=int)            # 0 tentative, 1 confirmed, -1 rejected
    hits = np.zeros(p)
    imp_sum, imp_n = np.zeros(p), np.zeros(p)
    it = 0
    for it in range(1, max_iter + 1):
        active = np.where(status != -1)[0]
        if len(active) == 0:                       # everything rejected
            break
        Xa = X[:, active]
        shadow = rng.permuted(Xa, axis=0)
        rf = _rf(seed + it, n_jobs, n_estimators=200).fit(np.hstack([Xa, shadow]), y)
        imp = rf.feature_importances_
        real, smax = imp[:len(active)], imp[len(active):].max()
        hits[active] += real > smax
        imp_sum[active] += real; imp_n[active] += 1
        if it >= 5:
            tent = np.where(status == 0)[0]
            if len(tent) == 0:
                break
            thr = alpha / len(tent)
            p_hi = binom.sf(hits[tent] - 1, it, 0.5)       # P(X >= hits)
            p_lo = binom.cdf(hits[tent], it, 0.5)          # P(X <= hits)
            status[tent[p_hi < thr]] = 1
            status[tent[p_lo < thr]] = -1
    mean_imp = imp_sum / np.maximum(imp_n, 1)
    order = sorted(range(p), key=lambda i: (-status[i], -hits[i], -mean_imp[i]))
    return status, hits, mean_imp, it, [Z.columns[i] for i in order]


def sel_boruta(X, y, k, seed=0, n_jobs=-1, **_):
    Z = prepare(X)
    status, hits, imp, n_iter, rank = boruta(Z, y, seed=seed, n_jobs=n_jobs)
    confirmed = int((status == 1).sum())
    kk = _k(k, confirmed, len(rank))
    return SelectionResult('boruta', rank[:kk], rank, confirmed,
                           {'n_confirmed': confirmed, 'n_tentative': int((status == 0).sum()),
                            'n_iter': int(n_iter),
                            'confirmed': [Z.columns[i] for i in np.where(status == 1)[0]]})


def sel_rfe_rf(X, y, k, seed=0, n_jobs=-1, **_):
    Z = prepare(X)
    rfe = RFE(_rf(seed, n_jobs, n_estimators=200), n_features_to_select=1, step=0.05).fit(Z.values, y)
    rank = list(Z.columns[np.argsort(rfe.ranking_, kind='stable')])
    kk = _k(k, 10, len(rank))
    return SelectionResult('rfe_rf', rank[:kk], rank, None)


# ─── Graph-FS ────────────────────────────────────────────────────────────────

def feature_graph(Z, alpha=None, link_isolated=True):
    """Feature-similarity graph: returns (|R| DataFrame, adjacency ndarray bool, threshold)."""
    alpha = config.GRAPH_FS_ALPHA if alpha is None else alpha
    R = Z.corr(method=config.GRAPH_FS_CORRELATION).abs().fillna(0.0)
    r = R.values.copy()
    np.fill_diagonal(r, 0.0)
    iu = np.triu_indices_from(r, 1)
    thr = float(r[iu].mean() + alpha * r[iu].std())
    A = r > thr
    if link_isolated:
        for i in np.where(A.sum(1) == 0)[0]:
            j = int(np.argmax(r[i]))
            A[i, j] = A[j, i] = True
    return R, A, thr


def _components(A):
    import networkx as nx
    G = nx.from_numpy_array(A.astype(int))
    return [sorted(c) for c in nx.connected_components(G)]


def graph_fs_cc_order(Z, alpha=None):
    """Representatives ordered by component size, then the remaining features round-robin
    over components by centrality. Returns (ranking, representatives, components, diag)."""
    R, A, thr = feature_graph(Z, alpha)
    names = list(Z.columns)
    deg = A.sum(1) / max(len(names) - 1, 1)                     # degree centrality
    strength = (R.values * A).sum(1)                            # tie-break: weighted degree
    comps = sorted(_components(A), key=lambda c: (-len(c), names[c[0]]))
    ordered = [sorted(c, key=lambda i: (-deg[i], -strength[i])) for c in comps]
    reps = [names[c[0]] for c in ordered]
    ranking, depth = [], 0
    while len(ranking) < len(names):
        for c in ordered:
            if depth < len(c):
                ranking.append(names[c[depth]])
        depth += 1
    diag = {'threshold': thr, 'n_edges': int(A.sum() // 2), 'n_components': len(comps),
            'components': [[names[i] for i in c] for c in ordered],
            'degree_centrality': {n: float(d) for n, d in zip(names, deg)}}
    return ranking, reps, A, R, diag


def sel_graph_fs_cc(X, y, k, **_):
    """Paper rule (alpha = 0.3). If that graph has fewer than k connected components (with the
    93 original features it is typically ONE component), alpha is raised in 0.1 steps until
    there are >= k components, so that k non-redundant representatives exist. alpha used is recorded."""
    Z = prepare(X)
    ranking, reps, A, R, diag = graph_fs_cc_order(Z, config.GRAPH_FS_ALPHA)
    native = len(reps)
    kk = _k(k, native, Z.shape[1])
    alpha = config.GRAPH_FS_ALPHA
    while len(reps) < kk and alpha < 4.0:
        alpha = round(alpha + 0.1, 2)
        ranking, reps, A, R, diag = graph_fs_cc_order(Z, alpha)
    sel = reps[:kk] if kk <= len(reps) else ranking[:kk]
    diag.update(adjacency=_edges(A, list(Z.columns)), alpha_used=alpha,
                n_components_at_paper_alpha=native)
    return SelectionResult('graph_fs_cc', sel, ranking, native, diag)


def _edges(A, names):
    i, j = np.where(np.triu(A, 1))
    return [[names[a], names[b]] for a, b in zip(i, j)]


def sel_graph_fs_spectral(X, y, k, seed=0, **_):
    Z = prepare(X)
    R, A, thr = feature_graph(Z)
    names = list(Z.columns)
    W = A.astype(float)
    L = np.diag(W.sum(1)) - W                                     # unnormalised Laplacian
    evals, evecs = np.linalg.eigh(L)
    # native number of clusters: largest eigengap among the first 20 eigenvalues
    gaps = np.diff(evals[:21])
    native = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    kk = _k(k, native, len(names))
    emb = evecs[:, :kk]
    km = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit(emb)
    d = np.linalg.norm(emb - km.cluster_centers_[km.labels_], axis=1)
    clusters = sorted(range(kk), key=lambda c: -(km.labels_ == c).sum())
    reps, members = [], []
    for c in clusters:
        idx = np.where(km.labels_ == c)[0]
        idx = idx[np.argsort(d[idx], kind='stable')]
        members.append([names[i] for i in idx])
        reps.append(names[idx[0]])
    ranking = list(reps)
    depth = 1
    while len(ranking) < len(names):
        for m in members:
            if depth < len(m):
                ranking.append(m[depth])
        depth += 1
    return SelectionResult('graph_fs_spectral', reps, ranking, native,
                           {'threshold': thr, 'n_edges': int(A.sum() // 2), 'n_clusters': kk,
                            'native_clusters': native, 'components': members,
                            'eigenvalues': [float(v) for v in evals[:21]],
                            'adjacency': _edges(A, names)})


def spectral_communities(A, n_clusters, seed=0):
    W = A.astype(float)
    L = np.diag(W.sum(1)) - W
    evals, evecs = np.linalg.eigh(L)
    emb = evecs[:, :n_clusters]
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(emb)
    return km.labels_, evals, emb, km


def sel_graph_fs_hybrid(X, y, k, seed=0, **_):
    """Graph-guided relevance selection (proposed): the Graph-FS feature graph is partitioned into
    C = max(2k, eigengap) spectral communities (label-free redundancy structure); the most
    outcome-relevant feature (ANOVA F, training data) of each community is its representative;
    the k most relevant representatives are kept -> at most one feature per community."""
    from sklearn.feature_selection import f_classif
    Z = prepare(X)
    names = list(Z.columns)
    R, A, thr = feature_graph(Z)
    F = pd.Series(np.nan_to_num(f_classif(Z.values, y)[0]), index=names)
    evals = np.linalg.eigvalsh(np.diag(A.sum(1)) - A.astype(float))
    gaps = np.diff(evals[:21])
    native = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    kk = _k(k, native, len(names))
    C = int(min(len(names), max(2 * kk, native)))
    labels, _, _, _ = spectral_communities(A, C, seed)
    members = [[names[i] for i in np.where(labels == c)[0]] for c in range(C)]
    members = [sorted(m, key=lambda f: -F[f]) for m in members if m]
    reps = sorted([m[0] for m in members], key=lambda f: -F[f])
    ranking = reps + [f for f in F.sort_values(ascending=False).index if f not in reps]
    return SelectionResult('graph_fs_hybrid', reps[:kk], ranking, native,
                           {'threshold': thr, 'n_edges': int(A.sum() // 2), 'n_communities': C,
                            'components': sorted(members, key=lambda m: -F[m[0]]),
                            'relevance_F': {f: float(v) for f, v in F.items()},
                            'adjacency': _edges(A, names)})


SELECTORS = {
    'table_s3': sel_table_s3, 'all': sel_all, 'mrmr': sel_mrmr, 'lasso': sel_lasso,
    'boruta': sel_boruta, 'rfe_rf': sel_rfe_rf, 'graph_fs_cc': sel_graph_fs_cc,
    'graph_fs_spectral': sel_graph_fs_spectral, 'graph_fs_hybrid': sel_graph_fs_hybrid,
}


def run_selector(method, X, y, task='LR', k='default', seed=0, n_jobs=-1):
    """X: DataFrame of TRAIN GTV features, y: 0/1 array. k='default' -> config.FS_K."""
    if method not in SELECTORS:
        raise ValueError(f"Unknown FS method '{method}'. Use one of {list(SELECTORS)}")
    k = config.FS_K if k == 'default' else k
    y = np.asarray(y).astype(int)
    return SELECTORS[method](X, y, k, task=task, seed=seed, n_jobs=n_jobs)
