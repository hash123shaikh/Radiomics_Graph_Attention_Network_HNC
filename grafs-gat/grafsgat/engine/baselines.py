"""
Non-graph baselines on EXACTLY the same split (Appendix S1 style random forests):

  rf_selected(task, method)  RF on the GTV features chosen by an FS method
                             -> isolates what the GRAPH adds on top of the features
  rf_clinical(task)          RF on the encoded clinical variables

Min-max scaling fitted on train; sampling strategy (under/over/intermediate) chosen by
validation AUC; threshold = Youden on validation. Outputs use the same layout as the GAT
(topology name 'RF'), so the benchmark summary treats them uniformly.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from grafsgat import config
from grafsgat.data.clinical import ClinicalEncoder, clinical_tables_for_mode, stacked_table
from grafsgat.data.features import gtv_matrix
from grafsgat.data.splits import get_or_create_split
from grafsgat.utils import safe_auc, find_optimal_threshold, calculate_metrics, bootstrap_auc_ci

STRATEGIES = ['under', 'over', 'intermediate']


def resample(X, y, strategy, seed):
    """Same semantics as graphs.dataset.ClassBalancedSampler (one draw)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    mino, majo = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
    t_min, t_maj = {'over': (len(majo), len(majo)), 'under': (len(mino), len(mino)),
                    'intermediate': ((len(mino) + len(majo)) // 2,) * 2}[strategy]

    def draw(idx, n):
        if n <= len(idx):
            return rng.choice(idx, n, replace=False)
        return np.concatenate([idx, rng.choice(idx, n - len(idx), replace=True)])
    keep = np.concatenate([draw(mino, t_min), draw(majo, t_maj)])
    return X[keep], y[keep]


def _fit(X, y, strategy, seed):
    Xs, ys = resample(X, y, strategy, seed)
    return RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS * 3, max_depth=config.RF_MAX_DEPTH,
                                  random_state=seed, n_jobs=-1).fit(Xs, ys)


def _run(task, X, y, name, method, use_clinical, seed):
    best = (-1, None)
    for s in STRATEGIES:
        a = safe_auc(y['val'], _fit(X['train'], y['train'], s, seed).predict_proba(X['val'])[:, 1])
        if a > best[0]:
            best = (a, s)
    rf = _fit(X['train'], y['train'], best[1], seed)
    p = {s: rf.predict_proba(X[s])[:, 1] for s in ('val', 'test')}
    thr = find_optimal_threshold(y['val'], p['val'])
    paths = config.exp_paths(task, method, 'RF', use_clinical, seed)
    res = {'method': method, 'topology': 'RF', 'use_clinical': use_clinical, 'seed': seed,
           'sampling': best[1], 'model': name}
    preds = []
    for s in ('val', 'test'):
        m = calculate_metrics(y[s], p[s], thr)
        m['auc_ci_low'], m['auc_ci_high'] = bootstrap_auc_ci(y[s], p[s])
        res[s] = m
        preds.append(pd.DataFrame({'patient_id': X[f'{s}_ids'], 'split': s, 'label': y[s], 'prob': p[s]}))
    paths['metrics'].write_text(json.dumps(res, indent=1))
    pd.concat(preds).to_csv(paths['predictions'], index=False)
    print(f"  {name:28s} val AUC {res['val']['auc']:.3f} | test AUC {res['test']['auc']:.3f} ({best[1]})")
    return res


def _labels(split):
    return {s: np.array([split['labels'][p] for p in split[s]]) for s in ('train', 'val', 'test')}


def rf_selected(task, method, seed=None):
    from grafsgat.selection.runner import load_selection
    seed = config.RF_RANDOM_STATE if seed is None else seed
    split = get_or_create_split(task, verbose=False)
    feats = load_selection(task, method)['selected']
    raw = {s: gtv_matrix(split[s])[feats].values for s in ('train', 'val', 'test')}
    sc = MinMaxScaler().fit(raw['train'])
    X = {s: sc.transform(v) for s, v in raw.items()}
    X.update({f'{s}_ids': split[s] for s in ('val', 'test')})
    return _run(task, X, _labels(split), f'RF [{method}]', method, False, seed)


def rf_clinical(task, seed=None):
    seed = config.RF_RANDOM_STATE if seed is None else seed
    split = get_or_create_split(task, verbose=False)
    tables, cols, quant = clinical_tables_for_mode(task)
    table = stacked_table(tables, split['train'] + split['val'] + split['test'])
    enc = ClinicalEncoder(cols, quant).fit(table, split['train'])
    X = {s: np.vstack([v for v in enc.transform(table, split[s]).values()]) for s in ('train', 'val', 'test')}
    X.update({f'{s}_ids': split[s] for s in ('val', 'test')})
    return _run(task, X, _labels(split), 'RF [clinical only]', 'clinical_only', True, seed)
