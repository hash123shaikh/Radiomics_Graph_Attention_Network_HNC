"""
GTV-only classical models (no peritumoral tissue): hyper-parameter grid on the study splits.

grid  feature set   orig (93) | orig+shape (107) | full (shape+orig+LoG+wavelet, 1223)
      selection     mrmr | lasso | ucorr (univariate AUC + |r|<0.8 redundancy filter)   (train only)
      k             2, 4, 6, 10, 15
      classifier    logistic L2 (C 0.03, 0.3, 3) | random forest (depth 3, 6)
      inputs        imaging | imaging+clinical
plus two simple references: GTV volume alone and clinical-only (logistic).

Model choice = highest VALIDATION AUC (pooled out-of-fold for MATEUS_CV, HMR+CHUM for MATEUS_COHORT);
MAASTRO test AUC is reported for every configuration but never used for choosing.

  python gtv_only/run_classical.py --tasks DM OS LR --n_jobs 6
-> outputs/gtv_only/results/classical_<task>_<design>.csv (+ predictions parquet-free CSV)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import FEATURE_SETS, RES_DIR, Prep, clinical_matrix, config, gtv_table, splits

warnings.filterwarnings('ignore')
KS = [2, 4, 6, 10, 15]
CLFS = {
    'lr_C0.03': lambda s: LogisticRegression(C=0.03, class_weight='balanced', max_iter=5000),
    'lr_C0.3' : lambda s: LogisticRegression(C=0.3, class_weight='balanced', max_iter=5000),
    'lr_C3'   : lambda s: LogisticRegression(C=3.0, class_weight='balanced', max_iter=5000),
    'rf_d3'   : lambda s: RandomForestClassifier(500, max_depth=3, min_samples_leaf=3, random_state=s,
                                                 class_weight='balanced_subsample', n_jobs=4),
    'rf_d6'   : lambda s: RandomForestClassifier(500, max_depth=6, min_samples_leaf=3, random_state=s,
                                                 class_weight='balanced_subsample', n_jobs=4),
}
FS = ['mrmr', 'lasso', 'ucorr']


def ucorr_ranking(Z, y):
    auc = np.array([abs(roc_auc_score(y, Z[:, j]) - 0.5) for j in range(Z.shape[1])])
    order, keep = np.argsort(-auc), []
    C = np.abs(np.nan_to_num(np.corrcoef(Z, rowvar=False)))
    for j in order:
        if all(C[j, i] < 0.8 for i in keep):
            keep.append(j)
        if len(keep) >= max(KS):
            break
    return keep


def ranking(method, Xtr, ytr, cols):
    from grafsgat.selection.methods import run_selector, prepare
    if method == 'ucorr':
        Z = prepare(pd.DataFrame(Xtr, columns=cols))
        return [list(Z.columns)[j] for j in ucorr_ranking(Z.values, ytr)]
    return run_selector(method, pd.DataFrame(Xtr, columns=cols), ytr, k=max(KS), seed=42).ranking


def run_split(task, design, fold, split, G):
    lab = lambda s: np.array([split['labels'][p] for p in split[s]])
    y = {s: lab(s) for s in ('train', 'val', 'test')}
    clin, _ = clinical_matrix(task, split)
    rows = []

    def fit_pred(name, Xs, model, seed=42, **meta):
        m = CLFS[model](seed).fit(Xs['train'], y['train'])
        for s in ('val', 'test'):
            for pid, pr, lb in zip(split[s], m.predict_proba(Xs[s])[:, 1], y[s]):
                rows.append({'config': name, 'fold': fold, 'split': s, 'patient_id': pid, 'label': lb,
                             'prob': pr, **meta})

    # references: volume alone, clinical alone
    vol = {s: np.log(G.loc[split[s], ['original_shape_MeshVolume']].values + 1) for s in y}
    mu, sd = vol['train'].mean(), vol['train'].std()
    fit_pred('volume|lr', {s: (v - mu) / sd for s, v in vol.items()}, 'lr_C0.3', featset='volume', fs='-', k=1,
             clf='lr_C0.3', inputs='imaging')
    fit_pred('clinical|lr', clin, 'lr_C0.3', featset='clinical', fs='-', k=0, clf='lr_C0.3', inputs='clinical')
    fit_pred('volume+clinical|lr', {s: np.hstack([(vol[s] - mu) / sd, clin[s]]) for s in y}, 'lr_C0.3',
             featset='volume', fs='-', k=1, clf='lr_C0.3', inputs='imaging+clinical')

    for fsname, keep in FEATURE_SETS.items():
        cols = [c for c in G.columns if keep(c)]
        raw = {s: G.loc[split[s], cols] for s in y}
        pre = Prep().fit(raw['train'])
        Z = {s: pd.DataFrame(pre.transform(raw[s]), columns=cols) for s in y}
        for meth in FS:
            rank = ranking(meth, raw['train'].values, y['train'], cols)
            for k in KS:
                feats = rank[:k]
                Xs = {s: Z[s][feats].values for s in y}
                for clf in CLFS:
                    for use_clin in (False, True):
                        X = {s: np.hstack([Xs[s], clin[s]]) for s in y} if use_clin else Xs
                        inp = 'imaging+clinical' if use_clin else 'imaging'
                        fit_pred(f'{fsname}|{meth}|k{k}|{clf}|{inp}', X, clf, featset=fsname, fs=meth, k=k,
                                 clf=clf, inputs=inp)
    return rows


def auc(d):
    return roc_auc_score(d.label, d.prob) if d.label.nunique() == 2 else np.nan


def summarise(P):
    out = []
    for cfg, d in P.groupby('config', sort=False):
        v, t = d[d.split == 'val'], d[d.split == 'test']
        te = t.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean')).reset_index()
        r = d.iloc[0][['featset', 'fs', 'k', 'clf', 'inputs']].to_dict()
        r.update(config=cfg, val_auc=auc(v), test_auc=auc(te),
                 val_auc_fold_mean=np.nanmean([auc(x) for _, x in v.groupby('fold', dropna=False)]),
                 test_auc_fold_mean=np.nanmean([auc(x) for _, x in t.groupby('fold', dropna=False)]),
                 test_n=len(te), test_events=int(te.label.sum()))
        out.append(r)
    return pd.DataFrame(out)


CLS_FEAT = __import__('os').environ.get('CLS_FEAT', 'primary')   # 'primary' GTV | 'total' GTV (primary + nodal)
PFX = 'classical' if CLS_FEAT == 'primary' else f'classical_{CLS_FEAT}'


def feature_table():
    if CLS_FEAT == 'total':
        return pd.read_csv(config.OUTPUT_DIR / 'popgraph' / 'total_gtv_features.csv', index_col=0)
    return gtv_table()


def run(task, design):
    G = feature_table()
    sp = [(k, s) for k, s in splits(task, design)]
    rows = []
    for k, s in sp:
        missing = [p for p in s['train'] + s['val'] + s['test'] if p not in G.index]
        if missing:
            raise RuntimeError(f'{len(missing)} patients lack GTV-only features: {missing[:5]}')
        rows += run_split(task, design, k if k else 0, s, G)
    P = pd.DataFrame(rows)
    P.to_csv(RES_DIR / f'{PFX}_pred_{task}_{design}.csv.gz', index=False)
    S = summarise(P).sort_values('val_auc', ascending=False)
    S.to_csv(RES_DIR / f'{PFX}_{task}_{design}.csv', index=False)
    return task, design


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['DM', 'OS', 'LR'])
    ap.add_argument('--designs', nargs='+', default=['MATEUS_CV', 'MATEUS_COHORT'])
    ap.add_argument('--n_jobs', type=int, default=6)
    a = ap.parse_args()
    feature_table()
    from joblib import Parallel, delayed
    jobs = [(t, d) for t in a.tasks for d in a.designs]
    for t, d in Parallel(n_jobs=min(a.n_jobs, len(jobs)), verbose=5)(delayed(run)(t, d) for t, d in jobs):
        S = pd.read_csv(RES_DIR / f'{PFX}_{t}_{d}.csv')
        print(f'\n=== {t} {d}: top 5 by validation AUC')
        print(S.head(5)[['config', 'val_auc', 'test_auc', 'test_auc_fold_mean']].to_string(index=False))
