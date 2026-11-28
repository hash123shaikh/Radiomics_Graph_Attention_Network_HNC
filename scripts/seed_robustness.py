"""
Seed robustness of the validation-chosen population-graph configurations.

Every configuration chosen in <prefix>_family_selection.csv (graph and MLP families, every task, design
and input set) is retrained with R fresh, disjoint blocks of N_SEEDS seeds (the reported models used
seeds 0..9; here blocks start at 1000). Each block is a complete re-run of the 10-seed ensemble, so the
spread of the MAASTRO AUC across blocks shows how much of the reported result is seed luck.

Step 8.
  python scripts/seed_robustness.py --region total --clinical burden --repeats 10 --n_jobs 4
-> outputs/results/population_graph/<prefix>_seed_robustness.csv (one row per configuration x block)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _paths  # noqa: F401
import train_population_graph as R


def features(cfg, ids, n_tr, y, G):
    featset, fs, k, _, _, inp = cfg.split('|')
    C = R.mateus_clinical(ids, G).values
    if inp == 'clinical':
        return C
    cols = [c for c in G.columns if R.FEATURE_SETS[featset](c)]
    raw = G.loc[ids, cols]
    pre = R.Prep().fit(raw.iloc[:n_tr])
    Z = pd.DataFrame(pre.transform(raw), columns=cols)
    rank = R.rank_features(fs, raw.iloc[:n_tr].values, y[:n_tr], cols, max(R.GRID['k']))
    X = Z[rank[:int(k)]].values.astype(np.float32)
    return np.hstack([X, C]) if inp == 'imaging+clinical' else X


def run(task, design, cfgs, repeats, region, clinical):
    R.configure(region, clinical)
    G = R.feature_table()
    rows = []
    for fold, split in R.splits(task, design):
        ids = split['train'] + split['val'] + split['test']
        n_tr, n_va = len(split['train']), len(split['val'])
        y = np.array([split['labels'][p] for p in ids], dtype=np.float32)
        tr_idx = list(range(n_tr))
        for cfg in cfgs:
            X = features(cfg, ids, n_tr, y, G)
            gk, kind = int(cfg.split('|')[3][1:]), cfg.split('|')[4]
            ei = R.knn_edges(X, tr_idx, gk)
            for r in range(repeats):
                seeds = range(1000 + r * R.N_SEEDS, 1000 + (r + 1) * R.N_SEEDS)
                p = np.mean([R.fit_predict(kind, X, y, tr_idx, ei, s) for s in seeds], axis=0)
                rows.append(pd.DataFrame({'config': cfg, 'repeat': r, 'fold': fold or 0,
                                          'split': ['val'] * n_va + ['test'] * len(split['test']),
                                          'patient_id': ids[n_tr:], 'label': y[n_tr:], 'prob': p[n_tr:]}))
    return task, design, pd.concat(rows)


def auc(d):
    return roc_auc_score(d.label, d.prob) if d.label.nunique() == 2 else np.nan


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--region', choices=['primary', 'total'], default='total')
    ap.add_argument('--clinical', choices=['mateus', 'burden'], default='burden')
    ap.add_argument('--n_jobs', type=int, default=4)
    a = ap.parse_args()
    R.configure(a.region, a.clinical)
    from joblib import Parallel, delayed
    Sel = pd.read_csv(R.OUT / f'{R.PREFIX}_family_selection.csv')
    Sel = Sel[Sel.family.isin(['graph', 'mlp'])]
    jobs = [(t, d, sorted(set(x.config))) for (t, d), x in Sel.groupby(['task', 'design'])]
    res = Parallel(n_jobs=a.n_jobs, verbose=10)(delayed(run)(t, d, c, a.repeats, a.region, a.clinical) for t, d, c in jobs)
    out = []
    for t, d, P in res:
        for (cfg, r), x in P.groupby(['config', 'repeat']):
            te = x[x.split == 'test']
            pooled = te.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
            ours = np.nanmean([auc(f) for _, f in te.groupby('fold')]) if d == 'MATEUS_CV' else auc(pooled)
            out.append(dict(task=t, design=d, config=cfg, repeat=r, val_auc=auc(x[x.split == 'val']),
                            ours=ours, pooled_test=auc(pooled)))
    O = pd.DataFrame(out)
    O = O.merge(Sel[['task', 'design', 'config', 'family', 'inputs', 'ours', 'mateus']]
                .rename(columns={'ours': 'reported'}), on=['task', 'design', 'config'])
    O.to_csv(R.OUT / f'{R.PREFIX}_seed_robustness.csv', index=False)
    s = O.groupby(['task', 'design', 'inputs', 'family']).agg(reported=('reported', 'first'),
                                                              mean=('ours', 'mean'), sd=('ours', 'std'),
                                                              min=('ours', 'min'), max=('ours', 'max'),
                                                              mateus=('mateus', 'first'))
    s['frac_beats'] = O.assign(b=O.ours > O.mateus).groupby(['task', 'design', 'inputs', 'family']).b.mean()
    pd.set_option('display.width', 250)
    print(s.round(3).to_string())
