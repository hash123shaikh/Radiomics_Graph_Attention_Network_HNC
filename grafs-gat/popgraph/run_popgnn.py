"""
Population-graph GNN (one node per patient) for LR / DM / OS on the study splits.

Nodes      every patient of the split (training, validation and MAASTRO test); node features are
           radiomic features of the whole GTV selected on the training patients only, optionally
           with the clinical variables of Mateus et al. (T stage, N stage, GTV volume quartile bins).
Edges      inductive k-nearest-neighbour graph in node-feature space: every node is linked to its
           k nearest TRAINING patients (never to validation or test patients), so validation and test
           predictions are computed from their own features and those of training neighbours only.
Models     GCN, GraphSAGE, GAT (2 layers) and an MLP without edges (control); probabilities averaged
           over N_SEEDS seeds.
Selection  highest VALIDATION AUC (pooled out-of-fold for MATEUS_CV, HMR+CHUM for MATEUS_COHORT);
           MAASTRO labels are never used for training or choosing.

  python popgraph/run_popgnn.py --tasks LR DM OS --n_jobs 20
  POPGNN_CLIN=burden python popgraph/run_popgnn.py     # clinical group with total + nodal GTV burden
-> outputs/popgraph/results/popgnn_<task>_<design>.csv and popgnn_pred_<task>_<design>.csv.gz
"""
import argparse
import os
import sys
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gtv_only'))
from common import FEATURE_SETS, Prep, config, gtv_table, splits  # noqa: E402

torch.set_num_threads(1)
OUT = config.OUTPUT_DIR / 'popgraph' / 'results'
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = 10
EPOCHS = 150
CLIN = os.environ.get('POPGNN_CLIN', 'mateus')      # 'mateus' | 'burden'
FEAT = os.environ.get('POPGNN_FEAT', 'primary')    # 'primary' GTV | 'total' GTV (primary + nodal)
PREFIX = 'popgnn' + ('' if CLIN == 'mateus' else f'_{CLIN}') + ('' if FEAT == 'primary' else f'_{FEAT}')


def feature_table():
    if FEAT == 'total':
        return pd.read_csv(config.OUTPUT_DIR / 'popgraph' / 'total_gtv_features.csv', index_col=0)
    return gtv_table()
VOL_BINS = [(0, 11000), (11000, 24000), (24000, 43000), (43000, np.inf)]   # mm^3, Mateus et al.


# ── features ────────────────────────────────────────────────────────────────────────────────────
def mateus_clinical(ids, G):
    """T stage (T1-T4), N stage (N0, N1, N2+) and GTV volume quartile bins, as in Mateus et al."""
    from grafsgat.data.clinical import encode_clinical
    H = pd.concat([encode_clinical('HNPET', harmonized=True)[0], encode_clinical('MAASTRO', harmonized=True)[0]])
    h = H.loc[ids]
    v = G.loc[ids, 'original_shape_MeshVolume'].values
    X = pd.DataFrame(index=ids)
    X['n0'], X['n1'], X['n2+'] = h.n_n0.values, h.n_n1.values, (h.n_n2 + h.n_n3).values
    for t in ('t1', 't2', 't3', 't4'):
        X[t] = h['t_' + t].values
    if CLIN == 'burden':                          # total GTV (primary + nodal) instead of primary only
        B = pd.read_csv(config.OUTPUT_DIR / 'popgraph' / 'tumour_burden.csv', index_col=0).loc[ids]
        v = B.vol_total.fillna(pd.Series(v, index=ids)).values
        X['log_nodal'] = np.log1p(B.vol_nodal.fillna(0).values) / 12
        X['n_nodal'] = np.minimum(B.n_nodal_gtv.fillna(0).values, 4) / 4
    for i, (a, b) in enumerate(VOL_BINS):
        X[f'vol{i}'] = ((v >= a) & (v < b)).astype(float)
    return X.astype(np.float32)


def rank_features(method, Xtr, ytr, cols, kmax):
    from grafsgat.selection.methods import prepare, run_selector
    if method == 'ucorr':
        Z = prepare(pd.DataFrame(Xtr, columns=cols))
        auc = np.array([abs(roc_auc_score(ytr, Z.values[:, j]) - 0.5) for j in range(Z.shape[1])])
        C = np.abs(np.nan_to_num(np.corrcoef(Z.values, rowvar=False)))
        keep = []
        for j in np.argsort(-auc):
            if all(C[j, i] < 0.8 for i in keep):
                keep.append(j)
            if len(keep) >= kmax:
                break
        return [list(Z.columns)[j] for j in keep]
    return run_selector(method, pd.DataFrame(Xtr, columns=cols), ytr, k=kmax, seed=42).ranking


# ── graph + models ──────────────────────────────────────────────────────────────────────────────
def knn_edges(X, train_idx, k):
    """Directed edges train-neighbour -> node; every node gets its k nearest training patients."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn[train_idx].T                                   # cosine similarity to training nodes
    pos = {j: i for i, j in enumerate(train_idx)}
    for i in train_idx:                                        # no self loops through the kNN
        S[i, pos[i]] = -np.inf
    nb = np.argsort(-S, axis=1)[:, :k]
    src = np.asarray(train_idx)[nb].ravel()
    dst = np.repeat(np.arange(len(X)), k)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


class Net(nn.Module):
    def __init__(self, kind, d_in, hidden=32, dropout=0.5):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == 'gcn':
            self.c1, self.c2 = GCNConv(d_in, hidden), GCNConv(hidden, hidden)
        elif kind == 'sage':
            self.c1, self.c2 = SAGEConv(d_in, hidden), SAGEConv(hidden, hidden)
        elif kind == 'gat':
            self.c1, self.c2 = GATConv(d_in, hidden // 4, heads=4), GATConv(hidden, hidden // 4, heads=4)
        else:
            self.c1, self.c2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.skip = nn.Linear(d_in, hidden)
        self.out = nn.Linear(hidden, 1)

    def conv(self, layer, x, ei):
        return layer(x) if self.kind == 'mlp' else layer(x, ei)

    def forward(self, x, ei):
        x = F.dropout(x, 0.2, self.training)
        h = F.elu(self.conv(self.c1, x, ei) + self.skip(x))
        h = F.dropout(h, self.dropout, self.training)
        h = F.elu(self.conv(self.c2, h, ei)) + h
        h = F.dropout(h, self.dropout, self.training)
        return self.out(h).squeeze(-1)


def fit_predict(kind, X, y, train_idx, ei, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tr = torch.tensor(train_idx)
    pw = torch.tensor((1 - y[train_idx]).sum() / max(y[train_idx].sum(), 1), dtype=torch.float32)
    m = Net(kind, X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-3)
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(x, ei)[tr], yt[tr], pos_weight=pw)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(x, ei)).numpy()


# ── grid ────────────────────────────────────────────────────────────────────────────────────────
GRID = dict(featset=['orig+shape', 'full'], fs=['mrmr', 'lasso', 'ucorr'], k=[2, 4, 8],
            graph_k=[5, 10, 20], model=['gcn', 'sage', 'gat', 'mlp'])


def run_split(task, fold, split, G):
    ids = split['train'] + split['val'] + split['test']
    n_tr, n_va = len(split['train']), len(split['val'])
    tr_idx = list(range(n_tr))
    y = np.array([split['labels'][p] for p in ids], dtype=np.float32)
    role = np.array(['train'] * n_tr + ['val'] * n_va + ['test'] * len(split['test']))
    C = mateus_clinical(ids, G).values
    rows = []

    def record(cfg, prob):
        for s in ('val', 'test'):
            msk = role == s
            rows.append(pd.DataFrame({'config': cfg, 'fold': fold, 'split': s, 'patient_id': np.array(ids)[msk],
                                      'label': y[msk], 'prob': prob[msk]}))

    # clinical only (Mateus variables), every model
    for gk, kind in product(GRID['graph_k'], GRID['model']):
        if kind == 'mlp' and gk != GRID['graph_k'][0]:
            continue
        ei = knn_edges(C, tr_idx, gk)
        p = np.mean([fit_predict(kind, C, y, tr_idx, ei, s) for s in range(N_SEEDS)], axis=0)
        record(f'clinical|-|0|g{gk}|{kind}|clinical', p)

    for fsname in GRID['featset']:
        cols = [c for c in G.columns if FEATURE_SETS[fsname](c)]
        raw = G.loc[ids, cols]
        pre = Prep().fit(raw.iloc[:n_tr])
        Z = pd.DataFrame(pre.transform(raw), columns=cols)
        for meth in GRID['fs']:
            rank = rank_features(meth, raw.iloc[:n_tr].values, y[:n_tr], cols, max(GRID['k']))
            for k in GRID['k']:
                Xi = Z[rank[:k]].values.astype(np.float32)
                for use_clin in (False, True):
                    X = np.hstack([Xi, C]) if use_clin else Xi
                    inp = 'imaging+clinical' if use_clin else 'imaging'
                    for gk, kind in product(GRID['graph_k'], GRID['model']):
                        if kind == 'mlp' and gk != GRID['graph_k'][0]:
                            continue
                        ei = knn_edges(X, tr_idx, gk)
                        p = np.mean([fit_predict(kind, X, y, tr_idx, ei, s) for s in range(N_SEEDS)], axis=0)
                        record(f'{fsname}|{meth}|{k}|g{gk}|{kind}|{inp}', p)
    return pd.concat(rows)


def auc(d):
    return roc_auc_score(d.label, d.prob) if d.label.nunique() == 2 else np.nan


def summarise(P):
    out = []
    for cfg, d in P.groupby('config', sort=False):
        v, t = d[d.split == 'val'], d[d.split == 'test']
        te = t.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean')).reset_index()
        f = cfg.split('|')
        out.append(dict(config=cfg, featset=f[0], fs=f[1], k=int(f[2]), graph_k=f[3], model=f[4], inputs=f[5],
                        val_auc=auc(v), test_auc=auc(te),
                        test_auc_fold_mean=np.nanmean([auc(x) for _, x in t.groupby('fold')]),
                        test_auc_fold_min=np.nanmin([auc(x) for _, x in t.groupby('fold')]),
                        test_auc_fold_max=np.nanmax([auc(x) for _, x in t.groupby('fold')]),
                        test_n=len(te), test_events=int(te.label.sum())))
    return pd.DataFrame(out)


def job(task, design, fold, split):
    return run_split(task, fold, split, feature_table())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['LR', 'DM', 'OS'])
    ap.add_argument('--designs', nargs='+', default=['MATEUS_CV', 'MATEUS_COHORT'])
    ap.add_argument('--n_jobs', type=int, default=20)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    feature_table()
    jobs = [(t, d, k or 0, s) for t in a.tasks for d in a.designs for k, s in splits(t, d)]
    res = Parallel(n_jobs=a.n_jobs, verbose=10)(delayed(job)(*j) for j in jobs)
    for t in a.tasks:
        for d in a.designs:
            P = pd.concat([r for (tt, dd, _, _), r in zip(jobs, res) if tt == t and dd == d])
            P.to_csv(OUT / f'{PREFIX}_pred_{t}_{d}.csv.gz', index=False)
            S = summarise(P).sort_values('val_auc', ascending=False)
            S.to_csv(OUT / f'{PREFIX}_{t}_{d}.csv', index=False)
            for inp in ('imaging', 'imaging+clinical', 'clinical'):
                b = S[S.inputs == inp].iloc[0]
                print(f'{t} {d:13} {inp:16} chosen {b.config:45} val {b.val_auc:.2f} '
                      f'test {b.test_auc:.2f} fold-mean {b.test_auc_fold_mean:.2f}')
