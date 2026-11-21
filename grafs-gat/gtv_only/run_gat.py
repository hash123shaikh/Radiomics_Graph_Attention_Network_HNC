"""
GTV-only graph attention network: the graph is built from INSIDE the tumour only.

  nodes    : whole-GTV node (read-out, node 0) + SLIC sub-regions inside the GTV (intratumoral habitats)
  features : k original radiomic features chosen by mRMR / Lasso on the training GTVs, train-fitted
             log/z-score transform, clip +/-5
  edges    : 'star' GTV <-> every sub-region | 'knn' star + spatial 4-NN between sub-regions
  global   : optional graph-level vector concatenated at read-out: GTV shape (log volume, sphericity,
             max 3-D diameter, surface/volume) and/or clinical variables
Hyper-parameter random search (same sampled configurations for every task/design), checkpoint = lowest
validation loss, choice of configuration = validation AUC; the chosen configuration is re-trained
with 3 seeds and the seed ensemble is reported on MAASTRO.

  python gtv_only/run_gat.py --task DM --n_configs 24
"""
import argparse
import itertools
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from common import RES_DIR, Prep, clinical_matrix, config, gtv_table, load_npz, splits
from grafsgat.graphs.dataset import ClassBalancedSampler
from grafsgat.models.gat import GRAFSGAT
from grafsgat.utils import set_seed, safe_auc

SHAPE = ['original_shape_MeshVolume', 'original_shape_Sphericity', 'original_shape_Maximum3DDiameter',
         'original_shape_SurfaceVolumeRatio']
GRID = {
    'fs': ['mrmr', 'lasso'], 'k': [4, 8], 'topology': ['star', 'knn'],
    'global': ['none', 'shape', 'shape+clinical'],
    'hidden': [16, 32], 'layers': [2, 3], 'heads': [2, 4], 'dropout': [0.2, 0.5],
    'lr': [1e-3, 3e-4], 'wd': [1e-4, 1e-3], 'sampling': ['over', 'intermediate'],
}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EPOCHS, PATIENCE, MIN_EPOCH, BS = 300, 40, 10, 32
SUB = {}


def sub(pid):
    if pid not in SUB:
        d = load_npz(pid)
        SUB[pid] = (d['sub'], [str(x) for x in d['sub_names']], d['sub_centroids'], d['gtv_centroid'])
    return SUB[pid]


def sample_configs(n, seed=0):
    allc = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    random.Random(seed).shuffle(allc)
    return allc[:n]


def build(split, task, cfg, G, rankings):
    ids = {s: split[s] for s in ('train', 'val', 'test')}
    feats = rankings[cfg['fs']][:cfg['k']]
    pre = Prep().fit(G.loc[ids['train'], feats])
    glob = {s: np.zeros((len(ids[s]), 0)) for s in ids}
    if cfg['global'] != 'none':
        sh = G.loc[ids['train'], SHAPE].copy()
        ps = Prep().fit(sh)
        glob = {s: ps.transform(G.loc[ids[s], SHAPE]) for s in ids}
    if 'clinical' in cfg['global']:
        clin, _ = clinical_matrix(task, split)
        glob = {s: np.hstack([glob[s], clin[s]]) for s in ids}
    out = {}
    for s in ids:
        gs = []
        gx = pre.transform(G.loc[ids[s], feats])
        for i, pid in enumerate(ids[s]):
            sf, sn, sc, gc = sub(pid)
            sx = pre.transform(pd.DataFrame(sf, columns=sn)[feats])
            x = np.vstack([gx[i:i + 1], sx]).astype(np.float32)
            n = len(x)
            e = {(0, j) for j in range(1, n)} | {(j, 0) for j in range(1, n)}
            if cfg['topology'] == 'knn' and n > 2:
                D = np.linalg.norm(sc[:, None] - sc[None], axis=-1)
                np.fill_diagonal(D, np.inf)
                for a in range(len(sc)):
                    for b in np.argsort(D[a])[:min(4, len(sc) - 1)]:
                        e |= {(a + 1, int(b) + 1), (int(b) + 1, a + 1)}
            ei = torch.tensor(sorted(e), dtype=torch.long).T
            g = Data(x=torch.from_numpy(x), edge_index=ei, y=torch.tensor([split['labels'][pid]]),
                     gtv_node_idx=torch.tensor([0]),
                     clinical=torch.tensor(glob[s][i:i + 1], dtype=torch.float))
            g.patient_id = pid
            gs.append(g)
        out[s] = gs
    return out, glob['train'].shape[1]


@torch.no_grad()
def predict(model, graphs):
    model.eval()
    p = []
    for b in DataLoader(graphs, batch_size=128):
        p.append(torch.sigmoid(model(b.to(DEV))).cpu().numpy())
    return np.concatenate(p)


def train_one(data, n_glob, cfg, seed):
    set_seed(seed)
    model = GRAFSGAT(data['train'][0].x.shape[1], n_glob, task='LR', hidden_dim=cfg['hidden'],
                     n_heads=cfg['heads'], n_layers=cfg['layers'], dropout=cfg['dropout']).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    ytr = np.array([int(g.y) for g in data['train']])
    yva = np.array([int(g.y) for g in data['val']])
    sampler = ClassBalancedSampler(ytr, cfg['sampling'], seed=seed)
    ld = DataLoader(data['train'], batch_size=BS, sampler=sampler)
    crit = nn.BCEWithLogitsLoss()
    yv_t = torch.tensor(yva, dtype=torch.float)
    best, best_loss, since = None, np.inf, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for b in ld:
            b = b.to(DEV)
            opt.zero_grad()
            loss = crit(model(b), b.y.float().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = predict(model, data['val'])
        vl = float(nn.functional.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), yv_t))
        if ep >= MIN_EPOCH:
            if vl < best_loss:
                best_loss, since = vl, 0
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best)
    return predict(model, data['val']), predict(model, data['test'])


def run_cfg(task, design, sp, G, rank, cfg, seeds):
    rows = []
    for fold, split in sp:
        data, ng = build(split, task, cfg, G, rank[fold])
        for s in seeds:
            pv, pt = train_one(data, ng, cfg, s)
            for name, p in (('val', pv), ('test', pt)):
                for g, pr in zip(data[name], p):
                    rows.append({'fold': fold or 0, 'seed': s, 'split': name, 'patient_id': g.patient_id,
                                 'label': int(g.y), 'prob': float(pr)})
    return pd.DataFrame(rows)


def score(P):
    v, t = P[P.split == 'val'], P[P.split == 'test']
    ve = v.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
    te = t.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
    return {'val_auc': safe_auc(ve.label, ve.prob), 'test_auc': safe_auc(te.label, te.prob),
            'test_auc_fold_mean': float(np.nanmean([safe_auc(x.label, x.prob) for _, x in
                                                     t.groupby(['fold', 'seed'])]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--designs', nargs='+', default=['MATEUS_CV', 'MATEUS_COHORT'])
    ap.add_argument('--n_configs', type=int, default=24)
    a = ap.parse_args()
    from grafsgat.selection.methods import run_selector
    G = gtv_table()
    orig = [c for c in G.columns if c.startswith('original_') and '_shape_' not in c]
    configs = sample_configs(a.n_configs)
    for design in a.designs:
        sp = splits(a.task, design)
        rank = {}
        for fold, s in sp:
            y = np.array([s['labels'][p] for p in s['train']])
            X = G.loc[s['train'], orig]
            rank[fold] = {m: run_selector(m, X, y, k=8, seed=42).ranking for m in GRID['fs']}
        rows, t0 = [], time.time()
        for i, cfg in enumerate(configs):
            P = run_cfg(a.task, design, sp, G, rank, cfg, [42])
            r = {**cfg, **score(P), 'seeds': 1}
            rows.append(r)
            print(f"[{a.task} {design}] {i + 1}/{len(configs)} val {r['val_auc']:.3f} test {r['test_auc']:.3f} "
                  f"{json.dumps(cfg)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        S = pd.DataFrame(rows).sort_values('val_auc', ascending=False)
        S.to_csv(RES_DIR / f'gat_search_{a.task}_{design}.csv', index=False)
        # chosen configuration (validation AUC) re-trained with 3 seeds, per input set
        final = []
        for inp, sub_ in (('imaging', S[S['global'] != 'shape+clinical']),
                          ('imaging+clinical', S[S['global'] == 'shape+clinical'])):
            if sub_.empty:
                continue
            cfg = {k: sub_.iloc[0][k] for k in GRID}
            cfg = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
                   for k, v in cfg.items()}
            P = run_cfg(a.task, design, sp, G, rank, cfg, [42, 43, 44])
            P.to_csv(RES_DIR / f'gat_pred_{a.task}_{design}_{inp}.csv.gz', index=False)
            final.append({'inputs': inp, **cfg, **score(P), 'seeds': 3})
            print(f"[{a.task} {design}] CHOSEN {inp}: {final[-1]}", flush=True)
        pd.DataFrame(final).to_csv(RES_DIR / f'gat_final_{a.task}_{design}.csv', index=False)


if __name__ == '__main__':
    main()
