"""
Patient graphs for one (feature-selection method, topology) pair.

  node features : the method's selected features (from selection/<method>.json),
                  z-scored with a scaler fitted on the GTV features of TRAINING patients
                  (authors' ScalerGeneration.py), applied to every node, clipped to +/-5
  nodes         : GTV (node 0, read-out) + K supervoxels
  topology      : config.TOPOLOGIES
      radiomic_star  K supervoxels nearest to the GTV in feature space; GTV <-> each (paper)
      spatial_star   K supervoxels nearest to the GTV in physical space;  GTV <-> each
      radiomic_knn   radiomic_star nodes; star edges + kNN edges between all nodes (feature space)
      spatial_knn    spatial_star nodes;  star edges + kNN edges between all nodes (physical space)
  edge_attr     : distance of the edge (feature or mm), not used by the default GAT

Saved to outputs/runs/<MODE>_<TASK>/graphs/<method>__<topology>/<pid>.pt + meta.json
"""

import json
import shutil

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from grafsgat import config
from grafsgat.data.features import load_feature_dict
from grafsgat.data.splits import get_or_create_split


class GTVScaler:
    def fit(self, X):
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def state(self):
        return {'mean': self.mean_.tolist(), 'scale': self.scale_.tolist()}


def _idx(names, wanted):
    missing = [w for w in wanted if w not in names]
    if missing:
        raise KeyError(f"Features not extracted: {missing}")
    return [names.index(w) for w in wanted]


def _knn_edges(P, k):
    """Undirected kNN edges between rows of P (both directions), excluding self loops."""
    n = len(P)
    if n < 2:
        return set()
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    e = set()
    for i in range(n):
        for j in np.argsort(D[i], kind='stable')[:min(k, n - 1)]:
            e.add((i, int(j))); e.add((int(j), i))
    return e


def build_patient_graph(fd, feat_idx, scaler, topology, label, pid, k=None, knn_k=None):
    k, knn_k = k or config.N_SUPERVOXELS_SELECTED, knn_k or config.KNN_K
    c = config.NODE_FEATURE_CLIP
    gtv = np.clip(scaler.transform(fd['gtv'][feat_idx][None, :])[0], -c, c)
    sv = np.clip(scaler.transform(fd['supervoxels'][:, feat_idx]), -c, c)
    pos_sv, pos_gtv = fd['centroids'], fd['gtv_centroid']

    feat_dist = np.linalg.norm(sv - gtv[None, :], axis=1)
    space_dist = np.linalg.norm(pos_sv - pos_gtv[None, :], axis=1)
    if topology.startswith('radiomic'):
        sel = np.argsort(feat_dist, kind='stable')[:min(k, len(sv))]
    elif topology.startswith('spatial'):
        sel = np.argsort(space_dist, kind='stable')[:min(k, len(sv))]
    else:
        raise ValueError(f"Unknown topology {topology}")

    x = np.vstack([gtv[None, :], sv[sel]]).astype(np.float32)
    pos = np.vstack([pos_gtv[None, :], pos_sv[sel]]).astype(np.float32)
    n = len(x)
    edges = {(0, i) for i in range(1, n)} | {(i, 0) for i in range(1, n)}
    if topology.endswith('knn'):
        edges |= _knn_edges(x if topology.startswith('radiomic') else pos, knn_k)
    edges = sorted(edges)
    ei = np.array(edges, dtype=np.int64).T
    metric = x if topology.startswith('radiomic') else pos
    ea = np.linalg.norm(metric[ei[0]] - metric[ei[1]], axis=1).astype(np.float32)[:, None]

    g = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea),
             y=torch.tensor([label], dtype=torch.long), pos=torch.from_numpy(pos),
             sv_ids=torch.from_numpy(np.concatenate([[0], fd['valid_sv_ids'][sel]]).astype(np.int64)),
             gtv_node_idx=torch.tensor([0], dtype=torch.long))
    g.patient_id = str(pid)
    return g


def build_graphs(task, method, topology, force=False):
    from grafsgat.selection.runner import load_selection
    split = get_or_create_split(task, verbose=False)
    sel = load_selection(task, method)
    if sel.get('split_key') != split['key']:
        raise RuntimeError(f"Selection for {method} was made on another split. Re-run 'select'.")
    feats = sel['selected']
    out = config.graph_dir(task, method, topology)
    meta_f = out / 'meta.json'
    if meta_f.exists() and not force:
        m = json.loads(meta_f.read_text())
        if m.get('version') == config.GRAPH_VERSION and m.get('split_key') == split['key'] \
                and m.get('features') == feats:
            return out

    X = []
    for p in split['train']:
        fd = load_feature_dict(p)
        X.append(fd['gtv'][_idx(fd['feature_names'], feats)])
    scaler = GTVScaler().fit(np.array(X))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    failed = {}
    ids = split['train'] + split['val'] + split['test']
    for p in tqdm(ids, desc=f'graphs {method}/{topology}', leave=False, disable=None):
        try:
            fd = load_feature_dict(p)
            g = build_patient_graph(fd, _idx(fd['feature_names'], feats), scaler, topology,
                                    split['labels'][p], p)
            g.cohort = split['cohort'][p]
            torch.save(g, out / f'{p}.pt')
        except Exception as e:
            failed[p] = f'{type(e).__name__}: {e}'
    if failed:
        print(f"  WARNING: {len(failed)} graphs failed: {list(failed.items())[:3]}")
    meta = {'mode': config.MODE, 'task': task, 'method': method, 'topology': topology,
            'version': config.GRAPH_VERSION, 'split_key': split['key'], 'features': feats,
            'k': config.N_SUPERVOXELS_SELECTED, 'knn_k': config.KNN_K, 'clip': config.NODE_FEATURE_CLIP,
            'scaler': scaler.state(), 'failed': failed}
    meta_f.write_text(json.dumps(meta, indent=1))
    return out


def load_graphs(task, method, topology):
    """Returns ({'train','val','test': [Data]}, meta, split)."""
    out = config.graph_dir(task, method, topology)
    meta_f = out / 'meta.json'
    if not meta_f.exists():
        raise FileNotFoundError(f"No graphs in {out}. Run the 'graph' stage first.")
    meta = json.loads(meta_f.read_text())
    if meta.get('version') != config.GRAPH_VERSION:
        raise RuntimeError(f"Graphs in {out} are outdated. Rebuild them.")
    split = get_or_create_split(task, verbose=False)
    if meta['split_key'] != split['key']:
        raise RuntimeError("Graphs were built on another split. Rebuild them.")
    graphs = {s: [torch.load(out / f'{p}.pt', weights_only=False) for p in split[s]
                  if p not in meta['failed']] for s in ('train', 'val', 'test')}
    return graphs, meta, split
