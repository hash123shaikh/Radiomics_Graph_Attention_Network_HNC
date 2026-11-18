"""
Evaluate a trained GAT on validation and test; optional attention extraction.

Writes  results/<exp>.json        val + test metrics (threshold from validation)
        predictions/<exp>.csv     per-patient probabilities for val AND test (split column)
"""

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from grafsgat import config
from grafsgat.engine.train import prepare_datasets, predict
from grafsgat.models.gat import GRAFSGAT
from grafsgat.utils import get_device, calculate_metrics, bootstrap_auc_ci


def load_model(task, method, topology, use_clinical, seed, device):
    path = config.exp_paths(task, method, topology, use_clinical, seed)['model']
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first.")
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['mode'] != config.MODE or ck['task'] != task:
        raise RuntimeError(f"Checkpoint is for {ck['mode']}/{ck['task']}")
    hp = dict(ck['hparams'])
    model = GRAFSGAT(hp.pop('n_node_features'), hp.pop('n_clinical'), task=hp.pop('task'), **hp)
    model.load_state_dict(ck['model_state'])
    return model.to(device).eval(), ck


def evaluate(task, method, topology, use_clinical=False, seed=None, verbose=True):
    seed = config.RANDOM_SEED if seed is None else seed
    device = get_device(verbose=False)
    model, ck = load_model(task, method, topology, use_clinical, seed, device)
    ds, meta, _ = prepare_datasets(task, method, topology, use_clinical, encoder_state=ck['encoder_state'])
    if meta['scaler'] != ck['graph_meta']['scaler']:
        raise RuntimeError("Graphs were rebuilt after this model was trained. Re-train.")
    thr, bs = ck['threshold'], config.get_gat_config(task)['batch_size']
    paths = config.exp_paths(task, method, topology, use_clinical, seed)
    res, preds = {'method': method, 'topology': topology, 'use_clinical': use_clinical, 'seed': seed,
                  'features': meta['features'], 'selected_epoch': ck['history']['best_epoch']}, []
    for split in ('val', 'test'):
        p, y, ids, loss = predict(model, DataLoader(ds[split], batch_size=bs, shuffle=False), device)
        m = calculate_metrics(y, p, thr)
        m['auc_ci_low'], m['auc_ci_high'] = bootstrap_auc_ci(y, p)
        m['loss'] = loss
        res[split] = m
        preds.append(pd.DataFrame({'patient_id': ids, 'split': split, 'label': y, 'prob': p}))
    paths['metrics'].write_text(json.dumps(res, indent=1))
    pd.concat(preds).to_csv(paths['predictions'], index=False)
    if verbose:
        print(f"    val AUC {res['val']['auc']:.3f} | test AUC {res['test']['auc']:.3f} "
              f"[{res['test']['auc_ci_low']:.3f}, {res['test']['auc_ci_high']:.3f}]")
    return res


@torch.no_grad()
def extract_attention(task, method, topology, use_clinical=False, seed=None, split='test'):
    """Final-layer attention paid BY the GTV read-out node TO every other node (edges j -> GTV)."""
    seed = config.RANDOM_SEED if seed is None else seed
    device = get_device(verbose=False)
    model, ck = load_model(task, method, topology, use_clinical, seed, device)
    ds, _, _ = prepare_datasets(task, method, topology, use_clinical, encoder_state=ck['encoder_state'])
    rows = []
    for b in DataLoader(ds[split], batch_size=64, shuffle=False):
        b = b.to(device)
        ei, alpha = model.attention_to_gtv(b)
        prob = torch.sigmoid(model(b)).cpu().numpy()
        gtv = (b.ptr[:-1] + b.gtv_node_idx.view(-1)).cpu().numpy()
        src, dst, a = ei[0].cpu().numpy(), ei[1].cpu().numpy(), alpha.cpu().numpy()
        for gi, gidx in enumerate(gtv):
            start = int(b.ptr[gi])
            keep = (dst == gidx) & (src != gidx)
            for s_, w in zip(src[keep], a[keep]):
                rows.append({'patient_id': b.patient_id[gi], 'label': int(b.y[gi]), 'prob': float(prob[gi]),
                             'node': int(s_ - start), 'supervoxel_label': int(b.sv_ids[s_]),
                             'attention': float(w), 'z_mm': float(b.pos[s_, 0]),
                             'y_mm': float(b.pos[s_, 1]), 'x_mm': float(b.pos[s_, 2])})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['attention_tertile'] = df.groupby('patient_id')['attention'].transform(
            lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'medium', 'high']).astype(str))
    out = config.sub_dir(task, 'attention') / f"{config.exp_name(method, topology, use_clinical, seed)}_{split}.csv"
    df.to_csv(out, index=False)
    return df
