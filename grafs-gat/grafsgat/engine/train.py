"""
Train one GRAFS-GAT network for (feature-selection method, topology, imaging[/clinical], seed).

Follows the authors' GAT_modeling.py + Table S2:
  - training set resampled every epoch (Table S2 strategy), unweighted BCE
  - Adam, linear warm-up, then ReduceLROnPlateau on validation loss
  - checkpoint (config.EPOCH_RULE), early stopping on validation loss in both cases:
      'val_loss' lowest validation loss after MIN_EPOCHS_BEFORE_SELECT (Bae et al. 2025)
      'auc_gap'  highest validation AUC among epochs whose train AUC - val AUC <= gap; gap starts at
                 AUC_GAP_START and grows by AUC_GAP_STEP until an epoch qualifies (adapted from Mateus et al.)
      'mateus'   exact rule of the Mateus et al. code: save when val AUC > best saved AND > threshold AND
                 |train AUC - val AUC| < max difference (config.MATEUS_SAVE_RULE); keep the LAST saved model;
                 stop once train AUC > config.MATEUS_STOP_TRAIN_AUC; nothing saved -> last epoch
  - decision threshold = Youden's J on VALIDATION (never on test)
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from grafsgat import config
from grafsgat.data.clinical import ClinicalEncoder, clinical_tables_for_mode, stacked_table
from grafsgat.graphs.builder import load_graphs
from grafsgat.graphs.dataset import PatientGraphDataset, make_loaders
from grafsgat.models.gat import GRAFSGAT
from grafsgat.utils import set_seed, get_device, safe_auc, find_optimal_threshold, plot_training_history


def prepare_datasets(task, method, topology, use_clinical, encoder_state=None):
    graphs, meta, split = load_graphs(task, method, topology)
    clin, encoder = {'train': None, 'val': None, 'test': None}, None
    if use_clinical:
        tables, cols, quant = clinical_tables_for_mode(task)
        all_ids = [g.patient_id for s in graphs.values() for g in s]
        table = stacked_table(tables, all_ids)
        encoder = (ClinicalEncoder.from_state(encoder_state) if encoder_state is not None else
                   ClinicalEncoder(cols, quant).fit(table, [g.patient_id for g in graphs['train']]))
        for s in clin:
            clin[s] = encoder.transform(table, [g.patient_id for g in graphs[s]])
    ds = {s: PatientGraphDataset(graphs[s], clin[s]) for s in ('train', 'val', 'test')}
    return ds, meta, encoder


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, ids, loss, n = [], [], [], 0.0, 0
    crit = nn.BCEWithLogitsLoss(reduction='sum')
    for b in loader:
        b = b.to(device)
        logit = model(b)
        y = b.y.float().view(-1)
        loss += crit(logit, y).item(); n += y.numel()
        probs.append(torch.sigmoid(logit).cpu().numpy()); labels.append(y.cpu().numpy())
        ids += list(b.patient_id)
    return (np.concatenate(probs) if probs else np.array([]),
            np.concatenate(labels).astype(int) if labels else np.array([], dtype=int),
            ids, loss / max(n, 1))


def _epoch(model, loader, opt, crit, device):
    model.train()
    tot, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        opt.zero_grad()
        loss = crit(model(b), b.y.float().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        opt.step()
        tot += loss.item() * b.num_graphs; n += b.num_graphs
    return tot / max(n, 1)


def train(task, method, topology, use_clinical=False, seed=None, n_epochs=None, sampling=None,
          verbose=True):
    seed = config.RANDOM_SEED if seed is None else seed
    set_seed(seed)
    device = get_device(verbose=False)
    ds, meta, encoder = prepare_datasets(task, method, topology, use_clinical)
    g = config.get_gat_config(task)
    strategy = sampling or g['sampling_strategy']
    if len(np.unique(ds['val'].labels)) < 2:
        raise ValueError("Validation set has a single class; cannot select a model")
    train_ld, val_ld, _ = make_loaders(ds['train'], ds['val'], ds['test'], task, strategy, seed=seed)

    n_clin = len(encoder.columns) if encoder else 0
    model = GRAFSGAT(ds['train'].graphs[0].x.shape[1], n_clin, task=task).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=g['learning_rate'], weight_decay=g['weight_decay'])
    warm = g['warmup_steps']
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1.0, (e + 1) / max(warm, 1)))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=config.SCHEDULER_PATIENCE,
                                                         factor=config.SCHEDULER_FACTOR, min_lr=config.MIN_LR)
    if strategy == 'none':
        y = ds['train'].labels
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)],
                                                            dtype=torch.float, device=device))
    else:
        crit = nn.BCEWithLogitsLoss()

    rule = config.EPOCH_RULE
    n_epochs = n_epochs or (config.MATEUS_MAX_EPOCHS if rule == 'mateus' else config.N_EPOCHS)
    if rule not in config.EPOCH_RULES:
        raise ValueError(f"Unknown EPOCH_RULE {rule}")
    # un-resampled training set, only needed for the train AUC of the 'auc_gap' / 'mateus' rules
    train_eval_ld = (DataLoader(ds['train'], batch_size=g['batch_size'], shuffle=False)
                     if rule in ('auc_gap', 'mateus') else None)
    m_thr, m_diff = config.MATEUS_SAVE_RULE[(task, bool(use_clinical))]
    m_best, m_saved, stop_reason = 0.0, 0, None
    hist = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'train_auc': [], 'lr': [], 'epoch_rule': rule,
            'best_epoch': None, 'best_val_loss': float('inf'), 'best_val_auc': None}
    best, since, t0, states, best_loss = None, 0, time.time(), {}, float('inf')
    for ep in range(1, n_epochs + 1):
        tr = _epoch(model, train_ld, opt, crit, device)
        p, y, _, vl = predict(model, val_ld, device)
        va = safe_auc(y, p)
        ta = np.nan
        if train_eval_ld is not None:
            tp, ty, _, _ = predict(model, train_eval_ld, device)
            ta = safe_auc(ty, tp)
        for key, v in (('train_loss', tr), ('val_loss', vl), ('val_auc', va), ('train_auc', ta),
                       ('lr', opt.param_groups[0]['lr'])):
            hist[key].append(v)
        warmup.step() if ep <= warm else plateau.step(vl)
        if ep >= config.MIN_EPOCHS_BEFORE_SELECT:
            if rule == 'auc_gap':
                states[ep] = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            if rule == 'mateus' and np.isfinite(va) and np.isfinite(ta) and va > m_best and va > m_thr \
                    and abs(va - ta) < m_diff:
                m_best, m_saved = va, m_saved + 1
                best = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                hist.update(best_epoch=ep, best_val_loss=vl, best_val_auc=va)
            if vl < best_loss:
                best_loss, since = vl, 0
                if rule == 'val_loss':
                    hist.update(best_epoch=ep, best_val_loss=vl, best_val_auc=va)
                    best = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            else:
                since += 1
        if verbose and (ep == 1 or ep % 50 == 0):
            print(f"    ep {ep:4d} | train {tr:.4f} | val {vl:.4f} AUC {va:.3f} | {time.time() - t0:.0f}s")
        if since >= config.EARLY_STOPPING_PATIENCE and rule != 'mateus':
            stop_reason = 'val_loss_patience'
            break
        if rule == 'mateus' and np.isfinite(ta) and ta > config.MATEUS_STOP_TRAIN_AUC:
            stop_reason = 'train_auc_above_stop'
            break
    if rule == 'auc_gap' and states:
        eps = np.array(sorted(states))
        tra, vaa = np.array(hist['train_auc'])[eps - 1], np.array(hist['val_auc'])[eps - 1]
        gap, ok = config.AUC_GAP_START, np.zeros(len(eps), bool)
        while not ok.any() and gap <= 1.0 + 1e-9:
            ok = np.isfinite(vaa) & (tra - vaa <= gap + 1e-12)
            gap = gap if ok.any() else gap + config.AUC_GAP_STEP
        if ok.any():
            e = int(eps[ok][np.nanargmax(vaa[ok])])
            best = states[e]
            hist.update(best_epoch=e, best_val_loss=hist['val_loss'][e - 1], best_val_auc=hist['val_auc'][e - 1],
                        auc_gap_used=round(float(gap), 3))
        states.clear()
    if rule == 'mateus':
        hist.update(mateus_threshold=m_thr, mateus_max_diff=m_diff, mateus_n_saved=m_saved,
                    mateus_fallback_last_epoch=m_saved == 0, stop_reason=stop_reason, last_epoch=ep)
    if best is None:
        best = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        hist.update(best_epoch=ep, best_val_loss=vl, best_val_auc=va)
    model.load_state_dict(best)
    p_val, y_val, _, _ = predict(model, val_ld, device)
    thr = find_optimal_threshold(y_val, p_val)

    paths = config.exp_paths(task, method, topology, use_clinical, seed)
    torch.save({'model_state': best, 'hparams': model.hparams,
                'encoder_state': encoder.state_dict() if encoder else None,
                'use_clinical': use_clinical, 'threshold': thr, 'mode': config.MODE, 'task': task,
                'method': method, 'topology': topology, 'seed': seed,
                'graph_meta': {k_: meta[k_] for k_ in ('features', 'scaler', 'split_key')},
                'history': hist}, paths['model'])
    paths['history'].write_text(json.dumps(hist))
    if verbose:
        print(f"    selected epoch {hist['best_epoch']} | val loss {hist['best_val_loss']:.4f} "
              f"AUC {hist['best_val_auc']:.3f} | {time.time() - t0:.0f}s")
    return paths['model']


def plot_history(task, method, topology, use_clinical, seed, save_path):
    h = json.loads(config.exp_paths(task, method, topology, use_clinical, seed)['history'].read_text())
    plot_training_history(h, save_path)
