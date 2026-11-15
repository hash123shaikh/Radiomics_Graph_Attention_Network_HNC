"""
Inference with saved GAT checkpoints: no retraining.

Every checkpoint (.pth) is self-contained: weights, hyper-parameters, the selected features, the
GTV z-score scaler fitted on that model's training patients, the clinical encoder (if used), the
validation-derived decision threshold, the graph topology and the task. A "model" here is an
ensemble of checkpoints (5 folds x 3 seeds for MATEUS_CV, 3 seeds for MATEUS_COHORT), exactly the
ensemble whose external test AUC is reported.

  find_checkpoints(task, design, ...)    -> list of .pth paths for one configuration
  ensure_features(pid, dataset)          -> preprocess + extract a cohort patient if not cached
  ensure_features_from_files(pid, ct, mask, label)  -> same for a new patient's NIfTI files
  predict_patients(checkpoints, pids, clinical=None) -> DataFrame of ensemble probabilities
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from radgraph import config
from radgraph.data.features import (feature_cache_valid, load_feature_dict, preprocess_images, preprocessed_id,
                                    _preprocess_one, SupervoxelFeatureExtractor)
from radgraph.graphs.builder import GTVScaler, build_patient_graph, _idx
from radgraph.graphs.dataset import RadGraphDataset
from radgraph.models.gat import RadGraphGAT
from radgraph.utils import get_device

FINAL_MODELS_DIR = config.OUTPUT_DIR / 'final_models'


def run_folder(task, design, epoch_rule='val_loss'):
    return config.RUNS_DIR / f"{design}_{task}{'' if epoch_rule == 'val_loss' else '_' + epoch_rule.replace('_', '')}"


def chosen_config(task, design, inputs='imaging', epoch_rule='val_loss'):
    """(method, topology, use_clinical) chosen on validation AUC in that run's results_summary.csv."""
    f = run_folder(task, design, epoch_rule) / 'results_summary.csv'
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run the study for {design} {task} first")
    df = pd.read_csv(f)
    b = df[df.best_by_val & (df.inputs == inputs) & (df.model == 'GAT')]
    if b.empty:
        raise ValueError(f"No chosen GAT configuration for {inputs} in {f}")
    r = b.iloc[0]
    return r.method, r.topology, bool(r.use_clinical)


def find_checkpoints(task, design, method, topology, use_clinical, epoch_rule='val_loss'):
    run = run_folder(task, design, epoch_rule)
    pattern = f"{method}__{topology}__{'img+clin' if use_clinical else 'img'}__s*.pth"
    files = sorted(run.glob(f'fold*/models/{pattern}')) or sorted(run.glob(f'models/{pattern}'))
    if not files:
        raise FileNotFoundError(f"No checkpoints {pattern} under {run}")
    return files


def checkpoints_in(folder):
    files = sorted(Path(folder).rglob('*.pth'))
    if not files:
        raise FileNotFoundError(f"No .pth checkpoints in {folder}")
    return files


# ─── Features for the patients to score ─────────────────────────────────────

def ensure_features(pid, dataset):
    """Cohort patient (images in that cohort's folder): preprocess + extract if not cached."""
    if feature_cache_valid(pid, dataset):
        return
    if preprocessed_id(pid, dataset) is None:
        _, err = _preprocess_one(pid, dataset)
        if err:
            raise RuntimeError(f"{pid}: preprocessing failed: {err}")
    SupervoxelFeatureExtractor().extract_patient(pid, dataset)


def ensure_features_from_files(pid, ct_path, mask_path, label=1, force=False):
    """New patient: CT (.nii/.nii.gz/.nrrd) + segmentation with the primary GTV as `label`."""
    import SimpleITK as sitk
    if not force and feature_cache_valid(pid, 'NEW'):
        return
    ct = sitk.ReadImage(str(ct_path))
    seg = sitk.Resample(sitk.ReadImage(str(mask_path)), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                        sitk.sitkUInt8)
    mask = sitk.BinaryThreshold(seg, label, label, 1, 0)
    if sitk.GetArrayViewFromImage(mask).sum() == 0:
        raise ValueError(f"{pid}: no voxels with label {label} in {mask_path}")
    preprocess_images(pid, ct, mask, f'label{label}', 'NEW')
    SupervoxelFeatureExtractor().extract_patient(pid, 'NEW')


# ─── Prediction ──────────────────────────────────────────────────────────────

def _load(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    hp = dict(ck['hparams'])
    model = RadGraphGAT(hp.pop('n_node_features'), hp.pop('n_clinical'), task=hp.pop('task'), **hp)
    model.load_state_dict(ck['model_state'])
    return model.to(device).eval(), ck


@torch.no_grad()
def _scores(model, graphs, clin, device, batch_size=64):
    ds = RadGraphDataset(graphs, clin)
    out = []
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False):
        b = b.to(device)
        out.append(torch.sigmoid(model(b)).cpu().numpy())
    return np.concatenate(out)


def predict_patients(checkpoints, pids, clinical_table=None, labels=None, train_ids=None, per_model=False):
    """
    checkpoints    : list of .pth paths (one ensemble)
    pids           : patient ids whose features are cached (see ensure_features*)
    clinical_table : encoded clinical DataFrame indexed by patient id (needed if the models use clinical data)
    labels         : optional {pid: label} (added to the output, e.g. for evaluation)
    train_ids      : optional set of ids used to train ANY of the checkpoints (flagged in the output)
    Returns one row per patient: prob (ensemble mean), prob_sd, n_models, threshold (mean of the
    models' validation Youden thresholds), predicted_high_risk, [label], [used_in_training], [p_<model>].
    """
    device = get_device(verbose=False)
    fds = {p: load_feature_dict(p) for p in pids}
    probs, thrs, names, meta = [], [], [], None
    for path in checkpoints:
        model, ck = _load(path, device)
        gm = ck['graph_meta']
        sc = GTVScaler()
        sc.mean_, sc.scale_ = np.asarray(gm['scaler']['mean']), np.asarray(gm['scaler']['scale'])
        graphs = []
        for p in pids:
            fd = fds[p]
            g = build_patient_graph(fd, _idx(fd['feature_names'], gm['features']), sc, ck['topology'], 0, p)
            graphs.append(g)
        clin = None
        if ck['use_clinical']:
            if clinical_table is None:
                raise ValueError(f"{Path(path).name} uses clinical data: pass the encoded clinical table")
            from radgraph.data.clinical import ClinicalEncoder
            clin = ClinicalEncoder.from_state(ck['encoder_state']).transform(clinical_table, pids)
        probs.append(_scores(model, graphs, clin, device))
        thrs.append(float(ck['threshold']))
        fold = next((q.name for q in (Path(path).parent, Path(path).parent.parent) if q.name.startswith('fold')), '')
        names.append(f"{fold}_s{ck['seed']}" if fold else f"s{ck['seed']}")
        meta = meta or {k: ck[k] for k in ('task', 'method', 'topology', 'use_clinical', 'mode')}
    P = np.vstack(probs)                                     # (n_models, n_patients)
    thr = float(np.mean(thrs))
    out = pd.DataFrame({'patient_id': pids, 'prob': P.mean(0), 'prob_sd': P.std(0), 'n_models': len(P),
                        'threshold': thr})
    out['predicted_high_risk'] = (out.prob >= thr).astype(int)
    if labels is not None:
        out['label'] = [labels.get(p, np.nan) for p in pids]
    if train_ids is not None:
        out['used_in_training'] = [p in train_ids for p in pids]
    if per_model:
        for n, row in zip(names, P):
            out[f'p_{n}'] = row
    for k, v in meta.items():
        out[k] = v
    return out


def training_ids(checkpoints):
    """Union of training+validation patients of the runs these checkpoints come from."""
    ids = set()
    for run in {Path(c).parent for c in checkpoints} | {Path(c).parent.parent for c in checkpoints}:
        f = run / 'split.json'
        if f.exists():
            s = json.loads(f.read_text())
            ids |= set(s['train']) | set(s['val'])
    return ids
