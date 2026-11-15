"""
Stage 1 + 2 of the pipeline.

Stage 1  preprocess_and_save_all():  CT + GTV -> 1 mm, 5 cm box, SLIC supervoxels
         -> outputs/cache/preprocessed/<pid>_preprocessed.npz
Stage 2  SupervoxelFeatureExtractor: PyRadiomics (93 non-shape features) for the GTV
         and every supervoxel -> outputs/cache/features/<pid>_features.npz

Both stages validate their caches: a file produced by an older/buggy version of the
code (or from a different preprocessed volume) is regenerated automatically.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from radgraph import config

logging.getLogger('radiomics').setLevel(logging.ERROR)

try:
    import radiomics
    from radiomics import featureextractor
    radiomics.setVerbosity(logging.ERROR)          # silence per-ROI info messages (e.g. GLCM symmetry note)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False


def _atomic_savez(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def preprocessed_path(pid):
    return config.PREPROCESSED_DIR / f'{pid}_preprocessed.npz'


def feature_path(pid):
    return config.FEATURE_CACHE_DIR / f'{pid}_features.npz'


# ─── Stage 1: preprocessing cache ────────────────────────────────────────────

def preprocessed_id(pid, dataset=None):
    """Identifier of a valid preprocessed file (current PREPROCESS_VERSION), or None if missing/stale."""
    f = preprocessed_path(pid)
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as d:
            if 'version' in d.files and int(d['version']) == config.PREPROCESS_VERSION:
                return str(d['created'])
    except Exception:
        return None
    return None


def preprocess_images(pid, ct_image, gtv_mask, gtv_name, dataset):
    """CT + binary GTV mask (SimpleITK images on the same grid) -> preprocessed cache file.
    Used for cohort patients and for new patients at inference time (predict.py)."""
    from radgraph.data.preprocessing import CTPreprocessor
    from radgraph.data.supervoxels import SupervoxelGenerator
    d = {'ct_image': ct_image, 'gtv_mask': gtv_mask, 'gtv_name': gtv_name}
    p = CTPreprocessor().preprocess_patient(d['ct_image'], d['gtv_mask'])
    labels, n_sv = SupervoxelGenerator().generate_supervoxels(
        p['ct_array'], p['region_array'], p['gtv_array'])
    _atomic_savez(
        preprocessed_path(pid),
        ct_array=p['ct_array'], gtv_array=p['gtv_array'],
        supervoxel_labels=labels.astype(np.int16 if n_sv < 32000 else np.int32),
        spacing=p['spacing'], bbox=p['bbox'], full_size=p['full_size'],
        origin=p['origin'], direction=p['direction'], cropped=np.array(p['cropped']),
        gtv_name=np.array(d['gtv_name']), dataset=np.array(dataset),
        version=np.array(config.PREPROCESS_VERSION),
        created=np.array(time.strftime('%Y%m%dT%H%M%S') + f'-{pid}'),
    )


def _preprocess_one(pid, dataset):
    from radgraph.data.loader import HNSCCDataLoader
    try:
        d = HNSCCDataLoader(dataset).load_patient_data(pid)
        preprocess_images(pid, d['ct_image'], d['gtv_mask'], d['gtv_name'], dataset)
        return pid, None
    except Exception as e:
        return pid, f'{type(e).__name__}: {e}'


def preprocess_and_save_all(dataset, patient_ids, force=False, n_jobs=1):
    todo = [p for p in patient_ids if force or preprocessed_id(p, dataset) is None]
    print(f"[preprocess] {dataset}: {len(patient_ids) - len(todo)} cached, {len(todo)} to do (n_jobs={n_jobs})")
    if n_jobs > 1 and todo:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_preprocess_one)(p, dataset) for p in todo)
    else:
        res = [_preprocess_one(p, dataset) for p in tqdm(todo, desc=f'Preprocess {dataset}')]
    failed = {p: e for p, e in res if e}
    for p, e in failed.items():
        print(f"  FAILED {p}: {e}")
    print(f"[preprocess] {dataset}: {len(todo) - len(failed)} new, {len(failed)} failed")
    if failed:
        log = config.LOGS_DIR / f'preprocess_failed_{dataset}.csv'
        pd.Series(failed, name='error').to_csv(log)
        print(f"  failure details -> {log}")
    return failed


# ─── Stage 2: radiomic features ──────────────────────────────────────────────

def _settings_hash():
    blob = json.dumps({'s': config.PYRADIOMICS_SETTINGS, 'c': config.FEATURE_CLASSES,
                       'v': config.FEATURE_VERSION, 'min': config.MIN_SUPERVOXEL_VOXELS},
                      sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def feature_cache_valid(pid, dataset):
    f = feature_path(pid)
    if not f.exists():
        return False
    pre_id = preprocessed_id(pid, dataset)
    try:
        with np.load(f, allow_pickle=False) as d:
            return ('settings_hash' in d.files
                    and str(d['settings_hash']) == _settings_hash()
                    and pre_id is not None
                    and str(d['preprocess_id']) == pre_id)
    except Exception:
        return False


def load_feature_dict(pid):
    with np.load(feature_path(pid), allow_pickle=False) as d:
        return {
            'gtv'          : d['gtv'].astype(np.float64),
            'supervoxels'  : d['supervoxels'].astype(np.float64),
            'centroids'    : d['centroids'].astype(np.float64),
            'gtv_centroid' : d['gtv_centroid'].astype(np.float64),
            'valid_sv_ids' : d['valid_sv_ids'],
            'feature_names': [str(x) for x in d['feature_names']],
        }


class SupervoxelFeatureExtractor:

    def __init__(self):
        if not HAS_PYRADIOMICS:
            raise ImportError("PyRadiomics required: pip install pyradiomics==3.0.1")
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**config.PYRADIOMICS_SETTINGS)
        self.extractor.disableAllFeatures()
        for fc in config.FEATURE_CLASSES:
            self.extractor.enableFeatureClassByName(fc)

    def _roi_features(self, ct_img, mask_arr, ref_img):
        m = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        m.CopyInformation(ref_img)
        res = self.extractor.execute(ct_img, m, label=1)
        names, vals = [], []
        for k, v in res.items():
            if k.startswith('diagnostics_') or '_shape_' in k:
                continue
            names.append(k)
            vals.append(float(v))
        return names, np.array(vals, dtype=np.float64)

    def extract_patient(self, pid, dataset):
        pre_id = preprocessed_id(pid, dataset)
        if pre_id is None:
            raise FileNotFoundError(f"No valid preprocessed file for {pid} (run the preprocess stage)")
        with np.load(preprocessed_path(pid), allow_pickle=False) as d:
            ct, gtv, labels = d['ct_array'], d['gtv_array'], d['supervoxel_labels']
            spacing = tuple(float(s) for s in d['spacing']) if 'spacing' in d.files else (1.0, 1.0, 1.0)

        ct_img = sitk.GetImageFromArray(ct.astype(np.float32))
        ct_img.SetSpacing(spacing)

        names, gtv_feats = self._roi_features(ct_img, gtv > 0, ct_img)
        gtv_centroid = np.argwhere(gtv > 0).mean(axis=0) * np.array(spacing[::-1])   # (z,y,x) mm

        sv_feats, cents, ids = [], [], []
        for sv in range(1, int(labels.max()) + 1):
            m = labels == sv
            if m.sum() < config.MIN_SUPERVOXEL_VOXELS:
                continue
            try:
                n2, f = self._roi_features(ct_img, m, ct_img)
            except Exception:
                continue
            if n2 != names:
                continue
            sv_feats.append(f)
            cents.append(np.argwhere(m).mean(axis=0) * np.array(spacing[::-1]))
            ids.append(sv)
        if not sv_feats:
            raise RuntimeError(f"No valid supervoxels for {pid}")

        sv_feats = np.nan_to_num(np.array(sv_feats), nan=0.0, posinf=0.0, neginf=0.0)
        gtv_feats = np.nan_to_num(gtv_feats, nan=0.0, posinf=0.0, neginf=0.0)
        _atomic_savez(
            feature_path(pid),
            gtv=gtv_feats, supervoxels=sv_feats, centroids=np.array(cents),
            gtv_centroid=gtv_centroid, valid_sv_ids=np.array(ids, dtype=np.int32),
            feature_names=np.array(names), settings_hash=np.array(_settings_hash()),
            preprocess_id=np.array(pre_id), dataset=np.array(dataset),
        )
        return len(ids)


def _extract_one(pid, dataset):
    try:
        n = SupervoxelFeatureExtractor().extract_patient(pid, dataset)
        return pid, None, n
    except Exception as e:
        return pid, f'{type(e).__name__}: {e}', 0


def extract_all_patients(dataset, patient_ids, n_jobs=1, force=False):
    todo = [p for p in patient_ids if force or not feature_cache_valid(p, dataset)]
    print(f"[extract] {dataset}: {len(patient_ids) - len(todo)} cached, {len(todo)} to extract "
          f"(n_jobs={n_jobs})")
    failed = {}
    if todo:
        if n_jobs > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_extract_one)(p, dataset) for p in todo)
        else:
            results = [_extract_one(p, dataset) for p in tqdm(todo, desc=f'Extract {dataset}')]
        failed = {p: err for p, err, _ in results if err}
    ok = [p for p in patient_ids if feature_cache_valid(p, dataset)]
    print(f"[extract] {dataset}: {len(ok)} patients with valid features, {len(failed)} failed")
    if failed:
        log = config.LOGS_DIR / f'extract_failed_{dataset}.csv'
        pd.Series(failed, name='error').to_csv(log)
        print(f"  failure details -> {log}")
    save_gtv_features_csv(dataset, ok)
    return ok, failed


def save_gtv_features_csv(dataset, patient_ids):
    """GTV-level features (one row per patient) for inspection / external tools."""
    rows = []
    for pid in patient_ids:
        fd = load_feature_dict(pid)
        rows.append(pd.Series(fd['gtv'], index=fd['feature_names'], name=pid))
    if rows:
        df = pd.DataFrame(rows)
        df.index.name = 'patient_id'
        out = config.CACHE_DIR / f'gtv_features_{dataset}.csv'
        df.to_csv(out)
        print(f"  GTV features -> {out} ({df.shape[0]} patients x {df.shape[1]} features)")


def gtv_matrix(patient_ids):
    """GTV radiomic features as a DataFrame (rows = patients, columns = 93 feature names)."""
    rows = []
    for pid in patient_ids:
        fd = load_feature_dict(pid)
        rows.append(pd.Series(fd['gtv'], index=fd['feature_names'], name=pid))
    return pd.DataFrame(rows)
