"""
Shared helpers for the GTV-only experiments: feature tables, the study splits (identical to the
main GRAFS-GAT runs: MATEUS_CV 5 folds and MATEUS_COHORT, external test = MAASTRO), clinical vectors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config                                         # noqa: E402

FEAT_DIR = config.OUTPUT_DIR / 'gtv_only' / 'features'
RES_DIR = config.OUTPUT_DIR / 'gtv_only' / 'results'
RES_DIR.mkdir(parents=True, exist_ok=True)
DESIGNS = ['MATEUS_CV', 'MATEUS_COHORT']


def load_npz(pid):
    with np.load(FEAT_DIR / f'{pid}.npz', allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def gtv_table():
    """All whole-GTV features (shape + original + LoG + wavelet), one row per patient (cached CSV)."""
    f = RES_DIR / 'gtv_features_all.csv'
    if f.exists():
        return pd.read_csv(f, index_col=0)
    rows = {}
    for p in sorted(FEAT_DIR.glob('*.npz')):
        if p.name.endswith('.tmp.npz'):
            continue
        d = load_npz(p.stem)
        rows[p.stem] = pd.Series(d['gtv'], index=[str(x) for x in d['gtv_names']])
    df = pd.DataFrame(rows).T
    df.index.name = 'patient_id'
    df.to_csv(f)
    return df


FEATURE_SETS = {
    'orig'      : lambda c: c.startswith('original_') and '_shape_' not in c,
    'orig+shape': lambda c: c.startswith('original_'),
    'full'      : lambda c: True,
}


def splits(task, design):
    """List of (fold, split dict) for the design; reuses the split files of the main study."""
    from grafsgat.data.splits import get_or_create_split
    config.set_mode(design)
    config.EPOCH_RULE = 'val_loss'
    config.set_task(task)
    out = []
    folds = range(1, config.N_FOLDS + 1) if config.split_scheme() == 'cv' else [None]
    for k in folds:
        config.set_fold(k)
        out.append((k, get_or_create_split(task, verbose=False)))
    config.set_fold(None)
    return out


def clinical_matrix(task, split):
    """Encoded clinical vectors (encoder fitted on the training patients of this split)."""
    from grafsgat.data.clinical import ClinicalEncoder, clinical_tables_for_mode, stacked_table
    tables, cols, quant = clinical_tables_for_mode(task)
    ids = split['train'] + split['val'] + split['test']
    table = stacked_table(tables, ids)
    enc = ClinicalEncoder(cols, quant).fit(table, split['train'])
    return {s: np.vstack(list(enc.transform(table, split[s]).values())) for s in ('train', 'val', 'test')}, cols


class Prep:
    """Train-fitted: median impute, signed log1p of heavy-tailed columns, z-score, clip +/-5."""

    def fit(self, X):
        X = X.astype(float)
        self.med = X.median()
        X = X.fillna(self.med)
        sk = X.skew().abs()
        self.heavy = list(sk.index[sk > 2.0])
        X = self._log(X)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1).fillna(1)
        return self

    def _log(self, X):
        X = X.copy()
        X[self.heavy] = np.sign(X[self.heavy]) * np.log1p(np.abs(X[self.heavy]))
        return X

    def transform(self, X):
        X = self._log(X.astype(float).fillna(self.med))
        return np.clip(((X - self.mu) / self.sd).values, -5, 5)
