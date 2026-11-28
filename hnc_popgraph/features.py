"""
Model inputs: radiomic feature tables, feature subsets, the training-fitted preprocessing of node
features, and the encoded clinical variables.
"""
import numpy as np
import pandas as pd

from hnc_popgraph import config


def primary_gtv_table():
    """Primary-GTV features (shape + original + LoG + wavelet), one row per patient. Built once from
    outputs/features/primary_gtv/<pid>.npz and cached as primary_gtv_features.csv."""
    f = config.PRIMARY_GTV_CSV
    if f.exists():
        return pd.read_csv(f, index_col=0)
    rows = {}
    for p in sorted(config.PRIMARY_GTV_DIR.glob('*.npz')):
        if p.name.endswith('.tmp.npz'):
            continue
        with np.load(p, allow_pickle=False) as d:
            rows[p.stem] = pd.Series(d['gtv'], index=[str(x) for x in d['gtv_names']])
    df = pd.DataFrame(rows).T
    df.index.name = 'patient_id'
    df.to_csv(f)
    return df


def whole_tumour_table():
    """Whole-tumour (primary + nodal GTV) features: shape + original + LoG, 479 per patient."""
    return pd.read_csv(config.WHOLE_TUMOUR_CSV, index_col=0)


def feature_table(region):
    """region: 'primary' (primary GTV) or 'total' (whole tumour)."""
    return whole_tumour_table() if region == 'total' else primary_gtv_table()


FEATURE_SETS = {
    'orig'      : lambda c: c.startswith('original_') and '_shape_' not in c,
    'orig+shape': lambda c: c.startswith('original_'),
    'full'      : lambda c: True,
}


class Prep:
    """Train-fitted: median imputation, signed log1p of heavy-tailed columns (|skewness| > 2), z-score, clip +/-5."""

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


def clinical_matrix(task, split):
    """Extended clinical set (age, sex, HPV, T, N, subsite, chemotherapy), encoder fitted on the training patients."""
    from hnc_popgraph.data.clinical import ClinicalEncoder, clinical_tables_for_mode, stacked_table
    tables, cols, quant = clinical_tables_for_mode(task)
    ids = split['train'] + split['val'] + split['test']
    table = stacked_table(tables, ids)
    enc = ClinicalEncoder(cols, quant).fit(table, split['train'])
    return {s: np.vstack(list(enc.transform(table, split[s]).values())) for s in ('train', 'val', 'test')}, cols
