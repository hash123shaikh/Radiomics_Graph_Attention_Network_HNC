"""
Clinical data: outcome labels + clinical feature encoding.

Single source of truth for every stage (splits, features, models), so that all of them see the
same patients, labels and clinical features.

  load_clinical(dataset)                  -> raw DataFrame indexed by patient id (str)
  compute_labels(dataset, task)           -> DataFrame[label, included, event, time_months]
  encode_clinical(dataset, harmonized)    -> (encoded DataFrame, quantitative columns)
  ClinicalEncoder                         -> fit on TRAIN rows: median imputation + min-max
"""

import re

import numpy as np
import pandas as pd

from hnc_popgraph import config


# ─── Raw data ────────────────────────────────────────────────────────────────

def _load_hnpet(cfg):
    """Head-Neck-PET-CT INFOclinical xlsx: one sheet per centre; IDs 'HN-CHUM-001' -> 'CHUM-001'."""
    frames = []
    for sh in cfg['clinical_sheets']:
        d = pd.read_excel(cfg['clinical_file'], sheet_name=sh)
        # normalise headers: en-dash, double spaces, 'follow-up(days)' vs 'follow-up (days)'
        d.columns = [re.sub(r'\s+', ' ', str(c).replace('\u2013', '-')).replace(' (', '(').strip()
                     for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df['Patient #'].astype(str).str.strip().str.match(r'^HN-[A-Z]+-\d+$')].copy()   # drop footnotes
    df['patient_id'] = df['Patient #'].astype(str).str.strip().str.replace(r'^HN-', '', regex=True)
    fu = pd.to_numeric(df['Time - diagnosis to last follow-up(days)'], errors='coerce')
    for ev, tcol, out in (('Locoregional', 'Time - diagnosis to LR(days)', 'lr'),
                          ('Distant', 'Time - diagnosis to DM(days)', 'dm')):
        e = pd.to_numeric(df[ev], errors='coerce')
        t = pd.to_numeric(df[tcol], errors='coerce')
        df[f'{out}_event'] = e
        # time = time to event for events, last follow-up otherwise
        df[f'{out}_time_days'] = np.where(e == 1, t, fu)
    death = pd.to_numeric(df['Death'], errors='coerce')
    df['os_event'] = death
    df['os_time_days'] = np.where(death == 1, pd.to_numeric(df['Time - diagnosis to Death(days)'], errors='coerce'), fu)
    return df


def load_clinical(dataset):
    cfg = config.dataset_cfg(dataset)
    if cfg['schema'] == 'HNPET':
        df = _load_hnpet(cfg)
    else:
        df = pd.read_csv(cfg['clinical_file'])
    df[cfg['patient_id_col']] = df[cfg['patient_id_col']].astype(str).str.strip()
    if df[cfg['patient_id_col']].duplicated().any():
        dups = df.loc[df[cfg['patient_id_col']].duplicated(), cfg['patient_id_col']].tolist()
        raise ValueError(f"{dataset}: duplicated patient IDs in clinical table: {dups[:10]}")
    return df.set_index(cfg['patient_id_col'], drop=False)


# ─── Labels ──────────────────────────────────────────────────────────────────

def compute_labels(dataset, task, verbose=True):
    """
    Binary outcome at a fixed horizon (config.horizon_months: 2 years for LR/DM, 4 years for OS,
    as in Mateus et al. 2023); patients followed for less than the horizon without an event are excluded.

      label 1 : event within the horizon
      label 0 : event-free for >= the horizon (later events count as 0)
      excluded: event-free follow-up shorter than the horizon, or missing event/time
    """
    cfg = config.dataset_cfg(dataset)
    if task not in cfg['outcomes']:
        raise ValueError(
            f"Task '{task}' is not available for {dataset} "
            f"(available: {list(cfg['outcomes'])}).")

    df = load_clinical(dataset)
    event_col, time_col = cfg['outcomes'][task]
    event = pd.to_numeric(df[event_col], errors='coerce')
    time  = pd.to_numeric(df[time_col],  errors='coerce')
    if cfg['time_unit'] == 'days':
        time_m = time / config.DAYS_PER_MONTH
    elif cfg['time_unit'] == 'months':
        time_m = time
    else:
        raise ValueError(f"Unknown time_unit {cfg['time_unit']}")

    H = config.horizon_months(task)
    known    = event.notna() & time_m.notna()
    positive = known & (event == 1) & (time_m <= H)
    negative = known & (time_m >= H) & ~positive
    included = positive | negative

    out = pd.DataFrame({
        'label'      : np.where(positive, 1, np.where(negative, 0, -1)),
        'included'   : included,
        'event'      : event,
        'time_months': time_m,
    }, index=df.index)

    if verbose:
        n_late = int((known & (event == 1) & (time_m > H)).sum())
        print(f"[labels] {dataset} {task}: {int(included.sum())}/{len(df)} included "
              f"({int(positive.sum())} positive, {int(negative.sum())} negative; "
              f"{int((~included).sum())} excluded: censored <{H} mo or missing; "
              f"{n_late} events after {H} mo counted as negative)")
    return out


def included_labels(dataset, task, verbose=True):
    """dict {patient_id: label} for included patients only."""
    lab = compute_labels(dataset, task, verbose=verbose)
    lab = lab[lab['included']]
    return {pid: int(v) for pid, v in lab['label'].items()}


# ─── Clinical feature encoding ───────────────────────────────────────────────

def _one_hot(series, categories, prefix):
    s = series.astype(str).str.strip().str.lower()
    return pd.DataFrame({f'{prefix}_{c}': (s == c).astype(np.float32) for c in categories},
                        index=series.index)


def _hpv_one_hot(pos_mask, neg_mask, index):
    """HPV as positive / negative / unknown (paper Table: Unknown is a category)."""
    return pd.DataFrame({
        'hpv_positive': pos_mask.astype(np.float32),
        'hpv_negative': neg_mask.astype(np.float32),
        'hpv_unknown' : (~(pos_mask | neg_mask)).astype(np.float32),
    }, index=index)


def _encode_maastro_full(df):
    """Paper clinical set: chemo, HPV, sex, age, AJCC stage, ECOG, subsite, tumour volume."""
    out = pd.DataFrame(index=df.index)
    chemo = df['concurrent_chemo'].astype(str).str.strip().str.lower()
    out['concurrent_chemo'] = chemo.isin(['concomitant', 'concurrent']).astype(np.float32)
    hpv = df['hpv_status'].astype(str).str.strip().str.lower()
    out = out.join(_hpv_one_hot(hpv == 'positive', hpv == 'negative', df.index))
    out['sex_male'] = (df['sex'].astype(str).str.strip().str.lower() == 'male').astype(np.float32)
    out = out.join(_one_hot(df['ajcc_stage'], ['i', 'ii', 'iii', 'iva', 'ivb', 'ivc'], 'ajcc'))
    out = out.join(_one_hot(df['tumor_subsite'], ['oropharynx', 'larynx'], 'subsite'))
    out['age']          = pd.to_numeric(df['age'], errors='coerce')
    out['ecog']         = pd.to_numeric(df['ecog_status'], errors='coerce')
    out['tumor_volume'] = pd.to_numeric(df['tumor_volume'], errors='coerce')
    return out, ['age', 'ecog', 'tumor_volume']


def _hnpet_basic(df):
    """Variables shared by the HNPET encodings."""
    out = pd.DataFrame(index=df.index)
    out['concurrent_chemo'] = df['Therapy'].astype(str).str.lower().str.contains('chemo').astype(np.float32)
    hpv = df['HPV status'].astype(str).str.strip()
    out = out.join(_hpv_one_hot(hpv == '+', hpv == '-', df.index))
    out['sex_male'] = (df['Sex'].astype(str).str.strip().str.upper() == 'M').astype(np.float32)
    t = df['T-stage'].astype(str).str.strip().str.lower().str[:2]      # T4a/T4b -> t4, Tx -> tx
    n = df['N-stage'].astype(str).str.strip().str.lower().str[:2]      # N2a/b/c -> n2
    site = df['Primary Site'].astype(str).str.strip().str.lower()
    return out, t, n, site


def _encode_hnpet_full(df):
    """All Head-Neck-PET-CT variables mapping onto the paper's clinical set (no ECOG/volume available)."""
    out, t, n, site = _hnpet_basic(df)
    for c in ['t1', 't2', 't3', 't4']:
        out[f't_{c}'] = (t == c).astype(np.float32)
    for c in ['n0', 'n1', 'n2', 'n3']:
        out[f'n_{c}'] = (n == c).astype(np.float32)
    for c in ['oropharynx', 'larynx', 'hypopharynx', 'nasopharynx']:
        out[f'subsite_{c}'] = (site == c).astype(np.float32)
    out['age'] = pd.to_numeric(df['Age'], errors='coerce')
    return out, ['age']


HARMONIZED_COLUMNS = ['concurrent_chemo', 'hpv_positive', 'hpv_negative', 'hpv_unknown', 'sex_male',
                      't_t1', 't_t2', 't_t3', 't_t4', 'n_n0', 'n_n1', 'n_n2', 'n_n3',
                      'subsite_oropharynx', 'subsite_larynx', 'age']


def _encode_harmonized(df, schema):
    """Variables present in BOTH MAASTRO and HNPET, identically encoded (cross-cohort modes)."""
    if schema == 'HNPET':
        out, t, n, site = _hnpet_basic(df)
        out['age'] = pd.to_numeric(df['Age'], errors='coerce')
    elif schema == 'MAASTRO':
        out = pd.DataFrame(index=df.index)
        chemo = df['concurrent_chemo'].astype(str).str.strip().str.lower()
        out['concurrent_chemo'] = chemo.isin(['concomitant', 'concurrent']).astype(np.float32)
        hpv = df['hpv_status'].astype(str).str.strip().str.lower()
        out = out.join(_hpv_one_hot(hpv == 'positive', hpv == 'negative', df.index))
        out['sex_male'] = (df['sex'].astype(str).str.strip().str.lower() == 'male').astype(np.float32)
        t = 't' + df['tstage'].astype(str).str.strip().str.lower().str[:1]
        n = 'n' + df['nstage'].astype(str).str.strip().str.lower().str[:1]
        site = df['tumor_subsite'].astype(str).str.strip().str.lower()
        out['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        raise ValueError(schema)
    for c in ['t1', 't2', 't3', 't4']:
        out[f't_{c}'] = (t == c).astype(np.float32)
    for c in ['n0', 'n1', 'n2', 'n3']:
        out[f'n_{c}'] = (n == c).astype(np.float32)
    for c in ['oropharynx', 'larynx']:
        out[f'subsite_{c}'] = (site == c).astype(np.float32)
    return out[HARMONIZED_COLUMNS], ['age']


def encode_clinical(dataset, harmonized=False):
    """
    Returns
    -------
    encoded : DataFrame indexed by patient id. Categorical -> 0/1 (one-hot);
              quantitative columns may contain NaN (imputed later by ClinicalEncoder).
    quant_cols : list[str]
    """
    df     = load_clinical(dataset)
    schema = config.dataset_cfg(dataset)['schema']
    if harmonized:
        enc, quant = _encode_harmonized(df, schema)
    elif schema == 'MAASTRO':
        enc, quant = _encode_maastro_full(df)
    elif schema == 'HNPET':
        enc, quant = _encode_hnpet_full(df)
    else:
        raise ValueError(schema)
    return enc.astype(np.float32), quant


class ClinicalEncoder:
    """
    Fit on training patients only:
      - quantitative columns: median imputation, then min-max to [0, 1] (Appendix S1)
      - categorical columns : already 0/1
    """

    def __init__(self, columns, quant_cols):
        self.columns    = list(columns)
        self.quant_cols = [c for c in quant_cols if c in self.columns]
        self.median_ = {}
        self.min_    = {}
        self.max_    = {}

    def fit(self, encoded_df, train_ids):
        tr = encoded_df.loc[[p for p in train_ids if p in encoded_df.index], self.columns]
        if len(tr) == 0:
            raise ValueError("ClinicalEncoder.fit: no training patients found in clinical table")
        for c in self.quant_cols:
            col = tr[c].astype(float)
            med = float(col.median()) if col.notna().any() else 0.0
            filled = col.fillna(med)
            self.median_[c] = med
            self.min_[c]    = float(filled.min())
            self.max_[c]    = float(filled.max())
        return self

    def transform(self, encoded_df, ids):
        missing = [p for p in ids if p not in encoded_df.index]
        if missing:
            raise KeyError(f"Clinical data missing for {len(missing)} patients, e.g. {missing[:5]}")
        X = encoded_df.loc[list(ids), self.columns].copy()
        for c in self.quant_cols:
            rng = self.max_[c] - self.min_[c]
            X[c] = (X[c].astype(float).fillna(self.median_[c]) - self.min_[c]) / (rng if rng > 0 else 1.0)
        X = X.fillna(0.0).astype(np.float32)
        return {pid: X.loc[pid].values for pid in ids}

    def state_dict(self):
        return {'columns': self.columns, 'quant_cols': self.quant_cols,
                'median': self.median_, 'min': self.min_, 'max': self.max_}

    @classmethod
    def from_state(cls, s):
        enc = cls(s['columns'], s['quant_cols'])
        enc.median_, enc.min_, enc.max_ = s['median'], s['min'], s['max']
        return enc


def clinical_tables_for_mode(task=None):
    """
    Encoded clinical tables for every cohort of the current mode, keyed by dataset.
    Multi-cohort modes use the harmonized schema so all columns match.
    Returns (tables, columns, quantitative columns).
    """
    harmonized = config.harmonized_clinical()
    tables, quant = {}, None
    for ds in config.datasets_in_mode():
        tables[ds], quant = encode_clinical(ds, harmonized=harmonized)
    cols = list(tables[config.TRAIN_DATASETS[0]].columns)
    for ds, t in tables.items():
        if list(t.columns) != cols:
            raise RuntimeError(f"Clinical columns differ between cohorts ({ds})")
    return tables, cols, quant


def stacked_table(tables, ids):
    """One encoded table containing the rows of `ids`, drawn from whichever cohort has them."""
    rows = []
    for t in tables.values():
        rows.append(t.loc[[p for p in ids if p in t.index]])
    return pd.concat(rows)


if __name__ == '__main__':
    for ds in config.DATASETS:
        for task in config.DATASETS[ds]['outcomes']:
            compute_labels(ds, task)
        enc, q = encode_clinical(ds)
        print(ds, 'full encoding:', enc.shape, 'quant:', q)
        enc, q = encode_clinical(ds, harmonized=True)
        print(ds, 'harmonized   :', enc.shape)
