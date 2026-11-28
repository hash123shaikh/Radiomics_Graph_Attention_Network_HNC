"""
Eligible patients and the train / validation / test split of every design, shared by all models.

  eligible  labelled patients (endpoint-specific inclusion) with a preprocessed CT and primary GTV
  test      every eligible MAASTRO patient
  train/val inside the Canadian cohort:
    'cv'      fold config.FOLD of a stratified N_FOLDS-fold CV (val = that fold)   [Mateus et al. 2023]
    'cohort'  by centre: config.COHORT_SPLIT['train'] / ['val']                    [Diamant et al. 2019]
Splits are saved in outputs/splits/ and recreated only when the patient set changes.
"""

import hashlib
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from hnc_popgraph import config
from hnc_popgraph.data.clinical import included_labels
from hnc_popgraph.data.preprocessing import is_preprocessed


def eligible(dataset, task, verbose=True):
    labels = included_labels(dataset, task, verbose=verbose)
    have = {p: l for p, l in labels.items() if is_preprocessed(p)}
    if verbose:
        print(f"[data] {dataset} {task}: {len(have)}/{len(labels)} labelled patients have a preprocessed CT + GTV")
    return have


def mode_labels(task, verbose=True):
    """Returns (train_labels, test_labels, cohort) for the current mode.
    train_labels/test_labels: {pid: label}; cohort: {pid: dataset}."""
    if not config.mode_task_available(task):
        raise ValueError(f"Task {task} is not available for every cohort of mode {config.MODE}")
    train, test, cohort = {}, {}, {}
    for ds in config.TRAIN_DATASETS:
        lab = eligible(ds, task, verbose)
        train.update(lab)
        cohort.update({p: ds for p in lab})
    for ds in config.TEST_DATASETS:
        lab = eligible(ds, task, verbose)
        test.update(lab)
        cohort.update({p: ds for p in lab})
    overlap = set(train) & set(test)
    if overlap:
        raise RuntimeError(f"Patient IDs present in train and test cohorts: {sorted(overlap)[:5]}")
    return train, test, cohort


def _hash(labels):
    return hashlib.md5(json.dumps(sorted(labels.items())).encode()).hexdigest()[:12]


def _strata(ids, labels, cohort, min_count=4):
    centre = [config.patient_centre(cohort[p], p) for p in ids]
    s = np.array([f'{c}_{labels[p]}' for c, p in zip(centre, ids)])
    _, counts = np.unique(s, return_counts=True)
    if counts.min() < min_count:              # stratum too small -> stratify on the label only
        s = np.array([str(labels[p]) for p in ids])
    return s


def _train_val(ids, labels, cohort, seed):
    """train/val inside the training cohort(s) of an external mode."""
    scheme = config.split_scheme()
    if scheme == 'cv':
        if config.FOLD is None:
            raise RuntimeError(f"{config.MODE} is a cross-validation mode: call config.set_fold(k) first")
        # outcome-stratified folds (Mateus et al.); centre x label when every stratum can fill all folds
        strata = _strata(ids, labels, cohort, min_count=config.N_FOLDS)
        folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=seed).split(ids, strata))
        tr, va = folds[config.FOLD - 1]
        return [ids[i] for i in tr], [ids[i] for i in va]
    if scheme == 'cohort':
        centre = {p: config.patient_centre(cohort[p], p) for p in ids}
        tr = [p for p in ids if centre[p] in config.COHORT_SPLIT['train']]
        va = [p for p in ids if centre[p] in config.COHORT_SPLIT['val']]
        unused = sorted({centre[p] for p in ids} - set(config.COHORT_SPLIT['train'] + config.COHORT_SPLIT['val']))
        if unused:
            raise ValueError(f"Centres {unused} are in neither COHORT_SPLIT['train'] nor ['val']")
        return tr, va
    raise ValueError(f'Unknown split scheme {scheme}')


def get_or_create_split(task, verbose=True):
    """Load the saved split of this run, or create it. Recreated if the patient set changes."""
    train_labels, test_labels, cohort = mode_labels(task, verbose)
    if not train_labels:
        raise RuntimeError(f"No preprocessed training patients in {config.PREPROCESSED_DIR}: run scripts/preprocess.py")
    keyed = {**train_labels, **{f'test:{p}': v for p, v in test_labels.items()}}
    keyed['_scheme'] = f'{config.split_scheme()}:{config.FOLD}:{config.N_FOLDS}:{config.COHORT_SPLIT}'
    key = _hash(keyed)
    f = config.split_file(task)
    if f.exists():
        s = json.loads(f.read_text())
        if s.get('key') == key:
            return s
        print(f"[split] patient set changed since {f} was made -> new split "
              f"(results produced with the old split are no longer comparable)")

    ids = sorted(train_labels)
    y = np.array([train_labels[p] for p in ids])
    if min(np.bincount(y, minlength=2)) < 3:
        raise ValueError(f"Too few patients per class to split: {np.bincount(y, minlength=2)}")
    seed = config.RANDOM_SEED

    train, val = _train_val(ids, train_labels, cohort, seed)
    test = sorted(test_labels)

    labels = {**train_labels, **test_labels}
    s = {'key': key, 'mode': config.MODE, 'task': task, 'scheme': config.split_scheme(), 'fold': config.FOLD,
         'train': sorted(train), 'val': sorted(val), 'test': sorted(test),
         'labels': {p: int(labels[p]) for p in sorted(labels)},
         'cohort': {p: cohort[p] for p in sorted(labels)}}
    f.write_text(json.dumps(s, indent=1))
    if verbose:
        for n in ('train', 'val', 'test'):
            print(f"[split] {n:5s}: {len(s[n]):4d} ({sum(labels[p] for p in s[n])} positive)")
    return s


def split_summary(split):
    """Per split x centre: n and positives (for reports)."""
    rows = []
    for name in ('train', 'val', 'test'):
        by = {}
        for p in split[name]:
            c = config.patient_centre(split['cohort'][p], p)
            n, k = by.get(c, (0, 0))
            by[c] = (n + 1, k + split['labels'][p])
        for c, (n, k) in sorted(by.items()):
            rows.append({'split': name, 'centre': c, 'n': n, 'positive': k})
    return rows


def study_splits(task, design):
    """[(fold, split)] for a design: five folds for 'MATEUS_CV', one split (fold None) for 'MATEUS_COHORT'."""
    config.set_mode(design)
    config.set_task(task)
    out = []
    folds = range(1, config.N_FOLDS + 1) if config.split_scheme() == 'cv' else [None]
    for k in folds:
        config.set_fold(k)
        out.append((k, get_or_create_split(task, verbose=False)))
    config.set_fold(None)
    return out
