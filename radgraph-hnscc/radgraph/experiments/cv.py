"""
Cross-fold summary of a cross-validation mode (MATEUS_CV), following Mateus et al., Sci Rep 2023:
5-fold stratified CV on the training cohort, every fold's model tested on the same external cohort.

Each fold k is an ordinary run in <mode folder>/fold<k>/ (its own split, feature selection, graphs,
GAT, RF baselines, report). This module combines them in <mode folder>/:

  test  (external cohort, identical in every fold)
      test_auc               AUC of the fold ensemble: seed-ensembled probabilities averaged over the
                             folds, with bootstrap 95% CI
      test_auc_fold_mean/sd/min/max   per-fold AUCs (Mateus et al. report this range)
  validation
      val_auc                pooled out-of-fold AUC (every training patient is validated exactly once)
  DeLong tests (fold-ensemble predictions): vs the paper reference and vs RF on the same features, Holm
  feature-selection stability ACROSS the folds (cv_jaccard, cv_kuncheva, cv_kendall_w, cv_core80),
  plus the per-fold bootstrap / cross-centre stability averaged over folds (boot_*, centre_*)
  best configuration = highest pooled out-of-fold validation AUC (per input set)

  -> results_summary.csv, results_per_fold.csv, selection_stability_cv.csv, figures/, report.html
"""

import json

import numpy as np
import pandas as pd

from radgraph import config
from radgraph.experiments.benchmark import (_ensembles, comparison_table, mark_best_by_val, print_table,
                                            result_figures)
from radgraph.selection.methods import FIXED
from radgraph.selection import stability as st


def _fold_dirs(task):
    return {k: config.mode_dir(task) / f'fold{k}' for k in range(1, config.N_FOLDS + 1)}


def collect(task):
    """Per-fold ensembles, per-fold summaries and splits. Raises if a fold is unfinished."""
    ens, rows, splits = {}, [], {}
    try:
        for k, d in _fold_dirs(task).items():
            if not (d / 'results_summary.csv').exists():
                raise RuntimeError(f"Fold {k} has no results_summary.csv yet ({d})")
            config.set_fold(k)
            ens[k], _ = _ensembles(task)
            r = pd.read_csv(d / 'results_summary.csv')
            r.insert(0, 'fold', k)
            rows.append(r)
            splits[k] = json.loads(config.split_file(task).read_text())
    finally:
        config.set_fold(None)
    return ens, pd.concat(rows, ignore_index=True), splits


def combine(fold_ens):
    """Fold ensemble: test = mean probability over folds; val = out-of-fold predictions."""
    keys = set.intersection(*[set(e) for e in fold_ens.values()])
    comb, spread = {}, {}
    for key in keys:
        parts = [fold_ens[k][key].assign(fold=k) for k in sorted(fold_ens)]
        cat = pd.concat(parts)
        test = cat[cat.split == 'test'].groupby('patient_id', as_index=False).agg(
            label=('label', 'first'), prob=('prob', 'mean'))
        test['split'] = 'test'
        val = cat[cat.split == 'val'][['patient_id', 'label', 'prob', 'split']]
        if val.patient_id.duplicated().any():
            raise RuntimeError("A patient is in the validation set of more than one fold")
        e = pd.concat([val, test], ignore_index=True)
        e['n_seeds'] = int(parts[0].n_seeds.iloc[0])
        comb[key] = e
        from radgraph.utils import safe_auc
        spread[key] = {sp: [safe_auc(p[p.split == sp].label, p[p.split == sp].prob) for p in parts]
                       for sp in ('val', 'test')}
    missing = set.union(*[set(e) for e in fold_ens.values()]) - keys
    if missing:
        print(f"  [cv] {len(missing)} configurations are missing in some fold and were skipped")
    return comb, spread


def selection_stability(task):
    """Stability of each FS method across the CV folds (5 selections on overlapping training sets)."""
    rows, per_fold = [], {}
    methods = config.fs_methods(task)
    for m in methods:
        sets, ranks = [], []
        for k, d in _fold_dirs(task).items():
            f = d / 'selection' / f'{m}.json'
            if f.exists():
                s = json.loads(f.read_text())
                sets.append(s['selected']); ranks.append(s['ranking'])
        if len(sets) < 2:
            continue
        per_fold[m] = sets
        if m in FIXED:
            continue
        rows.append(st._summary(m, sets, ranks, len(ranks[0]), f'{len(sets)} CV folds'))
    return pd.DataFrame(rows), per_fold


def summarize_cv(task, make_figures=True):
    config.set_fold(None)
    config.set_task(task)
    fold_ens, per_fold, splits = collect(task)
    comb, spread = combine(fold_ens)
    df = comparison_table(comb, spread, 'fold')

    out_dir = config.mode_dir(task)
    stab, per_fold_sel = selection_stability(task)
    sel_dir = config.sub_dir(task, 'selection')
    stab.to_csv(sel_dir / 'stability_cv_folds.csv', index=False)
    (sel_dir / 'selected_per_fold.json').write_text(json.dumps(per_fold_sel, indent=1))
    if not stab.empty:
        s = stab.set_index('method')
        for c in ('jaccard', 'dice', 'kuncheva', 'kendall_w', 'core80'):
            df[f'cv_{c}'] = df.method.map(s[c])
    # per-fold bootstrap / cross-centre stability, averaged over folds
    for fname, prefix in (('stability_bootstrap.csv', 'boot'), ('stability_centres.csv', 'centre')):
        fs = [d / 'selection' / fname for d in _fold_dirs(task).values()]
        fs = [pd.read_csv(f) for f in fs if f.exists()]
        if fs:
            s = pd.concat(fs).groupby('method').mean(numeric_only=True)
            s.reset_index().to_csv(sel_dir / f'{fname[:-4]}_fold_mean.csv', index=False)
            for c in ('jaccard', 'dice', 'kuncheva', 'kendall_w', 'core80'):
                if c in s:
                    df[f'{prefix}_{c}'] = df.method.map(s[c])
    nf = per_fold.groupby('method').n_features.mean()
    df['n_features'] = df.method.map(nf)

    df = mark_best_by_val(df)
    df.to_csv(out_dir / 'results_summary.csv', index=False)
    per_fold.to_csv(out_dir / 'results_per_fold.csv', index=False)
    print(f"[cv summary] {len(df)} configurations x {len(fold_ens)} folds -> {out_dir / 'results_summary.csv'}")
    print("  val_auc = pooled out-of-fold AUC; test_auc = fold-ensemble AUC on the external cohort")
    print_table(df)

    if make_figures:
        from radgraph.viz import result_plots as rp
        from radgraph.viz import selection_plots as sp
        from radgraph.viz.report import build_cv_report
        result_figures(task, df, comb)
        fd = config.figures_dir(task, 'results')
        for clin in (False, True):
            tag = 'img+clin' if clin else 'img'
            d = df[df.use_clinical == clin]
            if not d.empty:
                rp.plot_fold_aucs(per_fold[per_fold.use_clinical == clin], d, fd / f'fold_aucs_{tag}.png')
                if 'cv_kuncheva' in d:
                    rp.plot_stability_vs_auc(d, fd / f'cv_stability_vs_auc_{tag}.png', metric='cv_kuncheva',
                                             xlabel='Feature-selection stability across CV folds (Kuncheva)')
        if not stab.empty:
            sp.plot_stability(stab, config.figures_dir(task, 'selection') / 'stability_cv_folds.png',
                              f'Stability across the {config.N_FOLDS} CV folds, {config.MODE} {task}')
        sp.plot_fold_selection(per_fold_sel, config.figures_dir(task, 'selection') / 'selected_per_fold.png')
        build_cv_report(task, df, per_fold, splits, stab)
    return df
