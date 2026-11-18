"""
Study-level export across tasks (LR, DM, OS), designs (MATEUS_CV, MATEUS_COHORT) and checkpoint rules.

outputs/results/
    main_results.csv                 per task x design x rule x inputs: chosen GAT, reference, best RF,
                                     clinical-only RF -> validation AUC, external test AUC [95% CI],
                                     single-fold range, DeLong p (Holm) vs reference
    all_configurations.csv           every configuration of every run (the full results tables)
    feature_selection_stability.csv  stability of every FS method (across CV folds, bootstrap, centres)
    cohort_summary.csv               patients and events per task / cohort / split
    predictions/<run>__<config>.csv  per-patient probabilities of the chosen models
                                     (out-of-fold validation + external test, each fold's test probability)
    figures/roc_<TASK>.png           ROC on the external test cohort, 95% bootstrap bands, AUC [CI] in legend
    figures/roc_cv_folds_<TASK>.png  ROC of every CV fold + mean ROC +/- 1 SD (chosen configuration)
    figures/auc_overview.png         test AUC [95% CI] of chosen GAT vs reference vs clinical, all tasks
outputs/final_models/<TASK>_<DESIGN>[_aucgap]/<imaging|imaging+clinical>/
    the chosen configuration's checkpoints (all folds x seeds) + model_card.json  (used by predict.py)
"""

import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from grafsgat import config
from grafsgat.selection.methods import PRETTY
from grafsgat.utils import safe_auc
from grafsgat.viz.style import CAT, INK, INK_2, MUTED, NEUTRAL, save

RESULTS_DIR = config.OUTPUT_DIR / 'results'
FINAL_DIR = config.OUTPUT_DIR / 'final_models'
DESIGNS = ['MATEUS_CV', 'MATEUS_COHORT']
RULES = ['val_loss', 'auc_gap']
DESIGN_NAMES = {'MATEUS_CV': '5-fold CV (fold ensemble)', 'MATEUS_COHORT': 'Cohort split (HGJ+CHUS / HMR+CHUM)'}
TOPO = {'radiomic_star': 'radiomic star', 'spatial_star': 'spatial star', 'radiomic_knn': 'radiomic kNN',
        'spatial_knn': 'spatial kNN', 'RF': 'random forest'}


def _run(task, design, rule):
    config.set_mode(design); config.set_task(task); config.EPOCH_RULE = rule
    return config.mode_dir(task)


def available_runs():
    out = []
    for task in config.TASKS:
        for design in DESIGNS:
            for rule in RULES:
                d = config.RUNS_DIR / f"{design}_{task}{'' if rule == 'val_loss' else '_aucgap'}"
                if (d / 'results_summary.csv').exists():
                    out.append((task, design, rule))
    return out


def ensembles(task, design, rule):
    """{key: DataFrame[patient_id, split, label, prob, (fold-wise test probs)]}, per-fold {k: {key: df}} or None."""
    from grafsgat.experiments.benchmark import _ensembles
    _run(task, design, rule)
    if config.split_scheme() == 'cv':
        from grafsgat.experiments.cv import collect, combine
        fold_ens, _, _ = collect(task)
        comb, _ = combine(fold_ens)
        return comb, fold_ens
    ens, _ = _ensembles(task)
    return ens, None


def _cfg_label(r):
    if r.method == 'clinical_only':
        return 'Clinical-only RF'
    return f"{PRETTY.get(r.method, r.method)} + {TOPO.get(r.topology, r.topology)}"


def _picks(df, clin):
    """Rows reported in main_results: chosen GAT, reference, best RF (by val), clinical-only RF."""
    d = df[df.use_clinical == clin]
    ref_m, ref_t = config.reference_config()
    out = []
    b = d[d.best_by_val & (d.model == 'GAT')]
    if len(b):
        out.append(('chosen GAT', b.iloc[0]))
    r = d[(d.method == ref_m) & (d.topology == ref_t)]
    if len(r):
        out.append(('reference GAT', r.iloc[0]))
    rf = df[(df.model == 'RF') & (df.method != 'clinical_only')]
    if len(rf) and not clin:
        out.append(('best RF', rf.loc[rf.val_auc.idxmax()]))
    c = df[df.method == 'clinical_only']
    if len(c) and clin:
        out.append(('clinical-only RF', c.iloc[0]))
    return out


# ─── Figures ─────────────────────────────────────────────────────────────────

FPR = np.linspace(0, 1, 201)


def _interp_roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    t = np.interp(FPR, fpr, tpr)
    t[0] = 0.0
    return t


def _roc_band(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    T = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        T.append(_interp_roc(y[i], p[i]))
    return np.percentile(T, 2.5, axis=0), np.percentile(T, 97.5, axis=0)


def roc_figure(task, panels, path):
    """panels: [(title, [(label, y, p, auc, lo, hi), ...]), ...] -> one ROC panel per design."""
    nc = 2 if len(panels) > 1 else 1
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.8 * nc, 5.9 * nr), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (title, curves) in zip(axes.flat, panels):
        for (lab, y, p, auc, lo, hi), col in zip(curves, CAT):
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            if lab.startswith('Chosen'):
                blo, bhi = _roc_band(y, p)
                ax.fill_between(FPR, blo, bhi, color=col, alpha=0.15, lw=0)
            ax.step(fpr, tpr, where='post', color=col, lw=2 if lab.startswith('Chosen') else 1.5,
                    label=f'{lab}\nAUC {auc:.3f} [{lo:.3f}, {hi:.3f}]')
        ax.plot([0, 1], [0, 1], color=NEUTRAL, ls='--', lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
        ax.set_xlabel('1 - specificity (false positive rate)'); ax.set_ylabel('Sensitivity (true positive rate)')
        ax.legend(loc='lower right', fontsize=7, handlelength=1.5)
        ax.set_title(title, loc='left', fontsize=10)
    fig.suptitle(f'{config.TASK_NAMES[task]} within {config.horizon_months(task) // 12} years: '
                 f'external test (MAASTRO); shaded = 95% bootstrap band of the chosen GAT',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, path, dpi=300)


def roc_folds_figure(task, fold_ens, key, label, path):
    """Every fold's test ROC (thin), mean ROC across folds +/- 1 SD (band), fold-ensemble ROC (dashed)."""
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    T, aucs = [], []
    for k in sorted(fold_ens):
        e = fold_ens[k].get(key)
        if e is None:
            continue
        t = e[e.split == 'test']
        T.append(_interp_roc(t.label, t.prob)); aucs.append(safe_auc(t.label, t.prob))
        ax.plot(FPR, T[-1], color=MUTED, lw=0.9, alpha=0.7,
                label=f'fold {k} (AUC {aucs[-1]:.3f})')
    T = np.array(T); m, s = T.mean(0), T.std(0)
    m[-1] = 1.0
    ax.fill_between(FPR, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=CAT[0], alpha=0.18, lw=0, label='± 1 SD')
    ax.plot(FPR, m, color=CAT[0], lw=2.2, label=f'mean ROC (AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})')
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls='--', lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
    ax.set_xlabel('1 - specificity (false positive rate)'); ax.set_ylabel('Sensitivity (true positive rate)')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(f'{config.TASK_NAMES[task]}: 5-fold CV models on MAASTRO\n{label}', loc='left', fontsize=10)
    return save(fig, path, dpi=300)


def auc_overview(main, path):
    d = main[(main.epoch_rule == 'val_loss')].copy()
    if d.empty:
        return None
    d['row'] = d.task + ' | ' + d.design.str.replace('MATEUS_', '') + ' | ' + d.inputs
    rows = list(dict.fromkeys(d.row))
    roles = list(dict.fromkeys(d.role))
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.55 * len(rows)))
    off = np.linspace(-0.25, 0.25, len(roles))
    for j, (role, col) in enumerate(zip(roles, CAT)):
        g = d[d.role == role]
        y = [rows.index(r) + off[j] for r in g.row]
        ax.errorbar(g.test_auc, y, xerr=[g.test_auc - g.test_auc_ci_low, g.test_auc_ci_high - g.test_auc],
                    fmt='o', color=col, ecolor=col, elinewidth=1.2, markersize=5, capsize=0, label=role)
    ax.axvline(0.5, color=NEUTRAL, ls='--', lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(0.2, 1.0)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('External test AUC (MAASTRO), 95% bootstrap CI')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.9 / len(rows) - 0.04), ncol=2, fontsize=7.5)
    ax.set_title('External test AUC across outcomes and validation designs', loc='left')
    return save(fig, path, dpi=300)


# ─── Export ──────────────────────────────────────────────────────────────────

def _rel(p):
    try:
        return p.relative_to(config.BASE_DIR)
    except ValueError:
        return p


def _run_preprocess(task, design, rule):
    """Preprocessing profile recorded in the run's provenance.json (falls back to the current one)."""
    f = _run(task, design, rule) / 'provenance.json'
    try:
        return json.loads(f.read_text())['settings']['preprocess']
    except Exception:
        return config.PREPROCESS


def _copy_models(task, design, rule, row, fold_ens):
    from grafsgat.inference import find_checkpoints
    ck = find_checkpoints(task, design, row.method, row.topology, bool(row.use_clinical), rule)
    dest = FINAL_DIR / f"{design}_{task}{'' if rule == 'val_loss' else '_aucgap'}" / row.inputs
    if dest.exists():
        shutil.rmtree(dest)
    for f in ck:
        sub = dest / f.parent.parent.name if f.parent.parent.name.startswith('fold') else dest
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, sub / f.name)
        if (f.parent.parent / 'split.json').exists() and not (sub / 'split.json').exists():
            shutil.copy2(f.parent.parent / 'split.json', sub / 'split.json')   # who trained these models
    card = {'task': task, 'outcome': config.TASK_NAMES[task], 'horizon_months': config.horizon_months(task),
            'design': design, 'epoch_rule': rule, 'inputs': row.inputs,
            'fs_method': row.method, 'topology': row.topology, 'n_checkpoints': len(ck),
            'validation_auc': row.val_auc, 'external_test_auc': row.test_auc,
            'external_test_auc_95ci': [row.test_auc_ci_low, row.test_auc_ci_high],
            'preprocess': _run_preprocess(task, design, rule),
            'how_to_use': (f"python predict.py --task {task} --models {_rel(dest)} --dataset MAASTRO "
                           f"--preprocess {_run_preprocess(task, design, rule)}")}
    (dest / 'model_card.json').write_text(json.dumps(card, indent=1, default=float))
    return dest


def _predictions(task, design, rule, row, comb, fold_ens):
    key = (row.method, row.topology, bool(row.use_clinical))
    e = comb[key].copy()
    e.insert(0, 'task', task); e.insert(1, 'design', design); e.insert(2, 'epoch_rule', rule)
    e['split'] = e.split.map({'val': 'validation (out-of-fold)' if fold_ens else 'validation',
                              'test': 'external test'})
    if fold_ens:
        for k in sorted(fold_ens):
            t = fold_ens[k][key]
            t = t[t.split == 'test'].set_index('patient_id').prob
            e[f'prob_fold{k}'] = e.patient_id.map(t).where(e.split == 'external test')
            v = fold_ens[k][key]
            vf = set(v[v.split == 'val'].patient_id)
            e.loc[e.patient_id.isin(vf), 'cv_fold'] = k
    name = f"{design}_{task}{'' if rule == 'val_loss' else '_aucgap'}__{row.method}__{row.topology}__" \
           f"{'img+clin' if row.use_clinical else 'img'}.csv"
    f = RESULTS_DIR / 'predictions' / name
    f.parent.mkdir(parents=True, exist_ok=True)
    e = e.drop(columns=['n_seeds'], errors='ignore')
    first = ['task', 'design', 'epoch_rule', 'patient_id', 'split'] + (['cv_fold'] if 'cv_fold' in e else []) + ['label', 'prob']
    e = e[first + [c for c in e.columns if c not in first]]
    e.rename(columns={'prob': 'prob_ensemble'}).to_csv(f, index=False)
    return f


def cohort_summary():
    from grafsgat.data.splits import eligible_with_features
    rows = []
    for task in config.TASKS:
        for ds, role in (('HNPET', 'train/validation (4 Canadian centres)'), ('MAASTRO', 'external test')):
            lab = eligible_with_features(ds, task, verbose=False)
            rows.append({'task': task, 'horizon_months': config.horizon_months(task), 'cohort': ds, 'role': role,
                         'patients': len(lab), 'events': sum(lab.values()),
                         'event_rate': round(sum(lab.values()) / max(len(lab), 1), 3)})
    return pd.DataFrame(rows)


def export_all(copy_models=True):
    (RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    runs = available_runs()
    if not runs:
        raise SystemExit("No finished runs (results_summary.csv) yet")
    cohort_summary().to_csv(RESULTS_DIR / 'cohort_summary.csv', index=False)
    allc, main, stab, rocs = [], [], [], {}
    for task, design, rule in runs:
        run = _run(task, design, rule)
        print(f"[export] {design} {task} ({rule})")
        df = pd.read_csv(run / 'results_summary.csv')
        df.insert(0, 'task', task); df.insert(1, 'design', design); df.insert(2, 'epoch_rule', rule)
        allc.append(df)
        comb, fold_ens = ensembles(task, design, rule)
        for clin in (False, True):
            for role, r in _picks(df, clin):
                main.append({'task': task, 'horizon_months': config.horizon_months(task), 'design': design,
                             'epoch_rule': rule, 'inputs': 'imaging+clinical' if clin else 'imaging', 'role': role,
                             'configuration': _cfg_label(r), 'fs_method': r.method, 'topology': r.topology,
                             'selected_on': 'validation AUC' if role in ('chosen GAT', 'best RF') else 'fixed',
                             'n_features': r.get('n_features'),
                             'val_auc': r.val_auc, 'test_auc': r.test_auc, 'test_auc_ci_low': r.test_auc_ci_low,
                             'test_auc_ci_high': r.test_auc_ci_high,
                             'test_auc_fold_min': r.get('test_auc_fold_min'), 'test_auc_fold_max': r.get('test_auc_fold_max'),
                             'p_vs_reference_holm': r.get('p_vs_reference_holm'),
                             'test_n': r.test_n, 'test_events': r.test_events})
                if role.startswith('chosen'):
                    _predictions(task, design, rule, r, comb, fold_ens)
                    if copy_models:
                        _copy_models(task, design, rule, r, fold_ens)
            if rule == 'val_loss':
                curves = []
                for role, r in _picks(df, clin):
                    e = comb[(r.method, r.topology, bool(r.use_clinical))]
                    t = e[e.split == 'test']
                    name = {'chosen GAT': 'Chosen GAT: ' + _cfg_label(r), 'reference GAT': 'Reference GAT: ' + _cfg_label(r),
                            'best RF': 'Best RF: ' + PRETTY.get(r.method, r.method),
                            'clinical-only RF': 'Clinical-only RF'}[role]
                    curves.append((name, t.label.values, t.prob.values, r.test_auc, r.test_auc_ci_low,
                                   r.test_auc_ci_high))
                rocs.setdefault(task, []).append((f"{DESIGN_NAMES[design]} | {'imaging + clinical' if clin else 'imaging'}",
                                                  curves))
                if fold_ens and not clin:
                    b = df[df.best_by_val & (df.use_clinical == clin) & (df.model == 'GAT')].iloc[0]
                    roc_folds_figure(task, fold_ens, (b.method, b.topology, clin), _cfg_label(b),
                                     RESULTS_DIR / 'figures' / f'roc_cv_folds_{task}.png')
        sel = run / 'selection'
        for f, kind in (('stability_cv_folds.csv', 'across CV folds'),
                        ('stability_bootstrap_fold_mean.csv', 'bootstrap (mean over folds)'),
                        ('stability_centres_fold_mean.csv', 'cross-centre (mean over folds)'),
                        ('stability_bootstrap.csv', 'bootstrap'), ('stability_centres.csv', 'cross-centre')):
            if (sel / f).exists():
                s = pd.read_csv(sel / f)
                s.insert(0, 'task', task); s.insert(1, 'design', design); s.insert(2, 'epoch_rule', rule)
                s.insert(3, 'stability_type', kind)
                stab.append(s)
    config.set_mode('MATEUS_CV'); config.EPOCH_RULE = 'val_loss'
    for task, panels in rocs.items():
        config.set_task(task)
        roc_figure(task, panels, RESULTS_DIR / 'figures' / f'roc_{task}.png')
    main = pd.DataFrame(main)
    main.to_csv(RESULTS_DIR / 'main_results.csv', index=False)
    pd.concat(allc).to_csv(RESULTS_DIR / 'all_configurations.csv', index=False)
    if stab:
        pd.concat(stab, ignore_index=True).to_csv(RESULTS_DIR / 'feature_selection_stability.csv', index=False)
    auc_overview(main, RESULTS_DIR / 'figures' / 'auc_overview.png')
    print(main[['task', 'design', 'epoch_rule', 'inputs', 'role', 'configuration', 'val_auc', 'test_auc',
                'test_auc_ci_low', 'test_auc_ci_high']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"[export] -> {RESULTS_DIR}" + (f" and {FINAL_DIR}" if copy_models else ''))
    return main
