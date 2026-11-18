"""Benchmark result figures: AUC heatmap (method x topology), forest plot with 95% CI,
stability-vs-discrimination scatter, ROC overlay of key configurations."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from grafsgat import config
from grafsgat.selection.methods import PRETTY
from grafsgat.viz.style import ACCENT, SECOND, CAT, SEQ, INK, INK_2, MUTED, NEUTRAL, save

TOPO_NAME = {'radiomic_star': 'Radiomic star', 'spatial_star': 'Spatial star',
             'radiomic_knn': 'Radiomic kNN', 'spatial_knn': 'Spatial kNN', 'RF': 'RF (no graph)'}


def _label(r):
    return f"{PRETTY.get(r.method, r.method)} | {TOPO_NAME.get(r.topology, r.topology)}"


def plot_auc_heatmap(d, path, split='test'):
    g = d[d.method != 'clinical_only']
    piv = g.pivot_table(index='method', columns='topology', values=f'{split}_auc')
    cols = [t for t in config.TOPOLOGIES + ['RF'] if t in piv.columns]
    rows = [m for m in config.FS_METHODS if m in piv.index]
    piv = piv.loc[rows, cols]
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * len(cols), 0.9 + 0.45 * len(rows)))
    ax.grid(False)
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0.4, vmax=0.9, aspect='auto')
    best = d[d.best_by_val]
    for i, m in enumerate(rows):
        for j, t in enumerate(cols):
            v = piv.values[i, j]
            if np.isfinite(v):
                star = '*' if ((best.method == m) & (best.topology == t)).any() else ''
                ax.text(j, i, f'{v:.3f}{star}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.7 else INK, fontweight='bold' if star else 'normal')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([TOPO_NAME.get(c, c) for c in cols], rotation=25, ha='right')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([PRETTY.get(m, m) for m in rows])
    fig.colorbar(im, ax=ax, fraction=0.04, label=f'{split} AUC')
    inp = d.inputs.iloc[0]
    ax.set_title(f'{split.capitalize()} AUC, {inp} ({config.run_label()}, seed-ensembled)\n'
                 f'* = configuration chosen on validation AUC', loc='left', fontsize=10)
    return save(fig, path)


def plot_forest(d, path):
    d = d.sort_values('test_auc')
    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.24 * len(d)))
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        col = INK if r.best_by_val else (SECOND if r.model == 'RF' else ACCENT)
        ax.plot([r.test_auc_ci_low, r.test_auc_ci_high], [i, i], color=col, lw=1.4, alpha=0.8)
        ax.scatter(r.test_auc, i, s=34 if r.best_by_val else 22, color=col, zorder=3,
                   marker='D' if r.best_by_val else 'o')
    ref = d[(d.method == config.reference_config()[0]) & (d.topology == config.reference_config()[1])]
    if len(ref):
        ax.axvline(ref.test_auc.iloc[0], color=MUTED, ls=':', lw=1)
    ax.axvline(0.5, color=NEUTRAL, ls='--', lw=1)
    ax.set_yticks(y); ax.set_yticklabels([_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_xlim(0.2, 1.0); ax.set_xlabel('Test AUC (95% bootstrap CI)')
    ax.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='o', color=ACCENT, label='GAT'),
         Line2D([0], [0], marker='o', color=SECOND, label='Random forest (same features)'),
         Line2D([0], [0], marker='D', color=INK, ls='', label='chosen on validation'),
         Line2D([0], [0], color=MUTED, ls=':', label='reference (' + PRETTY.get(config.reference_config()[0], '')
                + ', radiomic star)')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.9 / max(len(d), 4) - 0.08),
              ncol=2, fontsize=7.5)
    ax.set_title(f'All configurations, {d.inputs.iloc[0]} ({config.run_label()})', loc='left')
    return save(fig, path)


def plot_stability_vs_auc(d, path, metric='boot_kuncheva', topology='radiomic_star',
                          xlabel='Feature-selection stability (Kuncheva index, bootstrap)'):
    g = d[(d.model == 'GAT') & (d.topology == topology) & d[metric].notna()]
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(g[metric], g.test_auc, yerr=[g.test_auc - g.test_auc_ci_low, g.test_auc_ci_high - g.test_auc],
                fmt='o', color=ACCENT, ecolor=NEUTRAL, elinewidth=1.2, markersize=7, capsize=0)
    for _, r in g.iterrows():
        ax.annotate(PRETTY.get(r.method, r.method), (r[metric], r.test_auc), xytext=(6, 4),
                    textcoords='offset points', fontsize=8, color=INK)
    ax.axhline(0.5, color=NEUTRAL, ls='--', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Test AUC (95% CI)')
    ax.set_title(f'Reproducible AND discriminative? ({TOPO_NAME[topology]}, {g.inputs.iloc[0]})', loc='left')
    return save(fig, path)


def plot_roc_overlay(d, ens, path):
    """Up to four lines: chosen GAT, paper reference, best RF, clinical-only RF."""
    picks = []
    b = d[d.best_by_val & (d.model == 'GAT')]
    if len(b):
        picks.append(('Chosen GAT: ' + _label(b.iloc[0]), b.iloc[0]))
    ref = d[(d.method == config.reference_config()[0]) & (d.topology == config.reference_config()[1])]
    if len(ref) and not (len(b) and ref.index[0] == b.index[0]):
        picks.append(('Reference: ' + _label(ref.iloc[0]), ref.iloc[0]))
    rf = d[(d.model == 'RF') & (d.method != 'clinical_only')]
    if len(rf):
        r = rf.loc[rf.val_auc.idxmax()]
        picks.append(('Best RF: ' + PRETTY.get(r.method, r.method), r))
    cl = d[d.method == 'clinical_only']
    if len(cl):
        picks.append(('Clinical-only RF', cl.iloc[0]))
    if not picks:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for (lab, r), col in zip(picks, CAT):
        e = ens[(r.method, r.topology, bool(r.use_clinical))]
        t = e[e.split == 'test']
        if t.label.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(t.label, t.prob)
        ax.plot(fpr, tpr, color=col, label=f'{lab} (AUC {r.test_auc:.3f})')
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls='--', lw=1)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(f'Test ROC, {d.inputs.iloc[0]} ({config.run_label()})', loc='left')
    ax.set_aspect('equal')
    return save(fig, path)


def plot_fold_aucs(per_fold, d, path):
    """Cross-validation modes: test AUC of every fold (grey dots) and of the fold ensemble
    (blue diamond, 95% CI line), one panel per topology, methods on the y axis."""
    cols = [t for t in config.TOPOLOGIES + ['RF'] if t in set(d.topology)]
    rows = [m for m in config.FS_METHODS if m in set(d.method)]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 0.9 + 0.42 * len(rows)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, cols):
        for i, m in enumerate(rows):
            f = per_fold[(per_fold.method == m) & (per_fold.topology == t)]
            ax.scatter(f.test_auc, np.full(len(f), i), s=14, color=MUTED, alpha=0.8, zorder=2)
            e = d[(d.method == m) & (d.topology == t)]
            if len(e):
                r = e.iloc[0]
                ax.plot([r.test_auc_ci_low, r.test_auc_ci_high], [i, i], color=ACCENT, lw=1, alpha=0.5)
                ax.scatter(r.test_auc, i, s=40, marker='D', color=INK if r.best_by_val else ACCENT, zorder=3)
        ax.axvline(0.5, color=NEUTRAL, ls='--', lw=1)
        ax.set_xlim(0.2, 1.0)
        ax.set_title(TOPO_NAME.get(t, t), fontsize=9.5, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([PRETTY.get(m, m) for m in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='o', color=MUTED, ls='', label='single fold'),
         Line2D([0], [0], marker='D', color=ACCENT, label='fold ensemble (95% CI)'),
         Line2D([0], [0], marker='D', color=INK, ls='', label='chosen on out-of-fold validation AUC')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'External test AUC per CV fold, {d.inputs.iloc[0]} ({config.MODE})',
                 x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, path)
