"""Figures for the selection stage: which features each method picks, how much methods agree,
and how stable / reproducible each method is."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grafsgat.selection.methods import PRETTY
from grafsgat.viz.style import ACCENT, SEQ, INK, INK_2, SURFACE, short, save, feature_class, FEATURE_CLASS_COLOUR


def _name(m):
    return PRETTY.get(m, m)


def plot_selection_matrix(selections, path, skip=('all',)):
    """Rows = features selected by at least one method, columns = methods; dot = selected."""
    ms = [m for m in selections if m not in skip]
    feats = sorted({f for m in ms for f in selections[m]['selected']},
                   key=lambda f: (-sum(f in selections[m]['selected'] for m in ms), f))
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(ms), 0.8 + 0.27 * len(feats)))
    for j, m in enumerate(ms):
        sel = selections[m]['selected']
        for i, f in enumerate(feats):
            if f in sel:
                ax.scatter(j, i, s=70, color=FEATURE_CLASS_COLOUR.get(feature_class(f), ACCENT),
                           edgecolors=SURFACE, linewidths=1.5, zorder=3)
    counts = [sum(f in selections[m]['selected'] for m in ms) for f in feats]
    for i, c in enumerate(counts):
        ax.text(len(ms) - 0.3, i, f'{c}', va='center', fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([_name(m) for m in ms], rotation=40, ha='right')
    ax.set_yticks(range(len(feats))); ax.set_yticklabels([short(f) for f in feats], fontsize=7.5)
    ax.set_xlim(-0.6, len(ms) - 0.1); ax.set_ylim(len(feats) - 0.5, -0.5)
    ax.set_title('Selected GTV features per method (dot colour = feature class; right: # methods)', loc='left')
    return save(fig, path)


def plot_method_agreement(M, path):
    M = M.copy()
    M.index = [_name(m) for m in M.index]; M.columns = [_name(m) for m in M.columns]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.grid(False)
    im = ax.imshow(M.values, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if v > 0.55 else INK)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index)
    fig.colorbar(im, ax=ax, fraction=0.045, label='Jaccard index')
    ax.set_title('Agreement between methods (Jaccard of selected sets)', loc='left')
    return save(fig, path)


def plot_stability(df, path, title):
    """Small multiples: one panel per metric, methods on the y axis (single hue)."""
    metrics = [c for c in ('jaccard', 'dice', 'kuncheva', 'kendall_w') if c in df and df[c].notna().any()]
    labels = {'jaccard': 'Jaccard', 'dice': 'Dice (DSI)', 'kuncheva': 'Kuncheva index',
              'kendall_w': "Kendall's W (rankings)"}
    d = df.set_index('method')
    order = d['jaccard'].sort_values().index
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.5 + 0.38 * len(d)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mtr in zip(axes, metrics):
        v = d.loc[order, mtr]
        ax.barh(range(len(v)), v.values, color=ACCENT, height=0.6)
        for i, x in enumerate(v.values):
            if np.isfinite(x):
                ax.text(max(x, 0) + 0.02, i, f'{x:.2f}', va='center', fontsize=7.5, color=INK_2)
        ax.set_xlim(min(0, np.nanmin(v.values) - 0.05), 1.12)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_title(labels[mtr], fontsize=9.5, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([_name(m) for m in order])
    fig.suptitle(title, x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def plot_selection_frequency(freqs, path, top=12):
    """Per method: how often each of its top features is selected across bootstrap resamples."""
    ms = list(freqs)
    if not ms:
        return None
    cols = 3
    rows = int(np.ceil(len(ms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 0.45 + 0.24 * top * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), ms):
        f = freqs[m].head(top)[::-1]
        ax.barh(range(len(f)), f.values, color=ACCENT, height=0.65)
        ax.set_yticks(range(len(f))); ax.set_yticklabels([short(x) for x in f.index], fontsize=7)
        ax.set_xlim(0, 1); ax.axvline(0.8, color=INK_2, ls='--', lw=0.8)
        ax.set_title(_name(m), fontsize=9.5, loc='left'); ax.grid(axis='y', visible=False)
    for ax in axes.ravel()[len(ms):]:
        ax.axis('off')
    fig.suptitle('Selection frequency across bootstrap resamples (dashed: 80%)', x=0.01, ha='left',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def plot_fold_selection(per_fold, path, skip=('all', 'table_s3'), top=30):
    """Cross-validation modes: how many of the folds selected each feature, per method."""
    ms = [m for m in per_fold if m not in skip]
    if not ms:
        return None
    n = max(len(per_fold[m]) for m in ms)
    cnt = pd.DataFrame({m: pd.Series([f for s in per_fold[m] for f in s]).value_counts() for m in ms}).fillna(0)
    cnt = cnt.loc[cnt.sum(1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(1.6 + 0.8 * len(ms), 0.9 + 0.26 * len(cnt)))
    ax.grid(False)
    cmap = SEQ.copy(); cmap.set_bad(SURFACE)                    # 0 folds = blank cell
    im = ax.imshow(np.ma.masked_equal(cnt.values, 0), cmap=cmap, vmin=0, vmax=n, aspect='auto')
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            v = int(cnt.values[i, j])
            if v:
                ax.text(j, i, str(v), ha='center', va='center', fontsize=7.5,
                        color='white' if v > n / 2 else INK)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([_name(m) for m in ms], rotation=40, ha='right')
    ax.set_yticks(range(len(cnt))); ax.set_yticklabels([short(f) for f in cnt.index], fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label=f'folds selecting the feature (of {n})')
    ax.set_title(f'Features selected in each CV fold (top {len(cnt)} by total count)', loc='left')
    return save(fig, path)
