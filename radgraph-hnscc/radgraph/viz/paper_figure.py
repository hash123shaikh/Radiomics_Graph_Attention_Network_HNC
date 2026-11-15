"""
Figure 3 of Bae et al. (published AUCs), for comparison with this study.
    python -m radgraph.viz.paper_figure

Figure 3 values were read from the published figure (400 dpi render) and agree with the
text: RadGraph LR imaging-only 0.57-0.66, +clinical 0.60-0.83; DM imaging-only 0.62-0.90,
+clinical 0.64-0.86.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from radgraph import config

MODEL_COLOURS = {
    'Clinical Baseline'    : '#1a1a1a',
    'Traditional Radiomics': '#7b6fa0',
    'Mateus et al. CNN'    : '#c77b8a',
    'Diamant et al. CNN'   : '#f08080',
    'RadGraph'             : '#f5b597',
}
DATASET_LABELS = ['HNPET', 'RADCURE', 'HN1', 'MDACC']

# {task: {panel: {model: [HNPET, RADCURE, HN1, MDACC]}}}
PAPER_AUCS = {
    'LR': {
        'imaging_only': {
            'Clinical Baseline'    : [0.682, 0.731, 0.664, 0.574],
            'Traditional Radiomics': [0.544, 0.589, 0.581, 0.625],
            'Mateus et al. CNN'    : [0.587, 0.641, 0.636, 0.643],
            'Diamant et al. CNN'   : [0.472, 0.441, 0.473, 0.467],
            'RadGraph'             : [0.571, 0.653, 0.655, 0.574],
        },
        'imaging_and_clinical': {
            'Clinical Baseline'    : [0.682, 0.731, 0.664, 0.574],
            'Traditional Radiomics': [0.646, 0.648, 0.643, 0.523],
            'Mateus et al. CNN'    : [0.754, 0.807, 0.803, 0.613],
            'Diamant et al. CNN'   : [0.502, 0.441, 0.450, 0.455],
            'RadGraph'             : [0.758, 0.834, 0.806, 0.603],
        },
    },
    'DM': {
        'imaging_only': {
            'Clinical Baseline'    : [0.587, 0.713, 0.832, 0.504],
            'Traditional Radiomics': [0.604, 0.623, 0.715, 0.596],
            'Mateus et al. CNN'    : [0.566, 0.717, 0.563, 0.651],
            'Diamant et al. CNN'   : [0.449, 0.385, 0.25, 0.427],
            'RadGraph'             : [0.719, 0.728, 0.898, 0.621],
        },
        'imaging_and_clinical': {
            'Clinical Baseline'    : [0.587, 0.713, 0.832, 0.504],
            'Traditional Radiomics': [0.606, 0.704, 0.567, 0.552],
            'Mateus et al. CNN'    : [0.770, 0.785, 0.867, 0.689],
            'Diamant et al. CNN'   : [0.418, 0.380, 0.24, 0.426],
            'RadGraph'             : [0.790, 0.786, 0.859, 0.641],
        },
    },
}


def polar_rose(ax, aucs, labels, title=''):
    n_ds, models = len(labels), list(aucs)
    centres = np.linspace(0, 2 * np.pi, n_ds, endpoint=False) + np.pi / n_ds
    width = (2 * np.pi / n_ds) / (len(models) + 1)
    for d in range(n_ds):
        best = max(models, key=lambda m: aucs[m][d])
        for i, m in enumerate(models):
            th = centres[d] + (i - (len(models) - 1) / 2) * width
            v = aucs[m][d]
            ax.bar(th, v, width=width * 0.95, color=MODEL_COLOURS.get(m, '#2196F3'), edgecolor='white')
            if m == best:
                ax.bar(th, v, width=width * 0.95, fill=False, edgecolor='#b00020', ls='--', lw=1.5)
            ax.text(th, v + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=6,
                    rotation=np.degrees(th) - 90, rotation_mode='anchor')
    ax.set_xticks(centres); ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 1.0); ax.set_yticks([0.5]); ax.set_yticklabels(['0.5'], fontsize=7)
    ax.set_title(title, pad=15, fontweight='bold')


def plot_paper_figure3(save_path):
    fig = plt.figure(figsize=(16, 16))
    for r, task in enumerate(('LR', 'DM')):
        for c, panel in enumerate(('imaging_only', 'imaging_and_clinical')):
            ax = fig.add_subplot(2, 2, 2 * r + c + 1, projection='polar')
            polar_rose(ax, PAPER_AUCS[task][panel], DATASET_LABELS,
                       f"{'Locoregional recurrence' if task == 'LR' else 'Distant metastasis'} — "
                       f"{'imaging only' if c == 0 else 'imaging + clinical'}")
    fig.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in MODEL_COLOURS.items()],
               loc='lower center', ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=config.FIGURE_DPI, facecolor='white'); plt.close()
    print(f"-> {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    plot_paper_figure3(config.OUTPUT_DIR / 'figure3_paper.png')


if __name__ == '__main__':
    main()
