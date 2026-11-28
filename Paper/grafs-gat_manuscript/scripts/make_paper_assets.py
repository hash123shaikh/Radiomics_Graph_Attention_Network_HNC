"""
Build every figure and LaTeX table of the manuscript from the study results, so the paper can be
regenerated after a re-run (e.g. once the full Head-Neck-PET-CT image set is available).

    python make_paper_assets.py          (project environment, e.g. conda activate GNNvenv)

Reads   $GRAFS_OUTPUT_DIR/results/*.csv (default grafs-gat/outputs/; written by export_results.py)
        and the clinical files
Writes  ../figures/*.pdf|png  and  ../tables/*.tex  (\\input by main.tex and supplementary.tex)
        ../tables/numbers.tex (\\newcommand macros for every number quoted in the text)
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
MS = HERE.parent
PROJ = MS.parent.parent / 'grafs-gat'
import os
OUT = Path(os.environ['GRAFS_OUTPUT_DIR']).expanduser() if os.environ.get('GRAFS_OUTPUT_DIR') else PROJ / 'outputs'
RES = OUT / 'results'
RUNS = OUT / 'runs'
FIG = MS / 'figures'
TAB = MS / 'tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJ))

# ─── style (white page, project palette) ─────────────────────────────────────
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#c9c8c2'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': NEUTRAL, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'legend.fontsize': 7.5, 'font.size': 8.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
})

TASKS = ['LR', 'DM', 'OS']
TASK_LONG = {'LR': 'Locoregional recurrence (2 y)', 'DM': 'Distant metastasis (2 y)',
             'OS': 'Overall survival (4 y)'}
DESIGN = {'MATEUS_CV': '5-fold CV', 'MATEUS_COHORT': 'Cohort split'}
METHODS = ['table_s3', 'all', 'mrmr', 'lasso', 'boruta', 'rfe_rf',
           'graph_fs_cc', 'graph_fs_spectral', 'graph_fs_hybrid']
MNAME = {'table_s3': 'Published feature set', 'all': 'All features', 'mrmr': 'mRMR', 'lasso': 'LASSO',
         'boruta': 'Boruta', 'rfe_rf': 'RFE-RF', 'graph_fs_cc': 'Graph-FS (CC)',
         'graph_fs_spectral': 'Graph-FS (spectral)', 'graph_fs_hybrid': 'Graph-FS (hybrid)',
         'clinical_only': 'Clinical only'}
TNAME = {'radiomic_star': 'Radiomic star', 'spatial_star': 'Spatial star',
         'radiomic_knn': 'Radiomic kNN', 'spatial_knn': 'Spatial kNN', 'RF': 'RF'}
SELECTORS = ['mrmr', 'lasso', 'boruta', 'rfe_rf', 'graph_fs_cc', 'graph_fs_spectral', 'graph_fs_hybrid']

main = pd.read_csv(RES / 'main_results.csv')
allc = pd.read_csv(RES / 'all_configurations.csv')
stab = pd.read_csv(RES / 'feature_selection_stability.csv')
coh = pd.read_csv(RES / 'cohort_summary.csv')
PRIMARY = 'val_loss'
# checkpoint-rule sensitivity analysis: exact rule of Mateus et al. when its runs exist, else the adapted rule
SENS_RULE = 'mateus' if (OUT / 'runs' / 'MATEUS_COHORT_OS_mateus' / 'results_summary.csv').exists() else 'auc_gap'

numbers = {}          # macro name -> text


def num(name, value):
    numbers[name] = value
    return value


def f2(x):
    return '--' if pd.isna(x) else f'{x:.2f}'


def ci(r, lo='test_auc_ci_low', hi='test_auc_ci_high', v='test_auc'):
    return f'{r[v]:.2f} [{r[lo]:.2f}, {r[hi]:.2f}]'


def save(fig, name):
    for a_ in fig.axes:
        a_.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def tex_escape(s):
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: study design
# ═════════════════════════════════════════════════════════════════════════════
def fig_design():
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 58); ax.axis('off')

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                    fc=colour + '1f', ec=colour, lw=1.1))
        ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=7.6, weight='bold', color=INK)
        ax.text(x + w / 2, y + h - 5.0, body, ha='center', va='top', fontsize=6.3, color=INK2,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, shrinkA=0, shrinkB=0))

    # row 1: data -> preprocessing -> features
    box(0.5, 38, 22, 18.5, 'Cohorts (TCIA)',
        'Head-Neck-PET-CT\nCHUM, CHUS, HGJ, HMR\n(training / validation)\n\nHN1 (MAASTRO)\n(external test only)', CAT[0])
    box(26, 38, 22, 18.5, 'Pre-processing',
        'CT + primary GTV\n1 mm isotropic resampling\n50 mm box around GTV\nSLIC-zero supervoxels\n(~100 per patient,\nGTV excluded)', CAT[0])
    box(51.5, 38, 22, 18.5, 'Radiomics',
        'PyRadiomics, 93 features\n(first order + 5 texture\nclasses, bin width 25 HU)\nGTV node + every\nsupervoxel node', CAT[0])
    box(77, 38, 22.5, 18.5, 'Outcomes',
        'LR, DM: event < 2 y\nOS: death < 4 y\npatients censored before\nthe horizon excluded\n(Mateus et al.)', CAT[0])
    arrow(23.3, 47, 25.2, 47); arrow(48.8, 47, 50.7, 47); arrow(74.3, 47, 76.2, 47)

    # row 2: grid
    box(0.5, 13, 30, 21.5, 'Feature selection (9)',
        'Published feature set (reference)\nAll features (no selection)\nmRMR, LASSO, Boruta, RFE-RF\nGraph-FS: CC, spectral, hybrid\n\nfitted on training patients only\n(k = 6 features)', CAT[1])
    box(34.5, 13, 30, 21.5, 'Graph topology (4)',
        'GTV hub + 20 supervoxels\nRadiomic star (reference)\nSpatial star\nRadiomic kNN (star + kNN, k = 4)\nSpatial kNN\n\nnodes z-scored on training GTVs', CAT[1])
    box(68.5, 13, 31, 21.5, 'Models',
        'GRAFS-GAT (published GAT settings)\nGTV read-out, imaging only or\nimaging + clinical, 3 seeds\n\nControls: RF on same features,\nclinical-only RF', CAT[1])
    arrow(31.3, 23.5, 33.7, 23.5); arrow(65.3, 23.5, 67.7, 23.5)
    arrow(50, 37.2, 50, 35.3)

    # row 3: validation design
    box(0.5, -4, 48.5, 14, 'Validation design (Mateus et al. 2023)',
        '5-fold stratified CV on the 4 Canadian centres\nor cohort split HGJ+CHUS / HMR+CHUM\n'
        'configuration chosen on validation AUC; MAASTRO used once', CAT[2])
    box(52.5, -4, 47, 14, 'Evaluation',
        'external AUC + 95% bootstrap CI\nDeLong test (Holm) vs reference configuration\n'
        'selection stability: Kuncheva, Jaccard, Kendall W', CAT[2])
    arrow(50, 12.2, 50, 10.8)
    save(fig, 'fig1_study_design')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: the four graph topologies (schematic on a synthetic patient)
# ═════════════════════════════════════════════════════════════════════════════
def fig_topologies():
    rng = np.random.default_rng(3)
    n = 40
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(12, 50, n)
    P = np.c_[rad * np.cos(ang), rad * np.sin(ang)]                   # physical position (mm)
    gtv_f = np.array([0.0, 0.0])
    F = rng.normal(0, 1, (n, 2)) + (P / 60.0)                          # feature vector (2-D toy)
    d_feat = np.linalg.norm(F - gtv_f, axis=1)
    d_phys = np.linalg.norm(P, axis=1)
    K, k = 10, 3

    def knn_edges(X):
        D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        return {tuple(sorted((i, int(j)))) for i in range(len(X)) for j in np.argsort(D[i])[:k]}

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.05))
    for ax, (topo, title) in zip(axs, TNAME.items()):
        sel = np.argsort(d_feat if topo.startswith('radiomic') else d_phys)[:K]
        ax.scatter(P[:, 0], P[:, 1], s=7, c=NEUTRAL, lw=0, zorder=1)
        Q = np.vstack([[0, 0], P[sel]])
        for j in range(1, K + 1):
            ax.plot([0, Q[j, 0]], [0, Q[j, 1]], color=CAT[0], lw=0.9, zorder=2)
        if topo.endswith('knn'):
            X = np.vstack([gtv_f, F[sel]]) if topo.startswith('radiomic') else Q
            for i, j in knn_edges(X):
                ax.plot(Q[[i, j], 0], Q[[i, j], 1], color=CAT[1], lw=0.8, zorder=2)
        ax.scatter(Q[1:, 0], Q[1:, 1], s=22, c=CAT[0], ec='white', lw=0.6, zorder=3)
        ax.scatter([0], [0], s=110, marker='*', c=CAT[3], ec=INK, lw=0.5, zorder=4)
        ax.set_title(title, fontsize=8.5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].plot([], [], color=CAT[0], lw=1, label='GTV–supervoxel edge')
    axs[0].plot([], [], color=CAT[1], lw=1, label='kNN edge')
    axs[0].scatter([], [], marker='*', s=60, c=CAT[3], ec=INK, lw=0.5, label='GTV (read-out node)')
    axs[0].scatter([], [], s=18, c=CAT[0], label='selected supervoxel')
    axs[0].scatter([], [], s=8, c=NEUTRAL, label='other supervoxel')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, 'fig2_topologies')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3/4/S*: ROC figures (copied from export_results.py output)
# ═════════════════════════════════════════════════════════════════════════════
def copy_existing():
    for t in TASKS:
        shutil.copy(RES / 'figures' / f'roc_{t}.png', FIG / f'roc_{t}.png')
        shutil.copy(RES / 'figures' / f'roc_cv_folds_{t}.png', FIG / f'roc_cv_folds_{t}.png')
        for inp in ('img', 'img+clin'):
            src = RUNS / f'MATEUS_CV_{t}' / 'figures' / 'results' / f'auc_heatmap_test_{inp}.png'
            if src.exists():
                shutil.copy(src, FIG / f'heatmap_cv_{t}_{inp.replace("+", "_")}.png')
    pg = sorted((RUNS / 'MATEUS_CV_LR' / 'fold1' / 'figures' / 'patient_graphs').glob(
        'reference_table_s3__radiomic_star__img__s42_HN*.png'))
    if pg:
        shutil.copy(pg[0], FIG / 'patient_attention_example.png')
    att = RUNS / 'MATEUS_CV_LR' / 'fold1' / 'figures' / 'patient_graphs' / \
        'reference_table_s3__radiomic_star__img__s42_attention_vs_distance.png'
    if att.exists():
        shutil.copy(att, FIG / 'attention_vs_distance.png')
    for f in ('feature_graph_graph_fs_spectral.png', 'selected_features_matrix.png'):
        src = RUNS / 'MATEUS_CV_LR' / 'fold1' / 'figures' / 'selection' / f
        if src.exists():
            shutil.copy(src, FIG / f'lr_fold1_{f}')
    src = RUNS / 'MATEUS_CV_LR' / 'figures' / 'selection' / 'selected_per_fold.png'
    if src.exists():
        shutil.copy(src, FIG / 'lr_selected_per_fold.png')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: external test AUC overview (all outcomes, designs, inputs)
# ═════════════════════════════════════════════════════════════════════════════
def fig_overview():
    m = main[main.epoch_rule == PRIMARY]
    roles = ['chosen GAT', 'reference GAT', 'best RF', 'clinical-only RF']
    colours = dict(zip(roles, CAT[:4]))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    rows = [(d, i) for d in DESIGN for i in ('imaging', 'imaging+clinical')]
    for ax, t in zip(axs, TASKS):
        for yi, (d, inp) in enumerate(rows):
            sub = m[(m.task == t) & (m.design == d) & (m.inputs == inp)]
            present = [r for r in roles if r in set(sub.role)]
            for k, role in enumerate(present):
                r = sub[sub.role == role].iloc[0]
                y = yi + (k - (len(present) - 1) / 2) * 0.2
                ax.plot([r.test_auc_ci_low, r.test_auc_ci_high], [y, y], color=colours[role], lw=1.3)
                ax.plot(r.test_auc, y, 'o', ms=4.2, color=colours[role], mec='white', mew=0.5)
        ax.axvline(0.5, color=NEUTRAL, ls='--', lw=0.8)
        ax.set_xlim(0, 1); ax.set_title({'LR': 'LR (2 years)', 'DM': 'DM (2 years)', 'OS': 'OS (4 years)'}[t])
        ax.set_xlabel('External test AUC [95% CI]')
        ax.set_ylim(len(rows) - 0.5, -0.5)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([f'{DESIGN[d]}\n{"imaging" if i == "imaging" else "imaging + clinical"}'
                            for d, i in rows])
    for role in roles:
        axs[0].plot([], [], 'o-', color=colours[role], ms=4, lw=1.3,
                    label={'chosen GAT': 'GAT, configuration chosen on validation',
                           'reference GAT': 'GAT, pre-specified reference',
                           'best RF': 'RF on selected GTV features',
                           'clinical-only RF': 'Clinical-only RF'}[role])
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig5_auc_overview')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: feature-selection stability (three settings) per method and task
# ═════════════════════════════════════════════════════════════════════════════
def stability_frame():
    s = stab[(stab.design == 'MATEUS_CV') & (stab.epoch_rule == PRIMARY)].copy()
    s['setting'] = s.stability_type.map(lambda x: 'CV folds' if x.startswith('across')
                                        else ('Bootstrap' if x.startswith('bootstrap') else 'Centres'))
    return s.groupby(['task', 'setting', 'method'])[['kuncheva', 'jaccard', 'kendall_w']].mean().reset_index()


def fig_stability():
    s = stability_frame()
    settings = ['CV folds', 'Bootstrap', 'Centres']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    w = 0.26
    for ax, setting in zip(axs, settings):
        for k, t in enumerate(TASKS):
            sub = s[(s.setting == setting) & (s.task == t)].set_index('method').reindex(SELECTORS)
            y = np.arange(len(SELECTORS)) + (k - 1) * w
            ax.barh(y, sub.kuncheva.values, height=w * 0.9, color=CAT[k], label=t)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title({'CV folds': 'Across 5 CV folds', 'Bootstrap': 'Bootstrap (B = 30)',
                      'Centres': 'Across centres (CHUS vs HGJ)'}[setting])
        ax.set_xlabel('Kuncheva index'); ax.set_xlim(-0.15, 1)
    axs[0].set_yticks(range(len(SELECTORS)))
    axs[0].set_yticklabels([MNAME[m] for m in SELECTORS])
    axs[0].invert_yaxis()
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig6_stability')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7: stability vs discrimination, and validation vs test AUC
# ═════════════════════════════════════════════════════════════════════════════
def fig_stab_vs_auc():
    s = stability_frame()
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & (allc.design == 'MATEUS_CV')]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    rho = {}
    for ax, t in zip(axs, TASKS):
        auc = g[g.task == t].groupby('method').test_auc.mean()
        kun = s[(s.setting == 'CV folds') & (s.task == t)].set_index('method').kuncheva
        ms = [m for m in SELECTORS if m in auc.index and m in kun.index]
        x, y = kun[ms].values, auc[ms].values
        placed = []
        for m, xi, yi in sorted(zip(ms, x, y), key=lambda z: -z[2]):
            c = CAT[1] if m.startswith('graph_fs') else CAT[0]
            ax.plot(xi, yi, 'o', ms=5, color=c, mec='white', mew=0.5)
            ly = yi
            while any(abs(ly - py) < 0.009 and abs(xi - px) < 0.25 for px, py in placed):
                ly -= 0.009
            placed.append((xi, ly))
            ax.annotate(MNAME[m].replace('Graph-FS ', 'G-FS '), (xi, yi), xytext=(xi + 0.02, ly + 0.002),
                        textcoords='data', fontsize=6, color=INK2)
        r = pd.Series(x).corr(pd.Series(y), method='spearman')
        rho[t] = r
        ax.axhline(0.5, color=NEUTRAL, ls='--', lw=0.8)
        ax.set_title(f'{t}  (Spearman ρ = {r:.2f})')
        ax.set_xlabel('Selection stability across folds\n(Kuncheva)')
        ax.set_xlim(0, 0.9)
    axs[0].set_ylabel('Mean external AUC\n(4 topologies × 2 input sets)')
    fig.tight_layout()
    save(fig, 'fig7_stability_vs_auc')
    for t in TASKS:
        num(f'rhoStab{t}', f'{rho[t]:.2f}')


def fig_val_vs_test():
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, sharex=True)
    for ax, t in zip(axs, TASKS):
        parts = []
        for k, d in enumerate(DESIGN):
            sub = g[(g.task == t) & (g.design == d)]
            r = sub[['val_auc', 'test_auc']].corr('spearman').iloc[0, 1]
            num(f'rhoVal{t}{"CV" if d == "MATEUS_CV" else "Coh"}', f'{r:.2f}')
            ax.scatter(sub.val_auc, sub.test_auc, s=9, color=CAT[k], alpha=0.75, lw=0,
                       label=f'{DESIGN[d]} (ρ = {r:.2f})')
        ax.plot([0, 1], [0, 1], color=NEUTRAL, ls='--', lw=0.8)
        ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95)
        ax.set_title(t); ax.set_xlabel('Validation AUC')
        ax.legend(loc='upper left', fontsize=6.3, handletextpad=0.2)
    axs[0].set_ylabel('External test AUC')
    fig.tight_layout()
    save(fig, 'fig8_val_vs_test')


# ═════════════════════════════════════════════════════════════════════════════
# Tables
# ═════════════════════════════════════════════════════════════════════════════
def table_cohort():
    """Patient characteristics of the analysed patients (LR-eligible with images) + outcome counts."""
    from grafsgat import config
    from grafsgat.data.clinical import load_clinical, compute_labels, encode_clinical
    from grafsgat.data.loader import HNSCCDataLoader
    config.set_mode('MATEUS_CV')
    rows = {}
    for ds in ('HNPET', 'MAASTRO'):
        df = load_clinical(ds).set_index('patient_id', drop=False)
        img = set(HNSCCDataLoader(ds).eligible_patients_any_task())
        enc, _ = encode_clinical(ds, harmonized=True)
        ids = [p for p in df.patient_id if p in img]
        e = enc.loc[[p for p in ids if p in enc.index]]
        age = e['age']
        col = {}
        col['Patients analysed'] = f'{len(ids)}'
        col['Age, median (IQR), y'] = f'{age.median():.0f} ({age.quantile(.25):.0f}--{age.quantile(.75):.0f})'
        col['Male sex'] = f'{int(e.sex_male.sum())} ({100 * e.sex_male.mean():.0f}\\%)'
        for k, lab in (('hpv_positive', 'HPV positive'), ('hpv_negative', 'HPV negative'),
                       ('hpv_unknown', 'HPV unknown')):
            col[lab] = f'{int(e[k].sum())} ({100 * e[k].mean():.0f}\\%)'
        for k, lab in (('subsite_oropharynx', 'Oropharynx'), ('subsite_larynx', 'Larynx')):
            col[lab] = f'{int(e[k].sum())} ({100 * e[k].mean():.0f}\\%)'
        oth = 1 - e.subsite_oropharynx - e.subsite_larynx
        col['Other subsite'] = f'{int(oth.sum())} ({100 * oth.mean():.0f}\\%)'
        t34 = e.t_t3 + e.t_t4
        col['T3--T4'] = f'{int(t34.sum())} ({100 * t34.mean():.0f}\\%)'
        n23 = e.n_n2 + e.n_n3
        col['N2--N3'] = f'{int(n23.sum())} ({100 * n23.mean():.0f}\\%)'
        col['Concurrent chemotherapy'] = f'{int(e.concurrent_chemo.sum())} ({100 * e.concurrent_chemo.mean():.0f}\\%)'
        for t in TASKS:
            c = coh[(coh.task == t) & (coh.cohort == ds)].iloc[0]
            col[f'{t}: analysed (events)'] = f'{c.patients} ({c.events}, {100 * c.event_rate:.1f}\\%)'
        rows[ds] = col
        if ds == 'HNPET':
            cen = pd.Series([p.split('-')[0] for p in ids]).value_counts()
            for c in ('CHUM', 'CHUS', 'HGJ', 'HMR'):
                num(f'n{c}', str(int(cen.get(c, 0))))
            num('nHNPETimg', str(len(ids)))
            num('nHNPETclin', str(len(df)))
    T = pd.DataFrame(rows)
    lines = [r'\begin{tabular}{@{}lll@{}}', r'\toprule',
             r'Characteristic & Canadian centres$^{a}$ & MAASTRO (HN1) \\',
             r' & training / validation & external test \\', r'\midrule']
    for k, r in T.iterrows():
        if k == 'LR: analysed (events)':
            lines.append(r'\midrule')
            lines.append(r'\multicolumn{3}{@{}l}{\textit{Outcome cohorts$^{b}$, patients (events, rate)}} \\')
        lines.append(f'{k} & {r.HNPET} & {r.MAASTRO} \\\\')
    lines += [r'\botrule', r'\end{tabular}']
    (TAB / 'table1_cohort.tex').write_text('\n'.join(lines) + '\n')
    for t in TASKS:
        for ds in ('HNPET', 'MAASTRO'):
            c = coh[(coh.task == t) & (coh.cohort == ds)].iloc[0]
            num(f'n{t}{ds}', str(c.patients)); num(f'e{t}{ds}', str(c.events))


def table_main():
    m = main[main.epoch_rule == PRIMARY]
    role_name = {'chosen GAT': 'GAT, chosen on validation', 'reference GAT': 'GAT, reference',
                 'best RF': 'RF, selected GTV features', 'clinical-only RF': 'RF, clinical only'}
    lines = [r'\begin{tabular}{@{}llllcc@{}}', r'\toprule',
             r'Outcome / design & Inputs & Model & Configuration & Val. AUC & Test AUC [95\% CI] \\',
             r'\midrule']
    for t in TASKS:
        for d in DESIGN:
            first = True
            for inp in ('imaging', 'imaging+clinical'):
                sub = m[(m.task == t) & (m.design == d) & (m.inputs == inp)]
                for _, r in sub.iterrows():
                    cfg = (MNAME.get(r.fs_method, r.fs_method) if r.role != 'clinical-only RF' else '--')
                    if r.role in ('chosen GAT', 'reference GAT'):
                        cfg += ' + ' + TNAME.get(r.topology, r.topology)
                    p = '' if pd.isna(r.p_vs_reference_holm) or r.role != 'chosen GAT' else \
                        (f'$^{{p={r.p_vs_reference_holm:.2f}}}$' if r.p_vs_reference_holm >= 0.01 else '$^{p<0.01}$')
                    lab = f'{t}, {DESIGN[d]}' if first else ''
                    first = False
                    lines.append(f'{lab} & {"Img" if inp == "imaging" else "Img+Clin"} & '
                                 f'{role_name[r.role]} & {cfg} & {f2(r.val_auc)} & {ci(r)}{p} \\\\')
            lines.append(r'\addlinespace')
        lines[-1] = r'\midrule'
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'table2_main.tex').write_text('\n'.join(lines) + '\n')

    # macros for the text
    def g(t, d, inp, role, col='test_auc'):
        r = m[(m.task == t) & (m.design == d) & (m.inputs == inp) & (m.role == role)]
        return r.iloc[0] if len(r) else None
    for t in TASKS:
        for d, dk in (('MATEUS_CV', 'CV'), ('MATEUS_COHORT', 'Coh')):
            for inp, ik in (('imaging', 'Img'), ('imaging+clinical', 'Clin')):
                for role, rk in (('chosen GAT', 'Chosen'), ('reference GAT', 'Ref'),
                                 ('best RF', 'RF'), ('clinical-only RF', 'ClinRF')):
                    r = g(t, d, inp, role)
                    if r is not None:
                        num(f'auc{t}{dk}{ik}{rk}', f'{r.test_auc:.2f}')
                        num(f'ci{t}{dk}{ik}{rk}', f'{r.test_auc_ci_low:.2f}--{r.test_auc_ci_high:.2f}')
    # fold-mean test AUC of the chosen CV models (comparable with Mateus et al.'s "mean (range)")
    a = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & (allc.design == 'MATEUS_CV') & allc.best_by_val]
    for _, r in a.iterrows():
        ik = 'Img' if r.inputs == 'imaging' else 'Clin'
        num(f'fold{r.task}{ik}', f'{r.test_auc_fold_mean:.2f}')
        num(f'foldRange{r.task}{ik}', f'{r.test_auc_fold_min:.2f}--{r.test_auc_fold_max:.2f}')


def table_mateus():
    """Our external AUCs next to Mateus et al. 2023 (Tables 3 and 4 of their paper)."""
    mateus = MATEUS_REPORTED
    a = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & allc.best_by_val]
    lines = [r'\begin{tabular}{@{}lllll@{}}', r'\toprule',
             r'Outcome & Design & Inputs & This study (GAT) & Mateus et al.\ CNN \\', r'\midrule']
    for t in TASKS:
        first = True
        for d in ('MATEUS_CV', 'MATEUS_COHORT'):
            for inp in ('imaging', 'imaging+clinical'):
                r = a[(a.task == t) & (a.design == d) & (a.inputs == inp)].iloc[0]
                ours = (f'{r.test_auc_fold_mean:.2f} ({r.test_auc_fold_min:.2f}--{r.test_auc_fold_max:.2f})'
                        if d == 'MATEUS_CV' else ci(r))
                lines.append(f'{TASK_LONG[t] if first else ""} & {DESIGN[d]} & '
                             f'{"Img" if inp == "imaging" else "Img+Clin"} & {ours} & {mateus[(d, inp)][t]} \\\\')
                first = False
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'table3_mateus.tex').write_text('\n'.join(lines) + '\n')


MATEUS_REPORTED = {
    ('MATEUS_CV', 'imaging'): {'DM': '0.83 (0.76--0.90)', 'LR': '0.53 (0.48--0.59)', 'OS': '0.63 (0.57--0.72)'},
    ('MATEUS_COHORT', 'imaging'): {'DM': '0.89 [0.79, 0.98]', 'LR': '0.45 [0.32, 0.57]', 'OS': '0.67 [0.57, 0.77]'},
    ('MATEUS_CV', 'imaging+clinical'): {'DM': '0.88 (0.86--0.90)', 'LR': '0.57 (0.53--0.60)', 'OS': '0.68 (0.63--0.71)'},
    ('MATEUS_COHORT', 'imaging+clinical'): {'DM': '0.93 [0.86, 0.99]', 'LR': '0.59 [0.47, 0.70]', 'OS': '0.69 [0.59, 0.79]'},
}

# Mateus et al. 2023, Table 4 ("Our ANN", testing row): clinical variables only, no imaging
MATEUS_ANN = {
    'MATEUS_CV': {'DM': '0.86 (0.81--0.89)', 'LR': '0.53 (0.50--0.54)', 'OS': '0.63 (0.61--0.64)'},
    'MATEUS_COHORT': {'DM': '0.87 [0.78, 0.95]', 'LR': '0.41 [0.29, 0.54]', 'OS': '0.63 [0.52, 0.73]'},
}


def _gtv_fmt(P, design):
    """Mateus format from a prediction table (split/fold/patient_id/label/prob[/seed]):
    CV -> mean (range) of per-fold external AUCs; cohort split -> AUC [95% CI]."""
    from grafsgat.utils import safe_auc, bootstrap_auc_ci
    t = P[P.split == 'test']
    if design == 'MATEUS_CV':
        a = []
        for _, x in t.groupby('fold'):
            e = x.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
            a.append(safe_auc(e.label, e.prob))
        return f'{np.mean(a):.2f} ({min(a):.2f}--{max(a):.2f})', float(np.mean(a))
    e = t.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
    v = safe_auc(e.label, e.prob)
    lo, hi = bootstrap_auc_ci(e.label.values, e.prob.values)
    return f'{v:.2f} [{lo:.2f}, {hi:.2f}]', float(v)


def table_gtv_only():
    """Sensitivity analysis without the peritumoral region (grafs-gat/gtv_only/)."""
    G = OUT / 'gtv_only' / 'results'
    if not (G / 'gtv_only_summary.csv').exists():
        print('  [gtv-only] no results, skipped')
        return
    S = pd.read_csv(G / 'gtv_only_summary.csv')
    a = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & allc.best_by_val]
    lines = [r'\begin{tabular}{@{}lllllllll@{}}', r'\toprule',
             r' & & & \multicolumn{4}{c}{This study} & \multicolumn{2}{c}{Mateus et al.} \\',
             r'\cmidrule(lr){4-7}\cmidrule(lr){8-9}',
             r'Outcome & Design & Inputs & GTV + peritumoral GAT & GTV radiomics & Intratumoral GAT & '
             r'Simple reference$^{a}$ & CNN & ANN$^{b}$ \\', r'\midrule']
    cfg_lines = [r'\begin{tabular}{@{}llllp{0.55\textwidth}@{}}', r'\toprule',
                 r'Outcome & Design & Inputs & Model & Configuration chosen on validation AUC \\', r'\midrule']
    short = {'LR': 'LR', 'DM': 'DM', 'OS': 'OS'}
    for t in TASKS:
        first = True
        for d in ('MATEUS_CV', 'MATEUS_COHORT'):
            dn = 'CV' if d == 'MATEUS_CV' else 'Coh'
            for inp in ('imaging', 'imaging+clinical'):
                tag = 'Img' if inp == 'imaging' else 'Clin'
                r = a[(a.task == t) & (a.design == d) & (a.inputs == inp)].iloc[0]
                peri = (f'{r.test_auc_fold_mean:.2f} ({r.test_auc_fold_min:.2f}--{r.test_auc_fold_max:.2f})'
                        if d == 'MATEUS_CV' else ci(r))
                Pc = pd.read_csv(G / f'classical_pred_{t}_{d}.csv.gz')
                sc = S[(S.task == t) & (S.design == d) &
                       (S.model == f'GTV classical, {inp} (chosen on val)')].iloc[0]
                cls, vc = _gtv_fmt(Pc[Pc.config == sc.config], d)
                Pg = pd.read_csv(G / f'gat_pred_{t}_{d}_{inp}.csv.gz')
                gat, vg = _gtv_fmt(Pg, d)
                ref_cfg = 'volume|lr' if inp == 'imaging' else 'clinical|lr'
                ref, vr = _gtv_fmt(Pc[Pc.config == ref_cfg], d)
                num(f'gtvCls{t}{dn}{tag}', f'{vc:.2f}')
                num(f'gtvGat{t}{dn}{tag}', f'{vg:.2f}')
                num(f'gtvRef{t}{dn}{tag}', f'{vr:.2f}')
                lines.append(f'{TASK_LONG[t] if first else ""} & {DESIGN[d]} & {"Img" if inp == "imaging" else "Img+Clin"}'
                             f' & {peri} & {cls} & {gat} & {ref} & {MATEUS_REPORTED[(d, inp)][t]}'
                             f' & {MATEUS_ANN[d][t] if inp == "imaging+clinical" else "--"} \\\\')
                fs_, meth, k, clf, _ = sc.config.split('|')
                clfn = {'lr_C0.03': 'logistic, $C=0.03$', 'lr_C0.3': 'logistic, $C=0.3$', 'lr_C3': 'logistic, $C=3$',
                        'rf_d3': 'random forest, depth 3', 'rf_d6': 'random forest, depth 6'}[clf]
                fsn = {'orig': '93 original', 'orig+shape': '93 original + 14 shape',
                       'full': '1223 (shape, original, LoG, wavelet)'}[fs_]
                methn = {'mrmr': 'mRMR', 'lasso': 'LASSO', 'ucorr': 'univariate AUC filter'}[meth]
                cfg_lines.append(f'{short[t] if first else ""} & {DESIGN[d]} & {"Img" if inp == "imaging" else "Img+Clin"}'
                                 f' & GTV radiomics & {fsn} features; {methn}, $k={k[1:]}$; {clfn} \\\\')
                gc = S[(S.task == t) & (S.design == d) &
                              (S.model == f'Intratumoral GAT, {inp} (chosen on val)')].iloc[0].config
                kv = dict(x.split('=') for x in gc.split())
                glob = {'none': 'none', 'shape': 'GTV shape', 'shape+clinical': 'GTV shape + clinical'}[kv['global']]
                cfg_lines.append(f' & & & Intratumoral GAT & {"mRMR" if kv["fs"] == "mrmr" else "LASSO"}, $k={kv["k"]}$; '
                                 f'{kv["topology"]} edges; read-out vector: {glob}; {kv["layers"]} layers, '
                                 f'{kv["heads"]} heads $\\times$ {kv["hidden"]} units; dropout {kv["dropout"]}; '
                                 f'learning rate {float(kv["lr"]):g}; weight decay {float(kv["wd"]):g}; '
                                 f'{kv["sampling"]} sampling \\\\')
                first = False
        lines.append(r'\midrule'); cfg_lines.append(r'\midrule')
    lines[-1] = r'\botrule'; cfg_lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}'); cfg_lines.append(r'\end{tabular}')
    (TAB / 'table_gtv_only.tex').write_text('\n'.join(lines) + '\n')
    (TAB / 'tabS_gtv_only_configs.tex').write_text('\n'.join(cfg_lines) + '\n')
    # grid sizes and spread (for the text)
    import glob as _g
    ncls = len(pd.read_csv(G / 'classical_DM_MATEUS_CV.csv'))
    ngat = len(pd.read_csv(G / 'gat_search_DM_MATEUS_CV.csv'))
    num('gtvNcls', str(ncls)); num('gtvNgat', str(ngat))
    sub_n = []
    for f in sorted((OUT / 'gtv_only' / 'features').glob('*.npz')):
        with np.load(f) as z:
            sub_n.append(len(z['sub']))
    num('gtvSubMed', str(int(np.median(sub_n)))); num('gtvSubMin', str(int(min(sub_n))))
    num('gtvSubMax', str(int(max(sub_n)))); num('gtvNpat', str(len(sub_n)))


def table_stability():
    s = stability_frame()
    lines = [r'\begin{tabular}{@{}l' + 'ccc' * 3 + r'@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Across CV folds} & \multicolumn{3}{c}{Bootstrap} & \multicolumn{3}{c}{Across centres} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r'Method & LR & DM & OS & LR & DM & OS & LR & DM & OS \\', r'\midrule']
    best = {}
    for setting in ('CV folds', 'Bootstrap', 'Centres'):
        for t in TASKS:
            sub = s[(s.setting == setting) & (s.task == t)].set_index('method').kuncheva
            best[(setting, t)] = sub.reindex(SELECTORS).max()
    for m in SELECTORS:
        cells = []
        for setting in ('CV folds', 'Bootstrap', 'Centres'):
            for t in TASKS:
                v = s[(s.setting == setting) & (s.task == t) & (s.method == m)].kuncheva
                v = v.iloc[0] if len(v) else np.nan
                txt = f2(v)
                if not pd.isna(v) and np.isclose(v, best[(setting, t)]):
                    txt = r'\textbf{' + txt + '}'
                cells.append(txt)
        lines.append(f'{MNAME[m]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\botrule', r'\end{tabular}']
    (TAB / 'table4_stability.tex').write_text('\n'.join(lines) + '\n')
    cv = s[s.setting == 'CV folds']
    for t in TASKS:
        sub = cv[cv.task == t].set_index('method').kuncheva
        num(f'kunSpec{t}', f2(sub.get('graph_fs_spectral')))
        num(f'kunLasso{t}', f2(sub.get('lasso')))
        num(f'kunRfe{t}', f2(sub.get('rfe_rf')))
        num(f'kunMrmr{t}', f2(sub.get('mrmr')))
        num(f'kunBoruta{t}', f2(sub.get('boruta')))
        num(f'kunCC{t}', f2(sub.get('graph_fs_cc')))
        num(f'kunHyb{t}', f2(sub.get('graph_fs_hybrid')))
    b = s[s.setting == 'Bootstrap']
    for t in TASKS:
        sub = b[b.task == t].set_index('method').kuncheva
        num(f'bootSpec{t}', f2(sub.get('graph_fs_spectral')))


def table_method_topology():
    """Supplementary: mean external AUC per method x topology (CV design, primary rule)."""
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY)]
    out = []
    for t in TASKS:
        for d in DESIGN:
            for inp in ('imaging', 'imaging+clinical'):
                sub = g[(g.task == t) & (g.design == d) & (g.inputs == inp)]
                piv = sub.pivot_table(index='method', columns='topology', values='test_auc')
                piv = piv.reindex([m for m in METHODS if m in piv.index])[list(TNAME)[:4]]
                lines = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
                         'Method & ' + ' & '.join(TNAME[c] for c in piv.columns) + r' \\', r'\midrule']
                for mth, r in piv.iterrows():
                    lines.append(f'{MNAME[mth]} & ' + ' & '.join(f2(v) for v in r.values) + r' \\')
                lines += [r'\botrule', r'\end{tabular}']
                name = f'tabS_grid_{t}_{"cv" if d == "MATEUS_CV" else "coh"}_{"img" if inp == "imaging" else "clin"}.tex'
                (TAB / name).write_text('\n'.join(lines) + '\n')
                out.append((t, d, inp, name))
    # overall means for the text
    for t in TASKS:
        mt = g[g.task == t].groupby('topology').test_auc.mean()
        for topo, v in mt.items():
            num(f'topo{t}{topo.replace("_", "")}', f'{v:.2f}')
        mi = g[g.task == t].groupby('inputs').test_auc.mean()
        num(f'meanImg{t}', f'{mi["imaging"]:.2f}'); num(f'meanClin{t}', f'{mi["imaging+clinical"]:.2f}')
        num(f'nConfig{t}', str(len(g[g.task == t])))
        num(f'nSig{t}', str(int((g[g.task == t].p_vs_reference_holm < 0.05).sum())))
    return out


def table_epoch_rule():
    """Supplementary: chosen / reference GAT under the checkpoint rules (primary, exact Mateus rule, adapted rule)."""
    m = main[main.role.isin(['chosen GAT', 'reference GAT'])]
    rules = [r for r in ('val_loss', SENS_RULE, 'auc_gap') if r in set(m.epoch_rule)]
    rules = list(dict.fromkeys(rules))
    head = {'val_loss': r'Lowest val.\ loss', 'mateus': r'Mateus et al.\ rule', 'auc_gap': r'Adapted AUC-gap rule'}
    lines = [r'\begin{tabular}{@{}llll' + 'c' * len(rules) + r'@{}}', r'\toprule',
             r'Outcome & Design & Inputs & Model & ' + ' & '.join(head[r] for r in rules) + r' \\', r'\midrule']
    for t in TASKS:
        first = True
        for d in DESIGN:
            for inp in ('imaging', 'imaging+clinical'):
                for role in ('chosen GAT', 'reference GAT'):
                    v = []
                    for rule in rules:
                        r = m[(m.task == t) & (m.design == d) & (m.inputs == inp) & (m.role == role) & (m.epoch_rule == rule)]
                        v.append(ci(r.iloc[0]) if len(r) else '--')
                    lines.append(f'{t if first else ""} & {DESIGN[d]} & {"Img" if inp == "imaging" else "Img+Clin"} & '
                                 f'{"chosen" if role == "chosen GAT" else "reference"} & ' + ' & '.join(v) + r' \\')
                    first = False
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'tabS_epoch_rule.tex').write_text('\n'.join(lines) + '\n')
    for t, d, inp, key in (('DM', 'MATEUS_COHORT', 'imaging+clinical', 'DMCohClin'),
                           ('LR', 'MATEUS_COHORT', 'imaging+clinical', 'LRCohClin')):
        r = m[(m.task == t) & (m.design == d) & (m.inputs == inp) & (m.role == 'chosen GAT') & (m.epoch_rule == SENS_RULE)]
        if len(r):
            num(f'ruleChosen{key}', f'{r.iloc[0].test_auc:.2f}')


def table_missing_images():
    """Supplementary: events lost because Head-Neck-PET-CT images were unavailable."""
    from grafsgat import config
    from grafsgat.data.clinical import load_clinical, compute_labels
    from grafsgat.data.loader import HNSCCDataLoader
    df = load_clinical('HNPET')
    img = set(HNSCCDataLoader('HNPET').eligible_patients_any_task())
    lines = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r'Outcome & \multicolumn{2}{c}{All patients in clinical file} & \multicolumn{2}{c}{Patients with CT + GTV} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & Patients & Events & Patients & Events \\', r'\midrule']
    for t in TASKS:
        lab = compute_labels('HNPET', t, verbose=False)
        d = df.join(lab)
        d = d[d.included]
        hi = d.patient_id.isin(img)
        lines.append(f'{TASK_LONG[t]} & {len(d)} & {int(d.label.sum())} & {int(hi.sum())} & {int(d[hi].label.sum())} \\\\')
        num(f'all{t}', str(len(d))); num(f'allEv{t}', str(int(d.label.sum())))
    lines += [r'\botrule', r'\end{tabular}']
    (TAB / 'tabS_missing_images.tex').write_text('\n'.join(lines) + '\n')


def write_numbers():
    import re
    out = ['% generated by scripts/make_paper_assets.py -- do not edit']
    for k, v in sorted(numbers.items()):
        name = re.sub(r'[^A-Za-z]', '', k.replace('0', 'Zero').replace('1', 'One').replace('2', 'Two')
                      .replace('3', 'Three').replace('4', 'Four').replace('5', 'Five'))
        out.append(f'\\newcommand{{\\{name}}}{{{v}}}')
    (TAB / 'numbers.tex').write_text('\n'.join(out) + '\n')


# ═════════════════════════════════════════════════════════════════════════════
# Clean versions of pipeline figures (baked-in titles removed; captions carry them)
# ═════════════════════════════════════════════════════════════════════════════
def crop_titles():
    from PIL import Image
    jobs = {  # file -> fraction of height removed at the top
        'roc_LR.png': 0.040, 'roc_DM.png': 0.040, 'roc_OS.png': 0.040,
        'roc_cv_folds_LR.png': 0.072, 'roc_cv_folds_DM.png': 0.072, 'roc_cv_folds_OS.png': 0.072,
        'lr_fold1_feature_graph_graph_fs_spectral.png': 0.068,
        'lr_selected_per_fold.png': 0.035, 'attention_vs_distance.png': 0.065,
    }
    for f, frac in jobs.items():
        p = FIG / f
        if not p.exists():
            continue
        im = Image.open(p)
        w, h = im.size
        im.crop((0, int(h * frac), w, h)).save(FIG / f.replace('.png', '_clean.png'))


# ═════════════════════════════════════════════════════════════════════════════
# Figure: GAT architecture (accurate fallback; replace by a designed figure if wanted)
# ═════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(-2, 40); ax.axis('off')

    # patient graph (star) on the left
    cx, cy, R = 9, 21, 7.5
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for a in ang:
        ax.plot([cx, cx + R * np.cos(a)], [cy, cy + R * np.sin(a)], color=CAT[0], lw=0.8, zorder=1)
        ax.scatter(cx + R * np.cos(a), cy + R * np.sin(a), s=30, color=CAT[0], ec='white', lw=0.5, zorder=2)
    ax.scatter(cx, cy, s=150, marker='*', color=CAT[3], ec=INK, lw=0.5, zorder=3)
    ax.text(cx, 12.0, 'GTV + 20 supervoxels\n$x_i\\in\\mathbb{R}^{k}$, $k=6$',
            ha='center', va='top', fontsize=6.2, color=INK2)
    ax.text(cx, 34.5, 'Input', ha='center', fontsize=7.6, weight='bold')

    def block(x, w, title, lines, colour, y=12, h=19):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.8',
                                    fc=colour + '1f', ec=colour, lw=1.0))
        ax.text(x + w / 2, y + h - 1.3, title, ha='center', va='top', fontsize=6.9, weight='bold', color=INK)
        ax.text(x + w / 2, y + h - 5.6, lines, ha='center', va='top', fontsize=5.9, color=INK2, linespacing=1.35)

    def arr(x0, x1, y=21.5):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))

    block(21, 19, 'GAT block  ×4', 'GATConv\n4 heads × 16 units\n(attention dropout 0.67)\n↓\nLayerNorm (per graph)\n↓\nReLU', CAT[0])
    block(43, 14, 'GAT layer 5', 'GATConv\n1 head × 16 units\n\n$h_v\\in\\mathbb{R}^{16}$\nfor every node', CAT[0])
    block(59.5, 13.5, 'GTV read-out', 'keep node 0\n$h_{GTV}\\in\\mathbb{R}^{16}$\n\ndropout 0.5', CAT[1])
    block(76, 10.5, 'Fusion', 'concatenate\nclinical $c\\in\\mathbb{R}^{16}$\n\n$z\\in\\mathbb{R}^{32}$', CAT[2])
    block(89, 10.5, 'Output', 'Linear\n32 → 1\nsigmoid\n\nP(event)', CAT[1])
    arr(17.5, 20.6); arr(40.3, 42.6); arr(57.3, 59.1); arr(73.3, 75.6); arr(86.8, 88.6)
    ax.text(30.5, 34.5, 'Message passing (5 layers)', ha='center', fontsize=7.6, weight='bold')
    ax.text(80.5, 34.5, 'Prediction', ha='center', fontsize=7.6, weight='bold')
    ax.text(50, 4.0,
            r'Attention: $\alpha_{ij}=\mathrm{softmax}_j\,\mathrm{LeakyReLU}\left(a^{\top}[Wh_i\,\|\,Wh_j]\right)$,  '
            r'$h_i^{\prime}=\Vert_{m=1}^{4}\sum_{j\in\mathcal{N}(i)\cup\{i\}}\alpha_{ij}^{m}W^{m}h_j$',
            ha='center', fontsize=6.6, color=INK)
    ax.text(50, 0.2, 'LR/OS settings shown (15,057 parameters with clinical input). DM: 3 layers, 8 units per head, '
            'dropout 0.89 (1,841 parameters). Imaging-only models omit the fusion step.',
            ha='center', fontsize=5.8, color=INK2)
    save(fig, 'fig_architecture')


# ═════════════════════════════════════════════════════════════════════════════
# Figure: method x topology heatmap (mean external AUC)
# ═════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', ['#c23a3a', '#ec8f8f', '#f0efec', '#86b6ef', '#256abf'])
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY)]
    topo = list(TNAME)[:4]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, t in zip(axs, TASKS):
        piv = g[g.task == t].pivot_table(index='method', columns='topology', values='test_auc', aggfunc='mean')
        piv = piv.reindex(METHODS)[topo]
        ax.imshow(np.ma.masked_invalid(piv.values), cmap=cmap, norm=TwoSlopeNorm(0.5, 0.3, 0.7), aspect='auto')
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, 'n/a', ha='center', va='center', fontsize=6, color=NEUTRAL)
                    continue
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.4,
                        color='white' if abs(v - 0.5) > 0.13 else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([TNAME[c].replace(' ', '\n') for c in topo], fontsize=6.5)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([MNAME[m] for m in METHODS], fontsize=6.8)
        ax.set_title(t); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, 'fig_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# Figure + table: GAT versus random forest on the same features (value of the graph)
# ═════════════════════════════════════════════════════════════════════════════
def fig_gat_vs_rf():
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & (allc.inputs == 'imaging')]
    r = allc[(allc.model == 'RF') & (allc.epoch_rule == PRIMARY) & (allc.method != 'clinical_only')]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    rows = []
    for ax, t in zip(axs, TASKS):
        ms = METHODS
        for k, d in enumerate(DESIGN):
            gm = g[(g.task == t) & (g.design == d)].groupby('method').test_auc.agg(['mean', 'min', 'max'])
            rm = r[(r.task == t) & (r.design == d)].set_index('method').test_auc
            sig = g[(g.task == t) & (g.design == d)].groupby('method').p_vs_rf_same_features_holm.min()
            for i, m in enumerate(ms):
                if m not in gm.index or m not in rm.index:
                    continue
                y = i + (k - 0.5) * 0.36
                ax.plot([gm.loc[m, 'min'], gm.loc[m, 'max']], [y, y], color=CAT[0], lw=0.8, alpha=0.45)
                ax.plot(gm.loc[m, 'mean'], y, 'o', ms=4, color=CAT[0], mec='white', mew=0.4,
                        marker='o' if k == 0 else 's')
                ax.plot(rm[m], y, 'D', ms=3.5, color=CAT[1], mec='white', mew=0.4)
            common = [m for m in gm.index if m in rm.index]
            n_better = int((gm.loc[common, 'mean'].values > rm.loc[common].values).sum())
            n = len(common)
            rows.append(dict(task=t, design=d, n_better=n_better, n=n,
                             gat=gm.loc[common, 'mean'].mean(), rf=rm.loc[common].mean(),
                             n_sig=int((sig < 0.05).sum())))
        ax.axvline(0.5, color=NEUTRAL, ls='--', lw=0.8)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([MNAME[m] for m in ms], fontsize=6.8)
        ax.set_title(t); ax.set_xlabel('External AUC (imaging only)'); ax.set_xlim(0.15, 0.85)
    axs[0].invert_yaxis()
    axs[0].plot([], [], 'o', color=CAT[0], ms=4, label='GAT, CV (mean, range over topologies)')
    axs[0].plot([], [], 's', color=CAT[0], ms=4, label='GAT, cohort split')
    axs[0].plot([], [], 'D', color=CAT[1], ms=3.5, label='RF on the same GTV features')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig_gat_vs_rf')

    T = pd.DataFrame(rows)
    lines = [r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
             r'Outcome & Design & GAT $>$ RF$^{a}$ & Mean AUC, GAT$^{b}$ & Mean AUC, RF & Significant$^{c}$ \\',
             r'\midrule']
    for t in TASKS:
        for k, d in enumerate(DESIGN):
            x = T[(T.task == t) & (T.design == d)].iloc[0]
            lines.append(f'{t if k == 0 else ""} & {DESIGN[d]} & {x.n_better}/{x.n} & {x.gat:.2f} & {x.rf:.2f} & {x.n_sig} \\\\')
            num(f'gatRf{t}{"CV" if d == "MATEUS_CV" else "Coh"}', f'{x.n_better}/{x.n}')
            num(f'gatRfSig{t}{"CV" if d == "MATEUS_CV" else "Coh"}', str(x.n_sig))
            num(f'gatMean{t}{"CV" if d == "MATEUS_CV" else "Coh"}', f'{x.gat:.2f}')
            num(f'rfMean{t}{"CV" if d == "MATEUS_CV" else "Coh"}', f'{x.rf:.2f}')
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'table_gat_vs_rf.tex').write_text('\n'.join(lines) + '\n')


def table_stability_full():
    """Supplementary: Jaccard, Kendall W and core-80 for every selector (CV design)."""
    s = stab[(stab.design == 'MATEUS_CV') & (stab.epoch_rule == PRIMARY)].copy()
    s['setting'] = s.stability_type.map(lambda x: 'CV folds' if x.startswith('across')
                                        else ('Bootstrap' if x.startswith('bootstrap') else 'Centres'))
    s = s.groupby(['task', 'setting', 'method'])[['jaccard', 'kendall_w', 'core80']].mean().reset_index()
    lines = [r'\begin{tabular}{@{}ll' + 'ccc' * 3 + r'@{}}', r'\toprule',
             r' & & \multicolumn{3}{c}{Across CV folds} & \multicolumn{3}{c}{Bootstrap} & \multicolumn{3}{c}{Across centres} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
             r'Outcome & Method & J & W & C80 & J & W & C80 & J & W & C80 \\', r'\midrule']
    for t in TASKS:
        for i, m in enumerate(SELECTORS):
            cells = []
            for setting in ('CV folds', 'Bootstrap', 'Centres'):
                x = s[(s.task == t) & (s.setting == setting) & (s.method == m)]
                cells += ([f2(x.jaccard.iloc[0]), f2(x.kendall_w.iloc[0]), f2(x.core80.iloc[0])]
                          if len(x) else ['--'] * 3)
            lines.append(f'{t if i == 0 else ""} & {MNAME[m]} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'tabS_stability_full.tex').write_text('\n'.join(lines) + '\n')


def extra_numbers():
    """Numbers quoted in the text that are not produced elsewhere."""
    g = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY)]
    for t in TASKS:
        x = g[g.task == t].test_auc
        num(f'gridMin{t}', f'{x.min():.2f}'); num(f'gridMax{t}', f'{x.max():.2f}')
        num(f'gridMed{t}', f'{x.median():.2f}')
        num(f'gridAbove{t}', f'{100 * (x > 0.5).mean():.0f}')
    m = main[main.role == 'reference GAT'].pivot_table(index=['task', 'design', 'inputs'],
                                                       columns='epoch_rule', values='test_auc')
    d = (m[SENS_RULE] - m.val_loss).abs()
    num('ruleRefMax', f'{d.max():.2f}'); num('ruleRefMed', f'{d.median():.2f}')
    # reference configuration validation AUC in the cohort split (LR) -- below chance
    r = main[(main.task == 'LR') & (main.design == 'MATEUS_COHORT') & (main.role == 'reference GAT') &
             (main.epoch_rule == PRIMARY) & (main.inputs == 'imaging')].iloc[0]
    num('valRefLRCohImg', f'{r.val_auc:.2f}')
    a = allc[(allc.model == 'GAT') & (allc.epoch_rule == PRIMARY) & (allc.design == 'MATEUS_CV') &
             allc.best_by_val & (allc.task == 'LR') & (allc.inputs == 'imaging')].iloc[0]
    num('valChosenLRCVImg', f'{a.val_auc:.2f}')


# ═════════════════════════════════════════════════════════════════════════════
# Figure: external ROC curves, all endpoints (rows) x design/inputs (columns), full width
# ═════════════════════════════════════════════════════════════════════════════
def fig_roc_all():
    from sklearn.metrics import roc_curve
    from grafsgat import config
    from grafsgat.experiments import export as ex
    cols = [('MATEUS_CV', False), ('MATEUS_CV', True), ('MATEUS_COHORT', False), ('MATEUS_COHORT', True)]
    short = {'chosen GAT': 'GAT, chosen', 'reference GAT': 'GAT, reference', 'best RF': 'RF, GTV features',
             'clinical-only RF': 'RF, clinical'}
    colour = {'chosen GAT': CAT[0], 'reference GAT': CAT[1], 'best RF': CAT[2], 'clinical-only RF': CAT[3]}
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.75))
    for i, t in enumerate(TASKS):
        cache = {}
        for j, (d, clin) in enumerate(cols):
            ax = axs[i, j]
            if d not in cache:
                comb, _ = ex.ensembles(t, d, PRIMARY)
                df = pd.read_csv(ex._run(t, d, PRIMARY) / 'results_summary.csv')
                cache[d] = (comb, df)
            comb, df = cache[d]
            config.set_mode(d); config.set_task(t)
            for role, r in ex._picks(df, clin):
                e = comb[(r.method, r.topology, bool(r.use_clinical))]
                e = e[e.split == 'test']
                y, p = e.label.values, e.prob.values
                fpr, tpr, _ = roc_curve(y, p)
                if role == 'chosen GAT':
                    lo, hi = ex._roc_band(y, p, n=500)
                    ax.fill_between(ex.FPR, lo, hi, color=CAT[0], alpha=0.14, lw=0)
                ax.step(fpr, tpr, where='post', color=colour[role], lw=1.3 if role == 'chosen GAT' else 1.0,
                        label=f'{short[role]} {r.test_auc:.2f}')
            ax.plot([0, 1], [0, 1], color=NEUTRAL, ls='--', lw=0.7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6)
            ax.legend(loc='lower right', fontsize=5.2, handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)
            if i == 0:
                ax.set_title(f'{DESIGN[d]}\n{"imaging + clinical" if clin else "imaging"}', fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f'{t}\nSensitivity', fontsize=7.5)
            if i == 2:
                ax.set_xlabel('1 - specificity', fontsize=7)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    save(fig, 'fig_roc_all')
    config.set_mode('MATEUS_CV')


# ═════════════════════════════════════════════════════════════════════════════
# Figure: Graph-FS workflow on the real training data (LR, CV fold 1)
# ═════════════════════════════════════════════════════════════════════════════
def fig_graphfs_workflow():
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    from grafsgat import config
    from grafsgat.selection.runner import train_matrix
    from grafsgat.selection import methods as M
    config.set_mode('MATEUS_CV'); config.set_task('LR'); config.set_fold(1)
    X, y, _, _ = train_matrix('LR')
    Z = M.prepare(X)
    names = list(Z.columns)
    R, A, thr = M.feature_graph(Z)
    spec = M.sel_graph_fs_spectral(X, y, 6)
    hyb = M.sel_graph_fs_hybrid(X, y, 6)
    config.set_fold(None)
    cls = [n.split('_')[1] for n in names]
    CLS = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    CLS_NAME = {'firstorder': 'First order', 'glcm': 'GLCM', 'glrlm': 'GLRLM', 'glszm': 'GLSZM',
                'gldm': 'GLDM', 'ngtdm': 'NGTDM'}
    ccol = dict(zip(CLS, CAT[:6]))
    short = lambda n: n.split('_', 2)[2]
    RED = '#d62728'

    # feature order: hybrid communities (largest first), members by relevance
    comms = sorted(hyb.info['components'], key=len, reverse=True)
    order = [f for c in comms for f in c]
    idx = [names.index(f) for f in order]
    bounds = np.cumsum([len(c) for c in comms])

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=0.93, bottom=0.07, wspace=0.5, hspace=0.38)

    # (a) data matrix
    ax = fig.add_subplot(gs[0, 0])
    rows = np.argsort(-y, kind='stable')
    M_ = np.clip(Z.values[rows][:, idx], -2.5, 2.5)
    ax.imshow(M_, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(f'{len(names)} radiomic features'); ax.set_ylabel(f'{len(y)} training patients')
    ax.set_title('Radiomic feature matrix', fontsize=8.5)
    ne = int(y.sum())
    ax.plot([-2.5, -2.5], [-0.5, ne - 0.5], color=RED, lw=3, clip_on=False)
    ax.text(-4, ne / 2, 'LR', color=RED, fontsize=6, ha='right', va='center', rotation=90)
    for j, f in enumerate(order):
        ax.add_patch(plt.Rectangle((j - 0.5, -6), 1, 4, color=ccol[f.split('_')[1]], clip_on=False, lw=0))

    # (b) similarity matrix
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(R.values[np.ix_(idx, idx)], cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    for b in bounds[:2]:
        ax.axhline(b - 0.5, color=INK, lw=0.6); ax.axvline(b - 0.5, color=INK, lw=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel('features'); ax.set_ylabel('features')
    ax.set_title('Similarity matrix |r|', fontsize=8.5)

    # (c) edge rule
    ax = fig.add_subplot(gs[0, 2])
    r = R.values.copy(); np.fill_diagonal(r, np.nan); v = r[np.triu_indices_from(r, 1)]
    ax.hist(v[v <= thr], bins=np.linspace(0, 1, 41), color=NEUTRAL)
    ax.hist(v[v > thr], bins=np.linspace(0, 1, 41), color=CAT[0])
    top = ax.get_ylim()[1]; ax.set_ylim(0, top * 1.38)
    ax.plot([thr, thr], [0, top * 1.18], color=RED, lw=1)
    ax.text(thr - 0.02, top * 1.33, f'edge if |r| > mean + 0.3 SD = {thr:.2f}', fontsize=6, color=RED, va='top', ha='center')
    ax.text(0.98, top * 1.12, f'{int(A.sum() // 2):,} edges', ha='right', fontsize=6.3, color=CAT[0], va='top')
    ax.set_xlabel('pairwise |r|'); ax.set_ylabel('feature pairs', labelpad=1)
    ax.set_title('Edge rule', fontsize=8.5)

    # graph layout shared by (d) and (e)
    G = nx.Graph(); G.add_nodes_from(range(len(names)))
    G.add_edges_from([(i, j) for i, j in zip(*np.where(np.triu(A, 1)))])
    pos = nx.spring_layout(G, seed=4, k=0.35, iterations=200)
    P = np.array([pos[i] for i in range(len(names))])
    E = np.array([(P[i], P[j]) for i, j in G.edges()])

    def draw_graph(ax, colours, highlight, title):
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(E, colors=NEUTRAL, linewidths=0.25, alpha=0.35))
        ax.scatter(P[:, 0], P[:, 1], s=14, c=colours, ec='white', lw=0.3, zorder=2)
        hi = [names.index(f) for f in highlight]
        ax.scatter(P[hi, 0], P[hi, 1], s=46, c=RED, ec=INK, lw=0.6, zorder=3)
        cx = np.median(P[:, 0])
        for i in hi:
            right = P[i, 0] < cx
            ax.annotate(short(names[i]), P[i], xytext=(4 if right else -4, 3), textcoords='offset points',
                        fontsize=5.3, color=INK, ha='left' if right else 'right')
        pad = 0.08 * (P[:, 0].max() - P[:, 0].min())
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, fontsize=8.5)

    # (d) feature graph coloured by class
    ax = fig.add_subplot(gs[1, 0])
    draw_graph(ax, [ccol[c] for c in cls], [], 'Feature graph')
    h = [Line2D([], [], marker='o', ls='', color=ccol[c], ms=4, label=CLS_NAME[c]) for c in CLS]
    ax.legend(handles=h, loc='lower left', fontsize=5.5, ncol=2, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0)

    # (e) spectral Graph-FS
    ax = fig.add_subplot(gs[1, 1])
    lab = np.zeros(len(names), int)
    for c, members in enumerate(spec.info['components']):
        for f in members:
            lab[names.index(f)] = c
    pal = [CAT[0]] + [CAT[1]] * 10
    draw_graph(ax, [pal[l] if l == 0 else CAT[1] for l in lab], spec.selected,
               'Spectral Graph-FS (k = 6)')
    ax.text(0.0, -0.06, f'cluster sizes: {", ".join(str(len(m)) for m in spec.info["components"])}',
            transform=ax.transAxes, fontsize=5.8, color=INK2)

    # (f) hybrid: relevance of community representatives
    ax = fig.add_subplot(gs[1, 2])
    F = pd.Series(hyb.info['relevance_F'])
    reps = [m[0] for m in hyb.info['components']]
    vals = F[reps].sort_values()
    colours = [RED if f in hyb.selected else NEUTRAL for f in vals.index]
    ax.barh(range(len(vals)), vals.values, color=colours, height=0.75)
    for k_, (f, v_) in enumerate(vals.items()):
        ax.text(v_ + vals.max() * 0.02, k_, f'{CLS_NAME[f.split("_")[1]]} {short(f)}', va='center', fontsize=5.0,
                color=INK if f in hyb.selected else INK2)
    ax.set_xlim(0, vals.max() * 2.15); ax.set_yticks([])
    ax.set_xlabel('relevance (ANOVA F)')
    ax.set_title(f'Hybrid Graph-FS ({len(reps)} communities)', fontsize=8.5)

    # panel letters and workflow arrows
    axes = fig.axes
    for a_, letter in zip(axes, 'abcdef'):
        a_.text(-0.16, 1.1, letter, transform=a_.transAxes, fontsize=10, weight='bold', va='bottom')

    def arrow(p0, p1):
        fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=10, color=CAT[0], lw=1.4))
    b = [a_.get_position() for a_ in axes]
    for i0, i1, gap in ((0, 1, 0.045), (1, 2, 0.075), (3, 4, 0.0), (4, 5, 0.0)):
        ym = (b[i0].y0 + b[i0].y1) / 2
        arrow((b[i0].x1 + 0.008, ym), (b[i1].x0 - gap - 0.008, ym))
    save(fig, 'fig_graphfs_workflow')
    num('gfsThr', f'{thr:.2f}'); num('gfsEdges', f'{int(A.sum() // 2):,}')
    num('gfsTrainN', str(len(y))); num('gfsTrainEv', str(int(y.sum())))
    num('gfsBigCluster', str(max(len(m) for m in spec.info['components'])))
    num('gfsHybComm', str(len(reps)))
    config.set_mode('MATEUS_CV')


def fig_selection_frequency():
    """Wide heatmap: how often (of 5 CV folds) each strategy selected each feature (LR)."""
    import json
    from matplotlib.colors import LinearSegmentedColormap
    runs = sorted((RUNS / 'MATEUS_CV_LR').glob('fold*/selection'))
    cnt = {}
    for m in SELECTORS:
        c = {}
        for r in runs:
            f = r / f'{m}.json'
            if f.exists():
                for x in json.loads(f.read_text())['selected']:
                    c[x] = c.get(x, 0) + 1
        cnt[m] = c
    T = pd.DataFrame(cnt).fillna(0).T.reindex(SELECTORS)
    cols = T.sum().sort_values(ascending=False).index[:30]
    T = T[cols]
    cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    ax.imshow(T.values, cmap=cmap, vmin=0, vmax=5, aspect='auto')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = int(T.values[i, j])
            if v:
                ax.text(j, i, v, ha='center', va='center', fontsize=5.5, color='white' if v >= 3 else INK)
    ax.set_yticks(range(len(SELECTORS))); ax.set_yticklabels([MNAME[m] for m in SELECTORS], fontsize=6.8)
    def nm(c):
        n = c.split('_', 2)[2]
        return ''.join(ch for ch in n if ch.isupper()) if len(n) > 20 else n
    pre = {'firstorder': 'FO', 'glcm': 'GLCM', 'glrlm': 'GLRLM', 'glszm': 'GLSZM', 'gldm': 'GLDM', 'ngtdm': 'NGTDM'}
    lab = [f"{pre[c.split('_')[1]]} {nm(c)}" for c in cols]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(lab, rotation=50, ha='right', fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(SELECTORS)), minor=True)
    ax.grid(which='minor', color=GRID, lw=0.5); ax.grid(which='major', visible=False)
    ax.tick_params(which='minor', length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, 'fig_selection_frequency')


TOPO_PATIENT = 'HN1260'     # external (MAASTRO) patient used to illustrate the topologies


def fig_topologies_ct(pid=None, out='fig2_topologies'):
    """The four graph topologies drawn on the real CT of one external patient (graphs used in fold 1)."""
    import torch
    from grafsgat import config
    from grafsgat.data.features import preprocessed_path
    pid = pid or TOPO_PATIENT
    config.set_mode('MATEUS_CV'); config.set_task('LR'); config.set_fold(1)
    graphs = {t: torch.load(config.graph_dir('LR', 'table_s3', t) / f'{pid}.pt', weights_only=False)
              for t in TNAME if t != 'RF'}
    config.set_fold(None)
    with np.load(preprocessed_path(pid), allow_pickle=False) as d:
        ct, gtv, sv = d['ct_array'], d['gtv_array'], d['supervoxel_labels']
    g0 = graphs['radiomic_star']
    z = int(np.clip(round(float(g0.pos[0, 0])), 0, ct.shape[0] - 1))
    if not gtv[z].any():                                        # slice with the largest GTV area
        z = int(np.argmax(gtv.reshape(gtv.shape[0], -1).sum(1)))
    img = np.clip(ct[z], -160, 240)                             # soft-tissue window (W400/L40)

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.25))
    for ax, (t, g) in zip(axs, graphs.items()):
        pos = g.pos.numpy()
        ax.imshow(img, cmap='gray', interpolation='bilinear')
        for node, sv_id in enumerate(g.sv_ids.numpy()):        # selected supervoxels on this slice
            if node == 0:
                continue
            m = sv[z] == sv_id
            if m.any():
                ax.contourf(m, levels=[0.5, 1.5], colors=[CAT[0]], alpha=0.28)
                ax.contour(m, levels=[0.5], colors=[CAT[0]], linewidths=0.4)
        ax.contourf(gtv[z], levels=[0.5, 1.5], colors=[CAT[1]], alpha=0.45)
        ax.contour(gtv[z], levels=[0.5], colors=[CAT[1]], linewidths=1.1)
        ei = g.edge_index.numpy()
        for s_, t_ in ei.T:
            if s_ < t_ and s_ != 0:
                ax.plot(pos[[s_, t_], 2], pos[[s_, t_], 1], color='#ffb000', lw=0.6, alpha=0.9)
        for node in range(1, len(pos)):
            ax.plot([pos[0, 2], pos[node, 2]], [pos[0, 1], pos[node, 1]], color='#9cc8ff', lw=0.55, alpha=0.9)
        near = np.abs(pos[1:, 0] - z) <= 6
        ax.scatter(pos[1:, 2], pos[1:, 1], s=np.where(near, 11, 5), color='#9cc8ff', ec=CAT[0], lw=0.4, zorder=3)
        ax.scatter(pos[0, 2], pos[0, 1], s=18, color=CAT[1], ec='white', lw=0.6, zorder=4)
        ax.set_title(TNAME[t], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)
    for ax, letter in zip(axs, 'abcd'):
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    fig.tight_layout(pad=0.2, w_pad=0.3)
    save(fig, out)
    config.set_mode('MATEUS_CV')
    num('topoPid', pid); num('topoSlice', str(z))


ATT_PATIENT = 'HN1792'      # external patient with locoregional recurrence, attention example


def fig_attention_ct(pid=None):
    """Attention of the GTV read-out on the real CT (reference configuration, fold 1)."""
    import torch
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from grafsgat import config
    from grafsgat.data.features import preprocessed_path
    pid = pid or ATT_PATIENT
    seq = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#6da7ec', '#256abf', '#0d366b'])
    config.set_mode('MATEUS_CV'); config.set_task('LR'); config.set_fold(1)
    g = torch.load(config.graph_dir('LR', 'table_s3', 'radiomic_star') / f'{pid}.pt', weights_only=False)
    att = pd.read_csv(config.run_dir('LR') / 'attention' / 'table_s3__radiomic_star__img__s42_test.csv')
    config.set_fold(None)
    a = att[att.patient_id == pid].set_index('node')['attention']
    prob, label = float(att[att.patient_id == pid].prob.iloc[0]), int(att[att.patient_id == pid].label.iloc[0])
    with np.load(preprocessed_path(pid), allow_pickle=False) as d:
        ct, gtv, sv = d['ct_array'], d['gtv_array'], d['supervoxel_labels']
    pos = g.pos.numpy()
    z = int(np.clip(round(float(pos[0, 0])), 0, ct.shape[0] - 1))
    norm = Normalize(a.min(), a.max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={'width_ratios': [1, 1.12]})
    ax1.imshow(np.clip(ct[z], -160, 240), cmap='gray', interpolation='bilinear')
    for node, sv_id in enumerate(g.sv_ids.numpy()):
        if node == 0:
            continue
        m = sv[z] == sv_id
        if m.any():
            ax1.contourf(m, levels=[0.5, 1.5], colors=[seq(norm(a.get(node, a.mean())))], alpha=0.7)
            ax1.contour(m, levels=[0.5], colors=['white'], linewidths=0.35)
    ax1.contourf(gtv[z], levels=[0.5, 1.5], colors=[CAT[1]], alpha=0.45)
    ax1.contour(gtv[z], levels=[0.5], colors=[CAT[1]], linewidths=1.2)
    for node in range(1, len(pos)):
        ax1.plot([pos[0, 2], pos[node, 2]], [pos[0, 1], pos[node, 1]], color='white', lw=0.4, alpha=0.6)
        ax1.scatter(pos[node, 2], pos[node, 1], s=12, color=seq(norm(a.get(node, a.mean()))), ec='white', lw=0.4, zorder=3)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    ax1.set_xlim(-0.5, ct.shape[2] - 0.5); ax1.set_ylim(ct.shape[1] - 0.5, -0.5)
    ax1.set_title('Axial CT with GTV and attention-shaded supervoxels', fontsize=7.8)

    d3 = np.linalg.norm(pos - pos[0], axis=1)
    ang = np.arctan2(-(pos[:, 1] - pos[0, 1]), pos[:, 2] - pos[0, 2])
    xy = np.c_[d3 * np.cos(ang), d3 * np.sin(ang)]; xy[0] = 0
    r = d3.max()
    for ring in np.linspace(r / 3, r, 3):
        ax2.add_patch(plt.Circle((0, 0), ring, fill=False, color=NEUTRAL, lw=0.6, ls=':'))
        ax2.text(0, ring, f'{ring:.0f} mm', fontsize=5.8, color=INK2, ha='center', va='bottom')
    for node in range(1, len(pos)):
        w = norm(a.get(node, a.mean()))
        ax2.plot(*xy[[0, node]].T, color=seq(w), lw=0.6 + 3.5 * w, zorder=2)
        ax2.scatter(*xy[node], s=30, color=seq(w), ec=INK, lw=0.4, zorder=3)
    ax2.add_patch(plt.Circle((0, 0), r * 0.07, color=CAT[1], ec=INK, lw=0.6, zorder=4))
    ax2.text(r * 0.1, -r * 0.1, 'GTV', fontsize=7, weight='bold', va='top')
    ax2.set_aspect('equal'); ax2.set_xlim(-1.1 * r, 1.1 * r); ax2.set_ylim(-1.1 * r, 1.1 * r)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    for s_ in ax2.spines.values():
        s_.set_visible(False)
    ax2.set_title('Patient graph (radius = distance from GTV)', fontsize=7.8)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=seq), ax=ax2, fraction=0.045, pad=0.02)
    cb.set_label('attention to GTV read-out', fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    for ax, letter in ((ax1, 'a'), (ax2, 'b')):
        ax.text(0.0, 1.06, letter, transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    fig.tight_layout(pad=0.3, w_pad=0.5)
    save(fig, 'fig_attention_ct')
    config.set_mode('MATEUS_CV')
    num('attPid', pid); num('attProb', f'{prob:.2f}')
    num('attMin', f'{a.min():.3f}'); num('attMax', f'{a.max():.3f}'); num('attUniform', f'{1 / (len(a) + 1):.3f}')


def crop_user_architecture():
    """Author's architecture figure: drop the captions baked into the image (LaTeX caption replaces them)."""
    from PIL import Image
    src = MS / 'figures' / 'source' / 'Radiomics GRaph Attention Network Architecture.png'
    if src.exists():
        im = Image.open(src).convert('RGB')
        w, h = im.size
        im.crop((0, 0, w, int(h * 0.918))).save(FIG / 'fig_architecture_author.png')


# ═════════════════════════════════════════════════════════════════════════════
# Sources of variation in external AUC (grafs-gat/tools/variance_components.py)
# ═════════════════════════════════════════════════════════════════════════════
# Independent retraining of the Mateus et al. CNN/ANN with three seeds (11, 22, 33), 5-fold CV, test = MAASTRO;
# mean test AUC over the folds that saved a checkpoint (number of folds in brackets). Source: reproduction
# repository hash123shaikh/reproducibility-hnc-cnn, results/statistical_reproducibility/cv_reports.
CNN_RETRAIN = {
    ('LR', 'CNN, Img'): [(0.58, 4), (0.51, 3), (0.55, 3)], ('LR', 'CNN, Img+Clin'): [(None, 0), (None, 0), (0.53, 3)],
    ('LR', 'ANN, Clin'): [(0.42, 5), (0.42, 5), (None, 0)],
    ('DM', 'CNN, Img'): [(0.80, 3), (0.79, 4), (0.76, 4)], ('DM', 'CNN, Img+Clin'): [(0.89, 3), (0.87, 2), (0.88, 2)],
    ('DM', 'ANN, Clin'): [(0.87, 2), (0.89, 2), (0.87, 4)],
    ('OS', 'CNN, Img'): [(0.65, 4), (0.64, 2), (0.62, 2)], ('OS', 'CNN, Img+Clin'): [(0.69, 3), (0.68, 2), (0.68, 3)],
    ('OS', 'ANN, Clin'): [(0.66, 2), (0.63, 3), (None, 0)],
}
CNN_REPORTED_CV = {('LR', 'CNN, Img'): 0.53, ('LR', 'CNN, Img+Clin'): 0.57, ('LR', 'ANN, Clin'): 0.53,
                   ('DM', 'CNN, Img'): 0.83, ('DM', 'CNN, Img+Clin'): 0.88, ('DM', 'ANN, Clin'): 0.86,
                   ('OS', 'CNN, Img'): 0.63, ('OS', 'CNN, Img+Clin'): 0.68, ('OS', 'ANN, Clin'): 0.63}


def table_cnn_retrain():
    lines = [r'\begin{tabular}{@{}lllllll@{}}', r'\toprule',
             r'Outcome & Model & Reported & Seed 11 & Seed 22 & Seed 33 & Spread \\', r'\midrule']
    for t in TASKS:
        first = True
        for m in ('CNN, Img', 'CNN, Img+Clin', 'ANN, Clin'):
            v = CNN_RETRAIN[(t, m)]
            cells = [f'{a:.2f} ({n}/5)' if a is not None else '--' for a, n in v]
            vals = [a for a, _ in v if a is not None]
            spread = f'{max(vals) - min(vals):.2f}' if len(vals) > 1 else '--'
            lines.append(f'{TASK_LONG[t] if first else ""} & {m} & {CNN_REPORTED_CV[(t, m)]:.2f} & '
                         + ' & '.join(cells) + f' & {spread} \\\\')
            first = False
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'tabS_cnn_retrain.tex').write_text('\n'.join(lines) + '\n')
    dm = [a for a, _ in CNN_RETRAIN[('DM', 'CNN, Img')]]
    num('cnnRetrainDMImgMin', f'{min(dm):.2f}'); num('cnnRetrainDMImgMax', f'{max(dm):.2f}')
    dm = [a for a, _ in CNN_RETRAIN[('DM', 'ANN, Clin')]]
    num('annRetrainDMMin', f'{min(dm):.2f}'); num('annRetrainDMMax', f'{max(dm):.2f}')
    dm = [a for a, _ in CNN_RETRAIN[('DM', 'CNN, Img+Clin')]]
    num('cnnRetrainDMClinMin', f'{min(dm):.2f}'); num('cnnRetrainDMClinMax', f'{max(dm):.2f}')


def fig_variance():
    f = RES / 'variance_components.csv'
    if not f.exists():
        print('  [variance] no variance_components.csv, skipped')
        return
    V = pd.read_csv(f)
    V = V[V.rule == PRIMARY].set_index(['task', 'design'])
    S = pd.read_csv(RES / 'single_model_auc.csv')
    S = S[S.rule == PRIMARY]
    E = pd.read_csv(RES / 'seed_ensemble_gain.csv')
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.78, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    comps = [('selector', 'Feature selector', CAT[0]), ('topology', 'Topology', CAT[2]),
             ('inputs', 'Clinical variables', CAT[3]), ('config_residual', 'Selector × topology × inputs', CAT[6]),
             ('between_fold', 'CV fold (training patients)', CAT[4]), ('between_seed', 'Random seed', CAT[7])]
    rows = [(t, d) for t in TASKS for d in DESIGN]
    for i, key in enumerate(rows):
        left = 0
        for c, lab, col in comps:
            w = max(float(V.loc[key, c]), 0) if np.isfinite(V.loc[key, c]) else 0
            ax.barh(i, w, left=left, color=col, height=0.62, label=lab if i == 0 else None, edgecolor='white', lw=0.5)
            if w >= 0.06:
                ax.text(left + w / 2, i, f'{w:.0%}', ha='center', va='center', fontsize=6.5,
                        color='white' if c in ('between_seed', 'selector', 'config_residual') else INK)
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{t}, {DESIGN[d]}' for t, d in rows])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_xlabel('Share of the variance of single-model external AUC')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.27), fontsize=6.8, handlelength=1.2)
    ax.set_title('a   Sources of variation', loc='left')
    groups = [(d, i) for d in DESIGN for i in ('imaging', 'imaging+clinical')]
    for j, t in enumerate(TASKS):
        a2 = fig.add_subplot(gs[1, j], sharey=None if j == 0 else fig.axes[1])
        data = [S[(S.task == t) & (S.design == d) & (S.inputs == i)].test_auc.values for d, i in groups]
        bp = a2.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
        for k, b in enumerate(bp['boxes']):
            b.set_facecolor(CAT[0] if groups[k][0] == 'MATEUS_CV' else CAT[1]); b.set_alpha(0.35); b.set_edgecolor(INK2)
        rng = np.random.default_rng(0)
        for k, x in enumerate(data):
            a2.scatter(k + 1 + rng.uniform(-0.18, 0.18, len(x)), x, s=1.5, color=INK2, alpha=0.25, lw=0)
            ref = MATEUS_REPORTED[(groups[k][0], groups[k][1])][t]
            a2.scatter(k + 1, float(ref.split()[0]), marker='D', s=22, color=CAT[7], zorder=5,
                       label='Mateus et al. CNN (reported)' if k == 0 and j == 0 else None)
        a2.axhline(0.5, color=NEUTRAL, ls='--', lw=0.8)
        a2.set_xticks(range(1, 5)); a2.set_xticklabels(['CV\nImg', 'CV\nImg+Clin', 'Coh.\nImg', 'Coh.\nImg+Clin'], fontsize=6.5)
        a2.set_ylim(0.15, 0.97); a2.set_title(('b   ' if j == 0 else '') + t, loc='left' if j == 0 else 'center')
        if j == 0:
            a2.set_ylabel('External AUC of single networks')
            a2.legend(loc='lower left', fontsize=6.5, handletextpad=0.2)
        else:
            plt.setp(a2.get_yticklabels(), visible=False)
    save(fig, 'fig_variance')
    # numbers for the text
    num('vcSeedMin', f'{V.between_seed.min() * 100:.0f}'); num('vcSeedMax', f'{V.between_seed.max() * 100:.0f}')
    num('vcSelMin', f'{V.selector.min() * 100:.0f}'); num('vcSelMax', f'{V.selector.max() * 100:.0f}')
    num('vcTopoMax', f'{V.topology.max() * 100:.0f}')
    num('vcCfgMin', f'{V.between_config.min() * 100:.0f}'); num('vcCfgMax', f'{V.between_config.max() * 100:.0f}')
    num('vcFoldMax', f'{V.between_fold.max() * 100:.0f}')
    num('vcNmodels', f'{int(V.n_models.sum()):,}'.replace(',', '{,}'))
    for t in TASKS:
        x = S[S.task == t].test_auc
        num(f'single{t}Min', f'{x.min():.2f}'); num(f'single{t}Max', f'{x.max():.2f}')
        sd = E[E.task == t].sd.median()
        num(f'seedSd{t}', f'{sd:.2f}')
        num(f'ensGain{t}', f'{E[E.task == t].gain.mean():.2f}')
    num('ensGainAll', f'{E.gain.mean():.2f}'); num('ensGainPos', f'{(E.gain > 0).mean() * 100:.0f}')
    dm = S[(S.task == 'DM') & (S.design == 'MATEUS_CV')]
    num('singleDMgeCNN', f'{(dm.test_auc >= 0.83).mean() * 100:.1f}')
    r = V.between_seed / V.between_config
    num('vcRatioMin', f'{r.min():.1f}'); num('vcRatioMax', f'{r.max():.1f}')
    pct = []
    for t in ('LR', 'OS'):
        for d in DESIGN:
            for i in ('imaging', 'imaging+clinical'):
                x = S[(S.task == t) & (S.design == d) & (S.inputs == i)].test_auc
                pct.append(100 * (x < float(MATEUS_REPORTED[(d, i)][t].split()[0])).mean())
    num('cnnPctMin', f'{min(pct):.0f}'); num('cnnPctMax', f'{max(pct):.0f}')


def table_mateus_style():
    """Supplementary: GRAFS-GAT reported exactly as Mateus et al. report their CNN (exact checkpoint rule, one seed,
    CV mean over the folds that saved a checkpoint), for each of the three seeds."""
    import json
    f = RES / 'mateus_style_report.csv'
    if not f.exists():
        print('  [mateus-style] no mateus_style_report.csv, skipped')
        return
    R = pd.read_csv(f)
    seeds = sorted(R.seed.unique())
    lines = [r'\begin{tabular}{@{}llll' + 'c' * len(seeds) + r'c@{}}', r'\toprule',
             r'Outcome & Design & Inputs & Model & ' + ' & '.join(f'Seed {s}' for s in seeds) + r' & Spread \\',
             r'\midrule']
    spreads = []
    for t in TASKS:
        first = True
        for d in DESIGN:
            for inp in ('imaging', 'imaging+clinical'):
                for role in ('chosen', 'reference'):
                    r = R[(R.task == t) & (R.design == d) & (R.inputs == inp) & (R.role == role)].sort_values('seed')
                    if r.empty:
                        continue
                    cells = []
                    for _, x in r.iterrows():
                        if d == 'MATEUS_CV':
                            cells.append(f'{x.auc_saved_mean:.2f} ({x.auc_saved_min:.2f}--{x.auc_saved_max:.2f}) '
                                         f'[{int(x.n_saved)}/5]')
                        else:
                            cells.append(f'{x.auc_saved_mean:.2f} [{x.ci_low:.2f}, {x.ci_high:.2f}]'
                                         + ('' if x.n_saved else r'$^{\dagger}$'))
                    sp = r.auc_saved_mean.max() - r.auc_saved_mean.min()
                    spreads.append((sp, t, d, inp, role, r.auc_saved_mean.min(), r.auc_saved_mean.max()))
                    lines.append(f'{t if first else ""} & {DESIGN[d]} & {"Img" if inp == "imaging" else "Img+Clin"} & '
                                 f'{role} & ' + ' & '.join(cells) + f' & {sp:.2f} \\\\')
                    first = False
        lines.append(r'\midrule')
    lines[-1] = r'\botrule'
    lines.append(r'\end{tabular}')
    (TAB / 'tabS_mateus_style.tex').write_text('\n'.join(lines) + '\n')
    sp = sorted(spreads)
    num('msSpreadMed', f'{np.median([x[0] for x in sp]):.2f}')
    num('msSpreadMax', f'{sp[-1][0]:.2f}')
    num('msSpreadMaxLo', f'{sp[-1][5]:.2f}'); num('msSpreadMaxHi', f'{sp[-1][6]:.2f}')
    num('msSpreadMaxWhat', f'{sp[-1][1]}, {DESIGN[sp[-1][2]].lower()}, '
        f'{"imaging" if sp[-1][3] == "imaging" else "imaging and clinical"} inputs, {sp[-1][4]} configuration')
    # how often the exact rule saved a checkpoint at all (all single networks)
    root = OUT / 'runs'
    for t in TASKS:
        hs = [json.loads(h.read_text()) for h in root.glob(f'MATEUS_*_{t}_mateus/**/histories/*__s*.json')]
        if hs:
            num(f'msSaved{t}', f'{100 * np.mean([not h.get("mateus_fallback_last_epoch", True) for h in hs]):.0f}')
    V = pd.read_csv(RES / 'variance_components.csv')
    V = V[V.rule == 'mateus']
    if len(V):
        num('vcSeedMtMin', f'{V.between_seed.min() * 100:.0f}'); num('vcSeedMtMax', f'{V.between_seed.max() * 100:.0f}')
        num('vcCfgMtMax', f'{V.between_config.max() * 100:.0f}')


def remove_unused_figures():
    """Delete intermediate / unused figure files: keep only what main.tex and supplementary.tex include."""
    import re
    tex = (MS / 'main.tex').read_text() + (MS / 'supplementary.tex').read_text()
    used = {Path(g).stem for g in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)}
    for f in FIG.iterdir():
        if f.is_file() and f.stem not in used:
            f.unlink()


if __name__ == '__main__':
    copy_existing(); crop_titles(); fig_overview()
    fig_stability(); fig_stab_vs_auc(); fig_val_vs_test()
    fig_heatmap(); fig_gat_vs_rf(); fig_roc_all(); crop_user_architecture(); fig_graphfs_workflow(); fig_selection_frequency(); fig_topologies_ct(); fig_attention_ct()
    table_cohort(); table_main(); table_mateus(); table_stability()
    table_method_topology(); table_epoch_rule(); table_gtv_only(); table_cnn_retrain(); fig_variance(); table_mateus_style(); table_missing_images(); table_stability_full()
    extra_numbers()
    write_numbers()
    remove_unused_figures()
    print(f'figures -> {FIG}\ntables  -> {TAB} ({len(numbers)} numbers)')
