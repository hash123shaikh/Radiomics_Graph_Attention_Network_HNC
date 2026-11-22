"""
Every table, figure and quoted number of the whole-tumour population-graph manuscript, generated from
the saved results (no retraining).

    ~/anaconda3/envs/GNNvenv/bin/python Paper/manuscript_popgraph/scripts/make_assets.py

Reads   grafs-gat/outputs/popgraph/results/*  (run_popgnn.py, graph_vs_mlp.py, seed_robustness.py)
        grafs-gat/outputs/gtv_only/results/classical_total_*  (run_classical.py, CLS_FEAT=total)
        grafs-gat/outputs/popgraph/tumour_burden.csv
Writes  ../figures/*.pdf|png, ../tables/*.tex, ../tables/numbers.tex
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MS = HERE.parent
PROJ = MS.parent.parent / 'grafs-gat'
OUT = Path(os.environ.get('GRAFS_OUTPUT_DIR', PROJ / 'outputs'))
PG = OUT / 'popgraph' / 'results'
CL = OUT / 'gtv_only' / 'results'
FIG, TAB = MS / 'figures', MS / 'tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'popgraph'))

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#e34948']
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#c9c8c2'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': NEUTRAL, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'legend.fontsize': 7.5, 'font.size': 8.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
})
TASKS = ['LR', 'DM', 'OS']
DESIGNS = ['MATEUS_CV', 'MATEUS_COHORT']
INPUTS = ['imaging', 'imaging+clinical']
DNAME = {'MATEUS_CV': '5-fold CV', 'MATEUS_COHORT': 'Cohort split'}
INAME = {'imaging': 'Imaging', 'imaging+clinical': 'Imaging + clinical', 'clinical': 'Clinical'}
TLONG = {'LR': 'Locoregional recurrence', 'DM': 'Distant metastasis', 'OS': 'Overall survival'}
MODEL = {'gcn': 'GCN', 'sage': 'GraphSAGE', 'gat': 'GAT', 'mlp': 'MLP'}
FS = {'mrmr': 'mRMR', 'lasso': 'LASSO', 'ucorr': 'univariate'}
FSET = {'orig+shape': 'original + shape', 'full': 'all (incl.\\ LoG)', 'orig': 'original'}
REPS = [('popgnn', 'Primary GTV', 'T, N, volume'), ('popgnn_burden', 'Primary GTV', 'T, N, burden'),
        ('popgnn_total', 'Primary + nodal GTV', 'T, N, volume'),
        ('popgnn_burden_total', 'Primary + nodal GTV', 'T, N, burden')]
MAIN = 'popgnn_burden_total'
NUM = {}


def num(k, v):
    NUM[k] = v


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def tex_table(name, body):
    (TAB / f'{name}.tex').write_text(body)


def fam(prefix):
    f = PG / f'{prefix}_family_selection.csv'
    return pd.read_csv(f) if f.exists() else None


def fmt(r, d):
    if d == 'MATEUS_CV':
        return f"{r.ours:.2f} ({r.fold_min:.2f}--{r.fold_max:.2f})"
    return f"{r.ours:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"


def cap(s):
    return s[0].upper() + s[1:]


def key(t, d, i):
    return f"{t}{'CV' if d == 'MATEUS_CV' else 'Coh'}{'Img' if i == 'imaging' else 'Clin'}"


# ── classical baseline on the same whole-tumour features ────────────────────────────────────────
def classical_chosen():
    from grafsgat.utils import bootstrap_auc_ci
    rows = []
    for t in TASKS:
        for d in DESIGNS:
            f = CL / f'classical_total_{t}_{d}.csv'
            if not f.exists():
                return None
            S = pd.read_csv(f)
            P = pd.read_csv(CL / f'classical_total_pred_{t}_{d}.csv.gz')
            for i in INPUTS + ['clinical']:
                b = S[S.inputs == i].sort_values('val_auc', ascending=False).iloc[0]
                te = P[(P.config == b.config) & (P.split == 'test')]
                fa = [__import__('sklearn.metrics', fromlist=['x']).roc_auc_score(x.label, x.prob)
                      for _, x in te.groupby('fold')]
                pooled = te.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
                lo, hi = bootstrap_auc_ci(pooled.label.values, pooled.prob.values)
                rows.append(dict(task=t, design=d, inputs=i, config=b.config, val_auc=b.val_auc,
                                 ours=b.test_auc_fold_mean if d == 'MATEUS_CV' else b.test_auc,
                                 fold_min=min(fa), fold_max=max(fa), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


# ── Table 1: tumour burden by cohort ───────────────────────────────────────────────────────────
def table_burden():
    B = pd.read_csv(OUT / 'popgraph' / 'tumour_burden.csv', index_col=0)
    B['cohort'] = np.where(B.index.str.startswith('HN1'), 'MAASTRO', 'Canadian')
    rows = []
    for c in ('Canadian', 'MAASTRO'):
        x = B[B.cohort == c]
        q = lambda v: f"{np.median(v):.1f} ({np.percentile(v, 25):.1f}--{np.percentile(v, 75):.1f})"
        rows.append([f'{len(x)}', f'{(x.n_nodal_gtv > 0).sum()} ({100 * (x.n_nodal_gtv > 0).mean():.0f}\\%)',
                     q(x.vol_primary / 1000), q(x.vol_nodal[x.n_nodal_gtv > 0] / 1000) if (x.n_nodal_gtv > 0).any() else '--',
                     q(x.vol_total / 1000), f"{100 * np.median(1 - x.vol_primary / x.vol_total):.0f}\\%",
                     f"{100 * (1 - x.vol_primary / x.vol_total)[x.n_nodal_gtv > 0].median():.0f}\\%"])
        num(f'nodalPct{c[:3]}', f'{100 * (x.n_nodal_gtv > 0).mean():.0f}')
        num(f'nodalFracMed{c[:3]}', f"{100 * (1 - x.vol_primary / x.vol_total)[x.n_nodal_gtv > 0].median():.0f}")
        num(f'nPat{c[:3]}', f'{len(x)}')
    lab = ['Patients with delineations', 'Patients with nodal GTV, n (\\%)', 'Primary GTV, cm$^3$',
           'Nodal GTV (if present), cm$^3$', 'Total GTV, cm$^3$', 'Nodal share of total GTV, median',
           'Nodal share when nodes present, median']
    body = ['\\begin{tabular}{@{}lll@{}}', '\\toprule',
            'Tumour burden & Canadian centres & MAASTRO (HN1) \\\\', '\\midrule']
    for j, l in enumerate(lab):
        body.append(f'{l} & {rows[0][j]} & {rows[1][j]} \\\\')
    body += ['\\botrule', '\\end{tabular}']
    tex_table('table_burden', '\n'.join(body) + '\n')


# ── Table 2: main comparison ───────────────────────────────────────────────────────────────────
def table_main(F, C):
    body = ['\\begin{tabular}{@{}lllllll@{}}', '\\toprule',
            'Endpoint & Design & Inputs & CNN$^{a}$ & Population GNN$^{b}$ & MLP, no graph$^{b}$ & Logistic/RF$^{c}$ \\\\',
            '\\midrule']
    wins = {'graph': 0, 'mlp': 0, 'cls': 0}
    for t in TASKS:
        first_t = True
        for d in DESIGNS:
            first_d = True
            for i in INPUTS:
                g = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == 'graph')].iloc[0]
                m = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == 'mlp')].iloc[0]
                ref = g.mateus
                cells = []
                for name, r in (('graph', g), ('mlp', m)):
                    s = fmt(r, d)
                    if r.ours > ref:
                        s = '\\textbf{' + s.split(' ')[0] + '} ' + ' '.join(s.split(' ')[1:])
                        wins[name] += 1
                    cells.append(s)
                if C is not None:
                    c = C[(C.task == t) & (C.design == d) & (C.inputs == i)].iloc[0]
                    num('cls' + key(t, d, i), f'{c.ours:.2f}')
                    s = fmt(c, d)
                    if c.ours > ref:
                        s = '\\textbf{' + s.split(' ')[0] + '} ' + ' '.join(s.split(' ')[1:])
                        wins['cls'] += 1
                    cells.append(s)
                else:
                    cells.append('--')
                body.append(f"{TLONG[t] if first_t else ''} & {DNAME[d] if first_d else ''} & {INAME[i]} & "
                            f"{ref:.2f} & {g.model and MODEL[g.model]} {cells[0]} & {cells[1]} & {cells[2]} \\\\")
                first_t = first_d = False
                num('main' + key(t, d, i), f'{g.ours:.2f}')
                num('mlp' + key(t, d, i), f'{m.ours:.2f}')
                num('cnn' + key(t, d, i), f'{ref:.2f}')
                if d == 'MATEUS_COHORT':
                    num('mainCI' + key(t, d, i), f'{g.ci_lo:.2f}--{g.ci_hi:.2f}')
        if t != TASKS[-1]:
            body.append('\\midrule')
    body += ['\\botrule', '\\end{tabular}']
    tex_table('table_main', '\n'.join(body) + '\n')
    num('winsGraph', str(wins['graph']))
    num('winsMLP', str(wins['mlp']))
    num('winsCls', str(wins['cls']))
    num('clsAvail', '1' if C is not None else '0')
    for t in TASKS:
        x = F[(F.task == t) & (F.family == 'graph')].iloc[0]
        num(f'nTest{t}', str(int(x.test_n)))
        num(f'eTest{t}', str(int(x.test_events)))
        if C is not None:
            c = C[(C.task == t) & (C.inputs != 'clinical')].ours
            num(f'cls{t}Min', f'{c.min():.2f}')
            num(f'cls{t}Max', f'{c.max():.2f}')
            cc = C[(C.task == t) & (C.inputs == 'clinical')].ours
            num(f'clsClin{t}Min', f'{cc.min():.2f}')
            num(f'clsClin{t}Max', f'{cc.max():.2f}')
        g = F[(F.task == t) & (F.family == 'graph') & (F.inputs != 'clinical')].ours
        num(f'main{t}Min', f'{g.min():.2f}')
        num(f'main{t}Max', f'{g.max():.2f}')
        cl = F[(F.task == t) & (F.family == 'graph') & (F.inputs == 'clinical')].ours
        num(f'clinOnly{t}Min', f'{cl.min():.2f}')
        num(f'clinOnly{t}Max', f'{cl.max():.2f}')


# ── Table 3: 2x2 representation ablation (graph family) ────────────────────────────────────────
def table_ablation():
    body = ['\\begin{tabular}{@{}llcccccccccccc@{}}', '\\toprule',
            ' & & \\multicolumn{4}{c}{LR} & \\multicolumn{4}{c}{DM} & \\multicolumn{4}{c}{OS} & \\\\',
            '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}',
            ' & & \\multicolumn{2}{c}{CV} & \\multicolumn{2}{c}{Cohort} & \\multicolumn{2}{c}{CV} & '
            '\\multicolumn{2}{c}{Cohort} & \\multicolumn{2}{c}{CV} & \\multicolumn{2}{c}{Cohort} & \\\\',
            'Tumour region & Clinical & Img & +Cl & Img & +Cl & Img & +Cl & Img & +Cl & Img & +Cl & Img & +Cl & Wins \\\\',
            '\\midrule']
    body[0] = '\\begin{tabular}{@{}llccccccccccccc@{}}'
    ref = []
    for t in TASKS:
        for d in DESIGNS:
            for i in INPUTS:
                ref.append(fam(MAIN).query('task==@t and design==@d and inputs==@i and family=="graph"').mateus.iloc[0])
    body.append('CNN (Mateus et al.) & -- & ' + ' & '.join(f'{r:.2f}' for r in ref) + ' & -- \\\\')
    body.append('\\midrule')
    abl = []
    for prefix, region, clin in REPS:
        F = fam(prefix)
        if F is None:
            continue
        vals, w = [], 0
        for t in TASKS:
            for d in DESIGNS:
                for i in INPUTS:
                    r = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == 'graph')].iloc[0]
                    vals.append(f'\\textbf{{{r.ours:.2f}}}' if r.ours > r.mateus else f'{r.ours:.2f}')
                    w += int(r.ours > r.mateus)
                    abl.append(dict(prefix=prefix, region=region, clin=clin, task=t, design=d, inputs=i,
                                    ours=r.ours, mateus=r.mateus))
        body.append(f'{region} & {clin} & ' + ' & '.join(vals) + f' & {w}/12 \\\\')
        num('wins' + {'popgnn': 'PrimVol', 'popgnn_burden': 'PrimBur', 'popgnn_total': 'TotVol',
                      'popgnn_burden_total': 'TotBur'}[prefix], str(w))
    body += ['\\botrule', '\\end{tabular}']
    tex_table('table_ablation', '\n'.join(body) + '\n')
    A = pd.DataFrame(abl)
    for t in TASKS:
        for pre, nm in (('popgnn', 'PrimVol'), ('popgnn_total', 'TotVol'), ('popgnn_burden_total', 'TotBur'),
                        ('popgnn_burden', 'PrimBur')):
            x = A[(A.prefix == pre) & (A.task == t)]
            if len(x):
                num(f'abl{nm}{t}Min', f'{x.ours.min():.2f}')
                num(f'abl{nm}{t}Max', f'{x.ours.max():.2f}')
    return A


# ── Table 4: graph versus MLP ──────────────────────────────────────────────────────────────────
def table_paired():
    P = pd.read_csv(PG / f'{MAIN}_graph_vs_mlp_paired.csv')
    M = pd.read_csv(PG / f'{MAIN}_graph_vs_mlp_matched.csv')
    body = ['\\begin{tabular}{@{}llllllll@{}}', '\\toprule',
            'Endpoint & Design & Inputs & Graph model & Graph AUC & MLP AUC & $\\Delta$ AUC [95\\% CI] & $p$ \\\\',
            '\\midrule']
    for t in TASKS:
        first_t = True
        for d in DESIGNS:
            first_d = True
            for i in INPUTS + ['clinical']:
                r = P[(P.task == t) & (P.design == d) & (P.inputs == i)].iloc[0]
                body.append(f"{TLONG[t] if first_t else ''} & {DNAME[d] if first_d else ''} & {INAME[i]} & "
                            f"{MODEL[r.graph_config.split('|')[4]]} & {r.graph_test:.3f} & {r.mlp_test:.3f} & "
                            f"{r.delta_test:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}] & {r.p_boot:.2f} \\\\")
                first_t = first_d = False
        if t != TASKS[-1]:
            body.append('\\midrule')
    body += ['\\botrule', '\\end{tabular}']
    tex_table('table_paired', '\n'.join(body) + '\n')
    num('pairedMinP', f'{P.p_boot.min():.2f}')
    num('pairedDeltaMin', f'{P.delta_test.min():+.3f}')
    num('pairedDeltaMax', f'{P.delta_test.max():+.3f}')
    num('pairedNsig', str(int((P.p_boot < 0.05).sum())))
    num('pairedN', str(len(P)))
    for t in TASKS:
        x = M[M.task == t]
        num(f'matchedValFrac{t}', f'{100 * x.val_frac_graph_better.mean():.0f}')
        num(f'matchedTestFrac{t}', f'{100 * x.test_frac_graph_better.mean():.0f}')
        num(f'matchedTestDelta{t}', f'{x.test_delta_median.mean():+.3f}')
    return P, M


# ── Supplementary: chosen configurations ───────────────────────────────────────────────────────
CLF = {'lr_C0.03': 'logistic ($C$=0.03)', 'lr_C0.3': 'logistic ($C$=0.3)', 'lr_C3': 'logistic ($C$=3)',
       'rf_d3': 'RF (depth 3)', 'rf_d6': 'RF (depth 6)'}


def table_configs(F, C):
    body = ['\\begin{tabular}{@{}lllllllll@{}}', '\\toprule',
            'Endpoint & Design & Inputs & Family & Features & Selector & $k$ & Graph & Val.\\ AUC \\\\',
            '\\midrule']
    for t in TASKS:
        for d in DESIGNS:
            for i in INPUTS + ['clinical']:
                for fm in ('graph', 'mlp'):
                    r = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == fm)].iloc[0]
                    f = r.config.split('|')
                    body.append(f"{t} & {DNAME[d]} & {INAME[i]} & {MODEL[f[4]]} & {FSET.get(f[0], '--')} & "
                                f"{FS.get(f[1], '--')} & {f[2] if f[2] != '0' else '--'} & "
                                f"{'--' if f[4] == 'mlp' else f[3][1:] + '-NN'} & {r.val_auc:.3f} \\\\")
                if C is not None and i != 'clinical':
                    c = C[(C.task == t) & (C.design == d) & (C.inputs == i)].iloc[0]
                    f = c.config.split('|')
                    if len(f) > 3:
                        body.append(f"{t} & {DNAME[d]} & {INAME[i]} & {CLF.get(f[3], f[3])} & "
                                    f"{FSET.get(f[0], f[0])} & {FS.get(f[1], f[1])} & {f[2].lstrip('k')} & -- & "
                                    f"{c.val_auc:.3f} \\\\")
                    else:
                        body.append(f"{t} & {DNAME[d]} & {INAME[i]} & logistic & {f[0].replace('+', ' + ')} only & "
                                    f"-- & -- & -- & {c.val_auc:.3f} \\\\")
        if t != TASKS[-1]:
            body.append('\\midrule')
    body += ['\\botrule', '\\end{tabular}']
    tex_table('tabS_configs', '\n'.join(body).replace('_', '\\_') + '\n')


def table_seed():
    f = PG / f'{MAIN}_seed_robustness.csv'
    if not f.exists():
        return None
    O = pd.read_csv(f)
    O = O[O.inputs != 'clinical']
    body = ['\\begin{tabular}{@{}llllllll@{}}', '\\toprule',
            'Endpoint & Design & Inputs & Family & Reported & Mean (SD) over re-runs & Range & Re-runs above CNN \\\\',
            '\\midrule']
    for t in TASKS:
        for d in DESIGNS:
            for i in INPUTS:
                for fm in ('graph', 'mlp'):
                    x = O[(O.task == t) & (O.design == d) & (O.inputs == i) & (O.family == fm)]
                    if not len(x):
                        continue
                    body.append(f"{t} & {DNAME[d]} & {INAME[i]} & {'GNN' if fm == 'graph' else 'MLP'} & "
                                f"{x.reported.iloc[0]:.3f} & {x.ours.mean():.3f} ({x.ours.std():.3f}) & "
                                f"{x.ours.min():.3f}--{x.ours.max():.3f} & "
                                f"{int((x.ours > x.mateus).sum())}/{len(x)} \\\\")
        if t != TASKS[-1]:
            body.append('\\midrule')
    body += ['\\botrule', '\\end{tabular}']
    tex_table('tabS_seed', '\n'.join(body) + '\n')
    sd = O.groupby(['task', 'design', 'inputs', 'family']).ours.std()
    num('seedSDmax', f'{sd.max():.3f}')
    num('seedSDmed', f'{sd.median():.3f}')
    num('seedRepeats', str(O.groupby(['task', 'design', 'config']).repeat.nunique().max()))
    g = O[O.family == 'graph']
    stable = g.groupby(['task', 'design', 'inputs']).apply(lambda x: (x.ours > x.mateus).mean())
    num('seedStableWins', str(int((stable == 1).sum())))
    num('seedStableLoss', str(int((stable == 0).sum())))
    return O


# ── figures ────────────────────────────────────────────────────────────────────────────────────
def fig_auc(F, C):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
    labels = [f'{DNAME[d]}\n{INAME[i]}' for d in DESIGNS for i in INPUTS]
    for ax, t in zip(axes, TASKS):
        for j, (d, i) in enumerate([(d, i) for d in DESIGNS for i in INPUTS]):
            g = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == 'graph')].iloc[0]
            m = F[(F.task == t) & (F.design == d) & (F.inputs == i) & (F.family == 'mlp')].iloc[0]
            lo, hi = (g.fold_min, g.fold_max) if d == 'MATEUS_CV' else (g.ci_lo, g.ci_hi)
            y = len(labels) - 1 - j
            ax.plot([lo, hi], [y + 0.12] * 2, color=CAT[0], lw=1.2)
            ax.scatter(g.ours, y + 0.12, color=CAT[0], s=24, zorder=3, label='Population GNN' if j == 0 else None)
            ax.scatter(m.ours, y - 0.12, color=CAT[2], marker='s', s=18, zorder=3, label='MLP (no graph)' if j == 0 else None)
            if C is not None:
                c = C[(C.task == t) & (C.design == d) & (C.inputs == i)].iloc[0]
                ax.scatter(c.ours, y - 0.32, color=CAT[3], marker='^', s=18, zorder=3,
                           label='Logistic/RF' if j == 0 else None)
            ax.scatter(g.mateus, y + 0.32, color=CAT[6], marker='D', s=22, zorder=3, label='CNN (Mateus et al.)' if j == 0 else None)
        ax.axvline(0.5, color=NEUTRAL, lw=0.8, ls='--')
        ax.set_title(TLONG[t])
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel('External AUC (MAASTRO)')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=7)
    axes[0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.22), ncol=4)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_auc_vs_cnn')


def fig_ablation(A):
    order = [p for p, _, _ in REPS if p in set(A.prefix)]
    lab = {p: f'{r}\n{c}' for p, r, c in REPS}
    cols = [(t, d, i) for t in TASKS for d in DESIGNS for i in INPUTS]
    M = np.array([[A[(A.prefix == p) & (A.task == t) & (A.design == d) & (A.inputs == i)].eval('ours - mateus').iloc[0]
                   for t, d, i in cols] for p in order])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cm = LinearSegmentedColormap.from_list('div', ['#b7322c', '#f3d9d4', '#f7f7f5', '#d3e4f7', '#2a63b0'])
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    im = ax.imshow(M, cmap=cm, norm=TwoSlopeNorm(0, -0.25, 0.25), aspect='auto')
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f'{M[a, b]:+.2f}', ha='center', va='center', fontsize=6.3,
                    color='white' if abs(M[a, b]) > 0.17 else INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab[p] for p in order], fontsize=6.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{'CV' if d == 'MATEUS_CV' else 'Coh'}\n{'Img' if i == 'imaging' else '+Cl'}"
                        for t, d, i in cols], fontsize=6.5)
    for k, t in enumerate(TASKS):
        ax.text(4 * k + 1.5, -0.75, TLONG[t], ha='center', fontsize=7.5, weight='bold')
        if k:
            ax.axvline(4 * k - 0.5, color='white', lw=2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('AUC difference vs CNN', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, 'fig_ablation')


def fig_matched():
    rows = []
    for t in TASKS:
        for d in DESIGNS:
            S = pd.read_csv(PG / f'{MAIN}_{t}_{d}.csv')
            S['fk'] = S.featset + '|' + S.fs + '|' + S.k.astype(str) + '|' + S.inputs
            col = 'test_auc_fold_mean' if d == 'MATEUS_CV' else 'test_auc'
            M = S[S.model == 'mlp'].set_index('fk')[['val_auc', col]]
            G = S[(S.model != 'mlp') & (S.inputs != 'clinical')].join(M, on='fk', rsuffix='_m')
            rows.append(pd.DataFrame({'task': t, 'design': d, 'model': G.model,
                                      'dval': G.val_auc - G.val_auc_m, 'dtest': G[col] - G[col + '_m']}))
    D = pd.concat(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (c, ttl) in zip(axes, (('dval', 'Validation'), ('dtest', 'External test (MAASTRO)'))):
        pos = 0
        ticks, tl = [], []
        for t in TASKS:
            for k, m in enumerate(('gcn', 'sage', 'gat')):
                x = D[(D.task == t) & (D.model == m)][c].dropna().values
                vp = ax.violinplot(x, [pos], widths=0.8, showextrema=False, showmedians=True)
                for b in vp['bodies']:
                    b.set_facecolor(CAT[k]); b.set_alpha(0.55); b.set_edgecolor('none')
                vp['cmedians'].set_color(INK)
                ticks.append(pos); tl.append(f'{MODEL[m]}\n{t}' if k == 1 else MODEL[m])
                pos += 1
            pos += 0.6
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels([MODEL[m] for _ in TASKS for m in ('gcn', 'sage', 'gat')], fontsize=6.3, rotation=90)
        for k, t in enumerate(TASKS):
            ax.text(k * 3.6 + 1, 1.02, t, transform=ax.get_xaxis_transform(), ha='center', fontsize=7.5, weight='bold')
        ax.set_title(ttl, pad=12)
    axes[0].set_ylabel('$\\Delta$AUC, graph $-$ MLP')
    fig.tight_layout()
    save(fig, 'fig_graph_vs_mlp')
    num('matchedN', str(len(D)))
    num('matchedTestFracAll', f'{100 * (D.dtest > 0).mean():.0f}')
    num('matchedValFracAll', f'{100 * (D.dval > 0).mean():.0f}')
    num('matchedTestMedAll', f'{D.dtest.median():+.3f}')


def fig_seed(O):
    if O is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=False)
    for ax, t in zip(axes, TASKS):
        pos, ticks, tl = 0, [], []
        for d in DESIGNS:
            for i in INPUTS:
                for k, fm in enumerate(('graph', 'mlp')):
                    x = O[(O.task == t) & (O.design == d) & (O.inputs == i) & (O.family == fm)]
                    if not len(x):
                        continue
                    ax.scatter(np.full(len(x), pos) + np.random.default_rng(0).uniform(-0.12, 0.12, len(x)), x.ours,
                               s=7, color=CAT[0] if fm == 'graph' else CAT[2], alpha=0.7, zorder=3)
                    ax.scatter(pos, x.reported.iloc[0], marker='_', s=120, color=INK, zorder=4)
                    pos += 1
                ax.plot([pos - 2.4, pos - 0.6], [x.mateus.iloc[0]] * 2, color=CAT[6], lw=1.2)
                ticks.append(pos - 1.5)
                tl.append(f"{'CV' if d == 'MATEUS_CV' else 'Coh'} {'Img' if i == 'imaging' else '+Cl'}")
                pos += 0.7
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl, fontsize=6.5)
        ax.set_title(TLONG[t])
        ax.set_ylabel('External AUC' if t == 'LR' else '')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=CAT[0], label='Population GNN, re-run'),
         Line2D([], [], marker='o', ls='', color=CAT[2], label='MLP, re-run'),
         Line2D([], [], marker='_', ls='', ms=10, color=INK, label='Reported (seeds 0–9)'),
         Line2D([], [], color=CAT[6], label='CNN (Mateus et al.)')]
    fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig_seed')


def fig_ct():
    """Primary versus primary + nodal GTV on CT for one Canadian and one MAASTRO patient."""
    import SimpleITK as sitk
    from total_gtv_features import total_mask
    from tumour_burden import nodal_map
    from grafsgat.data.loader import HNSCCDataLoader, rtstruct_mask_by_coordinates
    import pydicom
    B = pd.read_csv(OUT / 'popgraph' / 'tumour_burden.csv', index_col=0)
    nm = nodal_map()
    picks = []
    for coh, pref in (('HNPET', ~B.index.str.startswith('HN1')), ('MAASTRO', B.index.str.startswith('HN1'))):
        x = B[pref & (B.n_nodal_gtv >= 1)]
        x = x[(x.vol_primary > 10000) & (x.vol_nodal > 5000)]
        x = x.iloc[(x.vol_nodal / x.vol_total - 0.45).abs().argsort()]
        picks.append((coh, x.index[0]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (coh, pid) in zip(axes, picks):
        L = HNSCCDataLoader(coh)
        ct, tot = total_mask(coh, pid, nm.get(pid, []))
        rs = pydicom.dcmread(str(L._find_rtstruct(pid, L._patient_dir(pid))))
        prim_rois = ['GTV-1'] if coh == 'MAASTRO' else list(L.roi_map[str(pid)])
        prim = rtstruct_mask_by_coordinates(rs, prim_rois, ct)
        a = sitk.GetArrayFromImage(ct)
        t = sitk.GetArrayFromImage(tot)
        nod = (t > 0) & (prim == 0)
        both = np.minimum(nod.sum((1, 2)), prim.sum((1, 2)))
        z = int(np.argmax(both)) if both.max() > 0 else int(np.argmax(nod.sum((1, 2))))
        ys, xs = np.where((t[z] > 0))
        cy, cx = int(ys.mean()), int(xs.mean())
        h = 110
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
        ax.imshow(np.clip(a[z][sl], -160, 240), cmap='gray', interpolation='bilinear')
        ax.contourf(prim[z][sl], levels=[0.5, 1.5], colors=[CAT[1]], alpha=0.35)
        ax.contour(prim[z][sl], levels=[0.5], colors=[CAT[1]], linewidths=1.3)
        if nod[z].any():
            ax.contourf(nod[z][sl].astype(float), levels=[0.5, 1.5], colors=[CAT[0]], alpha=0.35)
            ax.contour(nod[z][sl].astype(float), levels=[0.5], colors=[CAT[0]], linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        b = B.loc[pid]
        ax.set_title(f"{'Canadian centre' if coh == 'HNPET' else 'MAASTRO'}: primary {b.vol_primary / 1000:.0f} cm$^3$, "
                     f"nodal {b.vol_nodal / 1000:.0f} cm$^3$", fontsize=7.6)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CAT[1], alpha=0.6, label='Primary GTV (main GRAFS-GAT analysis)'),
                        Patch(color=CAT[0], alpha=0.6, label='Nodal GTV (added in the whole-tumour representation)')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    for ax, letter in zip(axes, 'ab'):
        ax.text(-0.04, 1.02, letter, transform=ax.transAxes, fontsize=9, weight='bold', ha='right')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig_ct_tumour')


def fig_design():
    """Real population graph: chosen DM cohort-split configuration, first two principal components."""
    os.environ['POPGNN_CLIN'], os.environ['POPGNN_FEAT'] = 'burden', 'total'
    import importlib
    import run_popgnn as R
    importlib.reload(R)
    import seed_robustness as SR
    F = fam(MAIN)
    cfg = F[(F.task == 'DM') & (F.design == 'MATEUS_COHORT') & (F.inputs == 'imaging') & (F.family == 'graph')].config.iloc[0]
    (_, split), = R.splits('DM', 'MATEUS_COHORT')
    G = R.feature_table()
    ids = split['train'] + split['val'] + split['test']
    n_tr, n_va = len(split['train']), len(split['val'])
    y = np.array([split['labels'][p] for p in ids], dtype=np.float32)
    X = SR.features(cfg, ids, n_tr, y, G)
    gk = int(cfg.split('|')[3][1:])
    ei = R.knn_edges(X, list(range(n_tr)), gk).numpy()
    Xc = X - X[:n_tr].mean(0)
    _, _, Vt = np.linalg.svd(Xc[:n_tr], full_matrices=False)
    Z = Xc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    show_te = np.arange(n_tr + n_va, len(ids))[np.random.default_rng(1).choice(len(ids) - n_tr - n_va, 3, replace=False)]
    for s_, d_ in ei.T:
        if d_ < n_tr:
            ax.plot(*Z[[s_, d_]].T, color=GRID, lw=0.25, alpha=0.6, zorder=1)
        elif d_ in show_te:
            ax.annotate('', xy=Z[d_], xytext=Z[s_], arrowprops=dict(arrowstyle='->', color=CAT[3], lw=0.5, alpha=0.8, shrinkA=2, shrinkB=4), zorder=2)
    tr = np.arange(n_tr)
    ax.scatter(*Z[tr[y[tr] == 0]].T, s=14, color=CAT[0], ec='white', lw=0.3, zorder=3, label='Training patient, no DM')
    ax.scatter(*Z[tr[y[tr] == 1]].T, s=22, color=CAT[6], ec='white', lw=0.3, zorder=3, label='Training patient, DM')
    ax.scatter(*Z[show_te].T, s=80, marker='*', color=CAT[3], ec=INK, lw=0.5, zorder=4,
               label=f'MAASTRO patient (label unseen)\nwith its {gk} training neighbours')
    ax.set_xlabel('principal component 1'); ax.set_ylabel('principal component 2')
    ax.set_title(f'Population graph for distant metastasis (cohort split, {MODEL[cfg.split("|")[4]]}, '
                 f'{cfg.split("|")[2]} features, {gk}-NN)', fontsize=8.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, 'fig_population_graph')
    num('figGraphK', str(gk))


def write_numbers():
    lines = [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(NUM.items())]
    (TAB / 'numbers.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    F = fam(MAIN)
    C = classical_chosen()
    table_burden()
    table_main(F, C)
    A = table_ablation()
    table_paired()
    table_configs(F, C)
    O = table_seed()
    fig_auc(F, C)
    fig_ablation(A)
    fig_matched()
    fig_seed(O)
    fig_design()
    if '--no-ct' not in sys.argv:
        fig_ct()
    write_numbers()
    print(f'{len(NUM)} numbers; tables: {sorted(p.name for p in TAB.glob("*.tex"))}')
