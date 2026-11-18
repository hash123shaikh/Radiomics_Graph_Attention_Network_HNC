"""Self-contained HTML report of one run (figures embedded) -> outputs/runs/<MODE>_<TASK>/report.html"""

import base64
import html
import json

import pandas as pd

from grafsgat import config
from grafsgat.data.splits import split_summary
from grafsgat.selection.methods import PRETTY

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:15px;margin:20px 0 6px}
p.sub{color:var(--ink2);margin:0 0 16px}.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;font-size:12.5px;width:100%}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--ink2);font-weight:600}
tr.best td{font-weight:700}img{max-width:100%;height:auto;border-radius:6px;background:#fcfcfb}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}
.kpi{display:flex;flex-wrap:wrap;gap:12px}.kpi div{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--ink2);font-size:12px}.note{color:var(--ink2);font-size:12.5px}
"""


def _img(p):
    if not p.exists():
        return ''
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure class="card"><img alt="{html.escape(p.stem)}" src="data:image/png;base64,{b}"></figure>'


def _table(df, best_col=None, floatfmt='{:.3f}'):
    cols = list(df.columns)
    h = '<div class="card"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in cols) + '</tr></thead><tbody>'
    for _, r in df.iterrows():
        cls = ' class="best"' if best_col and r.get(best_col) else ''
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'<td>{floatfmt.format(v) if isinstance(v, float) else html.escape(str(v))}</td>')
        h += f'<tr{cls}>' + ''.join(cells) + '</tr>'
    return h + '</tbody></table></div>'


def build_report(task, df):
    run = config.run_dir(task)
    fig = run / 'figures'
    split = json.loads(config.split_file(task).read_text())
    parts = [f'<h1>GRAFS-GAT: feature selection x graph topology</h1>'
             f'<p class="sub">Mode <b>{config.run_label()}</b> (train {", ".join(config.TRAIN_DATASETS)}; '
             f'test {", ".join(config.TEST_DATASETS) or "internal split"}), task <b>{task}</b> '
             f'(event within {config.horizon_months(task)} months). Every number is computed on the same split; '
             f'feature selection, scalers and encoders are fitted on the training patients only; the '
             f'configuration is chosen on validation AUC and reported once on test.</p>']

    lab = split['labels']
    kpis = ''.join(f'<div><b>{len(split[s])}</b><span>{s} patients, {sum(lab[p] for p in split[s])} events</span></div>'
                   for s in ('train', 'val', 'test'))
    parts.append(f'<div class="kpi">{kpis}</div>')
    parts.append('<h3>Patients per centre</h3>' + _table(pd.DataFrame(split_summary(split))))

    # headline
    parts.append('<h2>1. Which configuration wins?</h2>')
    for clin in (False, True):
        b = df[(df.use_clinical == clin) & df.best_by_val]
        if len(b):
            r = b.iloc[0]
            parts.append(f'<p><b>{r.inputs}</b>: {PRETTY.get(r.method, r.method)} + {r.topology} '
                         f'(val AUC {r.val_auc:.3f}) &rarr; <b>test AUC {r.test_auc:.3f}</b> '
                         f'[{r.test_auc_ci_low:.3f}, {r.test_auc_ci_high:.3f}]</p>')
    cols = ['method_name', 'topology', 'inputs', 'n_features', 'val_auc', 'test_auc', 'test_auc_ci_low',
            'test_auc_ci_high', 'test_auc_seed_sd', 'p_vs_reference_holm', 'p_vs_rf_same_features_holm',
            'boot_kuncheva', 'centre_jaccard', 'best_by_val']
    parts.append(_table(df[[c for c in cols if c in df]].round(4), best_col='best_by_val'))
    parts.append('<p class="note">p values: paired DeLong test on the test set, Holm-corrected within each input set. '
                 'Reference = Table S3 features + radiomic star graph (Bae et al. 2025). '
                 'test_auc_seed_sd = spread of single-seed AUCs (the table uses seed-averaged predictions).</p>')
    for clin in ('img', 'img+clin'):
        parts.append('<div class="grid">' + _img(fig / 'results' / f'auc_heatmap_test_{clin}.png')
                     + _img(fig / 'results' / f'auc_heatmap_val_{clin}.png') + '</div>')
        parts.append('<div class="grid">' + _img(fig / 'results' / f'forest_test_auc_{clin}.png')
                     + _img(fig / 'results' / f'roc_overlay_{clin}.png') + '</div>')

    parts.append('<h2>2. Which feature selection is reproducible?</h2>')
    sel = run / 'selection'
    for f, title in (('stability_bootstrap.csv', 'Within-cohort stability (bootstrap)'),
                     ('stability_centres.csv', 'Cross-centre reproducibility')):
        if (sel / f).exists():
            parts.append(f'<h3>{title}</h3>' + _table(pd.read_csv(sel / f).round(3)))
    parts.append('<div class="grid">' + _img(fig / 'selection' / 'stability_bootstrap.png')
                 + _img(fig / 'selection' / 'stability_centres.png') + '</div>')
    parts.append('<div class="grid">' + _img(fig / 'results' / 'stability_vs_auc_img.png')
                 + _img(fig / 'results' / 'stability_vs_auc_img+clin.png') + '</div>')
    parts.append(_img(fig / 'selection' / 'selection_frequency_bootstrap.png'))

    parts.append('<h2>3. What did each method select?</h2>')
    parts.append('<div class="grid">' + _img(fig / 'selection' / 'selected_features_matrix.png')
                 + _img(fig / 'selection' / 'method_agreement_jaccard.png') + '</div>')
    for m in ('graph_fs_cc', 'graph_fs_spectral', 'graph_fs_hybrid'):
        parts.append(_img(fig / 'selection' / f'feature_graph_{m}.png'))

    parts.append('<h2>4. What does the graph attend to?</h2>')
    parts.append('<p class="note">Test patients from the chosen configuration (and the paper reference). Left: CT slice '
                 'with the graph\'s supervoxels coloured by the attention the GTV read-out pays them. Right: the graph '
                 'itself (radius = distance from the GTV).</p>')
    for p in sorted((fig / 'patient_graphs').glob('*.png')) if (fig / 'patient_graphs').exists() else []:
        parts.append(_img(p))
    for p in sorted((fig / 'training').glob('*.png')) if (fig / 'training').exists() else []:
        parts.append(_img(p))

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>GRAFS-GAT {config.run_label()} {task}</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(parts)}</main></body></html>')
    out = run / 'report.html'
    out.write_text(page)
    print(f"[report] -> {out}")
    return out


def build_cv_report(task, df, per_fold, splits, stab):
    """Cross-fold report of a cross-validation mode -> <mode folder>/report.html (links to fold reports)."""
    run = config.mode_dir(task)
    fig = run / 'figures'
    k = len(splits)
    s1 = next(iter(splits.values()))
    lab = s1['labels']
    n_tv = len(s1['train']) + len(s1['val'])
    parts = [f'<h1>GRAFS-GAT: feature selection x graph topology ({k}-fold CV)</h1>'
             f'<p class="sub">Mode <b>{config.MODE}</b>, task <b>{task}</b> (event within '
             f'{config.horizon_months(task)} months). Design of Mateus et al. (Sci Rep 2023): stratified '
             f'{k}-fold cross-validation on {", ".join(config.TRAIN_DATASETS)} for training/validation, every '
             f'fold&#39;s models tested on the external cohort {", ".join(config.TEST_DATASETS)}. Feature selection, '
             f'scalers and encoders are refitted inside every fold on its training patients only. '
             f'Checkpoint rule: <b>{config.EPOCH_RULE}</b>.</p>']
    kpis = (f'<div><b>{n_tv}</b><span>train+val patients, {sum(lab[p] for p in s1["train"] + s1["val"])} events</span></div>'
            f'<div><b>{k}</b><span>folds, {len(s1["val"])} validation patients each</span></div>'
            f'<div><b>{len(s1["test"])}</b><span>external test patients, {sum(lab[p] for p in s1["test"])} events</span></div>')
    parts.append(f'<div class="kpi">{kpis}</div>')
    fold_tab = pd.DataFrame([{'fold': f, **{f'{n}': f'{len(s[n])} ({sum(s["labels"][p] for p in s[n])} events)'
                                            for n in ('train', 'val', 'test')}} for f, s in splits.items()])
    parts.append('<h3>Folds</h3>' + _table(fold_tab))
    parts.append('<p class="note">Fold reports: ' + ' | '.join(
        f'<a href="fold{f}/report.html">fold {f}</a>' for f in splits) + '</p>')

    parts.append('<h2>1. Which configuration wins?</h2>')
    parts.append('<p class="note"><b>val_auc</b> = pooled out-of-fold AUC (every training patient validated once); '
                 '<b>test_auc</b> = AUC of the fold ensemble (probabilities averaged over folds and seeds) on the '
                 'external cohort, 95% bootstrap CI; <b>test_auc_fold_min/max</b> = range of the single-fold test AUCs '
                 '(as reported by Mateus et al.). The configuration is chosen on val_auc only.</p>')
    for clin in (False, True):
        b = df[(df.use_clinical == clin) & df.best_by_val]
        if len(b):
            r = b.iloc[0]
            parts.append(f'<p><b>{r.inputs}</b>: {PRETTY.get(r.method, r.method)} + {r.topology} '
                         f'(out-of-fold val AUC {r.val_auc:.3f}) &rarr; <b>external test AUC {r.test_auc:.3f}</b> '
                         f'[{r.test_auc_ci_low:.3f}, {r.test_auc_ci_high:.3f}]; single folds '
                         f'{r.test_auc_fold_min:.3f}&ndash;{r.test_auc_fold_max:.3f}</p>')
    cols = ['method_name', 'topology', 'inputs', 'n_features', 'val_auc', 'test_auc', 'test_auc_ci_low',
            'test_auc_ci_high', 'test_auc_fold_mean', 'test_auc_fold_min', 'test_auc_fold_max',
            'p_vs_reference_holm', 'p_vs_rf_same_features_holm', 'cv_kuncheva', 'boot_kuncheva', 'best_by_val']
    parts.append(_table(df[[c for c in cols if c in df]].round(4), best_col='best_by_val'))
    for clin in ('img', 'img+clin'):
        parts.append(_img(fig / 'results' / f'fold_aucs_{clin}.png'))
        parts.append('<div class="grid">' + _img(fig / 'results' / f'auc_heatmap_test_{clin}.png')
                     + _img(fig / 'results' / f'auc_heatmap_val_{clin}.png') + '</div>')
        parts.append('<div class="grid">' + _img(fig / 'results' / f'forest_test_auc_{clin}.png')
                     + _img(fig / 'results' / f'roc_overlay_{clin}.png') + '</div>')

    parts.append('<h2>2. Which feature selection is reproducible?</h2>')
    if not stab.empty:
        parts.append(f'<h3>Across the {k} CV folds</h3>' + _table(stab.round(3)))
    sel = run / 'selection'
    for f, title in (('stability_bootstrap_fold_mean.csv', 'Within-fold bootstrap stability (mean over folds)'),
                     ('stability_centres_fold_mean.csv', 'Cross-centre reproducibility (mean over folds)')):
        if (sel / f).exists():
            parts.append(f'<h3>{title}</h3>' + _table(pd.read_csv(sel / f).round(3)))
    parts.append('<div class="grid">' + _img(fig / 'selection' / 'stability_cv_folds.png')
                 + _img(fig / 'selection' / 'selected_per_fold.png') + '</div>')
    parts.append('<div class="grid">' + _img(fig / 'results' / 'cv_stability_vs_auc_img.png')
                 + _img(fig / 'results' / 'cv_stability_vs_auc_img+clin.png') + '</div>')
    parts.append('<p class="note">Patient-level attention graphs are in each fold&#39;s report.</p>')

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>GRAFS-GAT {config.MODE} {task}</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(parts)}</main></body></html>')
    out = run / 'report.html'
    out.write_text(page)
    print(f"[report] -> {out}")
    return out
