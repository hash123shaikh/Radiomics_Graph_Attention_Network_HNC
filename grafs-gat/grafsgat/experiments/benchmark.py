"""
Feature-selection x graph-topology benchmark for one run (<MODE>_<TASK>).

  run_grid()   build graphs, train + evaluate the GAT for every
               (method, topology, imaging/imaging+clinical, seed); RF baselines
  summarize()  seed-ensembled AUCs with 95% CI, DeLong vs the paper reference
               (Table S3 + radiomic star), GAT vs RF on the same features, Holm correction,
               stability metrics joined in, best configuration chosen on VALIDATION AUC
               -> results_summary.csv + figures + report.html
"""

import json
import time

import numpy as np
import pandas as pd

from grafsgat import config
from grafsgat.selection.methods import PRETTY
from grafsgat.stats import delong_test, holm
from grafsgat.utils import safe_auc, bootstrap_auc_ci


def run_grid(task, methods=None, topologies=None, variants=(False, True), seeds=None,
             n_epochs=None, force=False, baselines=True):
    config.set_task(task)
    from grafsgat.graphs.builder import build_graphs
    from grafsgat.engine.train import train
    from grafsgat.engine.evaluate import evaluate
    methods = methods or config.fs_methods(task)
    topologies = topologies or config.TOPOLOGIES
    seeds = seeds or config.BENCHMARK_SEEDS
    total = len(methods) * len(topologies) * len(variants) * len(seeds)
    done, t0 = 0, time.time()
    for m in methods:
        for topo in topologies:
            build_graphs(task, m, topo)
            for clin in variants:
                for s in seeds:
                    done += 1
                    paths = config.exp_paths(task, m, topo, clin, s)
                    if paths['metrics'].exists() and not force:
                        continue
                    print(f"[{done}/{total}] {config.exp_name(m, topo, clin, s)} "
                          f"({(time.time() - t0) / 60:.1f} min elapsed)")
                    train(task, m, topo, clin, seed=s, n_epochs=n_epochs, verbose=False)
                    evaluate(task, m, topo, clin, seed=s)
    if baselines:
        from grafsgat.engine.baselines import rf_selected, rf_clinical
        print("[baselines] random forests on the same split")
        for m in methods:
            rf_selected(task, m)
        rf_clinical(task)


# ─── Summary ─────────────────────────────────────────────────────────────────

def _ensembles(task):
    """{(method, topology, use_clinical): DataFrame[patient_id, split, label, prob, n_seeds]}"""
    groups = {}
    for f in sorted(config.sub_dir(task, 'results').glob('*.json')):
        r = json.loads(f.read_text())
        key = (r['method'], r['topology'], bool(r['use_clinical']))
        groups.setdefault(key, []).append(r['seed'])
    ens, per_seed = {}, {}
    for key, seeds in groups.items():
        dfs = [pd.read_csv(config.exp_paths(task, key[0], key[1], key[2], s)['predictions']) for s in seeds]
        per_seed[key] = {s: {sp: safe_auc(d[d.split == sp].label, d[d.split == sp].prob)
                             for sp in ('val', 'test')} for s, d in zip(seeds, dfs)}
        cat = pd.concat(dfs)
        e = cat.groupby(['patient_id', 'split'], as_index=False).agg(label=('label', 'first'),
                                                                     prob=('prob', 'mean'))
        e['n_seeds'] = len(seeds)
        ens[key] = e
    return ens, per_seed


def _aligned(a, b, split):
    x = a[a.split == split].set_index('patient_id')
    y = b[b.split == split].set_index('patient_id')
    ids = x.index.intersection(y.index)
    return x.loc[ids, 'label'].values, x.loc[ids, 'prob'].values, y.loc[ids, 'prob'].values


def comparison_table(ens, spread, spread_name='seed'):
    """One row per (method, topology, inputs) from ensembled predictions.
    ens    : {(method, topology, use_clinical): DataFrame[patient_id, split, label, prob, n_seeds]}
    spread : {key: {'val': [auc, ...], 'test': [auc, ...]}} of the members (seeds or folds)
    AUC + bootstrap CI per split, DeLong vs the reference and vs RF on the same features (test), Holm."""
    ref_m, ref_t = config.reference_config()
    rows = []
    for (m, t, clin), e in ens.items():
        row = {'method': m, 'method_name': PRETTY.get(m, m), 'topology': t,
               'model': 'RF' if t == 'RF' else 'GAT', 'inputs': 'imaging+clinical' if clin else 'imaging',
               'use_clinical': clin, 'n_seeds': int(e.n_seeds.iloc[0])}
        for sp in ('val', 'test'):
            d = e[e.split == sp]
            row[f'{sp}_auc'] = safe_auc(d.label, d.prob)
            lo, hi = bootstrap_auc_ci(d.label.values, d.prob.values)
            row[f'{sp}_auc_ci_low'], row[f'{sp}_auc_ci_high'] = lo, hi
            row[f'{sp}_n'], row[f'{sp}_events'] = len(d), int(d.label.sum())
            a = np.array(spread[(m, t, clin)][sp], dtype=float)
            row[f'{sp}_auc_{spread_name}_mean'] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            row[f'{sp}_auc_{spread_name}_sd'] = float(np.nanstd(a)) if np.isfinite(a).any() else np.nan
            if spread_name == 'fold':
                row[f'{sp}_auc_fold_min'] = float(np.nanmin(a)) if np.isfinite(a).any() else np.nan
                row[f'{sp}_auc_fold_max'] = float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
        ref = ens.get((ref_m, ref_t, clin))
        if ref is not None and (m, t) != (ref_m, ref_t) and t != 'RF':
            y, p1, p2 = _aligned(e, ref, 'test')
            row['p_vs_reference'] = delong_test(y, p1, p2)[3]
        if t != 'RF' and (m, 'RF', False) in ens and m != 'clinical_only':
            y, p1, p2 = _aligned(e, ens[(m, 'RF', False)], 'test')
            row['p_vs_rf_same_features'] = delong_test(y, p1, p2)[3]
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ('p_vs_reference', 'p_vs_rf_same_features'):
        if col in df:
            for clin in (False, True):
                msk = (df.use_clinical == clin) & df[col].notna()
                df.loc[msk, col + '_holm'] = holm(df.loc[msk, col].values)
    return df


def mark_best_by_val(df):
    """Pre-specified selection rule: best GAT configuration by validation AUC (per input set)."""
    df['best_by_val'] = False
    for clin in (False, True):
        g = df[(df.model == 'GAT') & (df.use_clinical == clin)]
        if len(g) and g.val_auc.notna().any():
            df.loc[g.val_auc.idxmax(), 'best_by_val'] = True
    return df.sort_values(['use_clinical', 'model', 'val_auc'], ascending=[True, True, False])


def print_table(df):
    show = df[['method', 'topology', 'inputs', 'val_auc', 'test_auc', 'test_auc_ci_low', 'test_auc_ci_high',
               'best_by_val']].head(40)
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def summarize(task, make_figures=True):
    config.set_task(task)
    ens, per_seed = _ensembles(task)
    if not ens:
        raise RuntimeError("No results yet. Run the 'benchmark' stage.")
    spread = {k: {sp: [v[sp] for v in d.values()] for sp in ('val', 'test')} for k, d in per_seed.items()}
    df = comparison_table(ens, spread, 'seed')

    # stability
    sel = config.sub_dir(task, 'selection')
    for fname, prefix in (('stability_bootstrap.csv', 'boot'), ('stability_centres.csv', 'centre')):
        f = sel / fname
        if f.exists():
            s = pd.read_csv(f).set_index('method')
            for c in ('jaccard', 'dice', 'kuncheva', 'kendall_w', 'core80'):
                if c in s:
                    df[f'{prefix}_{c}'] = df.method.map(s[c])
    # number of features
    n_feat = {}
    for m in df.method.unique():
        f = config.selection_file(task, m)
        if f.exists():
            n_feat[m] = len(json.loads(f.read_text())['selected'])
    df['n_features'] = df.method.map(n_feat)

    df = mark_best_by_val(df)
    out = config.run_dir(task) / 'results_summary.csv'
    df.to_csv(out, index=False)
    print(f"[summary] {len(df)} configurations -> {out}")
    print_table(df)

    if make_figures:
        make_all_figures(task, df, ens)
    return df


def result_figures(task, df, ens):
    from grafsgat.viz import result_plots as rp
    fd = config.figures_dir(task, 'results')
    for clin in (False, True):
        tag = 'img+clin' if clin else 'img'
        d = df[df.use_clinical == clin]
        if d.empty:
            continue
        rp.plot_auc_heatmap(d, fd / f'auc_heatmap_test_{tag}.png', 'test')
        rp.plot_auc_heatmap(d, fd / f'auc_heatmap_val_{tag}.png', 'val')
        rp.plot_forest(d, fd / f'forest_test_auc_{tag}.png')
        if 'boot_kuncheva' in d:
            rp.plot_stability_vs_auc(d, fd / f'stability_vs_auc_{tag}.png')
        rp.plot_roc_overlay(d, ens, fd / f'roc_overlay_{tag}.png')


def make_all_figures(task, df, ens=None):
    from grafsgat.viz.report import build_report
    if ens is None:
        ens, _ = _ensembles(task)
    result_figures(task, df, ens)
    try:
        best_patient_graphs(task, df)
    except Exception as e:                                    # figures must not kill the summary
        print(f"  [figures] patient graphs skipped: {type(e).__name__}: {e}")
    build_report(task, df)


def best_patient_graphs(task, df, n_examples=4):
    """Attention-coloured patient graphs (test set) for the best configuration and the reference."""
    from grafsgat.engine.evaluate import extract_attention
    from grafsgat.viz.patient_graph import plot_patient_examples
    from grafsgat.engine.train import plot_history
    ref_m, ref_t = config.reference_config()
    picks = []
    best = df[(df.model == 'GAT') & df.best_by_val]
    for _, r in best.iterrows():
        picks.append((r.method, r.topology, bool(r.use_clinical), 'best'))
    if not ((best.method == ref_m) & (best.topology == ref_t)).any():
        picks.append((ref_m, ref_t, False, 'reference'))
    for m, t, clin, label in picks:
        seed = config.BENCHMARK_SEEDS[0]
        if not config.exp_paths(task, m, t, clin, seed)['model'].exists():
            continue
        att = extract_attention(task, m, t, clin, seed)
        name = config.exp_name(m, t, clin, seed)
        plot_patient_examples(task, m, t, att, config.figures_dir(task, 'patient_graphs'),
                              prefix=f'{label}_{name}', n=n_examples)
        plot_history(task, m, t, clin, seed, config.figures_dir(task, 'training') / f'{label}_{name}.png')
