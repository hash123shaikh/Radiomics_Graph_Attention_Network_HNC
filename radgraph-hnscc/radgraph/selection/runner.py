"""
Selection stage of a run: fit every FS method on the TRAINING split, save the results,
run the stability / reproducibility analysis and draw the selection figures.

    outputs/runs/<MODE>_<TASK>/selection/<method>.json
    outputs/runs/<MODE>_<TASK>/selection/stability_bootstrap.csv
    outputs/runs/<MODE>_<TASK>/selection/stability_centres.csv
    outputs/runs/<MODE>_<TASK>/selection/method_agreement.csv
"""

import json

import numpy as np
import pandas as pd

from radgraph import config
from radgraph.data.features import gtv_matrix
from radgraph.data.splits import get_or_create_split
from radgraph.selection.methods import run_selector, PRETTY
from radgraph.selection import stability as st


def train_matrix(task, split=None):
    split = split or get_or_create_split(task, verbose=False)
    ids = split['train']
    X = gtv_matrix(ids)
    y = np.array([split['labels'][p] for p in ids])
    centres = [config.patient_centre(split['cohort'][p], p) for p in ids]
    return X, y, centres, split


def load_selection(task, method):
    f = config.selection_file(task, method)
    if not f.exists():
        raise FileNotFoundError(f"No selection for {method}. Run the 'select' stage first.")
    return json.loads(f.read_text())


def run_selection(task, methods=None, k='default', force=False):
    config.set_task(task)
    methods = methods or config.fs_methods(task)
    X, y, _, split = train_matrix(task)
    print(f"[select] {config.run_label()} {task}: {X.shape[0]} training patients ({int(y.sum())} events), "
          f"{X.shape[1]} GTV features, k={config.FS_K if k == 'default' else k}")
    out = {}
    for m in methods:
        f = config.selection_file(task, m)
        if f.exists() and not force:
            d = json.loads(f.read_text())
            if d.get('split_key') == split['key'] and d.get('k_requested') == (config.FS_K if k == 'default' else k):
                out[m] = d
                print(f"  {m:18s} cached: {len(d['selected'])} features")
                continue
        r = run_selector(m, X, y, task, k, seed=config.RANDOM_SEED)
        d = r.to_dict()
        d.update(split_key=split['key'], k_requested=config.FS_K if k == 'default' else k,
                 n_train=int(len(y)), n_events=int(y.sum()))
        f.write_text(json.dumps(d, indent=1))
        out[m] = d
        shown = r.selected if len(r.selected) <= 10 else r.selected[:5] + ['...']
        print(f"  {m:18s} -> {len(r.selected):2d} features: {shown}")
    agree = st.method_agreement({m: d['selected'] for m, d in out.items()})
    agree.to_csv(config.sub_dir(task, 'selection') / 'method_agreement.csv')

    from radgraph.viz import selection_plots as sp
    sp.plot_selection_matrix(out, config.figures_dir(task, 'selection') / 'selected_features_matrix.png')
    sp.plot_method_agreement(agree, config.figures_dir(task, 'selection') / 'method_agreement_jaccard.png')
    from radgraph.viz.feature_graph import plot_feature_graph
    for m in ('graph_fs_cc', 'graph_fs_spectral', 'graph_fs_hybrid'):
        if m in out:
            plot_feature_graph(out[m], config.figures_dir(task, 'selection') / f'feature_graph_{m}.png',
                               title=f"{PRETTY[m]} feature-similarity graph ({config.run_label()}, {task})")
    return out


def run_stability(task, methods=None, k='default', B=None, n_jobs=8):
    methods = methods or config.fs_methods(task)
    X, y, centres, _ = train_matrix(task)
    print(f"[stability] {config.run_label()} {task}: bootstrap B={B or config.STABILITY_N_BOOTSTRAP}")
    boot, freqs, _ = st.bootstrap_stability(X, y, methods, task, k, B, n_jobs=n_jobs)
    d = config.sub_dir(task, 'selection')
    boot.to_csv(d / 'stability_bootstrap.csv', index=False)
    pd.DataFrame(freqs).fillna(0).to_csv(d / 'selection_frequency_bootstrap.csv')
    cen, per_centre = st.centre_reproducibility(X, y, centres, methods, task, k)
    if not cen.empty:
        cen.to_csv(d / 'stability_centres.csv', index=False)
        (d / 'selection_per_centre.json').write_text(json.dumps(per_centre, indent=1))

    from radgraph.viz import selection_plots as sp
    fd = config.figures_dir(task, 'selection')
    sp.plot_stability(boot, fd / 'stability_bootstrap.png', f'Within-cohort stability (bootstrap), {config.run_label()} {task}')
    if not cen.empty:
        sp.plot_stability(cen, fd / 'stability_centres.png', f'Cross-centre reproducibility, {config.run_label()} {task}')
    sp.plot_selection_frequency(freqs, fd / 'selection_frequency_bootstrap.png')
    return boot, cen
