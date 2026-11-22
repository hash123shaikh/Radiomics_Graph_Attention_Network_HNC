"""
Does the patient graph help?  Re-selection of the population-graph grid restricted to model families,
using the predictions already saved by run_popgnn.py (no retraining).

  graph   : GCN, GraphSAGE, GAT only (best validation AUC among them)
  mlp     : the same network without edges (no-graph control)
  all     : every model (the original selection)

For every task x design x input set the validation-chosen configuration of each family is compared with
the published CNN of Mateus et al. (CV: mean of per-fold MAASTRO AUCs; cohort: MAASTRO AUC + 95% CI),
and graph-chosen versus MLP-chosen is tested with a paired patient-level bootstrap on MAASTRO.
A matched analysis contrasts every graph configuration with the MLP that shares its features
(feature set, selector, k, inputs), on validation and on test.

  python popgraph/graph_vs_mlp.py [--prefix popgnn_burden_total]
-> outputs/popgraph/results/<prefix>_family_selection.csv, <prefix>_graph_vs_mlp_paired.csv,
   <prefix>_graph_vs_mlp_matched.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mateus import MATEUS, RES  # noqa: E402
from grafsgat.utils import bootstrap_auc_ci  # noqa: E402

FAMILIES = {'graph': ('gcn', 'sage', 'gat'), 'mlp': ('mlp',), 'all': ('gcn', 'sage', 'gat', 'mlp')}
N_BOOT = 2000


def test_probs(P, cfg):
    """MAASTRO probabilities of one configuration (averaged over CV folds; same patients in every fold)."""
    te = P[(P.config == cfg) & (P.split == 'test')]
    return te.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))


def paired_boot(y, p1, p2, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='popgnn_burden_total')
    a = ap.parse_args()
    sel, paired, matched = [], [], []
    for t in ('LR', 'DM', 'OS'):
        for d in ('MATEUS_CV', 'MATEUS_COHORT'):
            S = pd.read_csv(RES / f'{a.prefix}_{t}_{d}.csv')
            P = pd.read_csv(RES / f'{a.prefix}_pred_{t}_{d}.csv.gz')
            for inp in ('imaging', 'imaging+clinical', 'clinical'):
                chosen = {}
                for fam, models in FAMILIES.items():
                    b = S[(S.inputs == inp) & S.model.isin(models)].sort_values('val_auc', ascending=False).iloc[0]
                    chosen[fam] = b
                    te = test_probs(P, b.config)
                    lo, hi = bootstrap_auc_ci(te.label.values, te.prob.values)
                    ours = b.test_auc_fold_mean if d == 'MATEUS_CV' else b.test_auc
                    ref = MATEUS.get((d, inp), {}).get(t, np.nan)
                    sel.append(dict(task=t, design=d, inputs=inp, family=fam, model=b.model, config=b.config,
                                    val_auc=b.val_auc, ours=ours,
                                    fold_min=b.test_auc_fold_min if d == 'MATEUS_CV' else np.nan,
                                    fold_max=b.test_auc_fold_max if d == 'MATEUS_CV' else np.nan,
                                    pooled_test=b.test_auc, ci_lo=lo, ci_hi=hi, mateus=ref,
                                    diff=ours - ref if ref == ref else np.nan,
                                    beats=bool(ours > ref) if ref == ref else np.nan,
                                    test_n=b.test_n, test_events=b.test_events))
                g, m = test_probs(P, chosen['graph'].config), test_probs(P, chosen['mlp'].config)
                m = m.loc[g.index]
                obs, lo, hi, p = paired_boot(g.label.values, g.prob.values, m.prob.values)
                paired.append(dict(task=t, design=d, inputs=inp, graph_config=chosen['graph'].config,
                                   mlp_config=chosen['mlp'].config,
                                   graph_val=chosen['graph'].val_auc, mlp_val=chosen['mlp'].val_auc,
                                   graph_test=chosen['graph'].test_auc, mlp_test=chosen['mlp'].test_auc,
                                   delta_test=obs, delta_lo=lo, delta_hi=hi, p_boot=p))
            # matched: each graph configuration vs the MLP with identical features
            S = S.copy()
            S['feat_key'] = S.featset + '|' + S.fs + '|' + S.k.astype(str) + '|' + S.inputs
            M = S[S.model == 'mlp'].set_index('feat_key')[['val_auc', 'test_auc', 'test_auc_fold_mean']]
            Gr = S[S.model != 'mlp'].join(M, on='feat_key', rsuffix='_mlp')
            for (inp, model), x in Gr.groupby(['inputs', 'model']):
                ours_col = 'test_auc_fold_mean' if d == 'MATEUS_CV' else 'test_auc'
                dv = x.val_auc - x.val_auc_mlp
                dt = x[ours_col] - x[ours_col + '_mlp']
                matched.append(dict(task=t, design=d, inputs=inp, model=model, n_configs=len(x),
                                    val_delta_median=dv.median(), val_frac_graph_better=(dv > 0).mean(),
                                    test_delta_median=dt.median(), test_frac_graph_better=(dt > 0).mean()))
    sel, paired, matched = pd.DataFrame(sel), pd.DataFrame(paired), pd.DataFrame(matched)
    sel.to_csv(RES / f'{a.prefix}_family_selection.csv', index=False)
    paired.to_csv(RES / f'{a.prefix}_graph_vs_mlp_paired.csv', index=False)
    matched.to_csv(RES / f'{a.prefix}_graph_vs_mlp_matched.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 50)
    v = sel[sel.inputs != 'clinical'].pivot_table(index=['task', 'design', 'inputs'], columns='family',
                                                  values='ours').round(3)
    v['mateus'] = sel[sel.family == 'graph'].set_index(['task', 'design', 'inputs']).mateus
    print(v.to_string())
    for fam in ('graph', 'mlp', 'all'):
        s = sel[(sel.family == fam) & (sel.inputs != 'clinical')]
        print(f'{fam:6} beats Mateus in {int(s.beats.sum())}/{len(s)}')
    print('\nchosen graph vs chosen MLP on MAASTRO (paired bootstrap):')
    print(paired[['task', 'design', 'inputs', 'graph_val', 'mlp_val', 'graph_test', 'mlp_test', 'delta_test',
                  'delta_lo', 'delta_hi', 'p_boot']].round(3).to_string(index=False))
    print('\nmatched graph - MLP (same features):')
    print(matched.groupby(['task', 'model'])[['val_delta_median', 'val_frac_graph_better', 'test_delta_median',
                                              'test_frac_graph_better']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
