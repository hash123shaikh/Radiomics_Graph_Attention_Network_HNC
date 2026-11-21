"""
Where does the variation in external AUC come from?

Every single GRAFS-GAT network (selector x topology x input set x CV fold x seed, or x seed in the cohort
split) is scored on MAASTRO on its own, i.e. exactly as a single training run is reported by Mateus et al.
(one model per fold, one seed). The external AUCs are then decomposed with a nested design:

  total = between configurations (selector, topology, input set and their interactions)
        + between CV folds within a configuration     (which training patients; CV design only)
        + between seeds within a configuration/fold   (random initialisation and resampling only)

and the between-configuration part is split further by a type-II ANOVA on the configuration means.
Output: outputs/results/single_model_auc.csv, outputs/results/variance_components.csv

  python tools/variance_components.py [--rules val_loss auc_gap mateus]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config  # noqa: E402

SEEDS = [42, 43, 44]


def single_model_aucs(task, design, rule):
    root = config.RUNS_DIR / f'{design}_{task}{config.rule_suffix(rule)}'
    if not (root / 'results_summary.csv').exists():
        return pd.DataFrame()
    folds = ([(k, root / f'fold{k}') for k in range(1, config.N_FOLDS + 1)] if design == 'MATEUS_CV'
             else [(0, root)])
    rows = []
    for fold, d in folds:
        for f in sorted((d / 'predictions').glob('*__s*.csv')):
            parts = f.stem.split('__')
            if len(parts) != 4 or not parts[3][1:].isdigit() or parts[1] == 'RF':
                continue
            method, topo, tag, s = parts
            p = pd.read_csv(f)
            t = p[p.split == 'test']
            v = p[p.split == 'val']
            if t.label.nunique() < 2:
                continue
            rows.append({'task': task, 'design': design, 'rule': rule, 'method': method, 'topology': topo,
                         'inputs': 'imaging+clinical' if tag == 'img+clin' else 'imaging', 'fold': fold,
                         'seed': int(s[1:]), 'test_auc': roc_auc_score(t.label, t.prob),
                         'val_auc': roc_auc_score(v.label, v.prob) if v.label.nunique() == 2 else np.nan})
    return pd.DataFrame(rows)


def decompose(d):
    """Nested sums of squares (fractions of the total) + ANOVA of configuration means."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    y = d.test_auc
    ss_tot = ((y - y.mean()) ** 2).sum()
    cfg = ['method', 'topology', 'inputs']
    m_cfg = d.groupby(cfg).test_auc.transform('mean')
    m_cf = d.groupby(cfg + ['fold']).test_auc.transform('mean')
    out = {'n_models': len(d), 'mean_auc': y.mean(), 'sd_auc': y.std(), 'min_auc': y.min(), 'max_auc': y.max(),
           'between_config': ((m_cfg - y.mean()) ** 2).sum() / ss_tot,
           'between_fold': ((m_cf - m_cfg) ** 2).sum() / ss_tot,
           'between_seed': ((y - m_cf) ** 2).sum() / ss_tot}
    # within-configuration spread of single models (what one training run can give for the SAME design)
    rng = d.groupby(cfg).test_auc.agg(lambda a: a.max() - a.min())
    sd = d.groupby(cfg).test_auc.std()
    out.update(within_config_range_median=rng.median(), within_config_sd_median=sd.median(),
               config_mean_range=d.groupby(cfg).test_auc.mean().max() - d.groupby(cfg).test_auc.mean().min())
    # split of the between-configuration part (configuration means, main effects, type II)
    cm = d.groupby(cfg, as_index=False).test_auc.mean()
    fit = smf.ols('test_auc ~ C(method) + C(topology) + C(inputs)', data=cm).fit()
    a = anova_lm(fit, typ=2)
    share = a['sum_sq'] / a['sum_sq'].sum() * out['between_config']
    out.update(selector=share.get('C(method)', np.nan), topology=share.get('C(topology)', np.nan),
               inputs=share.get('C(inputs)', np.nan), config_residual=share.get('Residual', np.nan),
               p_selector=a.loc['C(method)', 'PR(>F)'], p_topology=a.loc['C(topology)', 'PR(>F)'],
               p_inputs=a.loc['C(inputs)', 'PR(>F)'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', nargs='+', default=['val_loss', 'auc_gap', 'mateus'])
    a = ap.parse_args()
    allm, comp = [], []
    for rule in a.rules:
        for task in config.TASKS:
            for design in ('MATEUS_CV', 'MATEUS_COHORT'):
                d = single_model_aucs(task, design, rule)
                if d.empty:
                    print(f'  {task} {design} {rule}: no runs')
                    continue
                allm.append(d)
                r = decompose(d)
                comp.append({'task': task, 'design': design, 'rule': rule, **r})
                print(f"{task} {design:13} {rule:8} n={r['n_models']:4d} AUC {r['mean_auc']:.2f} "
                      f"[{r['min_auc']:.2f}-{r['max_auc']:.2f}] | config {r['between_config']:.0%} "
                      f"(sel {r['selector']:.0%}, topo {r['topology']:.0%}, inputs {r['inputs']:.0%}) "
                      f"fold {r['between_fold']:.0%} seed {r['between_seed']:.0%}")
    res = config.OUTPUT_DIR / 'results'
    pd.concat(allm).to_csv(res / 'single_model_auc.csv', index=False)
    pd.DataFrame(comp).to_csv(res / 'variance_components.csv', index=False)
    print('->', res / 'variance_components.csv')


if __name__ == '__main__':
    main()
