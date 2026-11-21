"""
Report GRAFS-GAT exactly as Mateus et al. (Sci Rep 2023) report their CNN, from the runs trained with
--epoch_rule mateus (checkpoint rule of their hn_cnn/fit.py):

  * one network per fold, ONE seed (no seed averaging)
  * 5-fold CV: mean (min-max) of the per-fold MAASTRO AUCs over the folds whose run SAVED a checkpoint
    (folds in which no epoch met the validation-AUC threshold and the train-val AUC difference are left out,
    as in their reporting); the number of contributing folds is given
  * cohort split: MAASTRO AUC with a 1000-resample bootstrap 95% CI; flagged if no checkpoint was saved
  * configuration: the one chosen on validation AUC under the same rule (and, separately, the reference)

The same numbers are also given for every seed and for all five folds, to show how much the reported value
depends on these choices.  ->  outputs/results/mateus_style_report.csv

  python tools/mateus_style_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config  # noqa: E402
from grafsgat.utils import bootstrap_auc_ci  # noqa: E402

SEEDS = [42, 43, 44]


def _one(d, name, seed):
    p = pd.read_csv(d / 'predictions' / f'{name}__s{seed}.csv')
    t = p[p.split == 'test']
    h = json.loads((d / 'histories' / f'{name}__s{seed}.json').read_text())
    return t, roc_auc_score(t.label, t.prob), not h.get('mateus_fallback_last_epoch', True)


def report(task, design, rule='mateus'):
    root = config.RUNS_DIR / f'{design}_{task}{config.rule_suffix(rule)}'
    S = pd.read_csv(root / 'results_summary.csv')
    S = S[S.model == 'GAT']
    rows = []
    for inp, tag in (('imaging', 'img'), ('imaging+clinical', 'img+clin')):
        sub = S[S.inputs == inp]
        chosen = sub.sort_values('val_auc', ascending=False).iloc[0]
        ref_m, ref_t = config.reference_config(task)
        refr = sub[(sub.method == ref_m) & (sub.topology == ref_t)]
        for role, r in (('chosen', chosen), ('reference', refr.iloc[0] if len(refr) else None)):
            if r is None:
                continue
            name = f'{r.method}__{r.topology}__{tag}'
            dirs = ([root / f'fold{k}' for k in range(1, config.N_FOLDS + 1)] if design == 'MATEUS_CV'
                    else [root])
            for seed in SEEDS:
                res = [_one(d, name, seed) for d in dirs]
                aucs = np.array([a for _, a, _ in res])
                saved = np.array([s for _, _, s in res])
                row = {'task': task, 'design': design, 'inputs': inp, 'role': role, 'method': r.method,
                       'topology': r.topology, 'seed': seed, 'n_saved': int(saved.sum()), 'n_folds': len(dirs)}
                if design == 'MATEUS_CV':
                    use = aucs[saved] if saved.any() else aucs
                    row.update(auc_saved_mean=use.mean(), auc_saved_min=use.min(), auc_saved_max=use.max(),
                               auc_all_mean=aucs.mean(), auc_all_min=aucs.min(), auc_all_max=aucs.max())
                else:
                    t = res[0][0]
                    lo, hi = bootstrap_auc_ci(t.label.values, t.prob.values)
                    row.update(auc_saved_mean=aucs[0], ci_low=lo, ci_high=hi, auc_all_mean=aucs[0])
                rows.append(row)
    return rows


def main():
    rows = []
    for task in config.TASKS:
        for design in ('MATEUS_CV', 'MATEUS_COHORT'):
            try:
                rows += report(task, design)
            except FileNotFoundError as e:
                print(f'  {task} {design}: not available ({e.filename})')
    R = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / 'results' / 'mateus_style_report.csv'
    R.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(R[R.seed == 42].round(3).to_string(index=False))
    print('->', out)


if __name__ == '__main__':
    main()
