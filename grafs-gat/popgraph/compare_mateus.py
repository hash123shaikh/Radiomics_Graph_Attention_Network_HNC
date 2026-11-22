"""
Validation-chosen population-graph GNN versus the published CNN of Mateus et al. (Sci Rep 2023).
5-fold CV: mean of the per-fold MAASTRO AUCs (their metric); cohort split: MAASTRO AUC with 95% CI.

  python popgraph/compare_mateus.py [--prefix popgnn]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config  # noqa: E402
from grafsgat.utils import bootstrap_auc_ci  # noqa: E402

RES = config.OUTPUT_DIR / 'popgraph' / 'results'
# Mateus et al. testing AUCs: Table 3 (CNN, imaging) and Table 4 (CNN with clinical data)
MATEUS = {('MATEUS_CV', 'imaging'): {'DM': 0.83, 'LR': 0.53, 'OS': 0.63},
          ('MATEUS_CV', 'imaging+clinical'): {'DM': 0.88, 'LR': 0.57, 'OS': 0.68},
          ('MATEUS_COHORT', 'imaging'): {'DM': 0.89, 'LR': 0.45, 'OS': 0.67},
          ('MATEUS_COHORT', 'imaging+clinical'): {'DM': 0.93, 'LR': 0.59, 'OS': 0.69}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='popgnn')
    a = ap.parse_args()
    rows = []
    for t in ('LR', 'DM', 'OS'):
        for d in ('MATEUS_CV', 'MATEUS_COHORT'):
            f = RES / f'{a.prefix}_{t}_{d}.csv'
            if not f.exists():
                continue
            S = pd.read_csv(f)
            P = pd.read_csv(RES / f'{a.prefix}_pred_{t}_{d}.csv.gz')
            for inp in ('imaging', 'imaging+clinical'):
                b = S[S.inputs == inp].sort_values('val_auc', ascending=False).iloc[0]
                te = P[(P.config == b.config) & (P.split == 'test')]
                te = te.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
                lo, hi = bootstrap_auc_ci(te.label.values, te.prob.values)
                ours = b.test_auc_fold_mean if d == 'MATEUS_CV' else b.test_auc
                ref = MATEUS[(d, inp)][t]
                rows.append(dict(task=t, design=d, inputs=inp, config=b.config, val_auc=round(b.val_auc, 3),
                                 ours=round(ours, 3),
                                 ours_range=(f'{b.test_auc_fold_min:.2f}-{b.test_auc_fold_max:.2f}'
                                             if d == 'MATEUS_CV' else f'[{lo:.2f}, {hi:.2f}]'),
                                 pooled_test=round(b.test_auc, 3), mateus=ref, diff=round(ours - ref, 3),
                                 beats=ours > ref))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(R.to_string(index=False))
    print(f'\nbeats Mateus in {int(R.beats.sum())}/{len(R)} settings')
    R.to_csv(RES / f'{a.prefix}_vs_mateus.csv', index=False)


if __name__ == '__main__':
    main()
