"""Summary of the GTV-only experiments -> outputs/gtv_only/results/gtv_only_summary.csv"""
import numpy as np, pandas as pd
from common import RES_DIR
from grafsgat.utils import bootstrap_auc_ci, safe_auc

def ci(P, cfg=None):
    t = P[P.split == 'test'] if cfg is None else P[(P.split == 'test') & (P.config == cfg)]
    e = t.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'mean'))
    return safe_auc(e.label, e.prob), *bootstrap_auc_ci(e.label.values, e.prob.values)

rows = []
for task in ('LR', 'DM', 'OS'):
    for d in ('MATEUS_CV', 'MATEUS_COHORT'):
        S = pd.read_csv(RES_DIR / f'classical_{task}_{d}.csv')
        P = pd.read_csv(RES_DIR / f'classical_pred_{task}_{d}.csv.gz')
        for name, sel in (('GTV volume only (LR)', S.config == 'volume|lr'),
                          ('Clinical only (LR)', S.config == 'clinical|lr')):
            r = S[sel].iloc[0]; a, lo, hi = ci(P, r.config)
            rows.append(dict(task=task, design=d, model=name, config=r.config, val_auc=r.val_auc, test_auc=a,
                             ci_low=lo, ci_high=hi, test_auc_fold_mean=r.test_auc_fold_mean))
        for inp in ('imaging', 'imaging+clinical'):
            g = S[(S.inputs == inp) & ~S.config.str.startswith('volume')].sort_values('val_auc', ascending=False)
            r = g.iloc[0]; a, lo, hi = ci(P, r.config)
            rows.append(dict(task=task, design=d, model=f'GTV classical, {inp} (chosen on val)', config=r.config,
                             val_auc=r.val_auc, test_auc=a, ci_low=lo, ci_high=hi,
                             test_auc_fold_mean=r.test_auc_fold_mean, grid_test_max=g.test_auc.max(),
                             grid_test_median=g.test_auc.median()))
            G = pd.read_csv(RES_DIR / f'gat_final_{task}_{d}.csv')
            f = G[G.inputs == inp].iloc[0]
            Pg = pd.read_csv(RES_DIR / f'gat_pred_{task}_{d}_{inp}.csv.gz')
            a, lo, hi = ci(Pg)
            cfg = ' '.join(f'{k}={f[k]}' for k in ('fs', 'k', 'topology', 'global', 'hidden', 'layers', 'heads',
                                                    'dropout', 'lr', 'wd', 'sampling'))
            sr = pd.read_csv(RES_DIR / f'gat_search_{task}_{d}.csv')
            rows.append(dict(task=task, design=d, model=f'Intratumoral GAT, {inp} (chosen on val)', config=cfg,
                             val_auc=f.val_auc, test_auc=a, ci_low=lo, ci_high=hi,
                             test_auc_fold_mean=f.test_auc_fold_mean, grid_test_max=sr.test_auc.max(),
                             grid_test_median=sr.test_auc.median()))
out = pd.DataFrame(rows)
out.to_csv(RES_DIR / 'gtv_only_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(out.drop(columns='config').round(3).to_string(index=False))
print(); print(out[out.model.str.contains('chosen')][['task', 'design', 'model', 'config']].to_string(index=False))
