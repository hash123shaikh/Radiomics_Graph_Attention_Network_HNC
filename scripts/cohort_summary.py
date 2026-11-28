"""
Step 5. Patients and events per endpoint and cohort (eligible patients of the study splits).

  python scripts/cohort_summary.py   -> outputs/results/cohort_summary.csv
"""
import pandas as pd

import _paths  # noqa: F401
from hnc_popgraph import config
from hnc_popgraph.data.splits import eligible

if __name__ == '__main__':
    rows = []
    for task in config.TASKS:
        config.set_task(task)
        for ds, role in (('HNPET', 'train/validation (4 Canadian centres)'), ('MAASTRO', 'external test')):
            lab = eligible(ds, task, verbose=False)
            rows.append({'task': task, 'horizon_months': config.horizon_months(task), 'cohort': ds, 'role': role,
                         'patients': len(lab), 'events': sum(lab.values()),
                         'event_rate': round(sum(lab.values()) / max(len(lab), 1), 3)})
    D = pd.DataFrame(rows)
    D.to_csv(config.COHORT_SUMMARY_CSV, index=False)
    print(D.to_string(index=False))
