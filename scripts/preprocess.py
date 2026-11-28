"""
Step 1. Resample CT and primary GTV to 1 mm and crop to the GTV box + 50 mm, for every patient with
outcome data in either cohort.

  python scripts/preprocess.py --n_jobs 4     -> outputs/preprocessed/<pid>_preprocessed.npz
"""
import argparse

import _paths  # noqa: F401
from hnc_popgraph.data.loader import HNSCCDataLoader
from hnc_popgraph.data.preprocessing import preprocess_cohort

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_jobs', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for ds in ('HNPET', 'MAASTRO'):
        preprocess_cohort(ds, HNSCCDataLoader(ds).eligible_patients_any_task(), force=a.force, n_jobs=a.n_jobs)
