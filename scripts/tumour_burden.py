"""
Step 3. Tumour burden from the delineations (mm^3): primary GTV volume, nodal GTV volume, number of nodal
GTV structures and total GTV volume (union of primary and nodal GTVs). Structures: see
hnc_popgraph/data/delineations.py.

  python scripts/tumour_burden.py --n_jobs 4   -> outputs/features/tumour_burden.csv
"""
import argparse
import re

import numpy as np
import pandas as pd
import pydicom

import _paths  # noqa: F401
from hnc_popgraph import config
from hnc_popgraph.data.delineations import nodal_map
from hnc_popgraph.data.loader import HNSCCDataLoader, rtstruct_mask_by_coordinates

OUT = config.TUMOUR_BURDEN_CSV


def one(dataset, pid, nodes_listed):
    L = HNSCCDataLoader(dataset)
    ct = L.load_ct_scan(pid)
    vox = float(np.prod(ct.GetSpacing()))
    pdir = L._patient_dir(pid)
    rs = pydicom.dcmread(str(L._find_rtstruct(pid, pdir)))
    names = [r.ROIName for r in rs.StructureSetROISequence]
    if dataset == 'MAASTRO':
        prim = ['GTV-1']
        nodes = sorted(n for n in names if re.fullmatch(r'GTV-[2-9]\d*', n))
    else:
        prim = L.roi_map[str(pid)]
        lower = {n.lower(): n for n in names}
        nodes = [lower[n.lower()] for n in nodes_listed if n.lower() in lower and n.lower() not in
                 {p.lower() for p in prim}]
    p = rtstruct_mask_by_coordinates(rs, prim, ct).astype(bool)
    out = {'patient_id': pid, 'vol_primary': p.sum() * vox, 'n_nodal_gtv': len(nodes), 'vol_nodal': 0.0,
           'vol_total': p.sum() * vox, 'nodes_missing': len(nodes_listed) - len(nodes) if dataset != 'MAASTRO' else 0}
    if nodes:
        n = np.zeros_like(p)
        for r in nodes:
            try:
                n |= rtstruct_mask_by_coordinates(rs, [r], ct).astype(bool)
            except ValueError:
                out['nodes_missing'] += 1
        out['vol_nodal'] = (n & ~p).sum() * vox
        out['vol_total'] = (n | p).sum() * vox
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_jobs', type=int, default=4)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    nm = nodal_map()
    jobs = [('HNPET', p, nm.get(p, [])) for p in HNSCCDataLoader('HNPET').eligible_patients_any_task()]
    jobs += [('MAASTRO', p, []) for p in HNSCCDataLoader('MAASTRO').eligible_patients_any_task()]

    def safe(*j):
        try:
            return one(*j)
        except Exception as e:                                   # noqa: BLE001
            print(f'  {j[1]}: {e}')
            return {'patient_id': j[1]}
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(safe)(*j) for j in jobs)
    D = pd.DataFrame(rows).set_index('patient_id')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT)
    print(D.describe().round(1).to_string())


if __name__ == '__main__':
    main()
