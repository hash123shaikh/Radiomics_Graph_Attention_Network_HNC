"""
Tumour burden from the delineations: primary GTV volume, nodal GTV volume, number of nodal GTV
structures and total GTV volume (union of primary and nodal GTVs), in mm^3.

  Head-Neck-PET-CT  primary = ROI(s) of gtv_primary_rois.csv; nodal = 'Name GTV Lymph Nodes' of the
                    TCIA list INFO_GTVcontours_HN.xlsx (comma separated; none listed -> 0)
  MAASTRO           primary = GTV-1; nodal = GTV-2 ... GTV-n

  python popgraph/tumour_burden.py --n_jobs 12   -> outputs/popgraph/tumour_burden.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config  # noqa: E402
from grafsgat.data.loader import HNSCCDataLoader, rtstruct_mask_by_coordinates  # noqa: E402

OUT = config.OUTPUT_DIR / 'popgraph' / 'tumour_burden.csv'


def nodal_map():
    x = pd.concat(pd.read_excel(config.HNPET_DIR / 'INFO_GTVcontours_HN.xlsx', sheet_name=None).values())
    return {p.replace('HN-', ''): ([s.strip() for s in str(n).split(',') if s.strip()] if pd.notna(n) else [])
            for p, n in zip(x['Patient'], x['Name GTV Lymph Nodes'])}


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
    ap.add_argument('--n_jobs', type=int, default=12)
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
