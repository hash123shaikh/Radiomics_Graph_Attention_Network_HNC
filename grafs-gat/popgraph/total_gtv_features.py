"""
Radiomic features of the TOTAL GTV (union of the primary and all nodal GTVs, as listed in
tumour_burden.py), defined identically in both cohorts: CT resampled to 1 mm isotropic (linear),
mask nearest neighbour, cropped to the mask with a 10 mm margin; PyRadiomics with the study
settings, shape + original + LoG (1, 2, 3, 5 mm).

  python popgraph/total_gtv_features.py --n_jobs 12   -> outputs/popgraph/total_gtv_features.csv
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grafsgat import config  # noqa: E402
from grafsgat.data.loader import HNSCCDataLoader, rtstruct_mask_by_coordinates  # noqa: E402
from tumour_burden import nodal_map  # noqa: E402

OUT = config.OUTPUT_DIR / 'popgraph' / 'total_gtv_features.csv'


def extractor():
    from radiomics import featureextractor
    import radiomics
    radiomics.setVerbosity(logging.ERROR)
    ex = featureextractor.RadiomicsFeatureExtractor(**dict(config.PYRADIOMICS_SETTINGS))
    ex.disableAllFeatures()
    ex.enableFeatureClassByName('shape')
    for fc in config.FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    ex.enableImageTypes(Original={}, LoG={'sigma': [1.0, 2.0, 3.0, 5.0]})
    return ex


def total_mask(dataset, pid, nodes_listed):
    L = HNSCCDataLoader(dataset)
    ct = L.load_ct_scan(pid)
    rs = pydicom.dcmread(str(L._find_rtstruct(pid, L._patient_dir(pid))))
    names = [r.ROIName for r in rs.StructureSetROISequence]
    if dataset == 'MAASTRO':
        rois = ['GTV-1'] + sorted(n for n in names if re.fullmatch(r'GTV-[2-9]\d*', n))
    else:
        lower = {n.lower(): n for n in names}
        rois = list(L.roi_map[str(pid)]) + [lower[n.lower()] for n in nodes_listed if n.lower() in lower]
    m = np.zeros(sitk.GetArrayViewFromImage(ct).shape, dtype=np.uint8)
    for r in dict.fromkeys(rois):
        try:
            m |= rtstruct_mask_by_coordinates(rs, [r], ct)
        except ValueError:
            pass
    mask = sitk.GetImageFromArray(m)
    mask.CopyInformation(ct)
    return ct, mask


def resample(img, interp, spacing=(1.0, 1.0, 1.0)):
    size = [int(round(s * sp / n)) for s, sp, n in zip(img.GetSize(), img.GetSpacing(), spacing)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), spacing, img.GetDirection(),
                         -1024 if interp != sitk.sitkNearestNeighbor else 0, img.GetPixelID())


def one(dataset, pid, nodes_listed):
    ct, mask = total_mask(dataset, pid, nodes_listed)
    ct = resample(sitk.Cast(ct, sitk.sitkFloat32), sitk.sitkLinear)
    mask = resample(mask, sitk.sitkNearestNeighbor)
    arr = sitk.GetArrayFromImage(mask)
    zz, yy, xx = np.where(arr > 0)
    lo = [max(0, int(a.min()) - 10) for a in (xx, yy, zz)]
    hi = [min(s, int(a.max()) + 11) for a, s in zip((xx, yy, zz), ct.GetSize())]
    roi = [h - l for h, l in zip(hi, lo)]
    ct, mask = sitk.RegionOfInterest(ct, roi, lo), sitk.RegionOfInterest(mask, roi, lo)
    r = extractor().execute(ct, mask, label=1)
    return {'patient_id': pid, **{k: float(v) for k, v in r.items() if not k.startswith('diagnostics_')}}


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
        except Exception as e:                                    # noqa: BLE001
            print(f'  {j[1]}: {type(e).__name__}: {e}')
            return {'patient_id': j[1]}
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(safe)(*j) for j in jobs)
    D = pd.DataFrame(rows).set_index('patient_id')
    D.to_csv(OUT)
    print(D.shape, 'missing rows:', int(D.isna().all(axis=1).sum()))


if __name__ == '__main__':
    main()
