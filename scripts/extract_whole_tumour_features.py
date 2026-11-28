"""
Step 4. Radiomic features of the whole tumour (union of the primary and all nodal GTVs), defined
identically in both cohorts: CT resampled to 1 mm isotropic (linear), mask nearest neighbour, cropped to
the mask with a 10 mm margin; PyRadiomics with the study settings, 14 shape + 93 original + 93 x 4 LoG
(sigma 1, 2, 3, 5 mm) = 479 features.

  python scripts/extract_whole_tumour_features.py --n_jobs 4   -> outputs/features/whole_tumour_features.csv
"""
import argparse
import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk

import _paths  # noqa: F401
from hnc_popgraph import config
from hnc_popgraph.data.delineations import nodal_map, whole_tumour_mask
from hnc_popgraph.data.loader import HNSCCDataLoader

OUT = config.WHOLE_TUMOUR_CSV


def extractor():
    from radiomics import featureextractor
    import radiomics
    radiomics.setVerbosity(logging.ERROR)
    ex = featureextractor.RadiomicsFeatureExtractor(**dict(config.PYRADIOMICS_SETTINGS))
    ex.disableAllFeatures()
    ex.enableFeatureClassByName('shape')
    for fc in config.FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    ex.enableImageTypes(Original={}, LoG={'sigma': config.LOG_SIGMAS_MM})
    return ex


def resample(img, interp, spacing=(1.0, 1.0, 1.0)):
    size = [int(round(s * sp / n)) for s, sp, n in zip(img.GetSize(), img.GetSpacing(), spacing)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), spacing, img.GetDirection(),
                         -1024 if interp != sitk.sitkNearestNeighbor else 0, img.GetPixelID())


def one(dataset, pid, nodes_listed):
    ct, mask = whole_tumour_mask(dataset, pid, nodes_listed)
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
    ap.add_argument('--n_jobs', type=int, default=4)
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
