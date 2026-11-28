"""
Step 2. PyRadiomics features of the primary GTV from the preprocessed volumes: 14 shape, 93 original,
93 x 4 LoG (sigma 1, 2, 3, 5 mm) and 93 x 8 wavelet features (1223 per patient).

  python scripts/extract_primary_gtv_features.py --n_jobs 4
-> outputs/features/primary_gtv/<pid>.npz and outputs/features/primary_gtv_features.csv
"""
import argparse
import logging
import os

import numpy as np
import SimpleITK as sitk

import _paths  # noqa: F401
from hnc_popgraph import config
from hnc_popgraph.features import primary_gtv_table

OUT = config.PRIMARY_GTV_DIR
VERSION = 1


def extractor():
    from radiomics import featureextractor
    import radiomics
    radiomics.setVerbosity(logging.ERROR)
    ex = featureextractor.RadiomicsFeatureExtractor(**dict(config.PYRADIOMICS_SETTINGS))
    ex.disableAllFeatures()
    ex.enableFeatureClassByName('shape')
    for fc in config.FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    ex.enableImageTypes(Original={}, LoG={'sigma': config.LOG_SIGMAS_MM}, Wavelet={})
    return ex


def one(pid):
    out = OUT / f'{pid}.npz'
    if out.exists():
        try:
            if int(np.load(out)['version']) == VERSION:
                return pid, None
        except Exception:
            pass
    try:
        with np.load(config.PREPROCESSED_DIR / f'{pid}_preprocessed.npz') as d:
            ct, gtv, ds = d['ct_array'], d['gtv_array'], str(d['dataset'])
        img = sitk.GetImageFromArray(ct.astype(np.float32))
        img.SetSpacing((1.0, 1.0, 1.0))
        m = sitk.GetImageFromArray((gtv > 0).astype(np.uint8))
        m.CopyInformation(img)
        r = extractor().execute(img, m, label=1)
        names = [k for k in r if not k.startswith('diagnostics_')]
        vals = np.array([float(r[k]) for k in names])
        tmp = out.with_name(out.name + '.tmp.npz')
        np.savez_compressed(tmp, gtv=np.nan_to_num(vals), gtv_names=np.array(names),
                            dataset=np.array(ds), version=np.array(VERSION))
        os.replace(tmp, out)
        return pid, None
    except Exception as e:
        return pid, f'{type(e).__name__}: {e}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_jobs', type=int, default=4)
    a = ap.parse_args()
    pids = sorted(f.name.replace('_preprocessed.npz', '') for f in config.PREPROCESSED_DIR.glob('*_preprocessed.npz'))
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(p) for p in pids)
    bad = {p: e for p, e in res if e}
    print(f'{len(pids) - len(bad)} ok, {len(bad)} failed', bad)
    print('table:', primary_gtv_table().shape)
