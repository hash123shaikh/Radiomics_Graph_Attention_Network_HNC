"""
GTV-only feature extraction (no peritumoral tissue).

For every patient with a valid preprocessed cache (outputs/cache/preprocessed, profile v4):
  whole GTV : 14 shape features + 93 original + LoG (sigma 1, 2, 3, 5 mm) + wavelet (8 sub-bands)
  sub-GTV   : SLIC (soft-tissue window) INSIDE the GTV mask only -> intratumoral sub-regions,
              93 original features + centroid per sub-region (nodes of the intratumoral graph)
-> outputs/gtv_only/features/<pid>.npz

  python gtv_only/extract.py --n_jobs 6
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grafsgat import config                                         # noqa: E402

OUT = config.OUTPUT_DIR / 'gtv_only' / 'features'
N_SUBREGIONS = 24          # SLIC target inside the GTV
MIN_SUB_VOXELS = 30        # 30 mm^3
VERSION = 1


def _extractor(kind):
    from radiomics import featureextractor
    import radiomics
    radiomics.setVerbosity(logging.ERROR)
    s = dict(config.PYRADIOMICS_SETTINGS)
    ex = featureextractor.RadiomicsFeatureExtractor(**s)
    ex.disableAllFeatures()
    if kind == 'gtv':
        ex.enableFeatureClassByName('shape')
        for fc in config.FEATURE_CLASSES:
            ex.enableFeatureClassByName(fc)
        ex.enableImageTypes(Original={}, LoG={'sigma': [1.0, 2.0, 3.0, 5.0]}, Wavelet={})
    else:
        for fc in config.FEATURE_CLASSES:
            ex.enableFeatureClassByName(fc)
        ex.enableImageTypes(Original={})
    return ex


def _run(ex, img, mask_arr):
    m = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    m.CopyInformation(img)
    r = ex.execute(img, m, label=1)
    names = [k for k in r if not k.startswith('diagnostics_')]
    return names, np.array([float(r[k]) for k in names])


def subregions(ct, gtv):
    from skimage.segmentation import slic
    lo, hi = config.PREPROCESS_PROFILES['v4']['SLIC_HU_WINDOW']
    img = np.clip((ct.astype(np.float32) - lo) / (hi - lo), 0, 1)
    m = gtv > 0
    n = int(np.clip(m.sum() / 400, 6, N_SUBREGIONS))           # >= ~400 voxels per sub-region
    lab = slic(img, mask=m, n_segments=n, slic_zero=True, channel_axis=None,
               enforce_connectivity=True, start_label=1)
    return np.where(m, lab, 0)


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
        gn, gf = _run(_extractor('gtv'), img, gtv > 0)
        lab = subregions(ct, gtv)
        ex = _extractor('sub')
        sf, cents, sn = [], [], None
        for k in range(1, int(lab.max()) + 1):
            mk = lab == k
            if mk.sum() < MIN_SUB_VOXELS:
                continue
            try:
                n2, f = _run(ex, img, mk)
            except Exception:
                continue
            sn = sn or n2
            if n2 != sn:
                continue
            sf.append(f); cents.append(np.argwhere(mk).mean(0))
        if len(sf) < 2:
            raise RuntimeError(f'only {len(sf)} sub-regions')
        tmp = out.with_name(out.name + '.tmp.npz')
        np.savez_compressed(tmp, gtv=np.nan_to_num(gf), gtv_names=np.array(gn),
                            sub=np.nan_to_num(np.array(sf)), sub_names=np.array(sn),
                            sub_centroids=np.array(cents), gtv_centroid=np.argwhere(gtv > 0).mean(0),
                            dataset=np.array(ds), version=np.array(VERSION))
        os.replace(tmp, out)
        return pid, None
    except Exception as e:
        return pid, f'{type(e).__name__}: {e}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_jobs', type=int, default=6)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pids = sorted(f.name.replace('_preprocessed.npz', '') for f in config.PREPROCESSED_DIR.glob('*_preprocessed.npz'))
    pids = pids[:a.limit] if a.limit else pids
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(p) for p in pids)
    bad = {p: e for p, e in res if e}
    print(f'{len(pids) - len(bad)} ok, {len(bad)} failed', bad)
