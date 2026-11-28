"""
CT preprocessing for the primary-GTV features:
  1. resample CT (linear) and GTV mask (nearest neighbour) to 1 mm isotropic on the same grid
  2. crop both to the GTV bounding box + config.CROP_MARGIN_MM on every side
-> outputs/preprocessed/<pid>_preprocessed.npz   (ct_array in HU and gtv_array, (Z, Y, X))
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from hnc_popgraph import config


class CTPreprocessor:

    def __init__(self, target_spacing=None):
        self.target_spacing = tuple(float(s) for s in (target_spacing or config.TARGET_SPACING))

    def _reference_grid(self, image):
        sp, sz = image.GetSpacing(), image.GetSize()
        new_size = [max(1, int(round(sz[i] * sp[i] / self.target_spacing[i]))) for i in range(3)]
        ref = sitk.Image(new_size, sitk.sitkUInt8)
        ref.SetSpacing(self.target_spacing)
        ref.SetOrigin(image.GetOrigin())
        ref.SetDirection(image.GetDirection())
        return ref

    @staticmethod
    def resample_image(image, reference, is_mask=False):
        """Masks: nearest neighbour, background 0. CT: linear, -1000 HU outside the image."""
        if is_mask:
            return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)

    @staticmethod
    def crop_box(gtv_mask, margin_mm):
        """GTV bounding box expanded by margin_mm on every side (clipped to the image): (x, y, z, sx, sy, sz)."""
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(gtv_mask)
        if 1 not in stats.GetLabels():
            raise ValueError('GTV mask is empty')
        x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(1)
        sp, size = gtv_mask.GetSpacing(), gtv_mask.GetSize()
        m = [int(np.ceil(margin_mm / sp[i])) for i in range(3)]
        lo = [max(0, x0 - m[0]), max(0, y0 - m[1]), max(0, z0 - m[2])]
        hi = [min(size[0], x0 + sx + m[0]), min(size[1], y0 + sy + m[1]), min(size[2], z0 + sz + m[2])]
        return (lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

    def preprocess_patient(self, ct_image, gtv_mask):
        ref = self._reference_grid(ct_image)
        ct_rs = self.resample_image(ct_image, ref, is_mask=False)
        gtv_rs = self.resample_image(gtv_mask, ref, is_mask=True)
        if sitk.GetArrayViewFromImage(gtv_rs).sum() == 0:
            raise ValueError('GTV vanished after resampling')
        bbox = self.crop_box(gtv_rs, config.CROP_MARGIN_MM)
        size, index = [int(b) for b in bbox[3:]], [int(b) for b in bbox[:3]]
        ct_out = sitk.RegionOfInterest(ct_rs, size=size, index=index)
        gtv_out = sitk.RegionOfInterest(gtv_rs, size=size, index=index)
        return {
            'ct_array' : np.clip(np.round(sitk.GetArrayFromImage(ct_out)), -32768, 32767).astype(np.int16),
            'gtv_array': (sitk.GetArrayFromImage(gtv_out) > 0).astype(np.uint8),
            'bbox'     : np.array(bbox, dtype=np.int32),
            'full_size': np.array(gtv_rs.GetSize(), dtype=np.int32),
            'origin'   : np.array(ct_out.GetOrigin(), dtype=np.float64),
            'direction': np.array(ct_out.GetDirection(), dtype=np.float64),
            'spacing'  : np.array(self.target_spacing, dtype=np.float32),
        }


def preprocessed_path(pid):
    return config.PREPROCESSED_DIR / f'{pid}_preprocessed.npz'


def is_preprocessed(pid):
    """A valid preprocessed file exists (current version and mask rule)."""
    f = preprocessed_path(pid)
    if not f.exists():
        return False
    try:
        with np.load(f, allow_pickle=False) as d:
            return (int(d['version']) == config.PREPROCESS_VERSION and str(d['mask_rule']) == config.MASK_RULE)
    except Exception:
        return False


def _save(path, **arrays):
    tmp = Path(path).with_name(Path(path).name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def preprocess_one(pid, dataset):
    from hnc_popgraph.data.loader import HNSCCDataLoader
    try:
        d = HNSCCDataLoader(dataset).load_patient_data(pid)
        p = CTPreprocessor().preprocess_patient(d['ct_image'], d['gtv_mask'])
        _save(preprocessed_path(pid), **p, cropped=np.array(True), gtv_name=np.array(d['gtv_name']),
              dataset=np.array(dataset), version=np.array(config.PREPROCESS_VERSION),
              mask_rule=np.array(config.MASK_RULE), created=np.array(time.strftime('%Y%m%dT%H%M%S') + f'-{pid}'))
        return pid, None
    except Exception as e:
        return pid, f'{type(e).__name__}: {e}'


def preprocess_cohort(dataset, patient_ids, force=False, n_jobs=1):
    todo = [p for p in patient_ids if force or not is_preprocessed(p)]
    print(f'[preprocess] {dataset}: {len(patient_ids) - len(todo)} done before, {len(todo)} to do')
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(preprocess_one)(p, dataset) for p in todo)
    failed = {p: e for p, e in res if e}
    for p, e in failed.items():
        print(f'  FAILED {p}: {e}')
    if failed:
        pd.Series(failed, name='error').to_csv(config.OUTPUT_DIR / f'preprocess_failed_{dataset}.csv')
    return failed
