"""
Image loading for RadGraph: CT + primary-tumour GTV mask.

Supported layouts (auto-detected per patient):
  NIfTI (HNPET) : <images_dir>/<pid>/<pid>__CT.nii.gz  and  <pid>.nii.gz  (label 1 = GTVp)
  MAASTRO (DICOM) : <images_dir>/<pid>/CT/*.dcm          and  <pid>/RTSTRUCT/*.dcm
"""

from pathlib import Path
import warnings

import numpy as np
import SimpleITK as sitk
import pydicom

from radgraph import config
from radgraph.data.clinical import compute_labels

warnings.filterwarnings('ignore', category=UserWarning)

try:
    from rt_utils import RTStructBuilder
    HAS_RT_UTILS = True
except ImportError:
    HAS_RT_UTILS = False


def select_gtv_roi(roi_names):
    """
    Pick the PRIMARY tumour GTV from a list of RTSTRUCT ROI names.

    1. exact (case-insensitive) match in config.GTV_PRIMARY_NAMES priority order
    2. otherwise the first name containing 'gtv' and none of config.GTV_EXCLUDE_TOKENS
    Never returns CTV/PTV or nodal GTVs. Returns None if nothing suitable exists.
    """
    lower = {n.strip().lower(): n for n in roi_names}
    for cand in config.GTV_PRIMARY_NAMES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for n in roi_names:
        nl = n.strip().lower()
        if 'gtv' in nl and not any(tok in nl for tok in config.GTV_EXCLUDE_TOKENS):
            return n
    return None


class HNSCCDataLoader:
    """Loads CT and GTV for one cohort (config.DATASETS[dataset])."""

    def __init__(self, dataset):
        self.dataset    = dataset
        self.cfg        = config.dataset_cfg(dataset)
        self.images_dir = Path(self.cfg['images_dir'])

    # ── patients ────────────────────────────────────────────────────────────

    def eligible_patients(self, task):
        """Patients with a valid 2-year label for `task` AND an image folder."""
        lab = compute_labels(self.dataset, task)
        ids = [p for p in lab.index[lab['included']]]
        have = [p for p in ids if self._patient_dir(p) is not None]
        if len(have) < len(ids):
            print(f"[data] {self.dataset}: {len(ids) - len(have)} labelled patients have no image folder")
        return have

    def eligible_patients_any_task(self):
        """Union over all tasks available for this cohort (preprocessing is task independent)."""
        ids = set()
        for task in self.cfg['outcomes']:
            ids |= set(self.eligible_patients(task))
        return sorted(ids)

    def _patient_dir(self, pid):
        for name in (str(pid), f"Patient_{pid}", f"patient_{pid}"):
            d = self.images_dir / name
            if d.is_dir():
                return d
        return None

    def _is_nifti(self, pdir):
        return any(pdir.glob('*.nii.gz'))

    # ── CT ──────────────────────────────────────────────────────────────────

    def load_ct_scan(self, pid):
        pdir = self._patient_dir(pid)
        if pdir is None:
            raise FileNotFoundError(f"Patient directory not found for {pid} in {self.images_dir}")

        if self._is_nifti(pdir):
            ct = pdir / f"{pid}__CT.nii.gz"
            if not ct.exists():
                hits = sorted(pdir.glob('*__CT.nii.gz'))
                if not hits:
                    raise FileNotFoundError(f"No *__CT.nii.gz for {pid}")
                ct = hits[0]
            return sitk.ReadImage(str(ct))

        ct_dir = pdir / 'CT' if (pdir / 'CT').is_dir() else pdir
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(ct_dir))
        if not series:
            raise FileNotFoundError(f"No DICOM CT series for {pid} in {ct_dir}")
        # take the series with the most slices (guards against scouts / extra series)
        files = max((reader.GetGDCMSeriesFileNames(str(ct_dir), s) for s in series), key=len)
        reader.SetFileNames(files)
        return reader.Execute()

    # ── GTV ─────────────────────────────────────────────────────────────────

    def _find_rtstruct(self, pid, pdir):
        cands = list((pdir / 'RTSTRUCT').glob('*.dcm')) if (pdir / 'RTSTRUCT').is_dir() else []
        cands += [f for f in pdir.glob('*.dcm')]
        for f in cands:
            try:
                if pydicom.dcmread(str(f), stop_before_pixels=True).Modality == 'RTSTRUCT':
                    return f
            except Exception:
                continue
        raise FileNotFoundError(f"RTSTRUCT not found for {pid}")

    def extract_gtv_mask(self, pid, ct_image):
        """Returns (mask sitk.Image on the CT grid, roi name)."""
        pdir = self._patient_dir(pid)

        if self._is_nifti(pdir):
            seg = pdir / f"{pid}.nii.gz"
            if not seg.exists():
                cands = [f for f in pdir.glob('*.nii.gz') if '__CT' not in f.name and '__PT' not in f.name]
                if not cands:
                    raise FileNotFoundError(f"Segmentation not found for {pid}")
                seg = cands[0]
            seg_img = sitk.ReadImage(str(seg))
            seg_img = sitk.Resample(seg_img, ct_image, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            lab = config.NIFTI_GTVP_LABEL
            mask = sitk.BinaryThreshold(seg_img, lab, lab, 1, 0)
            return mask, f'label{lab}(GTVp)'

        if not HAS_RT_UTILS:
            raise ImportError("rt-utils is required for DICOM RTSTRUCT (pip install rt-utils==1.2.7)")
        rt_file = self._find_rtstruct(pid, pdir)
        ct_dir  = pdir / 'CT' if (pdir / 'CT').is_dir() else pdir
        rt      = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rt_file))
        names   = rt.get_roi_names()
        gtv     = select_gtv_roi(names)
        if gtv is None:
            raise ValueError(f"No primary GTV ROI for {pid}. ROIs: {names}")
        arr = rt.get_roi_mask_by_name(gtv)                       # (rows, cols, slices) = (Y, X, Z)
        arr = np.transpose(arr, (2, 0, 1)).astype(np.uint8)       # -> (Z, Y, X) for SimpleITK
        # rt-utils orders slices by its own sorting; make sure it matches the CT
        if arr.shape != sitk.GetArrayFromImage(ct_image).shape:
            raise ValueError(f"{pid}: GTV mask shape {arr.shape} != CT shape "
                             f"{sitk.GetArrayFromImage(ct_image).shape}")
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(ct_image)
        return mask, gtv

    def load_patient_data(self, pid):
        ct = self.load_ct_scan(pid)
        mask, name = self.extract_gtv_mask(pid, ct)
        if sitk.GetArrayViewFromImage(mask).sum() == 0:
            raise ValueError(f"{pid}: GTV mask '{name}' is empty")
        return {'patient_id': pid, 'ct_image': ct, 'gtv_mask': mask, 'gtv_name': name}


if __name__ == '__main__':
    for ds in config.DATASETS:
        dl = HNSCCDataLoader(ds)
        ids = dl.eligible_patients_any_task()
        print(ds, len(ids), 'eligible; first:', ids[:3])
        d = dl.load_patient_data(ids[0])
        print('  CT', d['ct_image'].GetSize(), d['ct_image'].GetSpacing(), 'GTV', d['gtv_name'],
              int(sitk.GetArrayViewFromImage(d['gtv_mask']).sum()), 'voxels')
