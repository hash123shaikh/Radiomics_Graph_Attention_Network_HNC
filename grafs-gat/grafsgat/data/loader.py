"""
Image loading for GRAFS-GAT: CT + primary-tumour GTV mask.

Supported layouts (auto-detected per patient):
  DICOM (MAASTRO, Head-Neck-PET-CT) : <images_dir>/<pid>/CT/*.dcm  and  <pid>/RTSTRUCT/*.dcm
                                      (Head-Neck-PET-CT folders are named HN-<pid>, e.g. HN-CHUM-001)
  NIfTI (legacy HECKTOR release)    : <images_dir>/<pid>/<pid>__CT.nii.gz  and  <pid>.nii.gz  (label 1 = GTVp)

Primary GTV in an RTSTRUCT:
  * if the cohort has a ROI map (config 'gtv_roi_map', Head-Neck-PET-CT), the listed ROI(s) are used
    (union when several) and contours are rasterised from their PHYSICAL coordinates onto the CT grid.
    This is required because 91 Head-Neck-PET-CT structure sets were drawn on the PET of the PET/CT
    (same frame of reference as the CT, but contour slices reference PET images, which rt-utils
    cannot map to CT slices).
  * otherwise (MAASTRO) the ROI is chosen by name (select_gtv_roi).
  Both cohorts use the same rasteriser (pixel-centre-inside rule), so train and test masks follow one
  convention. (Up to v4 the MAASTRO masks came from rt-utils, which also fills pixels the contour
  line only touches: ~10% larger masks, Dice ~0.94 with the present rule.)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from skimage.draw import polygon as _fill_polygon

from grafsgat import config
from grafsgat.data.clinical import compute_labels

warnings.filterwarnings('ignore', category=UserWarning)



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


def load_gtv_roi_map(path):
    """{patient_id: [ROI names]} from a gtv_primary_rois.csv (patient_id may carry the 'HN-' prefix)."""
    if path is None or not Path(path).is_file():
        return {}
    df = pd.read_csv(path, dtype=str).fillna('')
    out = {}
    for pid, rois in zip(df['patient_id'], df['gtv_primary_rois']):
        names = [r.strip() for r in rois.split(';') if r.strip()]
        if names:
            pid = pid.strip()
            out[pid] = names
            out[pid[3:] if pid.startswith('HN-') else f'HN-{pid}'] = names
    return out


def rtstruct_mask_by_coordinates(rs, roi_names, ct_image):
    """
    Rasterise RTSTRUCT contours of `roi_names` (union) onto the grid of `ct_image`, using the
    contour points' physical coordinates only (no reliance on referenced SOP instances).

    A pixel belongs to a contour if its CENTRE lies inside the polygon. Each CT slice takes the
    contours of the nearest contour plane if that plane lies within half a contour-plane spacing; a
    slice inside the ROI's extent whose plane was skipped by the contouring (gap of ONE plane, e.g.
    HN-HGJ-003) takes the nearest plane, while larger gaps (separate lesions) are left empty.
    Contours on one plane are combined with XOR (holes stay holes), planes and ROIs with OR.
    Returns a uint8 array (Z, Y, X).
    """
    nx, ny, nz = ct_image.GetSize()
    num = {int(r.ROINumber): r.ROIName for r in rs.StructureSetROISequence}
    wanted = {n.lower() for n in roi_names}
    planes = {}                                               # ROI -> {z_mm: [polygons (N,2) in index x,y]}
    for rc in rs.ROIContourSequence:
        name = num.get(int(rc.ReferencedROINumber), '')
        if name.lower() not in wanted or not getattr(rc, 'ContourSequence', None):
            continue
        per = planes.setdefault(name, {})
        for c in rc.ContourSequence:
            pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            if len(pts) < 3:
                continue
            idx = np.array([ct_image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
            per.setdefault(round(float(np.median(idx[:, 2])), 3), []).append(idx[:, :2])
    missing = wanted - {n.lower() for n in planes}
    if missing:
        raise ValueError(f"ROI(s) without contours: {sorted(missing)}; available: {sorted(num.values())}")
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    for per in planes.values():
        zs = np.array(sorted(per))
        step = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
        for k in range(nz):
            j = int(np.argmin(np.abs(zs - k)))
            dist = abs(zs[j] - k)
            if dist > step / 2 + 1e-3:
                inside = zs[0] < k < zs[-1]
                lo, hi = zs[zs < k].max(initial=-np.inf), zs[zs > k].min(initial=np.inf)
                if not (inside and hi - lo <= 2 * step + 1e-3):
                    continue
            sl = np.zeros((ny, nx), dtype=np.uint8)
            for poly in per[zs[j]]:
                rr, cc = _fill_polygon(poly[:, 1], poly[:, 0], shape=(ny, nx))
                tmp = np.zeros_like(sl); tmp[rr, cc] = 1
                sl ^= tmp
            mask[k] |= sl
    return mask


def _fix_nonuniform_z(img, files, pid=''):
    """
    SimpleITK assumes equally spaced slices. If a series has a missing slice (e.g. HN-CHUM-001/002,
    one 3.27 mm slice absent in the TCIA release), every slice after the gap would be misplaced.
    Rebuild such a series on a uniform z grid (median spacing) with linear interpolation along z.
    Only axial series are handled (all cohort CTs are axial).
    """
    z_files = np.array([float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2])
                        for f in files])                    # slice positions in array order
    order = np.argsort(z_files)
    z = z_files[order]
    dz = np.diff(z)
    if len(z) < 3 or np.abs(dz - np.median(dz)).max() < 0.01 * abs(np.median(dz)):
        return img
    if not np.allclose(np.array(img.GetDirection()), np.eye(3).ravel(), atol=1e-3):
        raise ValueError(f"{pid}: non-uniform slice spacing in a non-axial CT series")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)[order]
    step = float(np.median(dz))
    zt = z[0] + step * np.arange(int(round((z[-1] - z[0]) / step)) + 1)
    hi = np.clip(np.searchsorted(z, zt), 1, len(z) - 1)
    w = ((zt - z[hi - 1]) / (z[hi] - z[hi - 1]))[:, None, None]
    out = (1 - w) * arr[hi - 1] + w * arr[hi]
    new = sitk.GetImageFromArray(np.round(out).astype(np.int16))
    new.SetSpacing((img.GetSpacing()[0], img.GetSpacing()[1], step))
    new.SetOrigin((img.GetOrigin()[0], img.GetOrigin()[1], float(z[0])))
    new.SetDirection(img.GetDirection())
    print(f"[data] {pid}: non-uniform slice spacing (max gap {dz.max():.2f} mm) -> resampled to "
          f"{len(zt)} slices at {step:.2f} mm")
    return new


class HNSCCDataLoader:
    """Loads CT and GTV for one cohort (config.DATASETS[dataset])."""

    def __init__(self, dataset):
        self.dataset    = dataset
        self.cfg        = config.dataset_cfg(dataset)
        self.images_dir = Path(self.cfg['images_dir'])
        self.roi_map    = load_gtv_roi_map(self.cfg.get('gtv_roi_map'))

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
        for name in (str(pid), f"HN-{pid}", f"Patient_{pid}", f"patient_{pid}"):
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
        img = reader.Execute()
        return _fix_nonuniform_z(img, files, pid)

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

        ct_dir = pdir / 'CT' if (pdir / 'CT').is_dir() else pdir
        rs = pydicom.dcmread(str(self._find_rtstruct(pid, pdir)))
        if self.roi_map:                                   # Head-Neck-PET-CT: names from TCIA list
            names = self.roi_map.get(str(pid))
            if not names:
                raise ValueError(f"{pid}: no entry in GTV ROI map {self.cfg.get('gtv_roi_map')}")
        else:                                              # MAASTRO: choose by name
            all_names = [r.ROIName for r in rs.StructureSetROISequence]
            gtv = select_gtv_roi(all_names)
            if gtv is None:
                raise ValueError(f"No primary GTV ROI for {pid}. ROIs: {all_names}")
            names = [gtv]
        ct_for = pydicom.dcmread(str(next(iter(sorted(ct_dir.glob('*.dcm'))))),
                                 stop_before_pixels=True).FrameOfReferenceUID
        rs_for = {f.FrameOfReferenceUID for f in getattr(rs, 'ReferencedFrameOfReferenceSequence', [])}
        if rs_for and ct_for not in rs_for:
            raise ValueError(f"{pid}: RTSTRUCT frame of reference does not match the CT")
        arr = rtstruct_mask_by_coordinates(rs, names, ct_image)
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(ct_image)
        return mask, '+'.join(names)

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
