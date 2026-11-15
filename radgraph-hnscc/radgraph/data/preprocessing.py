"""
CT preprocessing for RadGraph:
  1. resample CT (linear) and GTV mask (nearest) to 1 mm isotropic on the SAME grid
  2. define the peritumoral region = GTV bounding box + 5 cm margin (paper)
  3. optionally crop everything to that region (config.CROP_TO_REGION)
"""

import numpy as np
import SimpleITK as sitk

from radgraph import config


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

    def resample_image(self, image, reference, is_mask=False):
        """Resample onto `reference` grid. Masks: nearest neighbour, background 0. CT: linear, -1000 HU."""
        if is_mask:
            return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkNearestNeighbor,
                                 0, sitk.sitkUInt8)
        return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear,
                             -1000.0, sitk.sitkFloat32)

    def define_peritumoral_region(self, gtv_mask, margin_mm=None):
        """Box = GTV bounding box expanded by margin_mm on every side (clipped to the image)."""
        margin_mm = config.PERITUMORAL_MARGIN_MM if margin_mm is None else margin_mm
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(gtv_mask)
        if 1 not in stats.GetLabels():
            raise ValueError("GTV mask is empty")
        x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(1)
        sp, size = gtv_mask.GetSpacing(), gtv_mask.GetSize()
        m = [int(np.ceil(margin_mm / sp[i])) for i in range(3)]
        lo = [max(0, x0 - m[0]), max(0, y0 - m[1]), max(0, z0 - m[2])]
        hi = [min(size[0], x0 + sx + m[0]), min(size[1], y0 + sy + m[1]), min(size[2], z0 + sz + m[2])]
        return (lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])   # (x,y,z,sx,sy,sz)

    @staticmethod
    def crop_to_region(image, bbox):
        return sitk.RegionOfInterest(image, size=[int(b) for b in bbox[3:]],
                                     index=[int(b) for b in bbox[:3]])

    def preprocess_patient(self, ct_image, gtv_mask, crop=None):
        crop = config.CROP_TO_REGION if crop is None else crop

        ref    = self._reference_grid(ct_image)
        ct_rs  = self.resample_image(ct_image, ref, is_mask=False)
        gtv_rs = self.resample_image(gtv_mask, ref, is_mask=True)
        if sitk.GetArrayViewFromImage(gtv_rs).sum() == 0:
            raise ValueError("GTV vanished after resampling")

        bbox = self.define_peritumoral_region(gtv_rs)
        size = gtv_rs.GetSize()

        if crop:
            ct_out  = self.crop_to_region(ct_rs, bbox)
            gtv_out = self.crop_to_region(gtv_rs, bbox)
            region  = np.ones(sitk.GetArrayViewFromImage(gtv_out).shape, dtype=np.uint8)
        else:
            ct_out, gtv_out = ct_rs, gtv_rs
            region = np.zeros(sitk.GetArrayViewFromImage(gtv_rs).shape, dtype=np.uint8)
            x, y, z, sx, sy, sz = bbox
            region[z:z + sz, y:y + sy, x:x + sx] = 1

        ct_arr = np.clip(np.round(sitk.GetArrayFromImage(ct_out)), -32768, 32767).astype(np.int16)
        return {
            'ct_array'    : ct_arr,                                         # HU, (Z, Y, X)
            'gtv_array'   : (sitk.GetArrayFromImage(gtv_out) > 0).astype(np.uint8),
            'region_array': region,
            'bbox'        : np.array(bbox, dtype=np.int32),
            'full_size'   : np.array(size, dtype=np.int32),
            'origin'      : np.array(ct_out.GetOrigin(), dtype=np.float64),  # physical origin of stored grid
            'direction'   : np.array(ct_out.GetDirection(), dtype=np.float64),
            'spacing'     : np.array(self.target_spacing, dtype=np.float32),
            'cropped'     : bool(crop),
        }


if __name__ == '__main__':
    from radgraph.data.loader import HNSCCDataLoader
    dl = HNSCCDataLoader('MAASTRO')
    pid = dl.eligible_patients('LR')[0]
    d = dl.load_patient_data(pid)
    p = CTPreprocessor().preprocess_patient(d['ct_image'], d['gtv_mask'])
    print(pid, d['gtv_name'], p['ct_array'].shape, 'GTV voxels', int(p['gtv_array'].sum()))
