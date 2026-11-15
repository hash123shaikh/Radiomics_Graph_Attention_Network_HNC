"""
Supervoxels with zero-parameter SLIC (SLICO), following the authors' RadiomicSupervoxels.py:
  - SLIC runs only inside the peritumoral box with the GTV removed
  - the GTV itself is a separate node (not segmented)

Refinements (PREPROCESS_VERSION 4, see config):
  - body mask: the box usually extends beyond the skin; supervoxels are restricted to the patient body
    (CT > BODY_HU_THRESHOLD, largest 3-D connected component, holes filled slice by slice, so the
    airway lumen stays inside the body while air outside the skin and the couch are excluded)
  - soft-tissue window: the CT is clipped to SLIC_HU_WINDOW and scaled to [0, 1] with fixed bounds,
    so SLIC responds to soft-tissue interfaces instead of air/bone boundaries
Radiomic features are still computed on the raw HU values; only the supervoxel partition changes.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic

from radgraph import config


def body_mask(ct_array, threshold=None):
    """Patient body inside the (cropped) CT: threshold, largest 3-D component, per-slice hole filling."""
    threshold = config.BODY_HU_THRESHOLD if threshold is None else threshold
    m = ct_array > threshold
    m = ndi.binary_opening(m, structure=np.ones((1, 3, 3), bool))       # detach thin couch/blanket contacts
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(m)
    sizes = ndi.sum(m, lab, index=range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    for z in range(m.shape[0]):                                         # airway lumen counts as body
        m[z] = ndi.binary_fill_holes(m[z])
    return m


def slic_intensity(ct_array, peri):
    """Intensity image given to SLIC: soft-tissue window (fixed bounds) or legacy min-max over the region."""
    ct = ct_array.astype(np.float32)
    if config.SLIC_HU_WINDOW is not None:
        lo, hi = (float(v) for v in config.SLIC_HU_WINDOW)
    else:
        vals = ct[peri]
        lo, hi = float(vals.min()), float(vals.max())
    return np.clip((ct - lo) / (hi - lo + 1e-6), 0.0, 1.0)


class SupervoxelGenerator:

    def __init__(self, n_segments=None):
        self.n_segments = n_segments or config.N_SUPERVOXELS_TARGET

    def peritumoral_mask(self, ct_array, region_mask, gtv_array):
        peri = (region_mask > 0) & (gtv_array == 0)
        if config.USE_BODY_MASK:
            peri &= body_mask(ct_array) | (gtv_array > 0)
        return peri

    def generate_supervoxels(self, ct_array, region_mask, gtv_array):
        """
        Returns
        -------
        labels : int32 array (Z, Y, X); 0 = outside the peritumoral body region or GTV, 1..n = supervoxels
        n      : number of supervoxels
        """
        peri = self.peritumoral_mask(ct_array, region_mask, gtv_array)
        if not peri.any():
            raise ValueError("Peritumoral region is empty")

        img = slic_intensity(ct_array, peri)
        labels = slic(img, mask=peri, n_segments=self.n_segments, slic_zero=True,
                      channel_axis=None, enforce_connectivity=True, start_label=1)
        labels = np.where(peri, labels, 0)

        # renumber to 1..n
        uniq, inv = np.unique(labels, return_inverse=True)
        inv = inv.reshape(labels.shape)
        if uniq[0] == 0:
            labels = inv.astype(np.int32)                 # 0 stays 0, others become 1..n
        else:
            labels = (inv + 1).astype(np.int32)
        n = int(labels.max())
        if n < config.N_SUPERVOXELS_SELECTED:
            print(f"  WARNING: only {n} supervoxels (< {config.N_SUPERVOXELS_SELECTED} needed for a full graph)")
        return labels, n
