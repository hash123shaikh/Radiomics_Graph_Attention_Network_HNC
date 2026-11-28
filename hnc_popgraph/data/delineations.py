"""
Primary and nodal GTV structures of each patient.

  Head-Neck-PET-CT  primary = ROI(s) of gtv_primary_rois.csv; nodal = 'Name GTV Lymph Nodes' of the TCIA
                    list INFO_GTVcontours_HN.xlsx (comma separated; none listed -> no nodal GTV)
  MAASTRO (HN1)     primary = GTV-1; nodal = GTV-2 ... GTV-n
"""
import re

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from hnc_popgraph import config
from hnc_popgraph.data.loader import HNSCCDataLoader, rtstruct_mask_by_coordinates


def nodal_map():
    """{Head-Neck-PET-CT patient id: [nodal GTV ROI names]}"""
    x = pd.concat(pd.read_excel(config.HNPET_DIR / 'INFO_GTVcontours_HN.xlsx', sheet_name=None).values())
    return {p.replace('HN-', ''): ([s.strip() for s in str(n).split(',') if s.strip()] if pd.notna(n) else [])
            for p, n in zip(x['Patient'], x['Name GTV Lymph Nodes'])}


def whole_tumour_mask(dataset, pid, nodes_listed):
    """CT image and the union of the primary and nodal GTV masks on the CT grid."""
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
