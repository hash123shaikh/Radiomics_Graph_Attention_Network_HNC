#!/usr/bin/env python3
"""
Arrange the public downloads into the folder layout the pipeline reads (see DATA.md).

  HN1 (MAASTRO), TCIA download (NBIA Data Retriever, any nesting of patient/study/series folders):
      python tools/organize_data.py hn1 --src <download folder> --clinical <HN1 clinical .csv> \
          --out data/MAASTRO
      -> data/MAASTRO/Images/<HN1xxxx>/CT/*.dcm, RTSTRUCT/*.dcm  and  data/MAASTRO/clinical_data.csv

  Head-Neck-PET-CT, HECKTOR NIfTI release (imagesTr/ + labelsTr/, or any folder holding
  <ID>__CT.nii.gz and <ID>.nii.gz files):
      python tools/organize_data.py hecktor --src <HECKTOR folder> \
          --clinical <INFOclinical_HN_Version2_30may2018.xlsx> --out "data/HECKTOR 2026 Training Data"
      -> <out>/Images/<ID>/<ID>__CT.nii.gz, <ID>.nii.gz  and  <out>/INFOclinical_HN_Version2_30may2018.xlsx
      Only the four Head-Neck-PET-CT centres (CHUM, CHUS, HGJ, HMR) are kept.

Files are symlinked by default (no extra disk space); use --copy to copy them instead.
Nothing in --src is modified.
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

HNPET_CENTRES = ('CHUM', 'CHUS', 'HGJ', 'HMR')


def _place(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(Path(src).resolve(), dst)


def organize_hn1(src, out, clinical, copy):
    import pydicom
    n = Counter()
    for f in sorted(Path(src).rglob('*.dcm')):
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=['PatientID', 'Modality'])
        except Exception:
            n['unreadable'] += 1
            continue
        mod = str(getattr(d, 'Modality', ''))
        if mod not in ('CT', 'RTSTRUCT'):
            n[f'skipped {mod or "?"}'] += 1
            continue
        pid = str(d.PatientID).strip()
        sub = Path(out) / 'Images' / pid / mod
        # keep series apart if a patient has several (the loader uses the longest CT series)
        name = f.name if not (sub / f.name).exists() else f"{f.parent.name}_{f.name}"
        _place(f, sub / name, copy)
        n[mod] += 1
    if clinical:
        _place(clinical, Path(out) / 'clinical_data.csv', True)
    pts = [p for p in (Path(out) / 'Images').iterdir() if p.is_dir()] if (Path(out) / 'Images').exists() else []
    print(f"[hn1] {len(pts)} patients | {dict(n)} -> {out}")
    missing = [p.name for p in pts if not (p / 'RTSTRUCT').is_dir() or not (p / 'CT').is_dir()]
    if missing:
        print(f"  WARNING: {len(missing)} patients lack CT or RTSTRUCT: {missing[:10]}")


def organize_hecktor(src, out, clinical, copy):
    src = Path(src)
    cts = {f.name[:-len('__CT.nii.gz')]: f for f in src.rglob('*__CT.nii.gz')}
    n = Counter()
    for pid, ct in sorted(cts.items()):
        if not pid.startswith(HNPET_CENTRES):
            n['other centre'] += 1
            continue
        segs = [s for s in src.rglob(f'{pid}.nii.gz')]
        if not segs:
            n['no segmentation'] += 1
            print(f"  {pid}: no {pid}.nii.gz segmentation found, skipped")
            continue
        d = Path(out) / 'Images' / pid
        _place(ct, d / f'{pid}__CT.nii.gz', copy)
        _place(segs[0], d / f'{pid}.nii.gz', copy)
        n[pid.split('-')[0]] += 1
    if clinical:
        _place(clinical, Path(out) / Path(clinical).name, True)
    print(f"[hecktor] {dict(n)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cohort', choices=['hn1', 'hecktor'])
    ap.add_argument('--src', required=True, help='downloaded data folder')
    ap.add_argument('--out', required=True, help='cohort folder to create (see DATA.md)')
    ap.add_argument('--clinical', default=None, help='clinical file to place next to Images/')
    ap.add_argument('--copy', action='store_true', help='copy files instead of symlinking')
    a = ap.parse_args()
    (organize_hn1 if a.cohort == 'hn1' else organize_hecktor)(a.src, a.out, a.clinical, a.copy)


if __name__ == '__main__':
    main()
