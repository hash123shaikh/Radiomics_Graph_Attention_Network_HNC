r"""
Step 0a. Arrange the public TCIA downloads into the folder layout the pipeline reads (see README).

  HN1 (MAASTRO), TCIA download (NBIA Data Retriever, any nesting of patient/study/series folders):
      python scripts/organize_data.py hn1 --src <download folder> --clinical <HN1 clinical .csv> \
          --out data/MAASTRO
      -> data/MAASTRO/Images/<HN1xxxx>/CT/*.dcm, RTSTRUCT/*.dcm  and  data/MAASTRO/clinical_data.csv

  Head-Neck-PET-CT, full TCIA DICOM download (298 patients; the layout used by the study):
      python scripts/organize_data.py hnpet --src <.../Head-Neck-PET-CT> \
          --clinical <INFOclinical_HN_Version2_30may2018.xlsx> --gtv-list <INFO_GTVcontours_HN.xlsx> \
          --out data/Head-Neck-PET-CT
      -> <out>/Images/HN-<centre>-<nnn>/CT/*.dcm, RTSTRUCT/*.dcm, <out>/gtv_primary_rois.csv
      Per patient: CT = the CT of the PET/CT (the CT series in the structure set's frame of reference;
      the TomoTherapy planning kVCT is not used); RTSTRUCT = 'RTstructCTsimCTPETCT' (GTV on that CT) or,
      when absent, 'RTstructCTsimPETPETCT' (GTV on the PET, same frame of reference). PET, dose,
      plan, registration and other structure sets are not used. gtv_primary_rois.csv lists the
      primary-GTV ROI name(s) from INFO_GTVcontours_HN.xlsx with the exact case found in the RTSTRUCT.

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


def organize_hnpet(src, out, clinical, gtv_list, copy):
    import pandas as pd
    import pydicom
    src, out = Path(src), Path(out)
    series = []                                         # one row per series folder
    for sdir in sorted({f.parent for f in src.rglob('*.dcm')}):
        files = sorted(sdir.glob('*.dcm'))
        d = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        row = dict(dir=sdir, files=files, pid=str(d.PatientID).strip(), mod=str(d.Modality),
                   for_uid=str(getattr(d, 'FrameOfReferenceUID', '')), desc=str(getattr(d, 'SeriesDescription', '')),
                   rois=[])
        if row['mod'] == 'RTSTRUCT':
            row['for_uid'] = str(d.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)
            row['rois'] = [r.ROIName for r in d.StructureSetROISequence]
        series.append(row)
    gl = pd.concat(pd.read_excel(gtv_list, sheet_name=None).values()) if gtv_list else None
    gmap = {} if gl is None else dict(zip(gl.iloc[:, 0].astype(str).str.strip(), gl.iloc[:, 1].astype(str).str.strip()))
    rows, n = [], Counter()
    by_pid = {}
    for r in series:
        by_pid.setdefault(r['pid'], []).append(r)
    for pid, ss in sorted(by_pid.items()):
        if not pid.split('-')[1:2] or pid.split('-')[1] not in HNPET_CENTRES:
            continue
        rss = [r for r in ss if r['mod'] == 'RTSTRUCT' and 'RTstructCTsimCTPETCT' in r['dir'].name] or \
              [r for r in ss if r['mod'] == 'RTSTRUCT' and 'RTstructCTsimPETPETCT' in r['dir'].name]
        # CT of the PET/CT = CT series in the structure set's frame of reference (planning kVCT excluded)
        cts = [r for r in ss if r['mod'] == 'CT' and rss and r['for_uid'] == rss[0]['for_uid']
               and 'kvct' not in (r['desc'] + r['dir'].name).lower()]
        if len(cts) != 1 or len(rss) != 1:
            n['skipped (CT/RTSTRUCT not unique)'] += 1
            print(f"  {pid}: {len(cts)} PET/CT CT series, {len(rss)} GTV structure sets -> skipped")
            continue
        ct, rs = cts[0], rss[0]
        lower = {x.lower(): x for x in rs['rois']}
        want = [w.strip() for w in gmap.get(pid, '').split(',') if w.strip() and w.strip() != 'nan']
        names = [lower.get(w.lower()) for w in want]
        if not want or None in names:
            print(f"  {pid}: primary GTV {want} not found in {rs['rois']} (fix gtv_primary_rois.csv by hand)")
            names = [x for x in names if x]
        for f in ct['files']:
            _place(f, out / 'Images' / pid / 'CT' / f.name, copy)
        for f in rs['files']:
            _place(f, out / 'Images' / pid / 'RTSTRUCT' / f.name, copy)
        n[pid.split('-')[1]] += 1
        rows.append(dict(patient_id=pid, centre=pid.split('-')[1], gtv_primary_rois=';'.join(names),
                         n_ct_slices=len(ct['files']), rtstruct_source=rs['dir'].name.split('-')[1],
                         original_ct_series=f"{ct['dir'].parent.name}/{ct['dir'].name}",
                         original_rtstruct_series=f"{rs['dir'].parent.name}/{rs['dir'].name}"))
    pd.DataFrame(rows).to_csv(out / 'gtv_primary_rois.csv', index=False)
    for f in (clinical, gtv_list):
        if f:
            _place(f, out / Path(f).name, True)
    print(f"[hnpet] {dict(n)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cohort', choices=['hn1', 'hnpet'])
    ap.add_argument('--src', required=True, help='downloaded data folder')
    ap.add_argument('--out', required=True, help='cohort folder to create, e.g. data/MAASTRO')
    ap.add_argument('--clinical', default=None, help='clinical file to place next to Images/')
    ap.add_argument('--gtv-list', default=None, help='hnpet: INFO_GTVcontours_HN.xlsx (primary GTV names)')
    ap.add_argument('--copy', action='store_true', help='copy files instead of symlinking')
    a = ap.parse_args()
    if a.cohort == 'hnpet':
        organize_hnpet(a.src, a.out, a.clinical, a.gtv_list, a.copy)
    else:
        organize_hn1(a.src, a.out, a.clinical, a.copy)


if __name__ == '__main__':
    main()
