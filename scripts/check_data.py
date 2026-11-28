"""
Step 0b. Check the installation and the data before running the study.

Verifies that the required Python packages import, that both cohort folders contain the expected files,
and that the number of patients and events per endpoint equals that of the paper.

  python scripts/check_data.py
"""
import importlib
import sys

import _paths  # noqa: F401
from hnc_popgraph import config

# labelled patients (events) per endpoint, as in the paper (Table 1)
EXPECTED = {('HNPET', 'LR'): (276, 30), ('HNPET', 'DM'): (276, 27), ('HNPET', 'OS'): (164, 44),
            ('MAASTRO', 'LR'): (121, 30), ('MAASTRO', 'DM'): (109, 7), ('MAASTRO', 'OS'): (135, 45)}
PACKAGES = ['numpy', 'pandas', 'sklearn', 'scipy', 'torch', 'torch_geometric', 'SimpleITK', 'pydicom',
            'radiomics', 'skimage', 'joblib', 'openpyxl', 'matplotlib']


def ok(flag, text):
    print(f"  [{'OK' if flag else 'MISSING'}] {text}")
    return flag


def main():
    good = True
    print('Python packages')
    for p in PACKAGES:
        try:
            importlib.import_module(p)
            good &= ok(True, p)
        except Exception as e:                                   # noqa: BLE001
            good &= ok(False, f'{p} ({type(e).__name__})')

    print(f'\nData folder: {config.DATA_DIR}')
    files = {'HNPET': [config.HNPET_DIR / 'Images', config.DATASETS['HNPET']['clinical_file'],
                       config.HNPET_DIR / 'INFO_GTVcontours_HN.xlsx', config.DATASETS['HNPET']['gtv_roi_map']],
             'MAASTRO': [config.MAASTRO_DIR / 'Images', config.DATASETS['MAASTRO']['clinical_file']]}
    for ds, fs in files.items():
        for f in fs:
            good &= ok(f.exists(), f)
        img = config.dataset_cfg(ds)['images_dir']
        if img.exists():
            pts = [p for p in img.iterdir() if p.is_dir()]
            full = [p for p in pts if (p / 'CT').is_dir() and (p / 'RTSTRUCT').is_dir()]
            good &= ok(len(full) == len(pts) > 0, f'{ds}: {len(pts)} patient folders, {len(full)} with CT and RTSTRUCT')
    if not good:
        sys.exit('\nFix the items marked MISSING (see README, "Data") and run this check again.')

    print('\nPatients with outcome data (events), compared with the paper')
    from hnc_popgraph.data.clinical import included_labels
    for (ds, task), (n, e) in EXPECTED.items():
        lab = included_labels(ds, task, verbose=False)
        n2, e2 = len(lab), sum(lab.values())
        good &= ok((n2, e2) == (n, e), f'{ds:8s} {task}: {n2} patients ({e2} events); paper {n} ({e})')
    print('\nAll checks passed: run ./run_study.sh' if good else '\nSome counts differ from the paper: check the downloads.')
    sys.exit(0 if good else 1)


if __name__ == '__main__':
    main()
