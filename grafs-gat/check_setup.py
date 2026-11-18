#!/usr/bin/env python3
"""
Environment + data check for GRAFS-GAT.   Usage:  python check_setup.py
"""

import sys


def header(t):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


def check_python():
    header("Python")
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    ok = (v.major, v.minor) >= (3, 8)
    print("OK" if ok else "Python 3.8+ required")
    return ok


def check_packages():
    header("Packages")
    missing = []
    for mod in ['numpy', 'pandas', 'torch', 'torch_geometric', 'sklearn', 'SimpleITK',
                'radiomics', 'pydicom', 'skimage', 'scipy', 'mrmr', 'matplotlib', 'tqdm', 'networkx', 'openpyxl', 'joblib']:
        try:
            m = __import__(mod)
            print(f"  OK  {mod:16s} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  --  {mod:16s} MISSING")
            missing.append(mod)
    return not missing


# Patients with CT + primary GTV, and patients (events) per endpoint, in the manuscript (Table 1).
# A different count means the local data differ from those of the study (see DATA.md).
EXPECTED = {
    # full TCIA Head-Neck-PET-CT (DICOM, 298 patients; 294 with a valid label for some endpoint)
    'HNPET':   {'patients': 294, 'LR': (276, 30), 'DM': (276, 27), 'OS': (164, 44)},
    'MAASTRO': {'patients': 137, 'LR': (121, 30), 'DM': (109, 7), 'OS': (135, 45)},
}


def check_data():
    header("Data")
    import contextlib, io
    from grafsgat import config
    from grafsgat.data.clinical import included_labels
    from grafsgat.data.loader import HNSCCDataLoader
    ok = True
    for name, cfg in config.DATASETS.items():
        img, clin = cfg['images_dir'], cfg['clinical_file']
        print(f"{name}:")
        print(f"  images   {img}  {'OK' if img.is_dir() else 'MISSING'}")
        print(f"  clinical {clin}  {'OK' if clin.exists() else 'MISSING'}")
        if not (img.is_dir() and clin.exists()):
            ok = False
            continue
        exp = EXPECTED.get(name, {})
        dl = HNSCCDataLoader(name)
        with contextlib.redirect_stdout(io.StringIO()):
            n_img = len(dl.eligible_patients_any_task())
        print(f"  {'':8s} {'found':>14s} {'paper':>14s}")
        line = lambda k, f, e: print(f"  {k:8s} {f:>14s} {e:>14s}  {'OK' if f == e else 'DIFFERENT'}")
        line('patients', str(n_img), str(exp.get('patients', '?')))
        for task in cfg['outcomes']:
            with contextlib.redirect_stdout(io.StringIO()):
                ids = dl.eligible_patients(task)
                lab = included_labels(name, task, verbose=False)
            found = f"{len(ids)} ({sum(lab[p] for p in ids)} ev)"
            e = exp.get(task)
            line(task, found, f"{e[0]} ({e[1]} ev)" if e else '?')
    print("\nDIFFERENT counts do not stop the pipeline, but results will then differ from the paper.")
    return ok


def check_gpu():
    header("Device")
    import torch
    print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (CPU)")
    return True


def main():
    results = [(n, f()) for n, f in [('python', check_python), ('packages', check_packages),
                                    ('device', check_gpu), ('data', check_data)]]
    header("Summary")
    for n, r in results:
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    if all(r for _, r in results):
        print("\nNext:  python smoke_test.py                      (pipeline test on synthetic data)"
              "\n       python verify_release.py --download       (reproduce the published external AUCs)"
              "\n       ./run_mateus_study.sh                     (full study)")


if __name__ == '__main__':
    main()
