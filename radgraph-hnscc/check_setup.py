#!/usr/bin/env python3
"""
Environment + data check for RadGraph.   Usage:  python check_setup.py
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
                'radiomics', 'pydicom', 'rt_utils', 'skimage', 'scipy', 'mrmr', 'matplotlib', 'tqdm', 'networkx', 'openpyxl', 'joblib']:
        try:
            m = __import__(mod)
            print(f"  OK  {mod:16s} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  --  {mod:16s} MISSING")
            missing.append(mod)
    return not missing


def check_data():
    header("Data")
    from radgraph import config
    from radgraph.data.clinical import compute_labels
    ok = True
    for name, cfg in config.DATASETS.items():
        img, clin = cfg['images_dir'], cfg['clinical_file']
        print(f"{name}:")
        print(f"  images   {img}  {'OK' if img.is_dir() else 'MISSING'}")
        print(f"  clinical {clin}  {'OK' if clin.exists() else 'MISSING'}")
        if not (img.is_dir() and clin.exists()):
            ok = False
            continue
        for task in cfg['outcomes']:
            compute_labels(name, task)
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
        print("\nNext:  python main.py --mode MATEUS_CV --task LR --all --n_jobs 12")


if __name__ == '__main__':
    main()
