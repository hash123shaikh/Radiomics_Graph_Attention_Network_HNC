"""
Collect every finished run (LR / DM / OS x MATEUS_CV / MATEUS_COHORT x checkpoint rule) into
outputs/results/ (CSV tables, ROC figures, per-patient predictions) and copy the chosen models'
weights to outputs/final_models/ for predict.py. Safe to re-run at any time; nothing is retrained.

    python export_results.py
    python export_results.py --no_models      # tables and figures only
"""

import argparse

from radgraph.experiments.export import export_all

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no_models', action='store_true', help='do not copy model weights to outputs/final_models')
    export_all(copy_models=not ap.parse_args().no_models)
