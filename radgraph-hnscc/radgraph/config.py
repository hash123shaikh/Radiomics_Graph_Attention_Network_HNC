"""
Central configuration.

Study: outcome prediction in HNSCC (locoregional recurrence LR, distant metastasis DM, overall
survival OS) with a RadGraph graph attention network (Bae et al., Radiology: Imaging Cancer 2025),
comparing feature-selection methods (incl. Graph-FS, Moradmand et al., Sci Rep 2025) x graph
topologies, for discrimination AND reproducibility.

Validation design of Mateus et al., Sci Rep 13, 18176 (2023): train/validate on the 4 Canadian
centres (Head-Neck-PET-CT), test on the external Dutch cohort (MAASTRO). Run modes (MODES):
  'MATEUS_CV'      Mateus et al., Sci Rep 2023: 5-fold stratified CV on the 4 Canadian centres
                   (HNPET) for train/val, every fold tested on MAASTRO; aggregated across folds
  'MATEUS_COHORT'  Mateus et al. / Diamant et al. 2019 cohort split: train HGJ + CHUS,
                   validate HMR + CHUM, test MAASTRO
The HNPET images live in the folder "HECKTOR 2026 Training Data"; the clinical file is the
Head-Neck-PET-CT INFOclinical xlsx (true LR, DM and death outcomes).
"""

import os
from pathlib import Path
import torch

# ============================================================================
# PATHS
#   Defaults are relative to this repository, so the code runs from any location. Override them with
#   environment variables (no code edits needed), e.g.
#     export RADGRAPH_DATA_DIR=/data/hnscc          # holds the HNPET and MAASTRO folders
#     export RADGRAPH_OUTPUT_DIR=/scratch/radgraph  # caches, runs, results, final models
#   RADGRAPH_HNPET_DIR / RADGRAPH_MAASTRO_DIR point to the two cohort folders individually.
# ============================================================================
def _env_path(var, default):
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


BASE_DIR   = Path(__file__).resolve().parent.parent          # radgraph-hnscc/
DATA_DIR   = _env_path('RADGRAPH_DATA_DIR', BASE_DIR / "data")
OUTPUT_DIR = _env_path('RADGRAPH_OUTPUT_DIR', BASE_DIR / "outputs")
LOGS_DIR   = _env_path('RADGRAPH_LOGS_DIR', BASE_DIR / "logs")
HNPET_DIR   = _env_path('RADGRAPH_HNPET_DIR', DATA_DIR / "HNPET" if (DATA_DIR / "HNPET").is_dir()
                        else DATA_DIR / "HECKTOR 2026 Training Data")     # either folder name works
MAASTRO_DIR = _env_path('RADGRAPH_MAASTRO_DIR', DATA_DIR / "MAASTRO")

CACHE_DIR         = OUTPUT_DIR / "cache"
PREPROCESSED_DIR  = CACHE_DIR / "preprocessed"     # per-patient CT/GTV/supervoxels (cohort independent)
FEATURE_CACHE_DIR = CACHE_DIR / "features"         # per-patient radiomic features
RUNS_DIR          = OUTPUT_DIR / "runs"            # one folder per <MODE>_<TASK>

for _d in (OUTPUT_DIR, LOGS_DIR, PREPROCESSED_DIR, FEATURE_CACHE_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# COHORTS
#   outcomes : task -> (event column, time column) in the table returned by
#              data.clinical.load_clinical (derived columns for HNPET)
#   time_unit: unit of the time column
#   centre   : how to get the acquisition centre of a patient (for cross-centre
#              reproducibility): 'prefix' = text before the first '-' of the ID
# ============================================================================
DATASETS = {
    'MAASTRO': {
        'images_dir'     : MAASTRO_DIR / "Images",
        'clinical_file'  : MAASTRO_DIR / "clinical_data.csv",
        'patient_id_col' : 'patient_id',
        'schema'         : 'MAASTRO',
        'outcomes'       : {
            'LR': ('locoregional_recurrence', 'locoregional_recurrence_in_days'),
            'DM': ('distant_metastasis',      'distant_metastases_in_days'),
            'OS': ('event_overall_survival',  'overall_survival_in_days'),
        },
        'time_unit'      : 'days',
        'centre'         : 'MAASTRO',
    },
    'HNPET': {
        'images_dir'     : HNPET_DIR / "Images",
        'clinical_file'  : HNPET_DIR / "INFOclinical_HN_Version2_30may2018.xlsx",
        'clinical_sheets': ['CHUM', 'CHUS', 'HGJ', 'HMR'],      # 'Excluded' sheet is ignored
        'patient_id_col' : 'patient_id',
        'schema'         : 'HNPET',
        'outcomes'       : {
            'LR': ('lr_event', 'lr_time_days'),
            'DM': ('dm_event', 'dm_time_days'),
            'OS': ('os_event', 'os_time_days'),
        },
        'time_unit'      : 'days',
        'centre'         : 'prefix',
    },
}
DATASET_ALIASES = {'HECKTOR': 'HNPET'}

MODES = {
    #  mode            : (train/val cohorts, external test cohorts)
    'MATEUS_CV'     : (['HNPET'], ['MAASTRO']),
    'MATEUS_COHORT' : (['HNPET'], ['MAASTRO']),
}
# how train/val are formed inside the training cohort
SPLIT_SCHEME = {'MATEUS_CV': 'cv', 'MATEUS_COHORT': 'cohort'}
N_FOLDS      = 5
COHORT_SPLIT = {'train': ['HGJ', 'CHUS'], 'val': ['HMR', 'CHUM']}   # Diamant et al. 2019
FOLD         = None      # current fold (1..N_FOLDS) in 'cv' modes, set by set_fold()
MODE_ALIASES = {}

MODE           = 'MATEUS_CV'
TRAIN_DATASETS = list(MODES[MODE][0])
TEST_DATASETS  = list(MODES[MODE][1])


def set_mode(mode):
    """Switch run mode. Every module reads config.MODE / TRAIN_DATASETS / TEST_DATASETS at call time."""
    global MODE, TRAIN_DATASETS, TEST_DATASETS
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use one of {list(MODES)}")
    MODE = mode
    TRAIN_DATASETS, TEST_DATASETS = list(MODES[mode][0]), list(MODES[mode][1])
    set_fold(None)


def split_scheme():
    return SPLIT_SCHEME.get(MODE, 'random')


def set_fold(fold):
    global FOLD
    if fold is not None and not 1 <= fold <= N_FOLDS:
        raise ValueError(f"fold must be 1..{N_FOLDS}")
    FOLD = fold


def run_label():
    """Mode (+ fold) for figure titles."""
    return f"{MODE} fold {FOLD}" if FOLD else MODE


def is_external():
    return bool(TEST_DATASETS)


def harmonized_clinical():
    """Clinical columns must match across cohorts whenever more than one cohort is involved."""
    return len(datasets_in_mode()) > 1


def dataset_cfg(name):
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Use one of {list(DATASETS)}")
    return DATASETS[name]


def datasets_in_mode(mode=None):
    tr, te = MODES[MODE_ALIASES.get(mode, mode) if mode else MODE]
    return list(tr) + list(te)


def mode_task_available(task, mode=None):
    return all(task_available(d, task) for d in datasets_in_mode(mode))


def task_available(dataset, task):
    return task in dataset_cfg(dataset)['outcomes']


def patient_centre(dataset, pid):
    c = dataset_cfg(dataset)['centre']
    return str(pid).split('-')[0] if c == 'prefix' else c


# ============================================================================
# OUTCOMES (paper: binary at 2 years, 2-year minimum follow-up)
#   LR = locoregional recurrence, DM = distant metastasis, OS = death from any cause
# ============================================================================
TASKS = ['LR', 'DM', 'OS']
TASK_NAMES = {'LR': 'Locoregional recurrence', 'DM': 'Distant metastasis', 'OS': 'Overall survival (death)'}
TASK = 'LR'          # current task, set by set_task() (used for task-dependent defaults)


def set_task(task):
    global TASK
    if task not in TASKS:
        raise ValueError(f"Unknown task {task}. Use one of {TASKS}")
    TASK = task


LABEL_HORIZON_MONTHS = 24                       # LR, DM (RadGraph paper; Mateus et al.)
HORIZON_BY_TASK      = {'OS': 48}               # OS: 4 years = median follow-up (Mateus et al. 2023)


def horizon_months(task=None):
    return HORIZON_BY_TASK.get(task or TASK, LABEL_HORIZON_MONTHS)
DAYS_PER_MONTH       = 30.4375

# ============================================================================
# PREPROCESSING
# ============================================================================
TARGET_SPACING        = (1.0, 1.0, 1.0)
PERITUMORAL_MARGIN_MM = 50
CROP_TO_REGION        = True

GTV_PRIMARY_NAMES = ['GTV-1', 'GTV_1', 'GTV1', 'GTVp', 'GTV-P', 'GTV_P',
                     'GTV_Primary', 'GTV-Primary', 'GTV_T', 'GTV-T', 'GTV']
GTV_EXCLUDE_TOKENS = ['gtvn', 'gtv-n', 'gtv_n', 'node', 'nodal', 'ln', 'gtv-2', 'gtv-3',
                      'gtv-4', 'gtv-5', 'gtv-6', 'gtv-7', 'gtv-8', 'gtv2', 'gtv3', 'ctv', 'ptv']

NIFTI_GTVP_LABEL = 1     # primary tumour in the NIfTI segmentation
NIFTI_GTVN_LABEL = 2     # nodal GTV (not used as input)


# ============================================================================
# SUPERVOXELS + RADIOMICS
# ============================================================================
N_SUPERVOXELS_TARGET  = 100
MIN_SUPERVOXEL_VOXELS = 8
# Stage 2 refinements (v4). The 50 mm box extends beyond the skin: without a body mask ~22% of the
# supervoxels were mostly air, and min-max scaling over air (-1024 HU) and metal (~3000 HU) squeezed
# soft tissue into ~2.5% of the SLIC intensity range.
BODY_HU_THRESHOLD  = -500          # body = CT > threshold, largest 3-D component, holes filled per slice

# Preprocessing profiles. 'v3' is the setting of the results reported in the manuscript and of the
# released models (v0.1.0-models); 'v4' adds a body mask and a soft-tissue window before SLIC.
#   select with  --preprocess v3|v4  (main.py, predict.py)  or  export RADGRAPH_PREPROCESS=v3
PREPROCESS_PROFILES = {
    'v3': {'PREPROCESS_VERSION': 3, 'USE_BODY_MASK': False, 'SLIC_HU_WINDOW': None},
    'v4': {'PREPROCESS_VERSION': 4, 'USE_BODY_MASK': True,  'SLIC_HU_WINDOW': (-160, 240)},  # W400/L40
}
DEFAULT_PREPROCESS = 'v4'
PREPROCESS = None
PREPROCESS_VERSION = USE_BODY_MASK = SLIC_HU_WINDOW = None


def set_preprocess(profile):
    """Switch the preprocessing profile (sets PREPROCESS_VERSION, USE_BODY_MASK, SLIC_HU_WINDOW).
    Cached files carry their version, so files of the other profile are recomputed, never mixed."""
    global PREPROCESS
    if profile not in PREPROCESS_PROFILES:
        raise ValueError(f"Unknown preprocessing profile '{profile}'. Use one of {list(PREPROCESS_PROFILES)}")
    PREPROCESS = profile
    globals().update(PREPROCESS_PROFILES[profile])


set_preprocess(os.environ.get('RADGRAPH_PREPROCESS', DEFAULT_PREPROCESS))

PYRADIOMICS_SETTINGS = {
    'binWidth'             : 25,
    'normalize'            : False,
    'minimumROIDimensions' : 1,
    'minimumROISize'       : 1,
}
FEATURE_CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
FEATURE_VERSION = 2

# Table S3 of the RadGraph paper (mRMR + RF on RADCURE) -> the 'table_s3' reference selector
SELECTED_FEATURES = {
    'LR': ['original_glszm_GrayLevelNonUniformity', 'original_glcm_Id',
           'original_glcm_Idn', 'original_glszm_ZoneEntropy'],
    'DM': ['original_ngtdm_Coarseness', 'original_firstorder_Mean',
           'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_glcm_ClusterProminence',
           'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SmallAreaEmphasis'],
}

# ============================================================================
# FEATURE SELECTION (radgraph/selection)
#   All selectors are fitted on the GTV features of TRAINING patients only.
# ============================================================================
FS_METHODS = ['table_s3', 'all', 'mrmr', 'lasso', 'boruta', 'rfe_rf',
              'graph_fs_cc', 'graph_fs_spectral', 'graph_fs_hybrid']


def fs_methods(task=None):
    """FS methods available for a task: 'table_s3' only where the RadGraph paper lists features (LR, DM)."""
    task = task or TASK
    return [m for m in FS_METHODS if m != 'table_s3' or task in SELECTED_FEATURES]
FS_K                 = 6        # features kept per method (fair comparison); None -> method's native size
FS_K_RANGE           = (3, 20)  # native sizes are clipped to this range
FS_VARIANCE_MIN      = 1e-8     # prefilter: drop (near-)constant features on train
FS_LOG_TRANSFORM     = True     # signed log1p of heavy-tailed features before statistics
GRAPH_FS_ALPHA       = 0.3      # edge if |r| > mean + alpha * std  (Moradmand et al.)
GRAPH_FS_CORRELATION = 'pearson'
BORUTA_MAX_ITER      = 60
BORUTA_ALPHA         = 0.05
RF_FS_TREES          = 300

# stability / reproducibility
STABILITY_N_BOOTSTRAP = 30
STABILITY_MIN_CENTRE  = 15      # minimum patients (and >=3 events) for a per-centre selection

# ============================================================================
# GRAPH TOPOLOGIES (radgraph/graphs/builder.py)
#   radiomic_star : GTV + K supervoxels nearest to the GTV in feature space, star edges (paper)
#   spatial_star  : GTV + K supervoxels nearest to the GTV in physical space, star edges
#   radiomic_knn  : radiomic_star nodes + kNN edges between all nodes in feature space
#   spatial_knn   : spatial_star nodes  + kNN edges between all nodes in physical space
# ============================================================================
TOPOLOGIES             = ['radiomic_star', 'spatial_star', 'radiomic_knn', 'spatial_knn']
N_SUPERVOXELS_SELECTED = 20
KNN_K                  = 4
NODE_FEATURE_CLIP      = 5.0
GRAPH_VERSION          = 3

# ============================================================================
# GAT (Table S2)
# ============================================================================
GAT_CONFIG = {
    'LR': {'n_layers': 5, 'hidden_dim': 16, 'n_heads': 4, 'dropout': 0.67,
           'learning_rate': 4e-4, 'weight_decay': 7e-4, 'warmup_steps': 10,
           'batch_size': 64, 'n_neighbors': 20, 'sampling_strategy': 'intermediate'},
    'DM': {'n_layers': 3, 'hidden_dim': 8, 'n_heads': 4, 'dropout': 0.89,
           'learning_rate': 3e-4, 'weight_decay': 8e-4, 'warmup_steps': 5,
           'batch_size': 64, 'n_neighbors': 20, 'sampling_strategy': 'over'},
}
# OS is not in the RadGraph paper (no Table S2 entry): the LR hyper-parameters are used unchanged
GAT_CONFIG['OS'] = dict(GAT_CONFIG['LR'])


def get_gat_config(task):
    return dict(GAT_CONFIG[task])


READOUT_DROPOUT = 0.5
USE_EDGE_ATTR   = False

# ============================================================================
# TRAINING / SPLITS
# ============================================================================
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
CROSS_VAL_RATIO = 0.20
RANDOM_SEED     = 42
BENCHMARK_SEEDS = [42, 43, 44]   # GAT re-trained with each seed; predictions averaged

N_EPOCHS                 = 500
MIN_EPOCHS_BEFORE_SELECT = 10
EARLY_STOPPING_PATIENCE  = 50
SCHEDULER_PATIENCE       = 20
SCHEDULER_FACTOR         = 0.5
MIN_LR                   = 1e-5
GRAD_CLIP_NORM           = 1.0
# checkpoint rule: 'val_loss' = lowest validation loss (RadGraph paper)
#                  'auc_gap'  = highest validation AUC among epochs with train AUC - val AUC <= gap,
#                               gap starts at 0.05 and grows by 0.05 until an epoch qualifies (Mateus et al.)
EPOCH_RULE    = 'val_loss'
AUC_GAP_START = 0.05
AUC_GAP_STEP  = 0.05

# ============================================================================
# EVALUATION
# ============================================================================
N_BOOTSTRAP      = 1000
CONFIDENCE_LEVEL = 0.95
REFERENCE_CONFIG = ('table_s3', 'radiomic_star')   # DeLong comparisons are against this (LR, DM)
# OS has no Table S3 feature list: mRMR (the paper's own selection approach) + radiomic star instead
REFERENCE_BY_TASK = {'OS': ('mrmr', 'radiomic_star')}


def reference_config(task=None):
    return REFERENCE_BY_TASK.get(task or TASK, REFERENCE_CONFIG)

# ============================================================================
# RF BASELINES (Appendix S1)
# ============================================================================
MRMR_N_CANDIDATES   = 20
BASELINE_SEARCH     = 'forward'
BASELINE_MAX_COMBOS = 20000
RF_N_ESTIMATORS     = 100
RF_MAX_DEPTH        = 10
RF_RANDOM_STATE     = 42

# ============================================================================
# COMPUTE / FIGURES
# ============================================================================
FIGURE_DPI  = 300
USE_CUDA    = True
DEVICE      = 'cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu'
NUM_WORKERS = 0


# ============================================================================
# RUN LAYOUT
#   outputs/runs/<MODE>_<TASK>/
#       split.json
#       selection/<method>.json          selected features (+ ranking, diagnostics)
#       selection/stability_*.csv
#       graphs/<method>__<topology>/     <pid>.pt + meta.json
#       models/<exp>.pth                 exp = <method>__<topology>__<img|img+clin>__s<seed>
#       results/<exp>.json, predictions/<exp>.csv
#       figures/...                      all PNGs
#       report.html
# ============================================================================
def mode_dir(task):
    """outputs/runs/<MODE>_<TASK>[_aucgap]/ (folds live underneath in 'cv' modes)."""
    d = RUNS_DIR / f"{MODE}_{task}{'' if EPOCH_RULE == 'val_loss' else '_' + EPOCH_RULE.replace('_', '')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(task):
    """Folder of the current run: <mode folder>/fold<k> while a fold is set ('cv' modes), else the
    mode folder (which holds the cross-fold summary in 'cv' modes)."""
    d = mode_dir(task) / f'fold{FOLD}' if FOLD else mode_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sub_dir(task, name):
    d = run_dir(task) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_file(task):
    return run_dir(task) / 'split.json'


def selection_file(task, method):
    return sub_dir(task, 'selection') / f'{method}.json'


def graph_dir(task, method, topology):
    return sub_dir(task, 'graphs') / f'{method}__{topology}'


def exp_name(method, topology, use_clinical, seed):
    return f"{method}__{topology}__{'img+clin' if use_clinical else 'img'}__s{seed}"


def exp_paths(task, method, topology, use_clinical, seed):
    e = exp_name(method, topology, use_clinical, seed)
    return {
        'model'      : sub_dir(task, 'models') / f'{e}.pth',
        'metrics'    : sub_dir(task, 'results') / f'{e}.json',
        'predictions': sub_dir(task, 'predictions') / f'{e}.csv',
        'history'    : sub_dir(task, 'histories') / f'{e}.json',
    }


def figures_dir(task, *parts):
    d = sub_dir(task, 'figures').joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def print_config(task='LR'):
    print("=" * 70)
    print(f"RadGraph-FS | mode={MODE} (train={TRAIN_DATASETS}, test={TEST_DATASETS or 'internal split'})"
          f" | task={task} | device={DEVICE} | preprocess={PREPROCESS}")
    print(f"Train/val scheme: {split_scheme()}"
          f"{f' ({N_FOLDS} folds)' if split_scheme() == 'cv' else ''}"
          f"{f' {COHORT_SPLIT}' if split_scheme() == 'cohort' else ''} | checkpoint rule: {EPOCH_RULE}")
    print(f"FS methods: {fs_methods(task)} (k={FS_K}) | topologies: {TOPOLOGIES} | reference: {reference_config(task)}")
    print(f"GAT: {get_gat_config(task)}")
    print(f"Run folder: {mode_dir(task)}")
    print("=" * 70)
