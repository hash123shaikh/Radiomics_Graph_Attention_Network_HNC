"""
Study configuration: cohorts, endpoints, validation designs, preprocessing and radiomics settings.

Development cohort  Head-Neck-PET-CT (TCIA): four Canadian centres (CHUM, CHUS, HGJ, HMR)
External test       HN1 (TCIA), MAASTRO Clinic
Endpoints           LR and DM within 2 years, death (OS) within 4 years, as in Mateus et al.,
                    Sci Rep 13, 18176 (2023)
Designs             'MATEUS_CV'     stratified 5-fold cross-validation on the Canadian patients
                    'MATEUS_COHORT' train HGJ + CHUS, validate HMR + CHUM (Diamant et al. 2019)
                    MAASTRO is the external test set in both designs.

Paths default to data/ and outputs/ in the repository; override with HNC_DATA_DIR and HNC_OUTPUT_DIR.
"""
import os
from pathlib import Path


def _env_path(var, default):
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


# ── paths ────────────────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = _env_path('HNC_DATA_DIR', BASE_DIR / 'data')
OUTPUT_DIR = _env_path('HNC_OUTPUT_DIR', BASE_DIR / 'outputs')

HNPET_DIR   = _env_path('HNC_HNPET_DIR', DATA_DIR / 'Head-Neck-PET-CT')
MAASTRO_DIR = _env_path('HNC_MAASTRO_DIR', DATA_DIR / 'MAASTRO')

PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed'                 # 1 mm CT + primary GTV per patient
FEATURES_DIR     = OUTPUT_DIR / 'features'
PRIMARY_GTV_DIR  = FEATURES_DIR / 'primary_gtv'                # per-patient primary-GTV features
PRIMARY_GTV_CSV  = FEATURES_DIR / 'primary_gtv_features.csv'
WHOLE_TUMOUR_CSV = FEATURES_DIR / 'whole_tumour_features.csv'  # primary + nodal GTVs
TUMOUR_BURDEN_CSV = FEATURES_DIR / 'tumour_burden.csv'
SPLITS_DIR       = OUTPUT_DIR / 'splits'
RESULTS_DIR      = OUTPUT_DIR / 'results'
GRAPH_RESULTS    = RESULTS_DIR / 'population_graph'
CLASSICAL_RESULTS = RESULTS_DIR / 'conventional'
COHORT_SUMMARY_CSV = RESULTS_DIR / 'cohort_summary.csv'

for _d in (PREPROCESSED_DIR, PRIMARY_GTV_DIR, SPLITS_DIR, GRAPH_RESULTS, CLASSICAL_RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# ── cohorts ──────────────────────────────────────────────────────────────────────────────────────
#   outcomes: task -> (event column, time column) of data.clinical.load_clinical
#   centre  : 'prefix' = text before the first '-' of the patient ID
DATASETS = {
    'MAASTRO': {
        'images_dir'     : MAASTRO_DIR / 'Images',
        'clinical_file'  : MAASTRO_DIR / 'clinical_data.csv',
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
        'images_dir'     : HNPET_DIR / 'Images',
        'clinical_file'  : HNPET_DIR / 'INFOclinical_HN_Version2_30may2018.xlsx',
        'clinical_sheets': ['CHUM', 'CHUS', 'HGJ', 'HMR'],
        # primary-GTV ROI name(s) per patient (from TCIA INFO_GTVcontours_HN.xlsx); ';' joins ROIs
        'gtv_roi_map'    : HNPET_DIR / 'gtv_primary_rois.csv',
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
DATASET_ALIASES = {}

# ── validation designs ───────────────────────────────────────────────────────────────────────────
MODES = {                                    # design: (development cohorts, external test cohorts)
    'MATEUS_CV'    : (['HNPET'], ['MAASTRO']),
    'MATEUS_COHORT': (['HNPET'], ['MAASTRO']),
}
DESIGNS      = list(MODES)
SPLIT_SCHEME = {'MATEUS_CV': 'cv', 'MATEUS_COHORT': 'cohort'}
N_FOLDS      = 5
COHORT_SPLIT = {'train': ['HGJ', 'CHUS'], 'val': ['HMR', 'CHUM']}
RANDOM_SEED  = 42

FOLD           = None
MODE           = 'MATEUS_CV'
TRAIN_DATASETS = list(MODES[MODE][0])
TEST_DATASETS  = list(MODES[MODE][1])


def set_mode(mode):
    global MODE, TRAIN_DATASETS, TEST_DATASETS
    if mode not in MODES:
        raise ValueError(f"Unknown design '{mode}'. Use one of {list(MODES)}")
    MODE = mode
    TRAIN_DATASETS, TEST_DATASETS = list(MODES[mode][0]), list(MODES[mode][1])
    set_fold(None)


def split_scheme():
    return SPLIT_SCHEME[MODE]


def set_fold(fold):
    global FOLD
    if fold is not None and not 1 <= fold <= N_FOLDS:
        raise ValueError(f'fold must be 1..{N_FOLDS}')
    FOLD = fold


def harmonized_clinical():
    """Clinical columns must match across cohorts whenever more than one cohort is involved."""
    return len(datasets_in_mode()) > 1


def dataset_cfg(name):
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Use one of {list(DATASETS)}")
    return DATASETS[name]


def datasets_in_mode(mode=None):
    tr, te = MODES[mode or MODE]
    return list(tr) + list(te)


def task_available(dataset, task):
    return task in dataset_cfg(dataset)['outcomes']


def mode_task_available(task, mode=None):
    return all(task_available(d, task) for d in datasets_in_mode(mode))


def patient_centre(dataset, pid):
    c = dataset_cfg(dataset)['centre']
    return str(pid).split('-')[0] if c == 'prefix' else c


def split_file(task):
    """outputs/splits/<design>_<task>[_fold<k>].json"""
    return SPLITS_DIR / (f'{MODE}_{task}' + (f'_fold{FOLD}' if FOLD else '') + '.json')


# ── endpoints ────────────────────────────────────────────────────────────────────────────────────
TASKS = ['LR', 'DM', 'OS']
TASK_NAMES = {'LR': 'Locoregional recurrence', 'DM': 'Distant metastasis', 'OS': 'Overall survival (death)'}
TASK = 'LR'
LABEL_HORIZON_MONTHS = 24          # LR, DM
HORIZON_BY_TASK = {'OS': 48}       # OS: 4 years (Mateus et al.)
DAYS_PER_MONTH = 30.4375


def set_task(task):
    global TASK
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Use one of {TASKS}')
    TASK = task


def horizon_months(task=None):
    return HORIZON_BY_TASK.get(task or TASK, LABEL_HORIZON_MONTHS)


# ── images and delineations ──────────────────────────────────────────────────────────────────────
TARGET_SPACING = (1.0, 1.0, 1.0)
CROP_MARGIN_MM = 50                # primary-GTV volumes are stored cropped to the GTV box + 50 mm
GTV_PRIMARY_NAMES = ['GTV-1', 'GTV_1', 'GTV1', 'GTVp', 'GTV-P', 'GTV_P',
                     'GTV_Primary', 'GTV-Primary', 'GTV_T', 'GTV-T', 'GTV']
GTV_EXCLUDE_TOKENS = ['gtvn', 'gtv-n', 'gtv_n', 'node', 'nodal', 'ln', 'gtv-2', 'gtv-3',
                      'gtv-4', 'gtv-5', 'gtv-6', 'gtv-7', 'gtv-8', 'gtv2', 'gtv3', 'ctv', 'ptv']
MASK_RULE = 'coord-centre-v1'      # RTSTRUCT contour -> voxel mask rule (stored in every preprocessed file)
PREPROCESS_VERSION = 4
NIFTI_GTVP_LABEL = 1

# ── radiomics ────────────────────────────────────────────────────────────────────────────────────
PYRADIOMICS_SETTINGS = {'binWidth': 25, 'normalize': False, 'minimumROIDimensions': 1, 'minimumROISize': 1}
FEATURE_CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
LOG_SIGMAS_MM = [1.0, 2.0, 3.0, 5.0]

# ── feature selection (fitted on training patients only) ─────────────────────────────────────────
FS_VARIANCE_MIN  = 1e-8            # drop (near-)constant features
FS_LOG_TRANSFORM = True            # signed log1p of heavy-tailed features
FS_K_RANGE       = (3, 20)

# ── evaluation ───────────────────────────────────────────────────────────────────────────────────
N_BOOTSTRAP      = 1000
CONFIDENCE_LEVEL = 0.95
