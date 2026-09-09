"""Models comparison pipeline for ICU mortality prediction experiments.

Orchestrates end-to-end training, evaluation, and comparison of machine
learning and clinical-score models across multiple configurations including
windowing strategies, feature sets, balancing methods, and calibration
modes. Supports both out-of-fold cross-validation and holdout evaluation.
"""

from types import SimpleNamespace

SEED = 42
seed = SEED
import os
import sys
import faulthandler
import traceback
from datetime import datetime
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.path.append(os.path.abspath('..'))
import json
from pathlib import Path
# ─────────────────────────────────────────────────────────────────────────────
# LOGGING LAYER
# All printed output and errors are duplicated to a log file.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_LOG_DIR = _SCRIPT_DIR / "pipeline_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}.log"
_LOG_FILE = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class _TeeStream:
    """Write simultaneously to the terminal and the log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            try:
                stream.write(message)
                stream.flush()
            except (OSError, ValueError):
                pass
        return len(message)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


sys.stdout = _TeeStream(_ORIGINAL_STDOUT, _LOG_FILE)
sys.stderr = _TeeStream(_ORIGINAL_STDERR, _LOG_FILE)

# Configure the standard logging module to write to the same log file so
# that warnings from features_extraction_utils appear alongside print() output.
import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(_LOG_FILE)],
)
logger = logging.getLogger(__name__)

faulthandler.enable(file=_LOG_FILE, all_threads=True)

CURRENT_RUN_CONTEXT = {
    "experiment": "Pipeline initialization",
    "stage": "Importing dependencies",
}


def update_progress(stage):
    """Update the current stage and display it immediately."""
    CURRENT_RUN_CONTEXT["stage"] = stage
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [STAGE] {stage}", flush=True)


print(f"[LOG] Log file for this run: {_LOG_PATH}", flush=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from functools import partial
import utilitaries.create_merged_dataset as create_merged_dataset
import utilitaries.extract_data_utils as extract
import utilitaries.features_extraction_utils as extract_feat
import utilitaries.config_utils as mo_utils
import utilitaries.preprocessing_utils as preproc
import utilitaries.postprocessing_utils as postproc
import utilitaries.timestamp_sampling_utils as tsu
import utilitaries.path_utils as path_utils
import utilitaries.show_fig_utils as sfu
import utilitaries.training_utils as training
import utilitaries.evaluate_utils as evaluate
import utilitaries.resampling_and_window_choice_pipeline as choice
import utilitaries.sequential_utils as sequential
import utilitaries.static_features_utils as static_utils

import warnings

# Targeted filter: ignore UserWarning messages containing "cannot be shown".
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*cannot be shown.*"
)

pl.Config.set_tbl_cols(-1)

mode_names = ["score", "wp1", "wp2", "wp3"]
mode_names = ["wp3_test"]
mode_names = ["LR"]
RUN_COMPARISON = False
RUN_TRAINING = True
RUN_LASSO = True
RUN_INTERPRETABILITY = True
RUN_TEST = True
POPULATION = "Tout"
SAVE_FIGURE = True
TRANSPARENT = False
DATASET_PATH = "/../../../data2/paquie.d/Datasets/output"
TARGET_COLUMNS = {
    "Survie à 28 jours": "isDeceased_lt_28d",
    "Survie à 24 heures": "isDeceased_lt_24h",
    "Survie à 7 jours": "isDeceased_lt_7d",
    "Survie à 3 mois": "isDeceased_lt_3m",
}
EXPERIMENT_INDEX = 0
TOTAL_EXPERIMENTS = "X"
print(f"[PIPELINE] {TOTAL_EXPERIMENTS} experiment(s) scheduled.", flush=True)
mode_duplicates = "prio_last"
keep_duplicates = False
for mode_run in mode_names:
    if mode_run == "score":
        mode_duplicates = "prio_first"
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        target_labels = ["Survie à 28 jours"]
        model_names = ["IGS2"]
        stratify_modes = ["target_col"]
        optuna_run_options = [False]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi", "Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]
    elif mode_run == "LR":
        mode_duplicates = "prio_first"
        target_labels = ["Survie à 28 jours"]
        model_names = ["XGBoost TSFEL"]
        stratify_modes = ["target_col"]
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        optuna_run_options = [True]
        feature_modes = ["Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]
    elif mode_run == "TSFEL":
        mode_duplicates = "prio_first"
        target_labels = ["Survie à 28 jours"]
        model_names = ["SVC TSFEL", "Logistic Regression Lasso TSFEL", "XGBoost TSFEL", "RandomForest TSFEL",]
        stratify_modes = ["target_col"]
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        optuna_run_options = [False, True]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi", "Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]
    elif mode_run == "wp1":
        mode_duplicates = "prio_first"
        target_labels = ["Survie à 28 jours"]
        model_names = ["VanillaTransformerModified", "InceptionTimeModified", "XGBoost TSFEL", "LstmTimeModified", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col"]
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        optuna_run_options = [False, True]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi", "Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]

    elif mode_run == "wp2":
        WINDOWING_MODE =  "24h fin réanimation sans remplissage"
        target_labels = ["Survie à 28 jours", "Survie à 24 heures", "Survie à 7 jours", "Survie à 3 mois"]
        model_names = ["VanillaTransformerModified", "InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col", "h24-j28"]
        optuna_run_options = [False]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode"]

    elif mode_run == "wp2_test":
        WINDOWING_MODE =  "24h fin réanimation sans remplissage"
        target_labels = ["Survie à 28 jours"]
        model_names = ["LstmTimeModified", "XGBoost TSFEL","Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col"]
        optuna_run_options = [False]
        feature_modes = ["Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode"]

    elif mode_run == "wp3":
        WINDOWING_MODE =  "resampling aléatoire 'lomax' prio 24h sans remplissage"
        target_labels = ["Survie à 28 jours", "Survie à 24 heures", "Survie à 7 jours", "Survie à 3 mois"]
        model_names = ["VanillaTransformerModified", "InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col", "h24-j28"]
        optuna_run_options = [False]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode"]
    
    elif mode_run == "wp3_test":
        WINDOWING_MODE =  "resampling aléatoire 'lomax' prio 24h sans remplissage"
        target_labels = ["Survie à 28 jours"]
        model_names = ["LstmTimeModified", "XGBoost TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col"]
        optuna_run_options = [False]
        feature_modes = ["Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode"]
    for target_label in target_labels:
        TARGET_LABEL = target_label
        for stratification_choice in stratify_modes:
            STRATIFY_MODE = stratification_choice
            for feature_mode in feature_modes:
                if feature_mode == "Mode IGS2":
                    n_jobs = -1
                else:
                    n_jobs = 8
                for model_choice in model_names:
                    if model_choice in ["Logistic Regression Lasso TSFEL"]:
                        optuna_choices = [False]
                    else:
                        optuna_choices = optuna_run_options
                    print(f"[CONFIG] Optuna choices: {optuna_choices}")
                    for balancing_choice in balancing_methods:
                        for optuna_choice in optuna_choices:
                            try:
                                DATA_TYPE = "modèle"
                                if mode_run == "score" :
                                    DATA_TYPE = "score"

                                MODEL_NAME = model_choice
                                SCORE_NAME = "IGS2"

                                CLEANING_MODE = "Enlever Surveillance Continue"
                                FEATURE_MODE = feature_mode
                                BALANCING_METHOD = balancing_choice
                                BORUTA_FILTER = True
                                KEEP_STATIC = True
                                USE_OPTUNA = optuna_choice
                                RUN_OPTUNA = optuna_choice
                                EXTRACT_TSFEL = False
                                CLASS_WEIGHT = "_balanced"

                                CUSTOM_FEATURES = []

                                EXPERIMENT_INDEX += 1
                                CURRENT_RUN_CONTEXT["experiment"] = (
                                    f"{EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS} | model={MODEL_NAME} | "
                                    f"features={FEATURE_MODE} | balancing={BALANCING_METHOD} | optuna={RUN_OPTUNA}"
                                )
                                print("\n" + "=" * 100, flush=True)
                                print(f"[EXPERIMENT {EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS}] STARTED", flush=True)
                                print(f"  Model        : {MODEL_NAME}", flush=True)
                                print(f"  Features     : {FEATURE_MODE}", flush=True)
                                print(f"  Balancing    : {BALANCING_METHOD}", flush=True)
                                print(f"  Optuna       : {RUN_OPTUNA}", flush=True)
                                print("=" * 100, flush=True)
                                update_progress("Creating the experiment configuration")

                                pipeline_config = sequential.create_pipeline_config(
                                    type_donnees=DATA_TYPE,
                                    mode_fenetrage=WINDOWING_MODE,
                                    model_name=MODEL_NAME,
                                    score_name=SCORE_NAME,
                                    nettoyage=CLEANING_MODE,
                                    cible=TARGET_LABEL,
                                    mode_features=FEATURE_MODE,
                                    equilibrage=BALANCING_METHOD,
                                    population=POPULATION,
                                    save_figure=SAVE_FIGURE,
                                    transparent=TRANSPARENT,
                                    boruta_filter=BORUTA_FILTER,
                                    use_optuna=USE_OPTUNA,
                                    extract_tsfel=EXTRACT_TSFEL,
                                    class_weight=CLASS_WEIGHT,
                                )
                                config_transparent = pipeline_config.config_transparent
                                config_cleaning = pipeline_config.config_cleaning
                                config_mode = pipeline_config.config_mode
                                type_donnees = pipeline_config.type_donnees
                                modex = pipeline_config.modex
                                config_balance = pipeline_config.config_balance
                                config_y = pipeline_config.config_y
                                save_figure = pipeline_config.save_figure
                                boruta_filter = pipeline_config.boruta_filter
                                config_boruta = pipeline_config.config_boruta
                                config_keep_pop = pipeline_config.config_keep_pop
                                use_optuna = pipeline_config.use_optuna
                                config_models = pipeline_config.config_models
                                calibration = pipeline_config.calibration
                                calibration_mode = pipeline_config.calibration_mode
                                extract_tsfel = pipeline_config.extract_tsfel
                                class_weight_choice = pipeline_config.class_weight_choice

                                update_progress("Initializing hardware and parameters")
                                seed = SEED
                                stratify_mode = SimpleNamespace(value=STRATIFY_MODE)
                                cuda_available = torch.cuda.is_available()
                                print(f'Is CUDA available? {cuda_available}')
                                current_device = torch.cuda.current_device() if cuda_available else 'CPU'
                                print(f'Current device: {current_device}')
                                if cuda_available:
                                    print(f'GPU name: {torch.cuda.get_device_name(0)}')

                                if not config_transparent:
                                    plt.rcParams["figure.facecolor"] = "white"
                                    plt.rcParams['axes.facecolor'] = "white"
                                    plt.rcParams['savefig.facecolor'] = "white"
                                
                                is_score_mode = type_donnees.value == "score"
                                patient_col = extract.ID_COL
                                time_col = extract.TIME_COL
                                target_col = TARGET_COLUMNS[TARGET_LABEL]
                                expected_length = extract.WINDOW_SIZE
                                generated_dummy_columns = []
                                population_suffix = ''
                                if config_keep_pop.keep_population != 'all_diseases':
                                    population_suffix = '_' + config_keep_pop.keep_population
                                balance_method_prefix = ''
                                if config_balance.balance_method:
                                    balance_method_prefix = config_balance.balance_method + '_'
                                extension = '.joblib' if config_models.extraction_type == 'TSFEL' else '.pt'
                                DEFAULT_PARAMS = {
                                    # Short-sequence defaults: smaller kernels, fewer blocks,
                                    # reduced channels to avoid overfitting on ~24 timesteps.
                                    'InceptionTimeModified': {
                                        'epochs': 150,
                                        'patience': 20,
                                        'lr': 0.001,
                                        'num_blocks': 4,
                                        'out_channels': 32,
                                        'bottleneck_channels': 16,
                                        'kernel_sizes': 11,
                                        'batch_size': 64,
                                    },
                                    'LstmTimeModified': {'epochs': 100, 'patience': 30, 'lr': 0.001},
                                    'VanillaTransformerModified': {
                                        'epochs': 150,
                                        'patience': 20,
                                        'lr': 0.001,
                                        'd_model': 64,
                                        'nhead': 4,
                                        'num_layers': 2,
                                        'dim_feedforward': 128,
                                        'batch_size': 64,
                                    },
                                    'RandomForest TSFEL': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': class_weight_choice.value[1:]}, 
                                    'RandomForest Imbalanced TSFEL': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': class_weight_choice.value[1:]}, 
                                    'XGBoost TSFEL': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
                                    'SVC TSFEL': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'class_weight': class_weight_choice.value[1:]}}

                                exp = path_utils.Experiment(config_mode.name, config_cleaning, config_y, balance_method_prefix, modex, class_weight_choice, population_suffix, seed, stratify_mode=stratify_mode.value, calibrated_mode=calibration_mode.value if calibration.value else "", keep_duplicates = keep_duplicates, mode_duplicates = mode_duplicates)

                                dataset_path = DATASET_PATH

                                update_progress("Loading and merging data")
                                # _path = os.path.join(dataset_path, 'df_static_ano_clean.parquet')
                                # df_static = pl.scan_parquet(_path)

                                # _path = os.path.join(dataset_path, 'df_dynamic_full_clean.parquet')
                                # df_dynamic = extract.extract_data_survie(_path)

                                # df_merged = create_merged_dataset.create_merged_dataset(df_static, df_dynamic, True, save=True, folder=dataset_path)
                                _path = os.path.join(dataset_path, 'merged_static_ano_and_dynamic.parquet')
                                df_merged = pl.scan_parquet(_path)
                                logger.debug(
                                    "LOG df_merged unique encounterIds: %d",
                                    df_merged.collect().select("encounterId").unique().shape[0],
                                )
                                if not keep_duplicates:
                                    _path = os.path.join(dataset_path, 'df_static_full_clean.parquet')
                                    df_without_anomalies = pl.scan_parquet(_path)
                                    df_merged = extract.remove_duplicates(df_without_anomalies, df_merged, mode_duplicates == "prio_last")
                                else:
                                    df_merged.collect()
                                logger.debug(
                                    "LOG df_merged without duplicates unique encounterIds: %d",
                                    df_merged.select("encounterId").unique().shape[0],
                                )
                                df_labeled = tsu.prepare_labels(df_merged, 'relative')

                                df_labeled = df_labeled.filter(pl.col('delta_hour') >= 0)
                                
                                df_labeled.columns
                                logger.debug(
                                    "LOG df_labeled unique encounterIds: %d",
                                    df_labeled.select("encounterId").unique().shape[0],
                                )

                                

                                targets = ['isDeceased_lt_28d', 'isDeceased_lt_24h', 'isDeceased_lt_7d', 'isDeceased_lt_3m']

                                update_progress("Preparing and cleaning the dataset")
                                df_clean, features_list, target_length = choice.prepare_dataset_from_config(df_merged=df_labeled, config_mode=config_mode, target_col=target_col, patient_col=patient_col, targets=targets, seed=seed)
                                logger.debug(
                                    "LOG df_clean size unique encounterIds: %d",
                                    df_clean.select("encounterId").unique().shape[0],
                                )

                                score_columns = ['NEWS', 'NEWS2', 'sapsii', 'sapsii_prob']
                                discarded_columns = ['endotracheal_tube', 'tracheo', 'installation', 'eer', 'hx_comorbidité_majeure', 'imc', 'neuro_status', 'ecmo_all', 'prone', 'plq']
                                unused_columns = ['arret_therapeutique', 'limitation_therapeutique', 'hematocrit', 'peak_pressure']
                                unused_icu_columns = ['icu_actes', 'icu_mode_sortie', 'icu_DA', 'hosp_admissionMode', 'hosp_primaryDiagnosis', 'hosp_primaryDiagnosisCode']
                                pmsi_columns = ['icu_ghm', 'icu_mode_entree', 'hx_respi_chronique', 'hx_cirrhose', 'hx_cancer', 'hx_insuff_cardiaque', 'hx_irc', 'hx_ttt_immunosuppresseur']
                                useful_icu_columns = ['icu_ghm', 'icu_DP_code', 'icu_DP', 'icu_mode_entree']
                                broken_columns = []
                                leakage_columns = ['category', 'heure_entiere', 'encounterId', 'delta_hour', 'heure_calibree', 'year_inTime', target_col, 'deces_datediff_days', 'isDeceased', 'adm_unit', 'out_unit', 'transition_units', 'los', 'adm_year', 'hosp_los', 'hosp_dischargeMode', 'deces_hosp', *targets]
                                columns_to_drop = [*score_columns, *discarded_columns, *unused_columns, *leakage_columns, *unused_icu_columns, *broken_columns]
                                existing_columns = [col for col in columns_to_drop if col in df_clean.columns]
                                if existing_columns:
                                    df_clean_keep = df_clean.drop(existing_columns)
                                else:
                                    df_clean_keep = df_clean.copy()
                                df_clean_keep = df_clean_keep.select([pl.col(target_label) for target_label in df_clean_keep.columns if not target_label.endswith('_detected_term')])

                                commonly_used_features = ['score_glasgow', 'is_conscious', 'heart_rate', 'creat', 'is_cvvhf', 'is_hdi', 'pao2', 'is_ventilated', 'fio2_corr', 'age', 'temp', 'urine_rate', 'pas', 'pam', 'pad', 'bili_tot', 'leucocytes', 'admission_type', 'fr', 'ph', 'sodium', 'potassium', 'num_plq', 'blood_urea', 'nad_dose_poids', 'dobu_dose_poids', 'hemoglobine', 'tp', 'spo2', 'hco3', 'glyc_cap']

                                custom_features = SimpleNamespace(value=CUSTOM_FEATURES if CUSTOM_FEATURES else df_clean_keep.columns)

                                with open('../../Preprocessing_pipeline/preprocessing-pipelines/json/dynamic_features.json', 'r') as file:
                                    feature_metadata = json.load(file)
                                if modex.value == 'Mode All':
                                    selected_features = df_clean_keep.columns.copy()
                                elif modex.value == 'Mode All Without pmsi':
                                    selected_features = [col for col in df_clean_keep.columns if not col.startswith('hx_') and not col.startswith('icu_')]
                                elif modex.value == 'Mode Commonly Used Without pmsi':
                                    selected_features = [col for col in commonly_used_features if col in df_clean_keep.columns]
                                elif modex.value == 'Mode Commonly Used':
                                    all_target_commonly = commonly_used_features + pmsi_columns
                                    selected_features = [col for col in all_target_commonly if col in df_clean_keep.columns]
                                elif modex.value == 'Mode Custom':
                                    selected_features = custom_features.value.copy()
                                else:
                                    selected_features = mo_utils.FEAT[modex.value].keep_feats.copy()
                                    if 'urine_rate' in selected_features:
                                        selected_features.remove('urine_rate')
                                selected_feature_descriptions = ''
                                if config_mode.name == 'resampling_X_points':
                                    selected_features.append('real_time_hours')
                                elif config_mode.name == 'resampling_x_points_alea_lomax_prio_24h_no-fill':
                                    selected_features.append('observed_duration')
                                else:
                                    if 'observed_duration' in selected_features:
                                        selected_features.remove('observed_duration')
                                    if 'real_time_hours' in selected_features:
                                        selected_features.remove('real_time_hours')
                                for feature_name in selected_features:
                                    if feature_name in feature_metadata:
                                        selected_feature_descriptions += f"- {feature_metadata[feature_name]['description']} \n"

                                df_clean_1 = df_clean.select(*selected_features, patient_col, time_col, target_col)

                                keep_features = selected_features.copy()

                                if config_mode.mode == 'windows':
                                    patients_before_filter = df_clean_1.select(patient_col).unique().shape[0]
                                    valid_patient_ids = df_clean_1.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
                                    patients_with_valid_length = valid_patient_ids.shape[0]
                                    df_clean_2 = df_clean_1.join(valid_patient_ids, on=patient_col, how='inner')
                                    patients_after_filter = df_clean_2.select(patient_col).unique().shape[0]
                                    print(
                                        f"[WINDOWS FILTER] expected_length={expected_length}, "
                                        f"patients before filter={patients_before_filter}, "
                                        f"patients with valid length={patients_with_valid_length}, "
                                        f"patients after filter={patients_after_filter}",
                                        flush=True,
                                    )
                                    if df_clean_2.is_empty():
                                        raise ValueError("No patient has exactly the expected sequence length.")
                                    df_clean_2 = df_clean_2.sort(patient_col, time_col)
                                else:
                                    df_clean_2 = df_clean_1

                                # Extract categorical source columns BEFORE filtering
                                # (gender is NOT in selected_features for Mode Commonly Used)
                                categorical_source_columns = [
                                    "admission_type",
                                    "icu_ghm",
                                    "icu_mode_entree",
                                    "gender",
                                ]
                                categorical_data = None
                                if modex.value == "Mode Commonly Used":
                                    categorical_data = df_clean.select([
                                        patient_col,
                                        target_col,
                                        *categorical_source_columns,
                                    ]).unique()

                                (
                                    df_clean_3,
                                    keep_features,
                                    generated_dummy_columns,
                                ) = static_utils.encode_categorical_features(
                                    dataframe=df_clean_2,
                                    feature_mode=modex.value,
                                    features_to_keep=keep_features,
                                )

                                final_features = list(
                                    dict.fromkeys(keep_features)
                                )
                                assert target_col not in final_features, f'Leakage warning: {target_col} is present in the feature list!'
                                print(f'Number of features sent to {config_models.models_name} : {len(final_features)} : {final_features}')

                                # ---- TableOne for static features (Mode Commonly Used only) ----
                                if modex.value == "Mode Commonly Used":
                                    update_progress("Generating TableOne for static features")
                                    tableone_output_dir = exp.get_compare_figs_path() / "tableone"
                                    # Exclude icu_ghm from TableOne display (keep in training pipeline)
                                    tableone_categorical_columns = [
                                        col for col in categorical_source_columns
                                        if col != "icu_ghm"
                                    ]
                                    _ = static_utils.build_tableone(
                                        df_clean=df_clean_3,
                                        categorical_data=categorical_data,
                                        patient_col=patient_col,
                                        target_col=target_col,
                                        final_features=final_features,
                                        generated_dummy_columns=generated_dummy_columns,
                                        categorical_source_columns=tableone_categorical_columns,
                                        config_mode_name=config_mode.name,
                                        output_dir=tableone_output_dir,
                                        df_static_source = pl.read_parquet(os.path.join(DATASET_PATH, "df_static_full_clean.parquet"))
                                    )

                                X_init = df_clean_3.select(final_features).to_numpy()
                                y_init = df_clean_3[target_col].to_numpy()

                                update_progress("Preparing model data and the initial holdout")
                                train_init_df = None
                                train_init_tsfel = None
                                if config_models.extraction_type == 'TSFEL':
                                    raw_global_tsfel_path = exp.get_tsfel_parquet_path()
                                    static_feats = static_utils.build_static_feature_list(
                                            dataframe=df_clean_3,
                                            generated_dummy_columns=generated_dummy_columns,
                                            config_mode_name=config_mode.name,
                                        )
                                    if extract_tsfel.value or not os.path.exists(raw_global_tsfel_path):
                                        print("Starting global TSFEL extraction for all patients")
                                        tsfel_features = [target_label for target_label in final_features if target_label not in static_feats]
                                        tsfel_global_df = extract_feat.extract_tsfel_per_patient(df_clean_3, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col, n_jobs = n_jobs)
                                        static_global_df = df_clean_3.select([extract.ID_COL, *static_feats]).unique()
    
                                        # robustness verification
                                        static_row_counts = (
                                            static_global_df
                                            .group_by(extract.ID_COL)
                                            .len()
                                        )
    
                                        patients_with_multiple_static_rows = (
                                            static_row_counts
                                            .filter(pl.col("len") > 1)
                                        )
    
                                        assert patients_with_multiple_static_rows.is_empty(), (
                                            "Some patients have multiple distinct static-feature rows. "
                                            "This would duplicate their TSFEL row during the join.\n"
                                            f"Number of affected patients: "
                                            f"{patients_with_multiple_static_rows.height}\n"
                                            f"Examples:\n"
                                            f"{patients_with_multiple_static_rows.head(10)}"
                                        )
    
                                        complete_tsfel_df = tsfel_global_df.join(static_global_df, on=extract.ID_COL, how='inner')
                                        complete_tsfel_df.write_parquet(raw_global_tsfel_path)
                                        print('Global extraction saved')
                                    else:
                                        complete_tsfel_df = pl.read_parquet(raw_global_tsfel_path)
                                    initial_variable_list = complete_tsfel_df.columns
                                    parent_directory = raw_global_tsfel_path.parent
                                    np.save(parent_directory / 'keepVariableList_0.npy', initial_variable_list)

                                    if boruta_filter:
                                        filename_train_boruta = exp.get_tsfel_boruta(
                                            "train",
                                            0,
                                        )
                                        filename_test_boruta = exp.get_tsfel_boruta(
                                            "test",
                                            0,
                                        )
                                        print(filename_train_boruta, filename_test_boruta, not (
                                            os.path.exists(filename_train_boruta)
                                            and os.path.exists(filename_test_boruta)
                                        ))

                                saps2_pred = None
                                saps2_true = None
                                score_df = None
                                if is_score_mode:
                                    score_df = (
                                        df_clean
                                        .select(
                                            patient_col,
                                            target_col,
                                            pl.col("sapsii_prob").cast(pl.Float64),
                                        )
                                        .filter(
                                            pl.col("sapsii_prob").is_not_null()
                                            & pl.col(target_col).is_not_null()
                                        )
                                        .group_by(patient_col)
                                        .agg(
                                            pl.col("sapsii_prob")
                                            .drop_nulls()
                                            .first()
                                            .alias("score_probability"),

                                            pl.col(target_col)
                                            .drop_nulls()
                                            .first()
                                            .alias(target_col),
                                        )
                                        .drop_nulls(
                                            ["score_probability", target_col]
                                        )
                                        .sort(patient_col)
                                    )

                                    if score_df.is_empty():
                                        raise ValueError(
                                            "No valid score probabilities were found."
                                        )

                                    duplicate_check = (
                                        score_df
                                        .group_by(patient_col)
                                        .len()
                                        .filter(pl.col("len") > 1)
                                    )

                                    assert duplicate_check.is_empty(), (
                                        "The score dataset must contain exactly one row per patient."
                                    )

                                    saps2_pred = (
                                        score_df["score_probability"]
                                        .to_numpy()
                                        .reshape(-1)
                                    )

                                    saps2_true = (
                                        score_df[target_col]
                                        .to_numpy()
                                        .reshape(-1)
                                    )

                                    print(
                                        "[SCORE] Score dataset prepared: "
                                        f"{score_df.height} patients, "
                                        f"{int(saps2_true.sum())} positive outcomes."
                                    )

                                sgkf_init = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                                if is_score_mode:
                                    score_y_init = (
                                        score_df[target_col]
                                        .to_numpy()
                                        .reshape(-1)
                                    )

                                    score_groups_init = (
                                        score_df[patient_col]
                                        .to_numpy()
                                        .reshape(-1)
                                    )

                                    score_X_init = (
                                        score_df["score_probability"]
                                        .to_numpy()
                                        .reshape(-1, 1)
                                    )

                                    train_init_idx, test_init_idx = next(
                                        sgkf_init.split(
                                            X=score_X_init,
                                            y=score_y_init,
                                            groups=score_groups_init,
                                        )
                                    )

                                    train_score_df = score_df[train_init_idx]
                                    test_holdout_score_df = score_df[test_init_idx]

                                    X = train_score_df
                                    y = (
                                        train_score_df[target_col]
                                        .to_numpy()
                                        .reshape(-1)
                                    )
                                    groups = (
                                        train_score_df[patient_col]
                                        .to_numpy()
                                        .reshape(-1)
                                    )

                                else:
                                    logger.debug(
                                        "LOG df_clean_3 unique encounterIds: %d",
                                        df_clean_3.select("encounterId").unique().shape[0],
                                    )
                                    logger.debug(
                                        "TIME INIT: sgkf_init.split X_init shape=%s, y_init shape=%s",
                                        X_init.shape,
                                        y_init.shape,
                                    )
                                    groups_init = df_clean_3[patient_col].to_numpy()
                                    logger.debug(
                                        "TIME INIT: groups_init shape=%s, dtype=%s",
                                        groups_init.shape,
                                        groups_init.dtype,
                                    )
                                    train_init_idx, test_init_idx = next(sgkf_init.split(X=X_init, y=y_init, groups=groups_init))
                                    logger.debug(
                                        "TIME INIT: train_init_idx len=%d, test_init_idx len=%d",
                                        len(train_init_idx),
                                        len(test_init_idx),
                                    )
                                    train_init_patients = df_clean_3[train_init_idx].select(patient_col).unique()
                                    test_init_patients = df_clean_3[test_init_idx].select(patient_col).unique()
                                    logger.debug(
                                        "TIME INIT: train=%d patients, test=%d patients",
                                        train_init_patients.shape[0],
                                        test_init_patients.shape[0],
                                    )
                                    if config_models.extraction_type == 'TSFEL':
                                        train_init_tsfel = complete_tsfel_df.join(train_init_patients, on=patient_col, how='inner').sort(patient_col)
                                        test_holdout_tsfel = complete_tsfel_df.join(test_init_patients, on=patient_col, how='inner').sort(patient_col)
                                        X = train_init_tsfel
                                        y = train_init_tsfel[target_col].to_numpy()
                                        groups = train_init_tsfel[patient_col].to_numpy()
                                    elif config_models.extraction_type == 'time':
                                        logger.debug(
                                            "TIME INIT: extraction_type='time', building train/holdout dataframes"
                                        )
                                        logger.debug(
                                            "TIME INIT: df_clean_3 type=%s, train_init_idx type=%s",
                                            type(df_clean_3).__name__,
                                            type(train_init_idx).__name__,
                                        )
                                        train_init_df = df_clean_3[train_init_idx].sort([patient_col, time_col])
                                        test_holdout_df = df_clean_3[test_init_idx].sort([patient_col, time_col])
                                        logger.debug(
                                            "TIME INIT: train_init_df rows=%d, test_holdout_df rows=%d",
                                            len(train_init_df),
                                            len(test_holdout_df),
                                        )
                                        X = train_init_df
                                        y = train_init_df[target_col].to_numpy()
                                        groups = train_init_df[patient_col].to_numpy()
                                        logger.debug(
                                            "TIME INIT: X type=%s, y shape=%s, groups shape=%s",
                                            type(X).__name__,
                                            y.shape,
                                            groups.shape,
                                        )

                                update_progress("Building and preprocessing the 5 folds")
                                sgkf = StratifiedGroupKFold(
                                    n_splits=5,
                                    shuffle=True,
                                    random_state=seed,
                                )

                                folds_X_train = []
                                folds_X_validation = []
                                folds_X_holdout = []
                                folds_y_train = []
                                folds_y_validation = []
                                folds_groups = []
                                folds_sampling_stats = []
                                folds_feature_names = []

                                y_test_holdout = None
                                holdout_patient_ids = None

                                fold_processing_config = {
                                    "patient_col": extract.ID_COL,
                                    "time_col": extract.TIME_COL,
                                    "target_col": target_col,
                                    "boruta_filter": config_boruta,
                                    "balance_method": config_balance.balance_method,
                                    "expected_length": expected_length,
                                    "final_features": final_features,
                                    "exp": exp,
                                    "static_feats": static_feats if config_models.extraction_type == "TSFEL" else [],
                                    "keep_static": KEEP_STATIC,
                                }

                                # Load or compute the optimal correlation threshold.
                                boruta_crossfold_features = None
                                if config_models.extraction_type == "TSFEL":
                                    compare_figs_dir = exp.get_compare_figs_path()
                                    threshold_path = exp.get_corr_threshold_path()

                                    if threshold_path.exists():
                                        corr_threshold = float(np.load(threshold_path))
                                        print(
                                            f"Loaded correlation threshold: "
                                            f"{corr_threshold:.4f}"
                                        )
                                    else:
                                        update_progress(
                                            "Computing optimal correlation threshold"
                                        )
                                        tsfel_features_for_corr = [
                                            col for col in train_init_tsfel.columns
                                            if col not in [extract.ID_COL, target_col]
                                        ]
                                        corr_df = train_init_tsfel.select(
                                            tsfel_features_for_corr
                                        ).to_pandas()
                                        corr_results = extract_feat.generate_correlation_analysis(
                                            corr_df,
                                            source_features=final_features,
                                            static_features=static_feats,
                                            output_folder=str(compare_figs_dir),
                                        )
                                        corr_threshold = corr_results["optimal_threshold"]
                                        np.save(threshold_path, np.array(corr_threshold))
                                        print(
                                            f"Optimal correlation threshold computed and saved: "
                                            f"{corr_threshold:.4f}"
                                        )

                                    fold_processing_config["corr_threshold"] = corr_threshold

                                    # ---- Global corr/var filtering (once, not per fold) ----
                                    tsfel_features_for_filter = [
                                        col for col in train_init_tsfel.columns
                                        if col not in [extract.ID_COL, target_col]
                                    ]

                                    # If Boruta cache exists, corr/var was already applied
                                    # during the first run; skip redundant re-execution.
                                    boruta_crossfold_path = exp.get_boruta_crossfold_path()
                                    boruta_cached = config_boruta and boruta_crossfold_path.exists()

                                    if boruta_cached:
                                        update_progress(
                                            "Skipping corr/var filtering "
                                            "(Boruta cache exists)"
                                        )
                                        fold_processing_config["train_init"] = train_init_tsfel
                                        fold_processing_config["holdout_init"] = test_holdout_tsfel
                                        fold_processing_config["corrvar_features"] = tsfel_features_for_filter
                                    else:
                                        update_progress(
                                            "Applying global correlation/variance filtering"
                                        )
                                        train_corrvar, test_corrvar, corrvar_features = (
                                            extract_feat.filtrage_corr_var(
                                                train_init_tsfel,
                                                test_holdout_tsfel,
                                                patient_col,
                                                target_col,
                                                corr_threshold=corr_threshold,
                                                static_features=static_feats,
                                                keep_static=KEEP_STATIC,
                                                log_correlation_decisions = False,
                                                variable_number_log_path = compare_figs_dir / Path("log_variable_number.json")
                                            )
                                        )
                                        print(
                                            f"[CORR/VAR GLOBAL] {len(tsfel_features_for_filter)} -> "
                                            f"{len(corrvar_features)} features after filtering."
                                        )

                                        # Add filtered train/holdout data to config before Phase 1.
                                        fold_processing_config["train_init"] = train_corrvar
                                        fold_processing_config["holdout_init"] = test_corrvar
                                        fold_processing_config["corrvar_features"] = corrvar_features

                                    # ---- Phase 1: Boruta cross-fold feature resolution ----
                                    if config_boruta:
                                        update_progress(
                                            "Resolving Boruta cross-fold features"
                                        )
                                        boruta_crossfold_features = (
                                            extract_feat.resolve_boruta_crossfold_features(
                                                X=X,
                                                y=y,
                                                groups=groups,
                                                seed=seed,
                                                sgkf=sgkf,
                                                compare_figs_dir=compare_figs_dir,
                                                **fold_processing_config,
                                            )
                                        )
                                        fold_processing_config[
                                            "boruta_crossfold_features"
                                        ] = boruta_crossfold_features

                                if config_models.extraction_type == "time":
                                    fold_processing_config["train_init"] = train_init_df
                                    fold_processing_config["holdout_init"] = test_holdout_df
                                    print(
                                        f"\n[TIME PIPELINE] extraction_type='time' detected for {config_models.models_name}. "
                                        f"Setting train_init_df rows={len(train_init_df)}, "
                                        f"holdout_init_df rows={len(test_holdout_df)}",
                                        flush=True,
                                    )
                                    print(
                                        f"[TIME PIPELINE] X (train_init_df) type={type(X).__name__}, "
                                        f"y shape={y.shape}, groups shape={groups.shape}",
                                        flush=True,
                                    )

                                # ---- Phase 2: Main fold processing with unified Boruta features ----
                                for fold_index, (train_idx, validation_idx) in enumerate(
                                    sgkf.split(X=X, y=y, groups=groups)
                                ):
                                    print(
                                        f"\n─────────────────── Processing Fold "
                                        f"{fold_index + 1}/5 ───────────────────"
                                    )

                                    if config_models.extraction_type == "time":
                                        print(
                                            f"[TIME PIPELINE] Fold {fold_index + 1}: "
                                            f"train_idx length={len(train_idx)}, "
                                            f"validation_idx length={len(validation_idx)}",
                                            flush=True,
                                        )

                                    if type_donnees.value == "score":
                                        train_score_fold = X[train_idx]
                                        validation_score_fold = X[validation_idx]

                                        x_train_processed = (
                                            train_score_fold["score_probability"]
                                            .to_numpy()
                                            .reshape(-1, 1)
                                        )
                                        x_validation_processed = (
                                            validation_score_fold["score_probability"]
                                            .to_numpy()
                                            .reshape(-1, 1)
                                        )
                                        x_holdout_processed = (
                                            test_holdout_score_df["score_probability"]
                                            .to_numpy()
                                            .reshape(-1, 1)
                                        )

                                        y_train_processed = (
                                            train_score_fold[target_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )
                                        y_validation_processed = (
                                            validation_score_fold[target_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )
                                        y_holdout_fold = (
                                            test_holdout_score_df[target_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )

                                        processed_groups = (
                                            train_score_fold[patient_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )
                                        holdout_patient_ids_fold = (
                                            test_holdout_score_df[patient_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )
                                        feature_names_fold = ["score_probability"]

                                        sampling_statistics = {
                                            "n_sains_avant": int(
                                                np.sum(y_train_processed == 0)
                                            ),
                                            "n_malades_avant": int(
                                                np.sum(y_train_processed == 1)
                                            ),
                                            "n_sains_apres": int(
                                                np.sum(y_train_processed == 0)
                                            ),
                                            "n_malades_apres": int(
                                                np.sum(y_train_processed == 1)
                                            ),
                                        }

                                    elif config_models.extraction_type == "TSFEL":
                                        (
                                            x_train_processed,
                                            x_validation_processed,
                                            x_holdout_processed,
                                            y_train_processed,
                                            y_validation_processed,
                                            y_holdout_fold,
                                            processed_groups,
                                            holdout_patient_ids_fold,
                                            feature_names_fold,
                                            sampling_statistics,
                                        ) = preproc.process_tsfel_fold(
                                            fold_index,
                                            train_idx,
                                            validation_idx,
                                            X,
                                            y,
                                            groups,
                                            seed,
                                            **fold_processing_config,
                                        )

                                    elif config_models.extraction_type == "time":
                                        print(
                                            f"[TIME PIPELINE] Fold {fold_index + 1}: Calling preproc.process_time_fold...",
                                            flush=True,
                                        )
                                        (
                                            x_train_processed,
                                            x_validation_processed,
                                            x_holdout_processed,
                                            y_train_processed,
                                            y_validation_processed,
                                            y_holdout_fold,
                                            processed_groups,
                                            holdout_patient_ids_fold,
                                            feature_names_fold,
                                            sampling_statistics,
                                        ) = preproc.process_time_fold(
                                            fold_index,
                                            train_idx,
                                            validation_idx,
                                            seed,
                                            **fold_processing_config,
                                        )
                                        print(
                                            f"[TIME PIPELINE] Fold {fold_index + 1}: process_time_fold returned. "
                                            f"x_train shape={x_train_processed.shape}, "
                                            f"x_val shape={x_validation_processed.shape}, "
                                            f"x_holdout shape={x_holdout_processed.shape}, "
                                            f"y_train positives={int(y_train_processed.sum())}/{len(y_train_processed)}",
                                            flush=True,
                                        )

                                    else:
                                        raise ValueError(
                                            "Unknown or unsupported extraction type: "
                                            f"{config_models.extraction_type}"
                                        )

                                    y_holdout_fold = np.asarray(
                                        y_holdout_fold
                                    ).reshape(-1)
                                    holdout_patient_ids_fold = np.asarray(
                                        holdout_patient_ids_fold
                                    ).reshape(-1)

                                    if y_test_holdout is None:
                                        y_test_holdout = y_holdout_fold.copy()
                                        holdout_patient_ids = (
                                            holdout_patient_ids_fold.copy()
                                        )
                                    else:
                                        np.testing.assert_array_equal(
                                            y_holdout_fold,
                                            y_test_holdout,
                                            err_msg=(
                                                "Holdout labels differ between folds."
                                            ),
                                        )
                                        np.testing.assert_array_equal(
                                            holdout_patient_ids_fold,
                                            holdout_patient_ids,
                                            err_msg=(
                                                "Holdout patient order differs "
                                                "between folds."
                                            ),
                                        )

                                    if x_holdout_processed.shape[0] != len(
                                        y_test_holdout
                                    ):
                                        raise ValueError(
                                            f"Fold {fold_index}: holdout X/y size "
                                            "mismatch: "
                                            f"{x_holdout_processed.shape[0]} rows "
                                            f"for {len(y_test_holdout)} labels."
                                        )

                                    folds_X_train.append(x_train_processed)
                                    folds_X_validation.append(
                                        x_validation_processed
                                    )
                                    folds_X_holdout.append(x_holdout_processed)
                                    folds_y_train.append(y_train_processed)
                                    folds_y_validation.append(
                                        y_validation_processed
                                    )
                                    folds_groups.append(processed_groups)
                                    folds_sampling_stats.append(
                                        sampling_statistics
                                    )
                                    folds_feature_names.append(
                                        list(feature_names_fold)
                                    )

                                np.save(exp.get_var_path(), final_features)
                                print("All 5 folds were computed successfully!")
                                logger.debug(
                                    "Fold groups: %s",
                                    [g.shape for g in folds_groups],
                                )
                                logger.debug(
                                    "Holdout shapes: %s",
                                    [X_fold.shape for X_fold in folds_X_holdout],
                                )
                                logger.debug(
                                    "Holdout patients: %d",
                                    len(holdout_patient_ids),
                                )

                                model_name = config_models.models_name
                                output_directory = exp.get_output_path(model_name)
                                HYPERPARAMS_FILE = output_directory / 'best_hyperparameters.json'
                                saved_configs = {}

                                if HYPERPARAMS_FILE.exists():
                                    try:
                                        with open(
                                            HYPERPARAMS_FILE,
                                            "r",
                                            encoding="utf-8",
                                        ) as config_file:
                                            saved_configs = json.load(config_file)
                                    except (json.JSONDecodeError, OSError) as error:
                                        print(
                                            f"[WARNING] Unable to read {HYPERPARAMS_FILE}: {error}. "
                                            "A new Optuna search may be required."
                                        )
                                        saved_configs = {}

                                hyperparameters_already_available = (
                                    model_name in saved_configs
                                    and isinstance(saved_configs[model_name], dict)
                                    and bool(saved_configs[model_name])
                                )

                                if RUN_OPTUNA and RUN_TRAINING:
                                    if hyperparameters_already_available:
                                        print(
                                            f"[OPTUNA] Existing hyperparameters found for {model_name} "
                                            f"in '{HYPERPARAMS_FILE}'. Search skipped."
                                        )
                                    else:
                                        print("[OPTUNA] No saved configuration found.")
                                        update_progress("Running Optuna hyperparameter search")
                                        print('[OPTUNA] Search started...')
                                        save_optuna_name = (
                                        f"{config_models.models_name}_{exp.shortdirname()}"
                                        f"_fresh_{datetime.now():%Y%m%d_%H%M%S_%f}"
                                        )
                                        print(f"[DISK-SAVE] The best parameters will be saved to '{HYPERPARAMS_FILE}'.")
                                        if config_models.models_name == 'InceptionTimeModified':
                                            from utilitaries.optuna.optuna_inception_utils import run_stage1_search
                                            study = run_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                                            best_parameters = study.best_params
                                            best_parameters['out_channels'] = 2 ** best_parameters.pop('out_channels_exp')
                                            best_parameters['bottleneck_channels'] = 2 ** best_parameters.pop('bottleneck_channels_exp')
                                            best_parameters['batch_size'] = 2 ** best_parameters.pop('batch_size_exp')
                                            saved_configs[config_models.models_name] = best_parameters
                                        elif config_models.models_name == 'LstmTimeModified':
                                            from utilitaries.optuna.optuna_lstm_utils import run_lstm_stage1_search
                                            study = run_lstm_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                                            best_parameters = study.best_params
                                            best_parameters['hidden_size'] = 2 ** best_parameters.pop('hidden_size_exp')
                                            best_parameters['batch_size'] = 2 ** best_parameters.pop('batch_size_exp')
                                            if best_parameters.get('clip_grad') == 0.0:
                                                best_parameters['clip_grad'] = None
                                            saved_configs[config_models.models_name] = best_parameters
                                        elif config_models.models_name == 'VanillaTransformerModified':
                                            from utilitaries.optuna.optuna_vanilla_utils import run_stage1_search
                                            study = run_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                                            best_parameters = study.best_params
                                            best_parameters['d_model'] = 2 ** best_parameters.pop('d_model_exp')
                                            best_parameters['dim_feedforward'] = 2 ** best_parameters.pop('dim_feedforward_exp')
                                            best_parameters['batch_size'] = 2 ** best_parameters.pop('batch_size_exp')
                                            saved_configs[config_models.models_name] = best_parameters
                                        elif config_models.models_name == 'XGBoost TSFEL':
                                            from utilitaries.optuna.optuna_xgb_utils import run_xgb_stage1_search
                                            study = run_xgb_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name, fixed_params = {"tree_method" : "hist", "device" : "cuda"})
                                            saved_configs[config_models.models_name] = study.best_params
                                        elif config_models.models_name == 'SVC TSFEL':
                                            from utilitaries.optuna.optuna_svc_utils import run_svc_stage1_search
                                            study = run_svc_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                                            saved_configs[config_models.models_name] = study.best_params
                                        elif config_models.models_name == 'RandomForest TSFEL':
                                            from utilitaries.optuna.optuna_rf_utils import run_rf_stage1_search
                                            study = run_rf_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                                            best_parameters = study.best_params
                                            if best_parameters.get('max_depth') == 0:
                                                best_parameters['max_depth'] = None
                                            saved_configs[config_models.models_name] = best_parameters
                                        else:
                                            raise ValueError(
                                                f"No Optuna search is configured for "
                                                f"{config_models.models_name}."
                                            )
                                        with open(HYPERPARAMS_FILE, 'w') as config_file:
                                            json.dump(saved_configs, config_file, indent=4)
                                        print(f"[DISK-SAVE] Best parameters saved to '{HYPERPARAMS_FILE}' for {config_models.models_name}.")

                                update_progress("Preparing training")
                                print('Training started')
                                parameters = {}
                                uses_optuna_config = False

                                if use_optuna.value:
                                    if model_name in saved_configs:
                                        print(
                                            f"[LOAD] Optuna configuration found in "
                                            f"{HYPERPARAMS_FILE} for {model_name}."
                                        )
                                        parameters = saved_configs[model_name].copy()
                                        uses_optuna_config = True
                                    else:
                                        print(
                                            f"[WARNING] No Optuna configuration found for "
                                            f"{model_name}. Using default parameters."
                                        )
                                        parameters = DEFAULT_PARAMS.get(
                                            model_name,
                                            {},
                                        ).copy()
                                else:
                                    parameters = DEFAULT_PARAMS.get(
                                        model_name,
                                        {},
                                    ).copy()
                                print(f"--> Applied parameters: {parameters}\n")
                                
                                if len(folds_X_train) == 0:
                                    raise ValueError('The fold lists are empty')
                                is_dl_model = config_models.extraction_type == 'time'
                                num_dimensions = len(folds_X_train[0].shape) if hasattr(folds_X_train[0], 'shape') else 0
                                if is_dl_model and num_dimensions != 3:
                                    raise ValueError(f"Shape mismatch: model {config_models.models_name} expects a 3D matrix [patients, time, features], but X_train has {num_dimensions} dimension(s). Is the pipeline configured in 'time' mode?")
                                elif not is_dl_model and num_dimensions != 2:
                                    raise ValueError(f"Shape mismatch: model {config_models.models_name} expects a 2D tabular matrix, but X_train has {num_dimensions} dimension(s). Is the pipeline configured in 'TSFEL' mode?")

                                if RUN_TRAINING and not is_score_mode:
                                    folds_X_fit_exact = []
                                    folds_y_fit_exact = []
                                    learning_curve_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
                                    lc_train_scores = np.zeros((5, len(learning_curve_fractions)))
                                    lc_val_scores = np.zeros((5, len(learning_curve_fractions)))
                                    lc_sample_sizes = []
                                    for training_fold_index in range(5):
                                        update_progress(f"Training fold {training_fold_index + 1}/5")
                                        print(f'\n─────────────────── Training Fold {training_fold_index + 1}/5 ───────────────────')
                                        x_training_fold = folds_X_train[training_fold_index]
                                        y_training_fold = folds_y_train[training_fold_index]
                                        training_groups_fold = folds_groups[training_fold_index]
                                        training_groups_fold = np.asarray(
                                            training_groups_fold
                                        ).reshape(-1)

                                        logger.debug(
                                            "FOLD GROUPS: rows=%d, unique=%d, dtype=%s, first=%s",
                                            len(training_groups_fold),
                                            np.unique(training_groups_fold).size,
                                            training_groups_fold.dtype,
                                            training_groups_fold[:10],
                                        )

                                        assert len(training_groups_fold) == len(x_training_fold), (
                                            "Group vector and training data are not aligned: "
                                            f"{len(training_groups_fold)} groups for "
                                            f"{len(x_training_fold)} rows."
                                        )

                                        assert np.unique(training_groups_fold).size >= 3, (
                                            "The processed fold does not contain valid patient identifiers. "
                                            f"Only {np.unique(training_groups_fold).size} unique group(s) found."
                                        )
                                        model_path_fold = exp.get_model_path(config_models.models_name, training_fold_index, extension, uses_optuna_config=uses_optuna_config)
                                        exact_x_path = exp.get_lasso_path('X', training_fold_index, 'parquet')
                                        exact_y_path = exp.get_lasso_path('y', training_fold_index, 'npy')
                                        if os.path.exists(model_path_fold) and os.path.getsize(model_path_fold) > 0:
                                            print(f'--> Previously trained model found at {model_path_fold}. Skipping to the next fold.')
                                            continue
                                        logger.debug(
                                            'TRAIN Fold %d: x_training_fold shape=%s',
                                            training_fold_index + 1,
                                            x_training_fold.shape,
                                        )
                                        X_train_final_fold, y_train_final = (x_training_fold, y_training_fold)
                                        groups_final_fold = training_groups_fold
                                        X_calib, y_calib = (None, None)
                                        if calibration.value:
                                            if config_balance.balance_method != '':
                                                print(f"    [INFO] Analytical prior calibration. Using 100% of the fold for training...")
                                                X_train_final_fold = x_training_fold
                                                y_train_final = y_training_fold
                                                groups_final_fold = training_groups_fold
                                                X_calib = None
                                                y_calib = None
                                            else:
                                                print(f'    [INFO] Calibration enabled ({calibration_mode.value}). Splitting the fold with StratifiedGroupKFold...')
                                                current_groups = np.asarray(training_groups_fold)
                                                logger.debug(
                                                    "GROUPS: rows=%d, unique=%d",
                                                    len(current_groups),
                                                    len(np.unique(current_groups)),
                                                )
                                                skf_calib = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                                                train_idx_calib, calib_idx = next(skf_calib.split(x_training_fold, y_training_fold, groups=training_groups_fold))
                                                X_train_final_fold = x_training_fold[train_idx_calib]
                                                y_train_final = y_training_fold[train_idx_calib]
                                                groups_final_fold = np.asarray(training_groups_fold)[train_idx_calib]
                                                X_calib = x_training_fold[calib_idx]
                                                y_calib = y_training_fold[calib_idx]
                                                logger.debug(
                                                    'Shapes - train: %s, calib: %s',
                                                    X_train_final_fold.shape,
                                                    X_calib.shape,
                                                )
                                        print(f'    [LEARNING CURVE] Starting the unified learning-curve stage loop...')
                                        X_validation_fold_lc = folds_X_validation[training_fold_index]
                                        X_val_np = X_validation_fold_lc.to_numpy() if hasattr(X_validation_fold_lc, 'to_numpy') else np.asarray(X_validation_fold_lc)
                                        y_val_np = np.asarray(folds_y_validation[training_fold_index])
                                        for fraction_index, fraction in enumerate(learning_curve_fractions):
                                            is_final_stage = fraction == 1.0
                                            x_subset, y_subset, subset_mask = training.get_learning_curve_chunk(X_train_final_fold, y_train_final, groups_final_fold, fraction, is_dl_model, seed)
                                            if training_fold_index == 0:
                                                lc_sample_sizes.append(len(x_subset))
                                            print(f'      -> Stage {int(fraction * 100)}% ({len(x_subset)} patients)' + (' [FINAL TRAINING & DISK SAVE]' if is_final_stage else ' [TEMPORARY]'))
                                            if len(np.unique(y_subset)) < 2:
                                                print(f'          [WARNING] Only one class is present; skipping this stage.')
                                                lc_train_scores[training_fold_index, fraction_index] = np.nan
                                                lc_val_scores[training_fold_index, fraction_index] = np.nan
                                                continue
                                            current_save_path = model_path_fold if is_final_stage else None
                                            groups_fold_array = np.asarray(groups_final_fold).ravel()
                                            lasso_args = {
                                                'X_raw': X_train_final_fold, 
                                                'y_raw': y_train_final, 
                                                'groups_mask': groups_fold_array[subset_mask] if subset_mask is not None else groups_fold_array,
                                                'file_X': exact_x_path, 
                                                'file_y': exact_y_path,
                                            } if config_models.models_name == 'Logistic Regression Lasso TSFEL' else None
                                            FC_UNITS_MAP = {'none': None, '64': (64,), '128': (128,), '256': (256,), '128_64': (128, 64), '256_128': (256, 128)}
                                            if 'fc_units' in parameters and isinstance(parameters['fc_units'], str):
                                                parameters['fc_units'] = FC_UNITS_MAP[parameters['fc_units']]
                                            final_model_to_save, train_auc, val_auc = training.fit_model_by_name(model_name=config_models.models_name, X_train=x_subset, y_train=y_subset, X_val=X_val_np, y_val=y_val_np, seed=seed, is_final_palier=is_final_stage, save_path=current_save_path, lasso_args=lasso_args, **parameters)
                                            lc_train_scores[training_fold_index, fraction_index] = train_auc
                                            lc_val_scores[training_fold_index, fraction_index] = val_auc
                                            if is_final_stage and (not is_dl_model) and (final_model_to_save is not None):
                                                if calibration.value:
                                                    if config_balance.balance_method != '':
                                                        sampling_stats = folds_sampling_stats[training_fold_index]
                                                        numerator = sampling_stats['n_malades_avant'] * sampling_stats['n_sains_apres']
                                                        denominator = sampling_stats['n_sains_avant'] * sampling_stats['n_malades_apres']
                                                        if denominator > 0 and numerator > 0:
                                                            beta = numerator / denominator
                                                            final_model_to_save = training.apply_prior_calibration(final_model_to_save, beta)
                                                            logger.debug(
                                                                'Prior Calibration OK (beta = %.4f)', beta
                                                            )
                                                        else:
                                                            print(f'    [WARNING] Cannot compute beta because a class is missing.')
                                                    else:
                                                        print(f'    [INFO] Applying calibration {calibration_mode.value} to the held-out calibration set...')
                                                        final_model_to_save = training.apply_model_calibration(final_model_to_save, X_calib, y_calib, calibration_mode.value)
                                                joblib.dump(final_model_to_save, model_path_fold)
                                                print(f'--> Final model saved to {model_path_fold}')
                                        folds_X_fit_exact.append(X_train_final_fold)
                                        folds_y_fit_exact.append(y_train_final)
                                    print('Cross-validation completed successfully. Five models were saved.')

                                output_dir = exp.get_output_path(config_models.models_name, uses_optuna_config=uses_optuna_config)
                                print('Output directory ready:', output_dir)

                                if RUN_TRAINING and not is_score_mode:
                                    try:
                                        sfu.plot_collected_learning_curve(lc_sample_sizes, lc_train_scores, lc_val_scores, config_models.models_name, savefig=save_figure.value, folder=output_dir, transparent=config_transparent)
                                    except:
                                        print("Non-blocking error: the learning curve could not be saved because it already exists.")
                                fold_metrics = []

                                all_validation_scores = []
                                all_train_scores = []
                                all_y_true_report = []
                                all_y_pred_report = []

                                # These remain Python lists throughout the fold loop.
                                all_y_validation_global = []
                                all_probas_uncalib = []
                                all_probas_calib = []

                                effective_calibration = (
                                    calibration.value
                                    if not is_score_mode
                                    else False
                                )

                                update_progress("Evaluating the 5 folds")

                                for fold_idx in range(5):
                                    update_progress(
                                        f"Evaluating fold {fold_idx + 1}/5"
                                    )

                                    print(
                                        f"\n─────────────────── "
                                        f"Evaluating Fold {fold_idx + 1}/5 "
                                        f"───────────────────"
                                    )

                                    # =========================================================
                                    # Score mode
                                    # =========================================================
                                    if is_score_mode:
                                        X_validation = folds_X_validation[fold_idx]
                                        y_validation_fold = folds_y_validation[fold_idx]

                                        fold_y_true = np.asarray(
                                            y_validation_fold,
                                            dtype=int,
                                        ).reshape(-1)

                                        fold_probas_uncalib = np.asarray(
                                            X_validation,
                                            dtype=float,
                                        ).reshape(-1)

                                        # The clinical score is already a probability.
                                        # No additional calibration is applied.
                                        fold_probas_calib = (
                                            fold_probas_uncalib.copy()
                                        )

                                        print(
                                            f"[SCORE] Fold {fold_idx + 1}: "
                                            f"{len(fold_y_true)} patients."
                                        )

                                    # =========================================================
                                    # Model mode
                                    # =========================================================
                                    else:
                                        X_train = folds_X_train[fold_idx]
                                        X_validation = folds_X_validation[fold_idx]

                                        y_train = folds_y_train[fold_idx]
                                        y_validation_fold = folds_y_validation[fold_idx]

                                        loaded_model = exp.get_model_path(
                                            config_models.models_name,
                                            fold_idx,
                                            extension,
                                            uses_optuna_config=uses_optuna_config,
                                        )

                                        print(
                                            "Loaded model:",
                                            loaded_model,
                                        )

                                        if (
                                            config_models.models_name
                                            == "InceptionTimeModified"
                                        ):
                                            evaluation_result = (
                                                evaluate.evaluate_inception_fold(
                                                    X_validation,
                                                    y_validation_fold,
                                                    loaded_model,
                                                )
                                            )

                                        elif (
                                            config_models.models_name
                                            == "LstmTimeModified"
                                        ):
                                            evaluation_result = (
                                                evaluate.evaluate_lstm_fold(
                                                    fold_idx,
                                                    X_validation,
                                                    y_validation_fold,
                                                    loaded_model,
                                                )
                                            )

                                        elif (
                                            config_models.models_name
                                            == "VanillaTransformerModified"
                                        ):
                                            evaluation_result = (
                                                evaluate.evaluate_vanilla_transformer_fold(
                                                    fold_idx,
                                                    X_validation,
                                                    y_validation_fold,
                                                    loaded_model,
                                                )
                                            )

                                        elif (
                                            config_models.extraction_type
                                            == "TSFEL"
                                        ):
                                            evaluation_result = (
                                                evaluate.evaluate_tsfel_fold(
                                                    fold_idx,
                                                    X_train,
                                                    X_validation,
                                                    y_train,
                                                    y_validation_fold,
                                                    loaded_model,
                                                    calibration.value,
                                                )
                                            )

                                            # Kept only for the accuracy summary and
                                            # classification report.
                                            all_validation_scores.append(
                                                evaluation_result["test_score"]
                                            )

                                            all_train_scores.append(
                                                evaluation_result["train_score"]
                                            )

                                            all_y_true_report.extend(
                                                evaluation_result["y_test"]
                                            )

                                            all_y_pred_report.extend(
                                                evaluation_result["y_pred_test"]
                                            )

                                        else:
                                            raise ValueError(
                                                "Unsupported model or extraction type: "
                                                f"{config_models.models_name}"
                                            )

                                        fold_y_true = np.asarray(
                                            evaluation_result["y_test"],
                                            dtype=int,
                                        ).reshape(-1)

                                        fold_probas_uncalib = np.asarray(
                                            evaluation_result["probas_uncalib"],
                                            dtype=float,
                                        ).reshape(-1)

                                        raw_fold_probas_calib = (
                                            evaluation_result.get("probas_calib")
                                        )

                                        # Some evaluation functions may return None when
                                        # calibration is disabled.
                                        if raw_fold_probas_calib is None:
                                            fold_probas_calib = (
                                                fold_probas_uncalib.copy()
                                            )
                                        else:
                                            fold_probas_calib = np.asarray(
                                                raw_fold_probas_calib,
                                                dtype=float,
                                            ).reshape(-1)

                                    # =========================================================
                                    # Checks shared by all models and score mode
                                    # =========================================================
                                    if (
                                        fold_y_true.shape[0]
                                        != fold_probas_uncalib.shape[0]
                                    ):
                                        raise ValueError(
                                            f"Fold {fold_idx + 1}: y_true and "
                                            "uncalibrated probabilities have different "
                                            f"lengths: {len(fold_y_true)} != "
                                            f"{len(fold_probas_uncalib)}."
                                        )

                                    if (
                                        fold_y_true.shape[0]
                                        != fold_probas_calib.shape[0]
                                    ):
                                        raise ValueError(
                                            f"Fold {fold_idx + 1}: y_true and "
                                            "calibrated probabilities have different "
                                            f"lengths: {len(fold_y_true)} != "
                                            f"{len(fold_probas_calib)}."
                                        )

                                    # Select the probabilities corresponding to the
                                    # configuration that will be reported.
                                    fold_probabilities = (
                                        fold_probas_calib
                                        if effective_calibration
                                        else fold_probas_uncalib
                                    )

                                    # =========================================================
                                    # Metrics computed independently on this fold
                                    # =========================================================
                                    current_fold_metrics = (
                                        sfu.compute_binary_metrics(
                                            probas=fold_probabilities,
                                            y_true=fold_y_true,
                                        )
                                    )

                                    fold_metrics.append(
                                        current_fold_metrics
                                    )

                                    print(
                                        f"[FOLD {fold_idx + 1}] "
                                        f"AUC={current_fold_metrics['auc']:.4f}, "
                                        f"AUPRC={current_fold_metrics['auprc']:.4f}, "
                                        f"Brier={current_fold_metrics['brier']:.4f}, "
                                        f"ICI={current_fold_metrics['ici']:.4f}"
                                    )

                                    # =========================================================
                                    # OOF pooling
                                    # Each validation patient is appended exactly once.
                                    # =========================================================
                                    all_y_validation_global.extend(
                                        fold_y_true.tolist()
                                    )

                                    all_probas_uncalib.extend(
                                        fold_probas_uncalib.tolist()
                                    )

                                    all_probas_calib.extend(
                                        fold_probas_calib.tolist()
                                    )


                                # =============================================================
                                # End of the five-fold loop
                                # =============================================================

                                if len(fold_metrics) != 5:
                                    raise RuntimeError(
                                        "Five fold metric dictionaries were expected, "
                                        f"but {len(fold_metrics)} were collected."
                                    )

                                # Mean and standard deviation across the five fold metrics.
                                fold_metrics_summary = (
                                    sfu.summarize_fold_metrics(
                                        fold_metrics,
                                        ddof=1,
                                    )
                                )

                                print(
                                    "\n"
                                    + "=" * 20
                                    + " FOLD METRICS SUMMARY "
                                    + "=" * 20
                                )

                                for metric_name, metric_summary in (
                                    fold_metrics_summary.items()
                                ):
                                    print(
                                        f"{metric_name}: "
                                        f"{metric_summary['mean']:.4f} "
                                        f"(± {metric_summary['std']:.4f})"
                                    )

                                # Convert pooled OOF data only after all folds have
                                # been appended.
                                all_y_validation_global = np.asarray(
                                    all_y_validation_global,
                                    dtype=int,
                                )

                                all_probas_uncalib = np.asarray(
                                    all_probas_uncalib,
                                    dtype=float,
                                )

                                all_probas_calib = np.asarray(
                                    all_probas_calib,
                                    dtype=float,
                                )

                                if not (
                                    len(all_y_validation_global)
                                    == len(all_probas_uncalib)
                                    == len(all_probas_calib)
                                ):
                                    raise RuntimeError(
                                        "The pooled OOF arrays are not aligned: "
                                        f"y_true={len(all_y_validation_global)}, "
                                        f"uncalibrated={len(all_probas_uncalib)}, "
                                        f"calibrated={len(all_probas_calib)}."
                                    )

                                print(
                                    "\n"
                                    + "=" * 20
                                    + " GLOBAL CROSS-VALIDATION SUMMARY "
                                    + "=" * 20
                                )

                                if is_score_mode:
                                    print(
                                        "Score evaluation completed on "
                                        f"{len(all_y_validation_global)} "
                                        "pooled validation patients."
                                    )

                                elif config_models.extraction_type == "TSFEL":
                                    mean_acc = np.mean(
                                        all_validation_scores
                                    )

                                    std_acc = np.std(
                                        all_validation_scores,
                                        ddof=1,
                                    )

                                    print(
                                        f"Mean accuracy: {mean_acc:.4f} "
                                        f"(± {std_acc:.4f})"
                                    )

                                    print(
                                        "\nAggregated classification report "
                                        "across all 5 folds:"
                                    )

                                    print(
                                        classification_report(
                                            all_y_true_report,
                                            all_y_pred_report,
                                            target_names=[
                                                "Alive",
                                                "Deceased",
                                            ],
                                            zero_division=0,
                                        )
                                    )

                                print("=" * 79)

                                print(
                                    "\nGenerating the pooled "
                                    "out-of-fold calibration curve..."
                                )

                                logger.debug(
                                    "OOF SIZES: y_true=%d, uncalib=%d, calib=%d",
                                    len(all_y_validation_global),
                                    len(all_probas_uncalib),
                                    len(all_probas_calib),
                                )

                                sfu.calibration_curve_homemade(
                                    all_probas_uncalib,
                                    all_probas_calib,
                                    all_y_validation_global,
                                    config_models.models_name,
                                    effective_calibration,
                                    save_figure.value,
                                    output_dir,
                                    config_transparent,
                                    calibration_mode.value,
                                )

                                # Probabilities used for all final OOF figures and metrics.
                                probabilities = (
                                    all_probas_calib
                                    if effective_calibration
                                    else all_probas_uncalib
                                )

                                y_validation = (
                                    all_y_validation_global
                                )
                                if RUN_LASSO and config_models.models_name == 'Logistic Regression Lasso TSFEL':
                                    # Check if all Lasso path figures already exist.
                                    existing_lasso_paths = [
                                        output_dir / f"L1_Log_path_fold_{i + 1}.png"
                                        for i in range(5)
                                    ]
                                    if all(path.exists() for path in existing_lasso_paths):
                                        print(
                                            "Lasso path figures already exist. "
                                            "Skipping regeneration."
                                        )
                                    else:
                                        print("Lasso path started")
                                        print(
                                            "Generating Lasso paths with warm "
                                            "start and multiprocessing..."
                                        )
                                        c_grid = np.logspace(-4, 4, 100)

                                        def process_single_fold(fold_index):
                                            file_X = exp.get_lasso_path('X', fold_index, 'parquet')
                                            file_y = exp.get_lasso_path('y', fold_index, 'npy')
                                            x_exact_fit = pl.read_parquet(file_X).to_numpy()
                                            y_exact_fit = np.load(file_y)
                                            model_path = exp.get_model_path('Logistic Regression Lasso TSFEL', fold_index, extension)
                                            l1_model = joblib.load(model_path)
                                            best_c = l1_model.C
                                            lr_path_model = LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=100001, random_state=seed, warm_start=True)
                                            coefficient_list = []
                                            sorted_Cs = np.sort(c_grid)
                                            for c_value in sorted_Cs:
                                                lr_path_model.set_params(C=c_value)
                                                lr_path_model.fit(x_exact_fit, y_exact_fit)
                                                coefficient_list.append(lr_path_model.coef_[0].copy())
                                            return (sorted_Cs, np.array(coefficient_list), best_c, x_exact_fit.shape[1])
                                        results = Parallel(n_jobs=-1)((delayed(process_single_fold)(f_idx) for f_idx in range(5)))
                                        for fold_idx_L1, (sorted_Cs, coefficient_path, best_c, n_features) in enumerate(results):
                                            plt.figure(figsize=(10, 6))
                                            plt.plot(sorted_Cs, coefficient_path, alpha=0.7)
                                            plt.axvline(x=best_c, color='black', linestyle='--', linewidth=2, label=f'C optimal (Fold {fold_idx_L1 + 1}) = {best_c:.4f}')
                                            plt.xscale('log')
                                            plt.xlabel('Regularization parameter C (Log Scale)')
                                            plt.ylabel(f'Coefficients ({n_features} features)')
                                            plt.title(f'L1 Regularization Path - Fold {fold_idx_L1 + 1}\nOptimized (Warm Start)')
                                            plt.grid(True, which='both', ls='-', alpha=0.5)
                                            plt.legend()
                                            if save_figure.value:
                                                filename = f'L1_Log_path_fold_{fold_idx_L1 + 1}.png'
                                                plt.savefig(output_dir / Path(filename), dpi=300, bbox_inches='tight', transparent=config_transparent)
                                            plt.show()
                                update_progress(
                                    "Generating final metrics and figures"
                                )

                                auc_final, fpr, tpr, thresholds_roc = (
                                    sfu.roc_curve_homemade(
                                        probabilities,
                                        y_validation,
                                        config_models.models_name,
                                        save_figure.value,
                                        output_dir,
                                        config_transparent,
                                    )
                                )

                                plt.close("all")

                                auprc_final, precision, recall, thresholds = (
                                    sfu.prc_curve_homemade(
                                        probabilities,
                                        y_validation,
                                        config_models.models_name,
                                        save_figure.value,
                                        output_dir,
                                        config_transparent,
                                    )
                                )

                                plt.close("all")

                                (
                                    non_overlap_area,
                                    asymmetric_uncertainty,
                                    mean_risk_diff,
                                    mean_p1,
                                ) = sfu.kde_plot_homemade(
                                    probabilities,
                                    y_validation,
                                    config_models.models_name,
                                    save_figure.value,
                                    output_dir,
                                    config_transparent,
                                )

                                plt.close("all")

                                best_f1, best_t = (
                                    sfu.f1_score_evolution(
                                        probabilities,
                                        y_validation,
                                        config_models.models_name,
                                        save_figure.value,
                                        output_dir,
                                        config_transparent,
                                    )
                                )

                                plt.close("all")

                                y_pred, mcc = (
                                    sfu.confusion_matrix_homemade(
                                        probabilities,
                                        y_validation,
                                        best_t,
                                        config_models.models_name,
                                        save_figure.value,
                                        output_dir,
                                        config_transparent,
                                    )
                                )

                                plt.close("all")

                                global_brier = sfu.brier_evolution(
                                    probabilities,
                                    y_validation,
                                    save_figure.value,
                                    output_dir,
                                    transparent=config_transparent,
                                )

                                plt.close("all")

                                sfu.calibration_per_risk_brackets(
                                    probabilities,
                                    y_validation,
                                    save_figure.value,
                                    output_dir,
                                    transparent=config_transparent,
                                )

                                calibration_stats = (
                                    sfu.get_calibration_stats(
                                        probabilities,
                                        y_validation,
                                    )
                                )

                                all_results = {
                                    # =========================================================
                                    # Pooled OOF predictions
                                    # =========================================================
                                    "y_true_oof": y_validation,

                                    "probas_oof": probabilities,

                                    "probas_uncalib_oof": (
                                        all_probas_uncalib
                                    ),

                                    "probas_calib_oof": (
                                        all_probas_calib
                                    ),
                                    "preds_oof": y_pred,

                                    # =========================================================
                                    # Threshold-dependent OOF metrics
                                    # =========================================================
                                    "f1_score_oof": best_f1,
                                    "mcc_oof": mcc,

                                    # =========================================================
                                    # Discrimination OOF metrics
                                    # =========================================================
                                    "auc_oof": auc_final,
                                    "auprc_oof": auprc_final,

                                    # =========================================================
                                    # Calibration OOF metrics
                                    # =========================================================
                                    "brier_oof": global_brier,

                                    "calibration_intercept_oof": (
                                        calibration_stats["intercept"]
                                    ),

                                    "calibration_slope_oof": (
                                        calibration_stats["slope"]
                                    ),

                                    "ici_oof": calibration_stats["ici"],

                                    "e90_oof": calibration_stats["e90"],

                                    # "eMax_oof": calibration_stats["eMax"],

                                    # =========================================================
                                    # Complete fold information
                                    # =========================================================
                                    "fold_metrics": fold_metrics,
                                    "fold_metrics_summary": (
                                        fold_metrics_summary
                                    ),

                                    # =========================================================
                                    # Other pooled OOF metrics
                                    # =========================================================
                                    "non_overlap_area": non_overlap_area,

                                    "asymetric_incertitude": (
                                        asymmetric_uncertainty
                                    ),

                                    "mean_risk_diff": mean_risk_diff,

                                    "mean_deaths_prediction": mean_p1,
                                }

                                # Add, for each metric:
                                # - the five independent fold values;
                                # - their mean;
                                # - their sample standard deviation.
                                for metric_name, metric_summary in (
                                    fold_metrics_summary.items()
                                ):
                                    all_results[
                                        f"{metric_name}_per_fold"
                                    ] = metric_summary["fold_values"]

                                    all_results[
                                        f"{metric_name}_mean"
                                    ] = metric_summary["mean"]

                                    all_results[
                                        f"{metric_name}_std"
                                    ] = metric_summary["std"]


                                # =============================================================
                                # Independent holdout evaluation
                                # The threshold is fixed from pooled OOF predictions.
                                # =============================================================
                                if RUN_TEST:
                                    update_progress(
                                        "Evaluating the independent holdout"
                                    )

                                    holdout_output_dir = (
                                        output_dir / "holdout"
                                    )
                                    holdout_output_dir.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                    holdout_probas_uncalib_per_model = []
                                    holdout_probas_calib_per_model = []

                                    # Fold-specific TSFEL estimators used later for interpretability.
                                    fold_models_for_interpretability = []

                                    # ============================================================
                                    # HOLDOUT PREDICTIONS FOR EACH FOLD MODEL
                                    # ============================================================

                                    for holdout_fold_idx in range(5):
                                        X_test_holdout_fold = (
                                            folds_X_holdout[
                                                holdout_fold_idx
                                            ]
                                        )

                                        model_path = exp.get_model_path(
                                            config_models.models_name,
                                            holdout_fold_idx,
                                            extension,
                                            uses_optuna_config=(
                                                uses_optuna_config
                                            ),
                                        )

                                        # --------------------------------------------------------
                                        # Clinical score mode
                                        # --------------------------------------------------------

                                        if is_score_mode:
                                            fold_uncalibrated_probabilities = (
                                                np.asarray(
                                                    X_test_holdout_fold,
                                                    dtype=float,
                                                ).reshape(-1)
                                            )

                                            fold_calibrated_probabilities = (
                                                fold_uncalibrated_probabilities.copy()
                                            )

                                        # --------------------------------------------------------
                                        # Temporal models
                                        # --------------------------------------------------------

                                        elif (
                                            config_models.models_name
                                            == "InceptionTimeModified"
                                        ):
                                            result = (
                                                evaluate.evaluate_inception_fold(
                                                    X_test_holdout_fold,
                                                    y_test_holdout,
                                                    model_path,
                                                )
                                            )

                                        elif (
                                            config_models.models_name
                                            == "LstmTimeModified"
                                        ):
                                            result = (
                                                evaluate.evaluate_lstm_fold(
                                                    holdout_fold_idx,
                                                    X_test_holdout_fold,
                                                    y_test_holdout,
                                                    model_path,
                                                )
                                            )

                                        elif (
                                            config_models.models_name
                                            == "VanillaTransformerModified"
                                        ):
                                            result = (
                                                evaluate.evaluate_vanilla_transformer_fold(
                                                    holdout_fold_idx,
                                                    X_test_holdout_fold,
                                                    y_test_holdout,
                                                    model_path,
                                                )
                                            )

                                        # --------------------------------------------------------
                                        # TSFEL models
                                        # --------------------------------------------------------

                                        elif (
                                            config_models.extraction_type
                                            == "TSFEL"
                                        ):
                                            result = (
                                                evaluate.evaluate_tsfel_fold(
                                                    holdout_fold_idx,
                                                    folds_X_train[
                                                        holdout_fold_idx
                                                    ],
                                                    X_test_holdout_fold,
                                                    folds_y_train[
                                                        holdout_fold_idx
                                                    ],
                                                    y_test_holdout,
                                                    model_path,
                                                    calibration.value,
                                                )
                                            )

                                            fold_models_for_interpretability.append(
                                                joblib.load(model_path)
                                            )

                                        else:
                                            raise ValueError(
                                                "Unsupported holdout model: "
                                                f"{config_models.models_name}"
                                            )

                                        # --------------------------------------------------------
                                        # Retrieve uncalibrated / calibrated probabilities
                                        # --------------------------------------------------------

                                        if not is_score_mode:
                                            fold_uncalibrated_probabilities = (
                                                np.asarray(
                                                    result[
                                                        "probas_uncalib"
                                                    ],
                                                    dtype=float,
                                                ).reshape(-1)
                                            )

                                            raw_fold_calibrated = (
                                                result.get(
                                                    "probas_calib"
                                                )
                                            )

                                            if raw_fold_calibrated is None:
                                                fold_calibrated_probabilities = (
                                                    fold_uncalibrated_probabilities.copy()
                                                )
                                            else:
                                                fold_calibrated_probabilities = (
                                                    np.asarray(
                                                        raw_fold_calibrated,
                                                        dtype=float,
                                                    ).reshape(-1)
                                                )

                                        if (
                                            len(
                                                fold_uncalibrated_probabilities
                                            )
                                            != len(y_test_holdout)
                                        ):
                                            raise ValueError(
                                                "Holdout prediction length mismatch "
                                                f"for fold {holdout_fold_idx + 1}."
                                            )

                                        holdout_probas_uncalib_per_model.append(
                                            fold_uncalibrated_probabilities
                                        )

                                        holdout_probas_calib_per_model.append(
                                            fold_calibrated_probabilities
                                        )

                                    # ============================================================
                                    # ENSEMBLE OF THE FIVE FOLD MODELS
                                    # ============================================================

                                    holdout_probas_uncalib_per_model = (
                                        np.stack(
                                            holdout_probas_uncalib_per_model,
                                            axis=0,
                                        )
                                    )

                                    holdout_probas_calib_per_model = (
                                        np.stack(
                                            holdout_probas_calib_per_model,
                                            axis=0,
                                        )
                                    )

                                    holdout_probas_uncalib = (
                                        holdout_probas_uncalib_per_model.mean(
                                            axis=0
                                        )
                                    )

                                    holdout_probas_calib = (
                                        holdout_probas_calib_per_model.mean(
                                            axis=0
                                        )
                                    )

                                    # ============================================================
                                    # CANONICAL HOLDOUT ORDER
                                    # ============================================================

                                    # Sort the holdout by patient ID before any holdout evaluation.
                                    #
                                    # This provides a deterministic patient order across experiments.
                                    # When two models are evaluated on the same holdout population and
                                    # bootstrap uses the same random seed, identical bootstrap indices
                                    # therefore correspond to identical patients.
                                    holdout_patient_ids = np.asarray(
                                        holdout_patient_ids
                                    ).reshape(-1)

                                    y_test_holdout = np.asarray(
                                        y_test_holdout
                                    ).reshape(-1)

                                    holdout_sort_idx = np.argsort(
                                        holdout_patient_ids
                                    )

                                    holdout_patient_ids = (
                                        holdout_patient_ids[
                                            holdout_sort_idx
                                        ]
                                    )

                                    y_test_holdout = (
                                        y_test_holdout[
                                            holdout_sort_idx
                                        ]
                                    )

                                    holdout_probas_uncalib = (
                                        holdout_probas_uncalib[
                                            holdout_sort_idx
                                        ]
                                    )

                                    holdout_probas_calib = (
                                        holdout_probas_calib[
                                            holdout_sort_idx
                                        ]
                                    )

                                    # Axis 0 = fold model
                                    # Axis 1 = patient
                                    holdout_probas_uncalib_per_model = (
                                        holdout_probas_uncalib_per_model[
                                            :,
                                            holdout_sort_idx,
                                        ]
                                    )

                                    holdout_probas_calib_per_model = (
                                        holdout_probas_calib_per_model[
                                            :,
                                            holdout_sort_idx,
                                        ]
                                    )

                                    holdout_probabilities = (
                                        holdout_probas_calib
                                        if calibration.value
                                        else holdout_probas_uncalib
                                    )

                                    # ============================================================
                                    # HOLDOUT PERFORMANCE FIGURES
                                    # ============================================================

                                    holdout_figure_results = (
                                        sfu.plot_all_figs(
                                            probas_uncalib=(
                                                holdout_probas_uncalib
                                            ),
                                            probas_calib=(
                                                holdout_probas_calib
                                                if calibration.value
                                                else None
                                            ),
                                            y_test=(
                                                y_test_holdout
                                            ),
                                            config_models=(
                                                config_models
                                            ),
                                            calibration=(
                                                calibration.value
                                            ),
                                            calibration_mode=(
                                                calibration_mode.value
                                                if calibration.value
                                                else ""
                                            ),
                                            save_figure=(
                                                save_figure.value
                                            ),
                                            output_dir=(
                                                holdout_output_dir
                                            ),
                                            transparent=(
                                                config_transparent
                                            ),
                                        )
                                    )

                                    # ============================================================
                                    # CALIBRATION CURVES: OOF VS HOLDOUT
                                    # ============================================================

                                    if not is_score_mode:
                                        curve_specs = [
                                            {
                                                "probas": (
                                                    all_probas_uncalib
                                                ),
                                                "y_true": (
                                                    all_y_validation_global
                                                ),
                                                "name": "Uncalibrated",
                                                "extra_info": "OOF",
                                            },
                                            {
                                                "probas": (
                                                    holdout_probas_uncalib
                                                ),
                                                "y_true": (
                                                    y_test_holdout
                                                ),
                                                "name": "Uncalibrated",
                                                "extra_info": "holdout",
                                            },
                                        ]

                                        if calibration.value:
                                            curve_specs.insert(
                                                1,
                                                {
                                                    "probas": (
                                                        all_probas_calib
                                                    ),
                                                    "y_true": (
                                                        all_y_validation_global
                                                    ),
                                                    "name": "Calibrated",
                                                    "extra_info": "OOF",
                                                },
                                            )

                                            curve_specs.append(
                                                {
                                                    "probas": (
                                                        holdout_probas_calib
                                                    ),
                                                    "y_true": (
                                                        y_test_holdout
                                                    ),
                                                    "name": "Calibrated",
                                                    "extra_info": "holdout",
                                                }
                                            )

                                        sfu.plot_calibration_curves(
                                            curve_specs=(
                                                curve_specs
                                            ),
                                            y_true=(
                                                all_y_validation_global
                                            ),
                                            title=(
                                                "Calibration: OOF vs Holdout"
                                            ),
                                            save_figure=(
                                                save_figure.value
                                            ),
                                            output_dir=(
                                                holdout_output_dir
                                            ),
                                            filename=(
                                                "calibration_oof_vs_holdout.png"
                                            ),
                                            transparent=(
                                                config_transparent
                                            ),
                                        )

                                        plt.close("all")

                                    # ============================================================
                                    # HOLDOUT METRICS + BOOTSTRAP
                                    # ============================================================

                                    # F1 and MCC are evaluated at the threshold selected
                                    # exclusively from pooled OOF predictions.
                                    holdout_metric_functions = {
                                         "auc": (
                                             roc_auc_score
                                         ),
                                         "auprc": (
                                             average_precision_score
                                         ),
                                         "brier": (
                                             brier_score_loss
                                         ),
                                         "f1": partial(
                                             postproc.f1_at_fixed_threshold,
                                             threshold=best_t,
                                         ),
                                         "mcc": partial(
                                             postproc.mcc_at_fixed_threshold,
                                             threshold=best_t,
                                         ),
                                         "calibration_slope": (
                                             postproc.calibration_slope
                                         ),
                                         "calibration_intercept": (
                                             postproc.calibration_intercept
                                         ),
                                         "ici": (
                                             postproc.ici_score
                                         ),
                                         "e90": (
                                             postproc.e90_score
                                         ),
                                        #  "eMax": (
                                        #      postproc.eMax_score
                                        #  ),
                                     }

                                    holdout_bootstrap_results = (
                                        postproc.bootstrap_holdout_metrics(
                                            y_true=(
                                                y_test_holdout
                                            ),
                                            probabilities=(
                                                holdout_probabilities
                                            ),
                                            metric_functions=(
                                                holdout_metric_functions
                                            ),
                                            n_bootstrap=2000,
                                            confidence_level=0.95,
                                            seed=seed,
                                        )
                                    )

                                    print(
                                        "\n"
                                        + "=" * 20
                                        + " HOLDOUT BOOTSTRAP SUMMARY "
                                        + "=" * 20
                                    )

                                    for (
                                        metric_name,
                                        metric_result,
                                    ) in (
                                        holdout_bootstrap_results[
                                            "summary"
                                        ].items()
                                    ):
                                        print(
                                            f"{metric_name}: "
                                            f"{metric_result['estimate']:.4f} "
                                            f"[95% CI "
                                            f"{metric_result['ci_lower']:.4f}, "
                                            f"{metric_result['ci_upper']:.4f}]"
                                        )

                                    # ============================================================
                                    # SAVE HOLDOUT RESULTS BEFORE INTERPRETABILITY
                                    # ============================================================

                                    interpretability_holdout_results = None

                                    all_results.update(
                                        {
                                            "y_true_holdout": (
                                                y_test_holdout
                                            ),
                                            "holdout_patient_ids": (
                                                holdout_patient_ids
                                            ),
                                            "probas_uncalib_holdout_per_model": (
                                                holdout_probas_uncalib_per_model
                                            ),
                                            "probas_calib_holdout_per_model": (
                                                holdout_probas_calib_per_model
                                            ),
                                            "probas_uncalib_holdout": (
                                                holdout_probas_uncalib
                                            ),
                                            "probas_calib_holdout": (
                                                holdout_probas_calib
                                            ),
                                            "probas_holdout": (
                                                holdout_probabilities
                                            ),
                                            "fixed_threshold_from_oof": (
                                                float(best_t)
                                            ),
                                            "bootstrap_holdout": (
                                                holdout_bootstrap_results
                                            ),
                                            "holdout_figure_results": (
                                                holdout_figure_results
                                            ),
                                            "interpretability_holdout": (
                                                None
                                            ),
                                        }
                                    )

                                    output_dir.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                    joblib.dump(
                                        all_results,
                                        output_dir / "all_res.joblib",
                                    )

                                    # ============================================================
                                    # INTERPRETABILITY
                                    # ============================================================

                                    if (
                                        not is_score_mode
                                        and config_models.extraction_type
                                        == "TSFEL"
                                    ):
                                        model_name = (
                                            config_models.models_name
                                        )

                                        # Check if interpretability artifacts already exist
                                        # to avoid expensive recomputation.
                                        interpretability_cached = False

                                        if model_name in {
                                            "Random Forest TSFEL",
                                            "XGBoost TSFEL",
                                        }:
                                            interpretability_cached = (
                                                holdout_output_dir
                                                / "holdout_ensemble_treeshap_importance.csv"
                                            ).exists()

                                        elif (
                                            model_name
                                            == "Logistic Regression Lasso TSFEL"
                                        ):
                                            interpretability_cached = (
                                                holdout_output_dir
                                                / "holdout_ensemble_linear_coefficients.csv"
                                            ).exists()

                                        if interpretability_cached:
                                            update_progress(
                                                "Interpretability artifacts already "
                                                "exist, loading from cache"
                                            )

                                            cached_results_path = (
                                                output_dir / "all_res.joblib"
                                            )

                                            if cached_results_path.exists():
                                                cached_results = joblib.load(
                                                    cached_results_path
                                                )

                                                interpretability_holdout_results = (
                                                    cached_results.get(
                                                        "interpretability_holdout"
                                                    )
                                                )
                                            else:
                                                interpretability_cached = False

                                        if (
                                            not interpretability_cached
                                            and RUN_INTERPRETABILITY
                                        ):
                                            # --------------------------------------------------------
                                            # Random Forest / XGBoost -> TreeSHAP
                                            # --------------------------------------------------------

                                            if model_name in {
                                                "Random Forest TSFEL",
                                                "XGBoost TSFEL",
                                            }:
                                                update_progress(
                                                    "Computing ensemble TreeSHAP values "
                                                    "on the holdout"
                                                )

                                                interpretability_holdout_results = (
                                                    postproc.shap_tree_holdout_ensemble(
                                                        models=(
                                                            fold_models_for_interpretability
                                                        ),
                                                        X_test_per_model=(
                                                            folds_X_holdout
                                                        ),
                                                        feature_names_per_model=(
                                                            folds_feature_names
                                                        ),
                                                        model_name=(
                                                            model_name
                                                        ),
                                                        savefig=(
                                                            save_figure.value
                                                        ),
                                                        folder=(
                                                            holdout_output_dir
                                                        ),
                                                        transparent=(
                                                            config_transparent
                                                        ),
                                                    )
                                                )
                                                postproc.shap_tsfel_importance_matrix(
                                                    interpretability_holdout_results,
                                                    savefig=True,
                                                    folder=holdout_output_dir,
                                                    top_raw_variables=None,
                                                    top_descriptors=None,
                                                )

                                                postproc.shap_tsfel_importance_matrix(
                                                    interpretability_holdout_results,
                                                    filename = "holdout_ensemble_treeshap_tsfel_all_matrix",
                                                    savefig=True,
                                                    all_descriptors = True,
                                                    folder=holdout_output_dir,
                                                    top_raw_variables=None,
                                                    top_descriptors=None,
                                                )

                                                postproc.shap_tsfel_importance_matrix(
                                                    interpretability_holdout_results,
                                                    savefig=True,
                                                    folder=holdout_output_dir,
                                                    top_raw_variables=None,
                                                    top_descriptors=None,
                                                    transpose = True
                                                )

                                                postproc.shap_tsfel_importance_matrix(
                                                    interpretability_holdout_results,
                                                    filename = "holdout_ensemble_treeshap_tsfel_all_matrix",
                                                    savefig=True,
                                                    all_descriptors = True,
                                                    folder=holdout_output_dir,
                                                    top_raw_variables=None,
                                                    top_descriptors=None,
                                                    transpose = True
                                                )

                                            # --------------------------------------------------------
                                            # Logistic regression + Lasso -> coefficients
                                            # --------------------------------------------------------

                                            elif (
                                                model_name
                                                == "Logistic Regression Lasso TSFEL"
                                            ):
                                                update_progress(
                                                    "Computing fold-aggregated "
                                                    "linear coefficients"
                                                )

                                                interpretability_holdout_results = (
                                                    postproc.linear_coefficients_holdout_ensemble(
                                                        models=(
                                                            fold_models_for_interpretability
                                                        ),
                                                        feature_names_per_model=(
                                                            folds_feature_names
                                                        ),
                                                        savefig=(
                                                            save_figure.value
                                                        ),
                                                        folder=(
                                                            holdout_output_dir
                                                        ),
                                                    )
                                                )
                                                csv_filepath = holdout_output_dir / "holdout_ensemble_linear_coefficients.csv"
                                                csv_filepath2 = holdout_output_dir / "holdout_ensemble_linear_cumulative_importance.csv"
                                                postproc.plot_odds_ratios_with_others(csv_file  = csv_filepath,
                                                                    save = True,
                                                                    folder = holdout_output_dir,
                                                                    title = None
                                                                    )
                                                postproc.plot_aggregated_odds_ratios(csv_file  = csv_filepath2,
                                                                    save = True,
                                                                    folder = holdout_output_dir,
                                                                    title = None
                                                                    )

                                            # --------------------------------------------------------
                                            # SVC -> no interpretability analysis
                                            # --------------------------------------------------------

                                            elif (
                                                model_name
                                                == "SVC TSFEL"
                                            ):
                                                update_progress(
                                                    "Skipping interpretability for SVC"
                                                )

                                        # ============================================================
                                        # FINAL SAVE
                                        # ============================================================

                                        all_results[
                                            "interpretability_holdout"
                                        ] = (
                                            interpretability_holdout_results
                                        )

                                        joblib.dump(
                                            all_results,
                                            output_dir / "all_res.joblib",
                                        )

                                if RUN_COMPARISON:
                                    # List of all candidate models to evaluate.
                                    model_candidates = [
                                        ("InceptionTimeModified", ""),
                                        ("LstmTimeModified", ""),
                                        ("RandomForest TSFEL", ""),
                                        ("XGBoost TSFEL", "balanced"),
                                        ("SVC TSFEL", ""),
                                        ("Logistic Regression Lasso TSFEL", ""),
                                        ("IGS2", "")
                                    ]
                                    
                                    comparisons = []
                                    for model_name, class_weight_override in model_candidates:
                                        try:
                                            # Attempt to load the model.
                                            loaded_model = exp.load_model(model_name, class_weight=class_weight_override)
                                            comparisons.append(loaded_model)
                                        except FileNotFoundError:
                                            # Silently skip the model when its file does not exist.
                                            print(f"[COMPARISON] Model unavailable (skipped): {model_name}", flush=True)
                                    
                                    # Generate the report only when at least one model was found.
                                    print(f"[COMPARISON] Loaded {len(comparisons)} models for comparison.", flush=True)
                                    if comparisons:
                                        print("[COMPARISON] Generating comparative report...", flush=True)
                                        comparison_output_directory = exp.get_comparison_path("Comparisons All")
                                        print(f"[COMPARISON] Output directory: {comparison_output_directory}", flush=True)

                                        # Global model comparison
                                        print("[COMPARISON] Running generate_comparative_report...", flush=True)
                                        sfu.old_generate_comparative_report(
                                            comparisons,
                                            save_dir=comparison_output_directory,
                                            table_format='fancy_grid'
                                        )
                                        print("[COMPARISON] Global report generated successfully.", flush=True)

                                        # --- Subgroup Analysis ---
                                        print("[COMPARISON] Starting subgroup analysis...", flush=True)
                                        # Holdout patient IDs are identical across all models (fixed holdout set)
                                        ref_config = comparisons[0][1]
                                        print(f"[COMPARISON] Reference config keys: {list(ref_config.keys())}", flush=True)
                                        holdout_pids = np.asarray(ref_config["holdout_patient_ids"]).ravel()
                                        print(f"[COMPARISON] Holdout patient IDs count: {len(holdout_pids)}", flush=True)

                                        # Load holdout static features and align by patient ID
                                        print("[COMPARISON] Loading holdout static features...", flush=True)
                                        holdout_static = pl.read_parquet(
                                            os.path.join(dataset_path, "df_static_full_clean.parquet")
                                        )
                                        print(f"[COMPARISON] Static features loaded: {holdout_static.shape}", flush=True)

                                        holdout_static = holdout_static.with_columns(
                                            pl.col("encounterId").cast(pl.Int64)
                                        )

                                        # Reorder to match holdout sample order via inner join with order DataFrame
                                        print("[COMPARISON] Reordering static features to match holdout order...", flush=True)

                                        # Clean conversion of numpy/list PIDs to Int64
                                        clean_pids = np.asarray(holdout_pids, dtype=np.int64).ravel()

                                        order_df = pl.DataFrame({
                                            "encounterId": clean_pids,
                                            "holdout_order": np.arange(len(clean_pids), dtype=np.int64)
                                        })

                                        holdout_static_aligned = (
                                            order_df
                                            .join(holdout_static, on="encounterId", how="inner")
                                            .sort("holdout_order")
                                            .drop("holdout_order")
                                        )
                                        print(f"[COMPARISON] Aligned static features: {holdout_static_aligned.shape}", flush=True)

                                        # --- Define subgroup label arrays ---

                                        # Age: < 40, [40-55], [55-70], > 70
                                        print("[COMPARISON] Computing Age labels...", flush=True)
                                        # Replace potential nulls with NaN for numeric type
                                        age_raw = holdout_static_aligned["age"].fill_null(np.nan).to_numpy()
                                        age_labels = np.where(
                                            np.isnan(age_raw), "Unknown",
                                            np.where(age_raw < 40, "< 40",
                                            np.where(age_raw < 55, "40-55",
                                            np.where(age_raw <= 70, "55-70", "> 70")))
                                        )
                                        print(f"[COMPARISON] Age labels distribution: {dict(zip(*np.unique(age_labels, return_counts=True)))}", flush=True)

                                        # Admission type: Medical, Scheduled Surgery, Unscheduled Surgery
                                        print("[COMPARISON] Computing Admission Type labels...", flush=True)
                                        # Replace nulls with "Unknown" directly in Polars
                                        admission_labels = (
                                            holdout_static_aligned["admission_type"]
                                            .fill_null("Unknown")
                                            .to_numpy()
                                        )
                                        print(f"[COMPARISON] Admission labels distribution: {dict(zip(*np.unique(admission_labels, return_counts=True)))}", flush=True)

                                        # ICU GHM categories
                                        print("[COMPARISON] Computing ICU GHM labels...", flush=True)
                                        
                                        # 1. Extract the first list element if it's a List(String), or direct conversion
                                        icu_ghm_series = holdout_static_aligned["icu_ghm"]
                                        
                                        if icu_ghm_series.dtype == pl.List:
                                            icu_ghm_series = icu_ghm_series.list.get(0)
                                            
                                        icu_ghm_raw = (
                                            icu_ghm_series
                                            .fill_null("Other")
                                            .to_numpy()
                                        )
                                        
                                        target_ghms = {
                                            "Respiratory Pathology",
                                            "Neurology",
                                            "Neurosurgery and neuro-embolisation",
                                            "Cardiovascular Surgery",
                                            "Polytrauma",
                                        }
                                        
                                        icu_ghm_labels = np.array(
                                            [str(g) if g in target_ghms else "Other" for g in icu_ghm_raw]
                                        )
                                        print(f"[COMPARISON] GHM labels distribution: {dict(zip(*np.unique(icu_ghm_labels, return_counts=True)))}", flush=True)

                                        # --- Subgroup definitions ---
                                        subgroup_configs = [
                                            {
                                                "name": "Age",
                                                "labels": age_labels,
                                                "names": {"< 40": "< 40", "40-55": "40-55", "55-70": "55-70", "> 70": "> 70"},
                                            },
                                            {
                                                "name": "Admission Type",
                                                "labels": admission_labels,
                                                "names": {
                                                    "Medical": "Medical",
                                                    "Scheduled Surgery": "Scheduled Surgery",
                                                    "Unscheduled Surgery": "Unscheduled Surgery",
                                                },
                                            },
                                            {
                                                "name": "ICU GHM",
                                                "labels": icu_ghm_labels,
                                                "names": {
                                                    "Respiratory Pathology": "Respiratory Pathology",
                                                    "Neurology": "Neurology",
                                                    "Neurosurgery and neuro-embolisation": "Neurosurgery",
                                                    "Cardiovascular Surgery": "Cardiovascular Surgery",
                                                    "Polytrauma": "Polytrauma",
                                                    "Other": "Other",
                                                },
                                            },
                                        ]

                                        # --- Generate subgroup reports per model ---
                                        print(f"[COMPARISON] Generating subgroup reports for {len(comparisons)} models across {len(subgroup_configs)} subgroups...", flush=True)
                                        for model_name, config in comparisons:
                                            print(f"[COMPARISON] Processing model: {model_name}", flush=True)
                                            probas = np.asarray(config["probas_holdout"], dtype=float).ravel()
                                            y_true = np.asarray(config["y_true_holdout"], dtype=int).ravel()
                                            print(f"[COMPARISON] Model {model_name}: probas shape={probas.shape}, y_true shape={y_true.shape}", flush=True)

                                            for sg in subgroup_configs:
                                                print(f"[COMPARISON]   -> Subgroup: {sg['name']}", flush=True)
                                                sfu.generate_subgroup_comparative_report(
                                                    probas=probas,
                                                    y_true=y_true,
                                                    subgroup_labels=sg["labels"],
                                                    subgroup_names=sg["names"],
                                                    save_dir=comparison_output_directory / f"Subgroup_{sg['name']}" / model_name,
                                                )
                                                print(f"[COMPARISON]   -> Subgroup {sg['name']} done.", flush=True)
                                            print(f"[COMPARISON] Model {model_name} done.", flush=True)
                                        print("[COMPARISON] All subgroup reports generated successfully.", flush=True)

                                    else:
                                        print("[COMPARISON] No model could be loaded. Report generation cancelled.", flush=True)

                                update_progress("Experiment completed successfully")
                                print(f"[EXPERIMENT {EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS}] COMPLETED SUCCESSFULLY", flush=True)
                            except Exception as e:
                                print(f"\n!!! [EXPERIMENT {EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS}] CRASHED !!!", flush=True)
                                print(f"Error encountered: {str(e)}", flush=True)
                                print("\n[STACKTRACE]", flush=True)
                                print(traceback.format_exc(), flush=True)
                                print("!" * 90 + "\n", flush=True)
                                update_progress(f"Experiment {EXPERIMENT_INDEX} failed (skipped)")
                                continue # Prevent one failed experiment from stopping all remaining experiments.
                            finally:
                                plt.close("all")

update_progress("All experiments have completed")
print(f"[PIPELINE] Normal completion. Full log: {_LOG_PATH}", flush=True)
