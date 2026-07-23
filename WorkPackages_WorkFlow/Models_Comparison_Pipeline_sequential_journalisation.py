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
faulthandler.enable(file=_LOG_FILE, all_threads=True)

CURRENT_RUN_CONTEXT = {
    "experiment": "Pipeline initialization",
    "stage": "Importing dependencies",
}


def update_progress(stage):
    """Update the current stage and display it immediately."""
    CURRENT_RUN_CONTEXT["stage"] = stage
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [STAGE] {stage}", flush=True)


def _uncaught_exception_hook(exc_type, exc_value, exc_traceback):
    """Add the experiment context before the standard Python traceback."""
    print("\n" + "!" * 90, file=sys.stderr, flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [CRASH] Uncaught exception", file=sys.stderr, flush=True)
    print(f"[CRASH] Experiment: {CURRENT_RUN_CONTEXT.get('experiment')}", file=sys.stderr, flush=True)
    print(f"[CRASH] Last known stage: {CURRENT_RUN_CONTEXT.get('stage')}", file=sys.stderr, flush=True)
    print(f"[CRASH] Full log: {_LOG_PATH}", file=sys.stderr, flush=True)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    print("!" * 90, file=sys.stderr, flush=True)


sys.excepthook = _uncaught_exception_hook
print(f"[LOG] Log file for this run: {_LOG_PATH}", flush=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import utilitaries.create_merged_dataset as create_merged_dataset
import utilitaries.extract_data_utils as extract
import utilitaries.features_extraction_utils as extract_feat
import utilitaries.marimo_utils as mo_utils
import utilitaries.preprocessing_utils as preproc
import utilitaries.postprocessing_utils as postproc
import utilitaries.timestamp_sampling_utils as tsu
import utilitaries.path_utils as path_utils
import utilitaries.new_show_fig_utils as sfu
import utilitaries.training_utils as training
import utilitaries.evaluate_utils as evaluate
import utilitaries.resampling_and_window_choice_pipeline as choice
import utilitaries.sequential_utils as sequential
import utilitaries.static_features_utils as static_utils
from datetime import datetime

import warnings

# Targeted filter: ignore UserWarning messages containing "cannot be shown".
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*cannot be shown.*"
)

pl.Config.set_tbl_cols(-1)

mode_names = ["score", "wp1", "wp2", "wp3"]
RUN_TEST = True
RUN_COMPARISON = False
RUN_TRAINING = True
RUN_LASSO = True
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
    if mode_run == "test_imbalance":
        mode_duplicates = "prio_first"
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        target_labels = ["Survie à 28 jours"]
        model_names = ["LstmTimeModified", "XGBoost TSFEL"]
        stratify_modes = ["target_col"]
        optuna_run_options = [False]
        feature_modes = ["Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode", "DownSampling 50-50", "UpSampling 50-50"]
        RUN_COMPARISON = True
    elif mode_run == "test_robustesse":
        mode_duplicates = "prio_first"
        target_labels = ["Survie à 28 jours"]
        model_names = ["LstmTimeModified", "XGBoost TSFEL"]
        stratify_modes = ["target_col"]
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        optuna_run_options = [False]
        feature_modes =["Mode IGS2", "Mode Commonly Used Without pmsi", "Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]
    elif mode_run == "wp1":
        mode_duplicates = "prio_first"
        target_labels = ["Survie à 28 jours"]
        model_names = ["InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col"]
        WINDOWING_MODE = "24h début réanimation sans remplissage"
        optuna_run_options = [False, True]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi", "Mode Commonly Used"]
        balancing_methods = ["Aucune Méthode"]

    elif mode_run == "wp2":
        WINDOWING_MODE =  "24h fin réanimation sans remplissage"
        target_labels = ["Survie à 28 jours", "Survie à 24 heures", "Survie à 7 jours", "Survie à 3 mois"]
        model_names = ["InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col", "h24-j28"]
        optuna_run_options = [False]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode", "DownSampling 50-50", "UpSampling 50-50"]

    elif mode_run == "wp3":
        WINDOWING_MODE =  "resampling aléatoire 'lomax' prio 24h sans remplissage"
        target_labels = ["Survie à 28 jours", "Survie à 24 heures", "Survie à 7 jours", "Survie à 3 mois"]
        model_names = ["InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
        stratify_modes = ["target_col", "h24-j28"]
        optuna_run_options = [False]
        feature_modes = ["Mode IGS2", "Mode Commonly Used Without pmsi"]
        balancing_methods = ["Aucune Méthode", "DownSampling 50-50", "UpSampling 50-50"]
    for target_label in target_labels:
        TARGET_LABEL = target_label
        for stratification_choice in stratify_modes:
            STRATIFY_MODE = stratification_choice
            for feature_mode in feature_modes:
                for model_choice in model_names:
                    if model_choice in ["InceptionTimeModified", "Logistic Regression Lasso TSFEL"]:
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
                                    'InceptionTimeModified': {'epochs': 100, 'patience': 30, 'lr': 0.001}, 
                                    'LstmTimeModified': {'epochs': 100, 'patience': 30, 'lr': 0.001}, 
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

                                if not keep_duplicates:
                                    _path = os.path.join(dataset_path, 'df_static_full_clean.parquet')
                                    df_without_anomalies = pl.scan_parquet(_path)
                                    df_merged = extract.remove_duplicates(df_without_anomalies, df_merged, mode_duplicates == "prio_last")
                                else:
                                    df_merged.collect()
                                df_labeled = tsu.prepare_labels(df_merged, 'relative')
                                # TODO: Move this line earlier in the pipeline.
                                df_labeled = df_labeled.filter(pl.col('delta_hour') >= 0)
                                df_labeled.columns

                                

                                targets = ['isDeceased_lt_28d', 'isDeceased_lt_24h', 'isDeceased_lt_7d', 'isDeceased_lt_3m']

                                update_progress("Preparing and cleaning the dataset")
                                df_clean, features_list, target_length = choice.prepare_dataset_from_config(df_merged=df_labeled, config_mode=config_mode, target_col=target_col, patient_col=patient_col, targets=targets, seed=seed)

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
                                
                                print("--- TYPES DEBUGGING ---")
                                print(f"selected_features : {type(selected_features)} -> {selected_features[:3]} (Example)")
                                print(f"patient_col       : {type(patient_col)} -> {repr(patient_col)}")
                                print(f"time_col          : {type(time_col)} -> {repr(time_col)}")
                                print(f"target_col        : {type(target_col)} -> {repr(target_col)}")
                                print("--------------------------")
                                df_clean_1 = df_clean.select(*selected_features, patient_col, time_col, target_col)

                                keep_features = selected_features.copy()

                                if config_mode.mode == 'windows':
                                    valid_patient_ids = df_clean_1.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
                                    df_clean_2 = df_clean_1.join(valid_patient_ids, on=patient_col, how='inner')
                                    if df_clean_2.is_empty():
                                        raise ValueError("No patient has exactly the expected sequence length.")
                                    df_clean_2 = df_clean_2.sort(patient_col, time_col)
                                else:
                                    df_clean_2 = df_clean_1

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
                                X_init = df_clean_3.select(final_features).to_numpy()
                                y_init = df_clean_3[target_col].to_numpy()

                                update_progress("Preparing model data and the initial holdout")
                                train_init_df = None
                                train_init_tsfel = None
                                if config_models.extraction_type == 'TSFEL':
                                    raw_global_tsfel_path = exp.get_tsfel_parquet_path()
                                    if extract_tsfel.value or not os.path.exists(raw_global_tsfel_path):
                                        print("Starting global TSFEL extraction for all patients")
                                        static_feats = static_utils.build_static_feature_list(
                                            dataframe=df_clean_3,
                                            generated_dummy_columns=generated_dummy_columns,
                                            config_mode_name=config_mode.name,
                                        )
                                        tsfel_features = [target_label for target_label in final_features if target_label not in static_feats]
                                        tsfel_global_df = extract_feat.extract_tsfel_per_patient(df_clean_3, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
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
                                    groups_init = df_clean_3[patient_col].to_numpy()
                                    train_init_idx, test_init_idx = next(sgkf_init.split(X=X_init, y=y_init, groups=groups_init))
                                    train_init_patients = df_clean_3[train_init_idx].select(patient_col).unique()
                                    test_init_patients = df_clean_3[test_init_idx].select(patient_col).unique()
                                    if config_models.extraction_type == 'TSFEL':
                                        train_init_tsfel = complete_tsfel_df.join(train_init_patients, on=patient_col, how='inner').sort(patient_col)
                                        test_holdout_tsfel = complete_tsfel_df.join(test_init_patients, on=patient_col, how='inner').sort(patient_col)
                                        X = train_init_tsfel
                                        y = train_init_tsfel[target_col].to_numpy()
                                        groups = train_init_tsfel[patient_col].to_numpy()
                                    elif config_models.extraction_type == 'time':
                                        train_init_df = df_clean_3[train_init_idx].sort([patient_col, time_col])
                                        test_holdout_df = df_clean_3[test_init_idx].sort([patient_col, time_col])
                                        X = train_init_df
                                        y = train_init_df[target_col].to_numpy()
                                        groups = train_init_df[patient_col].to_numpy()

                                update_progress("Building and preprocessing the 5 folds")
                                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                                folds_X_train, folds_X_validation = ([], [])
                                folds_y_train, folds_y_validation = ([], [])
                                folds_groups = []
                                folds_sampling_stats = []
                                fold_processing_config = {
                                    'patient_col': extract.ID_COL,
                                    'time_col': extract.TIME_COL,
                                    'target_col': target_col,
                                    'boruta_filter': config_boruta,
                                    'balance_method': config_balance.balance_method,
                                    'expected_length': expected_length,
                                    'final_features': final_features,
                                    'exp': exp,
                                }
                                if 'train_init_df' in locals() and train_init_df is not None:
                                    fold_processing_config['train_init'] = train_init_df
                                else:
                                    fold_processing_config['train_init'] = train_init_tsfel
                                for fold_index, (train_idx, validation_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
                                    print(f'\n─────────────────── Processing Fold {fold_index + 1}/5 ───────────────────')
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

                                        processed_groups = (
                                            train_score_fold[patient_col]
                                            .to_numpy()
                                            .reshape(-1)
                                        )

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
                                    elif config_models.extraction_type == 'TSFEL':
                                        x_train_processed, x_validation_processed, y_train_processed, y_validation_processed, processed_groups, sampling_statistics = preproc.process_tsfel_fold(fold_index, train_idx, validation_idx, X, y, groups, seed, **fold_processing_config)
                                    elif config_models.extraction_type == 'time':
                                        x_train_processed, x_validation_processed, y_train_processed, y_validation_processed, processed_groups, sampling_statistics = preproc.process_time_fold(fold_index, train_idx, validation_idx, seed, **fold_processing_config)
                                    else:
                                        raise ValueError(f"Unknown or unsupported extraction type: {config_models.extraction_type}")
                                    folds_X_train.append(x_train_processed)
                                    folds_X_validation.append(x_validation_processed)
                                    folds_y_train.append(y_train_processed)
                                    folds_y_validation.append(y_validation_processed)
                                    folds_groups.append(processed_groups)
                                    folds_sampling_stats.append(sampling_statistics)
                                np.save(exp.get_var_path(), final_features)
                                print('All 5 folds were computed successfully!')
                                print(f"[DEBUG] Fold groups: {folds_groups}")

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

                                        print(
                                            f"[DEBUG FOLD GROUPS] "
                                            f"rows={len(training_groups_fold)}, "
                                            f"unique_groups={np.unique(training_groups_fold).size}, "
                                            f"dtype={training_groups_fold.dtype}, "
                                            f"first_values={training_groups_fold[:10]}"
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
                                        print(f'\n[DEBUG TRAIN - Fold {training_fold_index + 1}] x_training_fold shape: {x_training_fold.shape}')
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
                                                print(f"[DEBUG GROUPS] Number of rows: {len(current_groups)}, Unique groups: {len(np.unique(current_groups))}")
                                                skf_calib = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                                                train_idx_calib, calib_idx = next(skf_calib.split(x_training_fold, y_training_fold, groups=training_groups_fold))
                                                X_train_final_fold = x_training_fold[train_idx_calib]
                                                y_train_final = y_training_fold[train_idx_calib]
                                                groups_final_fold = np.asarray(training_groups_fold)[train_idx_calib]
                                                X_calib = x_training_fold[calib_idx]
                                                y_calib = y_training_fold[calib_idx]
                                                print(f'    [DEBUG] Shapes - base training set: {X_train_final_fold.shape}, calibration set: {X_calib.shape}')
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
                                                            print(f'    [DEBUG] Prior Calibration OK (beta = {beta:.4f})')
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

                                print(
                                    "DEBUG OOF SIZES -> "
                                    f"y_true: {len(all_y_validation_global)}, "
                                    f"uncalib: {len(all_probas_uncalib)}, "
                                    f"calib: {len(all_probas_calib)}"
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
                                    print('Lasso path started')
                                    print('Generating Lasso paths with warm start and multiprocessing...')
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
                                        plt.xlabel('Paramètre de régularisation C (Log Scale)')
                                        plt.ylabel(f'Coefficients ({n_features} features)')
                                        plt.title(f'L1 Regularization Path - Fold {fold_idx_L1 + 1}\nOptimisé (Warm Start)')
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

                                    "eMax_oof": calibration_stats["eMax"],

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


                                update_progress(
                                    "Saving final results"
                                )

                                output_dir.mkdir(
                                    parents=True,
                                    exist_ok=True,
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
                                    if comparisons:
                                        comparison_output_directory = exp.get_comparison_path("Comparisons All")
                                        sfu.générer_rapport_comparatif(comparisons, save_dir=comparison_output_directory, table_format='fancy_grid')
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



