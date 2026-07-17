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
# SURCOUCHE DE JOURNALISATION (aucun impact sur la logique de la pipeline)
# Tous les print et toutes les erreurs sont dupliqués dans un fichier de log.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_LOG_DIR = _SCRIPT_DIR / "pipeline_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}.log"
_LOG_FILE = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class _TeeStream:
    """Écrit simultanément dans le terminal et dans le fichier de log."""

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
    "experiment": "Initialisation de la pipeline",
    "stage": "Import des dépendances",
}


def update_progress(stage):
    """Met à jour l'étape courante et l'affiche immédiatement."""
    CURRENT_RUN_CONTEXT["stage"] = stage
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ÉTAPE] {stage}", flush=True)


def _uncaught_exception_hook(exc_type, exc_value, exc_traceback):
    """Ajoute le contexte de l'expérience avant le traceback Python standard."""
    print("\n" + "!" * 90, file=sys.stderr, flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [CRASH] Exception non interceptée", file=sys.stderr, flush=True)
    print(f"[CRASH] Expérience : {CURRENT_RUN_CONTEXT.get('experiment')}", file=sys.stderr, flush=True)
    print(f"[CRASH] Dernière étape connue : {CURRENT_RUN_CONTEXT.get('stage')}", file=sys.stderr, flush=True)
    print(f"[CRASH] Journal complet : {_LOG_PATH}", file=sys.stderr, flush=True)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    print("!" * 90, file=sys.stderr, flush=True)


sys.excepthook = _uncaught_exception_hook
print(f"[LOG] Journal de cette exécution : {_LOG_PATH}", flush=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import utilitaries.create_merged_dataset as create_merged_dataset
import utilitaries.extract_data_utils as extract
import utilitaries.features_extraction_utils as extract_feat
import utilitaries.marimo_utils as mo_utils
import utilitaries.preprocessing_utils as preproc
import utilitaries.timestamp_sampling_utils as tsu
import utilitaries.path_utils as path_utils
import utilitaries.show_fig_utils as sfu
import utilitaries.training_utils as training
import utilitaries.evaluate_utils as evaluate
import utilitaries.resampling_and_window_choice_pipeline as choice
import utilitaries.sequential_utils as sequential
from datetime import datetime
pl.Config.set_tbl_cols(-1)

mode_run = "wp1"
# MODE WP1
print("lancement de tous les entraînements du mode IGS2")
if mode_run == "wp1":
    CIBLE = "Survie à 28 jours"
    # models_name = ["InceptionTimeModified", "LstmTimeModified", "XGBoost TSFEL", "RandomForest TSFEL", "SVC TSFEL", "Logistic Regression Lasso TSFEL"]
    models_name = ["XGBoost TSFEL"]
    STRATIFY_MODE = "target_col"
    runs_optuna = [False, True]
    modes_features = ["Mode IGS2", "Mode Commonly Used Without pmsi"]
    equilibrages = ["Aucune Méthode", "DownSampling 50-50", "UpSampling 50-50"]
mode_names = [""]
RUN_COMPARISON = False
RUN_TRAINING = True
RUN_LASSO = True
POPULATION = "Tout"
SAVE_FIGURE = True
TRANSPARENT = False
DATASET_PATH = "/../../../data2/paquie.d/Datasets/output"
TOTAL_EXPERIMENTS = sum(
    len(equilibrages) * (1 if model == "InceptionTimeModified" else len(runs_optuna))
    for _mode in modes_features
    for model in models_name
)
EXPERIMENT_INDEX = 0
print(f"[PIPELINE] {TOTAL_EXPERIMENTS} expérience(s) planifiée(s).", flush=True)
for mod in modes_features:
    for m in models_name:
        print(m, m == "InceptionTimeModified")
        if m == "InceptionTimeModified":
            ru = [False]
        else:
            ru = runs_optuna
        print(ru)
        for eq in equilibrages:
            for opt in ru:
                TYPE_DONNEES = "modèle"

                MODE_FENETRAGE = "24h début réanimation sans remplissage"

                MODEL_NAME = m
                SCORE_NAME = "IGS2"

                NETTOYAGE = "Enlever Surveillance Continue"
                CIBLE = "Survie à 28 jours"
                MODE_FEATURES = mod
                EQUILIBRAGE = eq
                BORUTA_FILTER = True
                USE_OPTUNA = opt
                RUN_OPTUNA = opt
                EXTRACT_TSFEL = False
                CLASS_WEIGHT = "_balanced"

                CUSTOM_FEATURES = []

                EXPERIMENT_INDEX += 1
                CURRENT_RUN_CONTEXT["experiment"] = (
                    f"{EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS} | modèle={MODEL_NAME} | "
                    f"features={MODE_FEATURES} | équilibrage={EQUILIBRAGE} | optuna={RUN_OPTUNA}"
                )
                print("\n" + "=" * 100, flush=True)
                print(f"[EXPÉRIENCE {EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS}] DÉMARRAGE", flush=True)
                print(f"  Modèle       : {MODEL_NAME}", flush=True)
                print(f"  Features     : {MODE_FEATURES}", flush=True)
                print(f"  Équilibrage  : {EQUILIBRAGE}", flush=True)
                print(f"  Optuna       : {RUN_OPTUNA}", flush=True)
                print("=" * 100, flush=True)
                update_progress("Création de la configuration de l'expérience")

                pipeline_config = sequential.create_pipeline_config(
                    type_donnees=TYPE_DONNEES,
                    mode_fenetrage=MODE_FENETRAGE,
                    model_name=MODEL_NAME,
                    score_name=SCORE_NAME,
                    nettoyage=NETTOYAGE,
                    cible=CIBLE,
                    mode_features=MODE_FEATURES,
                    equilibrage=EQUILIBRAGE,
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

                update_progress("Initialisation du matériel et des paramètres")
                seed = SEED
                stratify_mode = SimpleNamespace(value=STRATIFY_MODE)
                cuda_dispo = torch.cuda.is_available()
                print(f'Est-ce que CUDA est disponible ? {cuda_dispo}')
                appareil_actuel = torch.cuda.current_device() if cuda_dispo else 'CPU'
                print(f'Appareil actuellement utilisé : {appareil_actuel}')
                if cuda_dispo:
                    print(f'Nom du GPU : {torch.cuda.get_device_name(0)}')

                if not config_transparent:
                    plt.rcParams["figure.facecolor"] = "white"
                    plt.rcParams['axes.facecolor'] = "white"
                    plt.rcParams['savefig.facecolor'] = "white"


                patient_col = extract.ID_COL
                time_col = extract.TIME_COL
                target_col = 'isDeceased_lt_28d'
                expected_length = extract.WINDOW_SIZE

                str_pop = ''
                if config_keep_pop.keep_population != 'all_diseases':
                    str_pop = '_' + config_keep_pop.keep_population
                str_balance_method = ''
                if config_balance.balance_method:
                    str_balance_method = config_balance.balance_method + '_'
                extension = '.joblib' if config_models.extraction_type == 'TSFEL' else '.pt'
                DEFAULT_PARAMS = {'InceptionTimeModified': {'epochs': 100, 'patience': 30, 'lr': 0.001}, 'LstmTimeModified': {'epochs': 100, 'patience': 30, 'lr': 0.001}, 'RandomForest TSFEL': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': class_weight_choice.value[1:]}, 'RandomForest Imbalanced TSFEL': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': class_weight_choice.value[1:]}, 'XGBoost TSFEL': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8}, 'SVC TSFEL': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'class_weight': class_weight_choice.value[1:]}}

                exp = path_utils.Experiment(config_mode.name, config_cleaning, config_y, str_balance_method, modex, class_weight_choice, str_pop, seed, stratify_mode=stratify_mode.value, calibrated_mode=calibration_mode.value if calibration.value else "")

                dataset_path = DATASET_PATH

                update_progress("Chargement et fusion des données")
                # _path = os.path.join(dataset_path, 'df_static_ano_clean.parquet')
                # df_static = pl.scan_parquet(_path)

                # _path = os.path.join(dataset_path, 'df_dynamic_full_clean.parquet')
                # df_dynamic = extract.extract_data_survie(_path)

                # df_merged = create_merged_dataset.create_merged_dataset(df_static, df_dynamic, True, save=True, folder=dataset_path)
                _path = os.path.join(dataset_path, 'merged_static_ano_and_dynamic.parquet')
                df_merged = pl.scan_parquet(_path)
                df_merged_1 = tsu.prepare_labels(df_merged, 'relative')
                df_merged_1 = df_merged_1.filter(pl.col('delta_hour') >= 0)
                df_merged_1 = df_merged_1.collect()
                df_merged_1.columns

                targets = ['isDeceased_lt_28d', 'isDeceased_lt_24h', 'isDeceased_lt_7d', 'isDeceased_lt_3m']

                update_progress("Préparation et nettoyage du dataset")
                df_clean, features_list, target_length = choice.prepare_dataset_from_config(df_merged=df_merged_1, config_mode=config_mode, target_col=target_col, patient_col=patient_col, targets=targets, seed=seed)

                score = ['NEWS', 'NEWS2', 'sapsii', 'sapsii_prob']
                dustbin = ['endotracheal_tube', 'tracheo', 'installation', 'eer', 'hx_comorbidité_majeure', 'imc', 'neuro_status', 'ecmo_all', 'prone', 'plq']
                useless = ['arret_therapeutique', 'limitation_therapeutique', 'hematocrit', 'peak_pressure']
                icu_useless = ['icu_actes', 'icu_mode_sortie', 'icu_DA', 'hosp_admissionMode', 'hosp_primaryDiagnosis', 'hosp_primaryDiagnosisCode']
                icu_useful = ['icu_ghm', 'icu_DP_code', 'icu_DP', 'icu_mode_entree']
                broken = []
                cheat = ['category', 'heure_entiere', 'encounterId', 'delta_hour', 'heure_calibree', 'year_inTime', target_col, 'deces_datediff_days', 'isDeceased', 'adm_unit', 'out_unit', 'transition_units', 'los', 'adm_year', 'hosp_los', 'hosp_dischargeMode', 'deces_hosp', *targets]
                cols_to_drop = [*score, *dustbin, *useless, *cheat, *icu_useless, *broken]
                existing_cols = [col for col in cols_to_drop if col in df_clean.columns]
                if existing_cols:
                    df_clean_keep = df_clean.drop(existing_cols)
                else:
                    df_clean_keep = df_clean.copy()
                df_clean_keep = df_clean_keep.select([pl.col(c) for c in df_clean_keep.columns if not c.endswith('_detected_term')])

                commonly_used = ['score_glasgow', 'is_conscious', 'heart_rate', 'creat', 'is_cvvhf', 'is_hdi', 'pao2', 'is_ventilated', 'fio2_corr', 'age', 'temp', 'urine_rate', 'pas', 'pam', 'pad', 'bili_tot', 'leucocytes', 'admission_type', 'fr', 'ph', 'sodium', 'potassium', 'num_plq', 'blood_urea', 'nad_dose_poids', 'dobu_dose_poids', 'hemoglobine', 'tp', 'spo2', 'hco3', 'glyc_cap']

                custom_features = SimpleNamespace(value=CUSTOM_FEATURES if CUSTOM_FEATURES else df_clean_keep.columns)

                with open('../../Preprocessing_pipeline/preprocessing-pipelines/json/dynamic_features.json', 'r') as file:
                    json_feat = json.load(file)
                if modex.value == 'Mode All':
                    keep_feats = df_clean_keep.columns
                elif modex.value == 'Mode All Without pmsi':
                    keep_feats = df_clean_keep.select(~cs.starts_with('hx_') & ~cs.starts_with('icu_')).columns
                elif modex.value == 'Mode Commonly Used Without pmsi':
                    keep_feats = df_clean_keep.select(commonly_used).columns
                elif modex.value == 'Mode Commonly Used':
                    keep_feats = df_clean_keep.select(commonly_used, *icu_useful, cs.starts_with('hx_')).columns
                elif modex.value == 'Mode Custom':
                    keep_feats = custom_features.value
                else:
                    keep_feats = mo_utils.FEAT[modex.value].keep_feats
                    if 'urine_rate' in keep_feats:
                        keep_feats.remove('urine_rate')
                str_keep_feats = ''
                if config_mode.name == 'resampling_X_points':
                    keep_feats.append('real_time_hours')
                elif config_mode.name == 'resampling_x_points_alea_lomax_prio_24h_no-fill':
                    keep_feats.append('observed_duration')
                else:
                    if 'observed_duration' in keep_feats:
                        keep_feats.remove('observed_duration')
                    if 'real_time_hours' in keep_feats:
                        keep_feats.remove('real_time_hours')
                for kf in keep_feats:
                    if kf in json_feat:
                        str_keep_feats += f"- {json_feat[kf]['description']} \n"

                df_clean_1 = df_clean.select(*keep_feats, patient_col, time_col, target_col)

                keep_features = keep_feats.copy()

                if config_mode.mode == 'windows':
                    valid_ids = df_clean_1.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
                    df_clean_2 = df_clean_1.join(valid_ids, on=patient_col, how='inner')
                    if df_clean_2.is_empty():
                        raise ValueError("Aucun patient n'a exactement la longueur attendue.")
                    df_clean_2 = df_clean_2.sort(patient_col, time_col)
                else:
                    df_clean_2 = df_clean_1

                df_clean_3 = df_clean_2.with_columns(pl.col('admission_type').fill_null('Unknown')).to_dummies(columns=['admission_type'])
                if modex.value in ['Mode Custom', 'Mode All Without pmsi', 'Mode All']:
                    df_clean_3 = df_clean_3.with_columns(pl.col('gender').fill_null('Unknown')).to_dummies(columns=['gender'])
                    cols_gender = [c for c in df_clean_3.columns if c.startswith('gender_')]
                    if 'gender' in keep_features:
                        keep_features.remove('gender')
                        keep_features.extend(cols_gender)
                df_clean_3 = df_clean_3.with_columns(cs.numeric().cast(pl.Float64), cs.boolean().cast(pl.Float64))
                cols_admission = [c for c in df_clean_3.columns if c.startswith('admission_type_')]
                if 'admission_type' in keep_features:
                    keep_features.remove('admission_type')
                    keep_features.extend(cols_admission)
                final_features = list(dict.fromkeys(keep_features))
                assert target_col not in final_features, f'Alerte Leakage : {target_col} est présente dans les features !'
                print(f'Nombre de features envoyées au {config_models.models_name} : {len(final_features)} : {final_features}')
                X_init = df_clean_3.select(final_features).to_numpy()
                y_init = df_clean_3[target_col].to_numpy()

                update_progress("Préparation des données modèle et du holdout initial")
                train_init_df = None
                train_init_tsfel = None
                if config_models.extraction_type == 'TSFEL':
                    filename_global_brut = exp.get_tsfel_parquet_path()
                    if extract_tsfel.value or not os.path.exists(filename_global_brut):
                        print("Lancement de l'extraction TSFEL globale sur tous les patients")
                        static_feats = ["admission_type_Medical", "admission_type_Scheduled Surgery", "admission_type_Unknown", "admission_type_Unscheduled Surgery", "score_glasgow", "age"]
                        if config_mode.name == "resampling_X_points":
                            static_feats.append("real_time_hours")
                        elif config_mode.name == "resampling_x_points_alea_lomax_prio_24h_no-fill":
                            static_feats.append("observed_duration")
                        print(static_feats)
                        tsfel_features = [c for c in final_features if c not in static_feats]
                        TSFEL_global_df = extract_feat.extract_tsfel_per_patient(df_clean_3, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
                        static_global = df_clean_3.select([extract.ID_COL, *static_feats]).unique()
                        df_tsfel_complet = TSFEL_global_df.join(static_global, on=extract.ID_COL, how='inner')
                        df_tsfel_complet.write_parquet(filename_global_brut)
                        print('Extraction globale sauvegardée')
                    else:
                        df_tsfel_complet = pl.read_parquet(filename_global_brut)
                    keepVariableList_0 = df_tsfel_complet.columns
                    parent_folder = filename_global_brut.parent
                    np.save(parent_folder / 'keepVariableList_0.npy', keepVariableList_0)
                groups_init = df_clean_3[patient_col].to_numpy()
                sgkf_init = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_init_idx, test_init_idx = next(sgkf_init.split(X=X_init, y=y_init, groups=groups_init))
                train_init_patients = df_clean_3[train_init_idx].select(patient_col).unique()
                test_init_patients = df_clean_3[test_init_idx].select(patient_col).unique()
                if config_models.extraction_type == 'TSFEL':
                    train_init_tsfel = df_tsfel_complet.join(train_init_patients, on=patient_col, how='inner').sort(patient_col)
                    test_holdout_tsfel = df_tsfel_complet.join(test_init_patients, on=patient_col, how='inner').sort(patient_col)
                    X = train_init_tsfel
                    y = train_init_tsfel[target_col].to_numpy()
                    groups = train_init_tsfel[patient_col].to_numpy()
                elif config_models.extraction_type == 'time' or type_donnees.value == 'score':
                    train_init_df = df_clean_3[train_init_idx].sort([patient_col, time_col])
                    test_holdout_df = df_clean_3[test_init_idx].sort([patient_col, time_col])
                    X = train_init_df
                    y = train_init_df[target_col].to_numpy()
                    groups = train_init_df[patient_col].to_numpy()

                saps2_pred = None
                saps2_true = None
                if type_donnees.value == 'score':
                    df_clean_saps2 = X.join(df_clean.select(['sapsii_prob', 'encounterId']).cast(pl.Float64), on='encounterId', how='inner').filter(pl.col('sapsii_prob').is_not_null()).group_by('encounterId').first().sort(by='encounterId')
                    df_clean_saps2.describe()
                    saps2_pred = df_clean_saps2['sapsii_prob'].to_numpy()
                    saps2_true = df_clean_saps2['isDeceased_lt_28d'].to_numpy()

                update_progress("Construction et prétraitement des 5 folds")
                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                folds_X_train, folds_X_test = ([], [])
                folds_y_train, folds_y_test = ([], [])
                folds_groups = []
                folds_sampling_stats = []
                pipeline_config = {'patient_col': extract.ID_COL, 'time_col': extract.TIME_COL, 'target_col': target_col, 'boruta_filter': config_boruta, 'balance_method': config_balance.balance_method, 'expected_length': expected_length, 'final_features': final_features, 'exp': exp}
                if 'train_init_df' in locals() and train_init_df is not None:
                    pipeline_config['train_init'] = train_init_df
                else:
                    pipeline_config['train_init'] = train_init_tsfel
                for fold_idxx, (train_idx, test_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
                    print(f'\n─────────────────── Traitement du Fold {fold_idxx + 1}/5 ───────────────────')
                    if config_models.extraction_type == 'TSFEL':
                        X_tr, X_te, y_tr, y_te, grp, stats = preproc.process_tsfel_fold(fold_idxx, train_idx, test_idx, X, y, groups, seed, **pipeline_config)
                    elif config_models.extraction_type == 'time':
                        X_tr, X_te, y_tr, y_te, grp, stats = preproc.process_time_fold(fold_idxx, train_idx, test_idx, seed, **pipeline_config)
                    else:
                        raise ValueError(f"Type d'extraction inconnu ou non implémenté : {config_models.extraction_type}")
                    folds_X_train.append(X_tr)
                    folds_X_test.append(X_te)
                    folds_y_train.append(y_tr)
                    folds_y_test.append(y_te)
                    folds_groups.append(grp)
                    folds_sampling_stats.append(stats)
                np.save(exp.get_var_path(), final_features)
                print('Les 5 folds ont été calculés avec succès !')
                print(folds_groups)

                model_name = config_models.models_name
                output_direc = exp.get_output_path(model_name)
                HYPERPARAMS_FILE = output_direc / 'best_hyperparameters.json'

                if RUN_OPTUNA:
                    update_progress("Recherche d'hyperparamètres Optuna")
                    print('Recherche lancée')
                    print('[OPTUNA] Début de la recherche...')
                    if HYPERPARAMS_FILE.exists():
                        with open(HYPERPARAMS_FILE, 'r') as fil:
                            saved_configs = json.load(fil)
                    else:
                        saved_configs = {}
                    save_optuna_name = (
                        f"{config_models.models_name}_{exp.shortdirname()}"
                        f"_fresh_{datetime.now():%Y%m%d_%H%M%S_%f}"
                    )
                    print(f"[DISK-SAVE] Les meilleurs paramètres seront sauvegardés dans '{HYPERPARAMS_FILE}'.")
                    if config_models.models_name == 'InceptionTimeModified':
                        from utilitaries.optuna.optuna_inception_utils import run_stage1_search
                        study = run_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                        bp = study.best_params
                        bp['out_channels'] = 2 ** bp.pop('out_channels_exp')
                        bp['bottleneck_channels'] = 2 ** bp.pop('bottleneck_channels_exp')
                        bp['batch_size'] = 2 ** bp.pop('batch_size_exp')
                        saved_configs[config_models.models_name] = bp
                    elif config_models.models_name == 'LstmTimeModified':
                        from utilitaries.optuna.optuna_lstm_utils import run_lstm_stage1_search
                        study = run_lstm_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                        bp = study.best_params
                        bp['hidden_size'] = 2 ** bp.pop('hidden_size_exp')
                        bp['batch_size'] = 2 ** bp.pop('batch_size_exp')
                        if bp.get('clip_grad') == 0.0:
                            bp['clip_grad'] = None
                        saved_configs[config_models.models_name] = bp
                    elif config_models.models_name == 'XGBoost TSFEL':
                        from utilitaries.optuna.optuna_xgb_utils import run_xgb_stage1_search
                        study = run_xgb_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                        saved_configs[config_models.models_name] = study.best_params
                    elif config_models.models_name == 'SVC TSFEL':
                        from utilitaries.optuna.optuna_svc_utils import run_svc_stage1_search
                        study = run_svc_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                        saved_configs[config_models.models_name] = study.best_params
                    elif config_models.models_name == 'RandomForest TSFEL':
                        from utilitaries.optuna.optuna_rf_utils import run_rf_stage1_search
                        study = run_rf_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
                        bp = study.best_params
                        if bp.get('max_depth') == 0:
                            bp['max_depth'] = None
                        saved_configs[config_models.models_name] = bp
                    with open(HYPERPARAMS_FILE, 'w') as fil:
                        json.dump(saved_configs, fil, indent=4)
                    print(f"[DISK-SAVE] Meilleurs paramètres sauvegardés dans '{HYPERPARAMS_FILE}' pour {config_models.models_name}.")

                update_progress("Préparation de l'entraînement")
                print('Entraînement lancé')
                parameters = {}
                config_optuna = False
                if use_optuna.value and HYPERPARAMS_FILE.exists():
                    with open(HYPERPARAMS_FILE, 'r') as fileh:
                        all_configs = json.load(fileh)
                    if model_name in all_configs:
                        print(f'[LOAD] Configuration Optuna trouvée dans {HYPERPARAMS_FILE} pour {model_name} !')
                        parameters = all_configs[model_name]
                        config_optuna = True
                    else:
                        print(f'[LOAD] Aucune config pour {model_name} dans ce fichier. Valeurs PAR DÉFAUT.')
                        parameters = DEFAULT_PARAMS.get(model_name, {})
                        config_optuna = False
                else:
                    print(f"[WARNING] Aucun fichier d'hyperparamètres trouvé à : {HYPERPARAMS_FILE}. Valeurs PAR DÉFAUT.")
                    parameters = DEFAULT_PARAMS.get(model_name, {})
                    config_optuna = False
                print(f'--> Paramètres appliqués : {parameters}\n')
                if len(folds_X_train) == 0:
                    raise ValueError('Les listes de folds sont vides')
                is_dl_model = config_models.extraction_type == 'time'
                n_dims = len(folds_X_train[0].shape) if hasattr(folds_X_train[0], 'shape') else 0
                if is_dl_model and n_dims != 3:
                    raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice 3D [patients, temps, features], mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'time' ?")
                elif not is_dl_model and n_dims != 2:
                    raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice tabulaire 2D, mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'TSFEL' ?")

                if RUN_TRAINING:
                    folds_X_fit_exact = []
                    folds_y_fit_exact = []
                    paliers_lc = [0.2, 0.4, 0.6, 0.8, 1.0]
                    lc_train_scores = np.zeros((5, len(paliers_lc)))
                    lc_val_scores = np.zeros((5, len(paliers_lc)))
                    lc_sample_sizes = []
                    for fold_idx_2 in range(5):
                        update_progress(f"Entraînement du fold {fold_idx_2 + 1}/5")
                        print(f'\n─────────────────── Entraînement du Fold {fold_idx_2 + 1}/5 ───────────────────')
                        X_train_fold_2 = folds_X_train[fold_idx_2]
                        y_train_fold_2 = folds_y_train[fold_idx_2]
                        groups_fold_2 = folds_groups[fold_idx_2]
                        model_path_fold = exp.get_model_path(config_models.models_name, fold_idx_2, extension, config_optuna=config_optuna)
                        file_X_exact = exp.get_lasso_path('X', fold_idx_2, 'parquet')
                        file_y_exact = exp.get_lasso_path('y', fold_idx_2, 'npy')
                        if os.path.exists(model_path_fold) and os.path.getsize(model_path_fold) > 0:
                            print(f'--> Modèle déjà entraîné trouvé à : {model_path_fold} (Passage au fold suivant)')
                            continue
                        print(f'\n[DEBUG TRAIN - Fold {fold_idx_2 + 1}] Shape de X_train_fold_2: {X_train_fold_2.shape}')
                        X_train_final_fold, y_train_final = (X_train_fold_2, y_train_fold_2)
                        groups_final_fold = groups_fold_2
                        X_calib, y_calib = (None, None)
                        if calibration.value:
                            if config_balance.balance_method != '':
                                print(f"    [INFO] Calibration Prior (analytique). Utilisation de 100% du fold pour l'entraînement...")
                                X_train_final_fold = X_train_fold_2
                                y_train_final = y_train_fold_2
                                groups_final_fold = groups_fold_2
                                X_calib = None
                                y_calib = None
                            else:
                                print(f'    [INFO] Calibration activée ({calibration_mode.value}) Séparation du fold via StratifiedGroupKFold...')
                                skf_calib = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                                train_idx_calib, calib_idx = next(skf_calib.split(X_train_fold_2, y_train_fold_2, groups=groups_fold_2))
                                X_train_final_fold = X_train_fold_2[train_idx_calib]
                                y_train_final = y_train_fold_2[train_idx_calib]
                                groups_final_fold = np.asarray(groups_fold_2)[train_idx_calib]
                                X_calib = X_train_fold_2[calib_idx]
                                y_calib = y_train_fold_2[calib_idx]
                                print(f'    [DEBUG] Shapes - Train Base: {X_train_final_fold.shape}, Calib: {X_calib.shape}')
                        print(f'    [LEARNING CURVE] Lancement de la boucle de paliers unifiée...')
                        X_test_fold_lc = folds_X_test[fold_idx_2]
                        X_val_np = X_test_fold_lc.to_numpy() if hasattr(X_test_fold_lc, 'to_numpy') else np.asarray(X_test_fold_lc)
                        y_val_np = np.asarray(folds_y_test[fold_idx_2])
                        for p_idx, p in enumerate(paliers_lc):
                            is_final_palier = p == 1.0
                            X_chunk_np, y_chunk_np, mask_chunk = training.get_learning_curve_chunk(X_train_final_fold, y_train_final, groups_final_fold, p, is_dl_model, seed)
                            if fold_idx_2 == 0:
                                lc_sample_sizes.append(len(X_chunk_np))
                            print(f'      -> Palier {int(p * 100)}% ({len(X_chunk_np)} patients)' + (' [ENTRAÎNEMENT FINAL & DISK-SAVE]' if is_final_palier else ' [ÉPHÉMÈRE]'))
                            if len(np.unique(y_chunk_np)) < 2:
                                print(f'          [WARNING] Une seule classe présente, saut de ce palier.')
                                lc_train_scores[fold_idx_2, p_idx] = np.nan
                                lc_val_scores[fold_idx_2, p_idx] = np.nan
                                continue
                            current_save_path = model_path_fold if is_final_palier else None
                            lasso_args = {'X_raw': X_train_final_fold, 'y_raw': y_train_final, 'groups_mask': groups_final_fold[mask_chunk] if mask_chunk is not None else None, 'file_X': file_X_exact, 'file_y': file_y_exact} if config_models.models_name == 'Logistic Regression Lasso TSFEL' else None
                            FC_UNITS_MAP = {'none': None, '64': (64,), '128': (128,), '256': (256,), '128_64': (128, 64), '256_128': (256, 128)}
                            if 'fc_units' in parameters and isinstance(parameters['fc_units'], str):
                                parameters['fc_units'] = FC_UNITS_MAP[parameters['fc_units']]
                            final_model_to_save, train_auc, val_auc = training.fit_model_by_name(model_name=config_models.models_name, X_train=X_chunk_np, y_train=y_chunk_np, X_val=X_val_np, y_val=y_val_np, seed=seed, is_final_palier=is_final_palier, save_path=current_save_path, lasso_args=lasso_args, **parameters)
                            lc_train_scores[fold_idx_2, p_idx] = train_auc
                            lc_val_scores[fold_idx_2, p_idx] = val_auc
                            if is_final_palier and (not is_dl_model) and (final_model_to_save is not None):
                                if calibration.value:
                                    if config_balance.balance_method != '':
                                        statsX = folds_sampling_stats[fold_idx_2]
                                        top = statsX['n_malades_avant'] * statsX['n_sains_apres']
                                        bottom = statsX['n_sains_avant'] * statsX['n_malades_apres']
                                        if bottom > 0 and top > 0:
                                            beta = top / bottom
                                            final_model_to_save = training.apply_prior_calibration(final_model_to_save, beta)
                                            print(f'    [DEBUG] Prior Calibration OK (beta = {beta:.4f})')
                                        else:
                                            print(f'    [WARNING] Impossible de calculer beta, classe manquante.')
                                    else:
                                        print(f'    [INFO] Application de la calibration {calibration_mode.value} sur le jeu held-out...')
                                        final_model_to_save = training.apply_model_calibration(final_model_to_save, X_calib, y_calib, calibration_mode.value)
                                joblib.dump(final_model_to_save, model_path_fold)
                                print(f'--> Modèle final enregistré à : {model_path_fold}')
                        folds_X_fit_exact.append(X_train_final_fold)
                        folds_y_fit_exact.append(y_train_final)
                    print('Cross Validation terminée ! 5 modèles ont été enregistrés avec succès.')

                output_dir = exp.get_output_path(config_models.models_name, config_optuna=config_optuna)
                print('Dossier de sortie prêt :', output_dir)

                asymetric_incertitude_score = None
                auc_final_score = None
                best_f1_score = None
                brier_score_score = None
                mcc_score = None
                mean_p1_score = None
                mean_risk_diff_score = None
                non_overlap_area_score = None
                y_pred_score = None
                if type_donnees.value == 'score':
                    auc_final_score, fpr_score, tpr_score, th_score, brier_score_score, best_f1_score, best_t_score, y_pred_score, mcc_score, non_overlap_area_score, asymetric_incertitude_score, mean_risk_diff_score, mean_p1_score = sfu.plot_all_figs(saps2_pred, saps2_true, config_models, False, save_figure.value, output_dir, config_transparent)

                if RUN_TRAINING:
                    try:
                        sfu.plot_collected_learning_curve(lc_sample_sizes, lc_train_scores, lc_val_scores, config_models.models_name, savefig=save_figure.value, folder=output_dir, transparent=config_transparent)
                    except:
                        print("Erreur (non blocante) : la courbe d'apprentissage ne peut pas être enregistrée car elle l'a déjà été !")

                all_test_scores, all_train_scores = ([], [])
                all_auc_scores, all_brier_scores = ([], [])
                all_y_true_report, all_y_pred_report = ([], [])
                all_y_test_global, all_probas_uncalib, all_probas_calib = ([], [], [])
                update_progress("Évaluation des 5 folds")
                for fold_idx in range(5):
                    update_progress(f"Évaluation du fold {fold_idx + 1}/5")
                    print(f'\n─────────────────── Évaluation du Fold {fold_idx + 1}/5 ───────────────────')
                    X_train, X_test = (folds_X_train[fold_idx], folds_X_test[fold_idx])
                    y_train, y_test = (folds_y_train[fold_idx], folds_y_test[fold_idx])
                    loaded_model = exp.get_model_path(config_models.models_name, fold_idx, extension, config_optuna=config_optuna)
                    print('Modèle chargé :', loaded_model)
                    if config_models.models_name == 'InceptionTimeModified':
                        res = evaluate.evaluate_inception_fold(X_test, y_test, loaded_model)
                        all_auc_scores.append(res['auc'])
                        all_brier_scores.append(res['brier'])
                    elif config_models.models_name == 'LstmTimeModified':
                        res = evaluate.evaluate_lstm_fold(fold_idx, X_test, y_test, loaded_model)
                        all_auc_scores.append(res['auc'])
                        all_brier_scores.append(res['brier'])
                    elif config_models.extraction_type == 'TSFEL':
                        res = evaluate.evaluate_tsfel_fold(fold_idx, X_train, X_test, y_train, y_test, loaded_model, calibration.value)
                        all_test_scores.append(res['test_score'])
                        all_train_scores.append(res['train_score'])
                        all_y_true_report.extend(res['y_test'])
                        all_y_pred_report.extend(res['y_pred_test'])
                    else:
                        raise ValueError(f"Modèle ou type d'extraction non pris en compte : {config_models.models_name}")
                    all_y_test_global.extend(res['y_test'])
                    all_probas_uncalib.extend(res['probas_uncalib'])
                    all_probas_calib.extend(res['probas_calib'])
                print('\n' + '=' * 20 + ' BILAN GLOBAL DE LA CROSS-VALIDATION ' + '=' * 20)
                all_y_test_global = np.array(all_y_test_global)
                all_probas_uncalib = np.array(all_probas_uncalib)
                all_probas_calib = np.array(all_probas_calib)
                if config_models.extraction_type == 'TSFEL':
                    mean_acc = np.mean(all_test_scores)
                    std_acc = np.std(all_test_scores)
                    print(f'Score moyen (Accuracy) : {mean_acc:.4f} (± {std_acc:.4f})')
                    print("\nRapport de classification cumulé (sur l'ensemble des 5 folds) :")
                    print(classification_report(all_y_true_report, all_y_pred_report, target_names=['Alive', 'Deceased'], zero_division=0))
                else:
                    mean_auc, std_auc = (np.mean(all_auc_scores), np.std(all_auc_scores))
                    mean_brier, std_brier = (np.mean(all_brier_scores), np.std(all_brier_scores))
                    print(f'AUC moyenne   : {mean_auc:.4f} (± {std_auc:.4f})')
                    print(f'Brier moyenne : {mean_brier:.4f} (± {std_brier:.4f})')
                print('=' * 79)
                print('\nGénération de la courbe de calibration poolée...')
                print(f'DEBUG SIZES -> y_true: {len(all_y_test_global)}, uncalib: {len(all_probas_uncalib)}, calib: {len(all_probas_calib)}')
                sfu.calibration_curve_homemade(all_probas_uncalib, all_probas_calib, all_y_test_global, config_models.models_name, config_models.extraction_type, calibration.value, save_figure.value, output_dir, config_transparent, calibration_mode.value)
                calibration_results = sfu.calibration_curve_advanced(all_probas_uncalib, all_probas_calib, all_y_test_global, config_models.models_name, config_models.extraction_type, calibration.value, save_figure.value, output_dir, config_transparent, calibration_mode.value)
                probas = all_probas_calib
                y_test = all_y_test_global

                if RUN_LASSO:
                    print('Lasso Path lancé')
                    if config_models.models_name == 'Logistic Regression Lasso TSFEL':
                        print('Génération ultra-rapide des Lasso Paths (Warm Start + Multi-processing)...')
                        Cs_grid = np.logspace(-4, 4, 100)

                        def process_single_fold(idx):
                            file_X = exp.get_lasso_path('X', idx, 'parquet')
                            file_y = exp.get_lasso_path('y', idx, 'npy')
                            X_pure_fit_np = pl.read_parquet(file_X).to_numpy()
                            y_pure_fit = np.load(file_y)
                            model_path = exp.get_model_path('Logistic Regression Lasso TSFEL', idx, extension)
                            model_L1 = joblib.load(model_path)
                            best_C2 = model_L1.C
                            lr_path_model = LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=100001, random_state=seed, warm_start=True)
                            coefs_list = []
                            sorted_Cs = np.sort(Cs_grid)
                            for c_val in sorted_Cs:
                                lr_path_model.set_params(C=c_val)
                                lr_path_model.fit(X_pure_fit_np, y_pure_fit)
                                coefs_list.append(lr_path_model.coef_[0].copy())
                            return (sorted_Cs, np.array(coefs_list), best_C2, X_pure_fit_np.shape[1])
                        results = Parallel(n_jobs=-1)((delayed(process_single_fold)(f_idx) for f_idx in range(5)))
                        for fold_idx_L1, (sorted_Cs, coefs_path, best_C2, n_features) in enumerate(results):
                            plt.figure(figsize=(10, 6))
                            plt.plot(sorted_Cs, coefs_path, alpha=0.7)
                            plt.axvline(x=best_C2, color='black', linestyle='--', linewidth=2, label=f'C optimal (Fold {fold_idx_L1 + 1}) = {best_C2:.4f}')
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

                update_progress("Génération des métriques et figures finales")
                auc_final, fpr, tpr, thresholds_roc = sfu.roc_curve_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)

                auprc_final, precision, recall, thresholds = sfu.prc_curve_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)

                non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1 = sfu.kde_plot_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)

                best_f1, best_t = sfu.f1_score_evolution(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)

                y_pred, mcc = sfu.confusion_matrix_homemade(probas, y_test, best_t, config_models.models_name, save_figure.value, output_dir, config_transparent)

                global_brier = sfu.brier_evolution(probas, y_test, save_figure.value, output_dir, transparent=config_transparent)

                if type_donnees.value == 'modèle':
                    calibration_stats = sfu.get_calibration_stats(
                        probas,
                        y_test,
                    )

                    all_res = {
                        "y_true": y_test,
                        "probas": probas,
                        "preds": y_pred,
                        "f1_score": best_f1,
                        "mcc": mcc,
                        "auc": auc_final,
                        "brier": global_brier,
                        "calibration_intercept": calibration_stats["intercept"],
                        "calibration_slope": calibration_stats["slope"],
                        "ici": calibration_stats["ici"],
                        "e90": calibration_stats["e90"],
                        "non_overlap_area": non_overlap_area,
                        "asymetric_incertitude": asymetric_incertitude,
                        "mean_risk_diff": mean_risk_diff,
                        "mean_deaths_prediction": mean_p1,
                    }

                else:
                    calibration_stats = sfu.get_calibration_stats(
                        saps2_pred,
                        saps2_true,
                    )

                    all_res = {
                        "y_true": saps2_true,
                        "probas": saps2_pred,
                        "preds": y_pred_score,
                        "f1_score": best_f1_score,
                        "mcc": mcc_score,
                        "auc": auc_final_score,
                        "brier": brier_score_score,
                        "calibration_intercept": calibration_stats["intercept"],
                        "calibration_slope": calibration_stats["slope"],
                        "ici": calibration_stats["ici"],
                        "e90": calibration_stats["e90"],
                        "non_overlap_area": non_overlap_area_score,
                        "asymetric_incertitude": asymetric_incertitude_score,
                        "mean_risk_diff": mean_risk_diff_score,
                        "mean_deaths_prediction": mean_p1_score,
                    }
                update_progress("Sauvegarde des résultats finaux")
                output_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(all_res, output_dir / 'all_res.joblib')
                resu = joblib.load(output_dir / 'all_res.joblib')
                if RUN_COMPARISON:
                    comparaisons = [exp.load_model("InceptionTimeModified"), exp.load_model("LstmTimeModified"), exp.load_model("RandomForest TSFEL", ""), exp.load_model("XGBoost TSFEL"), exp.load_model("SVC TSFEL"), exp.load_model("Logistic Regression Lasso TSFEL", ""), exp.load_model("IGS2")]
                    sfu.générer_rapport_comparatif(comparaisons, save_dir="Comparaison ALl", table_format='fancy_grid')

                update_progress("Expérience terminée avec succès")
                print(f"[EXPÉRIENCE {EXPERIMENT_INDEX}/{TOTAL_EXPERIMENTS}] TERMINÉE AVEC SUCCÈS", flush=True)
                plt.close("all")

update_progress("Toutes les expériences sont terminées")
print(f"[PIPELINE] Fin normale. Journal complet : {_LOG_PATH}", flush=True)
