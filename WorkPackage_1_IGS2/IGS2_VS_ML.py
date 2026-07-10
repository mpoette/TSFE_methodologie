import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importation des bibliothèques
    """)
    return


@app.cell
def _():
    # 1. Configuration stricte de la reproductibilité 
    seed = 42
    import os
    import sys

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Optionnel : Bloquer le multi-threading si on veut du 100% déterministe en CPU
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. Ajout du chemin pour les modules locaux
    sys.path.append(os.path.abspath(".."))

    # 3. Imports de la bibliothèque standard
    import json
    import math
    import time
    from pathlib import Path

    # 4. Librairies tierces (Data Science, Visualisation, ML)
    import joblib
    import marimo as mo
    import matplotlib.pyplot as plt
    import optuna
    import pandas as pd
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns
    import tsfel
    from sklearn.metrics import (
        brier_score_loss,
        classification_report,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import GridSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from joblib import Parallel, delayed
    from collections import Counter
    # 5. Imports locaux (modules customs)
    import utilitaries.create_merged_dataset as create_merged_dataset
    import utilitaries.extract_data_utils as extract
    import utilitaries.features_extraction_utils as extract_feat
    import utilitaries.marimo_utils as mo_utils
    import utilitaries.optuna.optuna_utils as optuna_utils
    import utilitaries.preprocessing_utils as preproc
    import utilitaries.preprocessing_utils as ui
    import utilitaries.path_utils as path_utils
    import utilitaries.show_fig_utils as sfu
    import utilitaries.training_utils as training
    import utilitaries.evaluate_utils as evaluate
    import utilitaries.shared_ui as spl
    from utilitaries.models.inceptionTimeModified import (
        evaluate_on_test,
        load_model_from_checkpoint,
        predict_proba,
        train_inception_time,
    )
    from utilitaries.models.lstmTimeModified import (
        evaluate_lstm_on_test,
        load_lstm_from_checkpoint,
        predict_proba_lstm,
        train_lstm_model,
    )
    pl.Config.set_tbl_cols(-1)
    return (
        LogisticRegression,
        Parallel,
        Path,
        StratifiedGroupKFold,
        classification_report,
        create_merged_dataset,
        cs,
        delayed,
        evaluate,
        extract,
        extract_feat,
        joblib,
        json,
        mo,
        mo_utils,
        np,
        os,
        path_utils,
        pl,
        plt,
        preproc,
        seed,
        sfu,
        spl,
        torch,
        training,
    )


@app.cell
def _(torch):
    # 1. Vérifier si le GPU (CUDA) est disponible
    cuda_dispo = torch.cuda.is_available()
    print(f"Est-ce que CUDA est disponible ? {cuda_dispo}")

    # 2. Voir sur quel appareil PyTorch est configuré par défaut
    appareil_actuel = torch.cuda.current_device() if cuda_dispo else "CPU"
    print(f"Appareil actuellement utilisé : {appareil_actuel}")

    if cuda_dispo:
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Widgets Marimo utilisés dans ce notebook
    """)
    return


@app.cell
def _(mo, spl):
    uid = mo.ui.dictionary(spl.create_pipeline_widgets())
    return (uid,)


@app.cell
def _(spl, uid):
    # 1. Gestion des widgets dynamiques dépendants
    models = spl.get_model_dropdown(uid["type_donnees"].value)
    return (models,)


@app.cell
def _(models, uid):
    save_figure = uid["save_figure"]
    config_mode = uid["mode"].value
    config_models = models.value
    config_cleaning = uid["cleaning"].value
    config_y = uid["y_dd"].value
    config_keep_pop = uid["keep_pop"].value
    config_balance = uid["balance"].value
    config_transparent = uid["transparent"].value
    boruta_filter = uid["boruta_filter"]
    config_boruta = boruta_filter.value
    use_optuna = uid["use_optuna"]
    modex = uid["modex"]
    type_donnees = uid["type_donnees"]
    return (
        boruta_filter,
        config_balance,
        config_boruta,
        config_cleaning,
        config_keep_pop,
        config_mode,
        config_models,
        config_transparent,
        config_y,
        modex,
        save_figure,
        type_donnees,
        use_optuna,
    )


@app.cell
def _(config_balance, models, spl):
    calibration, calibration_mode = spl.get_calibration_widgets(models.value, config_balance.balance_method)
    return calibration, calibration_mode


@app.cell
def _(config_transparent, plt):
    if not config_transparent:
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams['axes.facecolor'] = "white"
        plt.rcParams['savefig.facecolor'] = "white"
    return


@app.cell
def _(config_models, mo):
    run = mo.ui.run_button(
        label=f"Lancer l'entraînement du modèle {config_models.models_name}"
    )
    return (run,)


@app.cell
def _(mo):
    run_lasso = mo.ui.run_button(
        label=f"Lancer la recherche du Lasso Path"
    )
    return (run_lasso,)


@app.cell
def _(mo):
    run_test = mo.ui.run_button(
        label=f"Comparer les AUC des modèles"
    )
    return (run_test,)


@app.cell
def _(mo):
    run_optuna = mo.ui.run_button(
        label=f"Lancer la recherche d'hyperparamètres"
    )
    return (run_optuna,)


@app.cell
def _(boruta_filter, calibration, calibration_mode, config_models, spl):
    ui_tsfel, extract_tsfel, class_weight_choice = spl.get_tsfel_ui_components(
        extraction_type=config_models.extraction_type,
        models_type=config_models.models_type,
        boruta_filter=boruta_filter,
        calibration=calibration,
        calibration_mode=calibration_mode,  
    )
    return class_weight_choice, extract_tsfel, ui_tsfel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration dynamique des chemins utilisés
    """)
    return


@app.cell
def _(
    calibration_mode,
    class_weight_choice,
    config_cleaning,
    config_mode,
    config_y,
    modex,
    path_utils,
    seed,
    str_balance_method,
    str_pop,
):
    exp = path_utils.Experiment(config_mode.name, config_cleaning, config_y, str_balance_method, modex, class_weight_choice, str_pop, seed, stratify_mode = "", calibrated_mode = calibration_mode.value)
    return (exp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extraction du dataset
    """)
    return


@app.cell
def _(pl):
    pl.Config.set_tbl_cols(-1)
    return


@app.cell
def _():
    dataset_path = "/../../../data2/paquie.d/Datasets/output"
    return (dataset_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chargement Dataframes et merge de statique et dynamique
    """)
    return


@app.cell
def _(dataset_path, os, pl):
    _path = os.path.join(dataset_path, "df_static_ano_clean.parquet")
    df_static = pl.scan_parquet(_path)
    return (df_static,)


@app.cell
def _(dataset_path, extract, os):
    _path = os.path.join(dataset_path, 'df_dynamic_full_clean.parquet')
    df_dynamic = extract.extract_data_survie(_path)
    return (df_dynamic,)


@app.cell
def _(create_merged_dataset, dataset_path, df_dynamic, df_static):
    df_merged = create_merged_dataset.create_merged_dataset(df_static, df_dynamic, True, save = True, folder = dataset_path)
    return (df_merged,)


@app.cell
def _(df_merged, pl):
    df_merged_1 = df_merged.with_columns([

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_28d") ]
                                      )
    df_merged_1 = df_merged_1.filter(pl.col("delta_hour") >= 0)
    df_merged_1 = df_merged_1.collect()
    return (df_merged_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extraction de la fenêtre temporelle de 24H
    """)
    return


@app.cell
def _(df_merged_1, extract, pl):
    df_clean = extract.prepare_data(df_merged_1, hour_offset = 0, random = False, max_hour = 6, strict_mode = True, target_col = "isDeceased_lt_28d", show_fig = True)

    df_clean = df_clean.with_columns(
        (pl.col("is_ventilated").fill_null(pl.lit(False))).alias("is_ventilated"),
        (pl.col("is_prone").fill_null(pl.lit(False))).alias("is_prone"),
        (pl.col("is_conscious").fill_null(pl.lit(False))).alias("is_conscious"),
        (pl.col("is_cvvhf").fill_null(pl.lit(False))).alias("is_cvvhf"),
        (pl.col("is_hdi").fill_null(pl.lit(False))).alias("is_hdi"),

    )
    df_clean = df_clean.filter(pl.col("taille").is_not_null() & pl.col("poids_admission").is_not_null())
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtrage du dataset/Prétraitement
    """)
    return


@app.cell
def _(mo, mo_utils, models):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        models,
        mo.md(mo_utils.config_end)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### On enlève les features inutiles/donnant trop d'informations
    """)
    return


@app.cell
def _(df_clean, pl, target_col):
    score = ["NEWS", "NEWS2", "sapsii", "sapsii_prob"]
    dustbin = ["endotracheal_tube","tracheo", "installation", "eer", "hx_comorbidité_majeure", "imc", "neuro_status", "ecmo_all", "prone", "plq"]
    useless = ["arret_therapeutique", "limitation_therapeutique", "hematocrit", "peak_pressure"]
    icu_useless = ["icu_actes", "icu_mode_sortie", "icu_DA", "hosp_admissionMode", "hosp_primaryDiagnosis", "hosp_primaryDiagnosisCode"]
    icu_useful = ["icu_ghm", "icu_DP_code", "icu_DP", "icu_mode_entree",]
    broken = []
    cheat = ["category", "heure_entiere", "encounterId", "delta_hour", "heure_calibree", "heure_entiere", "year_inTime", target_col, "deces_datediff_days", "isDeceased", "adm_unit", "out_unit", "transition_units", "los", "adm_year", "hosp_los", "hosp_dischargeMode", "deces_hosp"]
    df_clean_keep = df_clean.drop([*score, *dustbin, *useless, *cheat, *icu_useless, *broken])
    df_clean_keep = df_clean_keep.select([pl.col(c) for c in df_clean_keep.columns if not c.endswith("_detected_term")])
    return df_clean_keep, icu_useful


@app.cell
def _():
    commonly_used = ["score_glasgow", "is_conscious", "heart_rate", "creat", "is_cvvhf", "is_hdi", "pao2", "is_ventilated", "fio2_corr", "age", "temp", "urine_rate", "pas", "pam", "pad", "bili_tot", "leucocytes", "admission_type", "fr", "ph", "sodium", "potassium", "num_plq", "blood_urea", "nad_dose_poids", "dobu_dose_poids", "hemoglobine", "tp", "spo2", "hco3", "glyc_cap"]
    return (commonly_used,)


@app.cell
def _(df_clean_keep, mo):
    custom_features = mo.ui.multiselect(
        options=df_clean_keep.columns,
        value= df_clean_keep.columns,
        label="(features sélectionnables)",
    )
    return (custom_features,)


@app.cell
def _(
    commonly_used,
    cs,
    custom_features,
    df_clean_keep,
    icu_useful,
    json,
    mo,
    mo_utils,
    modex,
):
    with open ("../../Preprocessing_pipeline/preprocessing-pipelines/json/dynamic_features.json", "r") as file:
        json_feat = json.load(file)

    if modex.value == "Mode All":
        keep_feats = df_clean_keep.columns

    elif modex.value == "Mode All Without pmsi":
        # On sélectionne TOUT, SAUF ce qui commence par "hx_" OU "icu_"
        keep_feats = df_clean_keep.select(
            ~cs.starts_with("hx_") & ~cs.starts_with("icu_")
        ).columns

    elif modex.value == "Mode Commonly Used Without pmsi":
        keep_feats = df_clean_keep.select(
            commonly_used,
        ).columns
    elif modex.value == "Mode Commonly Used":
        keep_feats = df_clean_keep.select(
            commonly_used,
            *icu_useful,
            cs.starts_with("hx_")
        ).columns
    elif modex.value == "Mode Custom":
        keep_feats = custom_features.value
    else:
        keep_feats = mo_utils.FEAT[modex.value].keep_feats
        if "urine_rate" in keep_feats:
            keep_feats.remove("urine_rate")

    str_keep_feats = ""

    for kf in keep_feats:
        if kf in json_feat:
            str_keep_feats += f"- {json_feat[kf]['description']} \n"

    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français (dynamic feature only):** \n{str_keep_feats}"),
        mo.md(mo_utils.config_end)
    ])
    return keep_feats, str_keep_feats


@app.cell
def _(df_clean, keep_feats, patient_col, target_col, time_col):
    df_clean_1 = df_clean.select(*keep_feats, patient_col, time_col, target_col)
    return (df_clean_1,)


@app.cell
def _(keep_feats):
    keep_features = keep_feats.copy()
    return (keep_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Liste des colonnes utiles
    """)
    return


@app.cell
def _(extract):
    patient_col = extract.ID_COL
    time_col = extract.TIME_COL
    target_col = "isDeceased_lt_28d"
    expected_length = extract.WINDOW_SIZE
    return expected_length, patient_col, target_col, time_col


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Filtrage des patients possédant moins de 24h de données
    """)
    return


@app.cell
def _(df_clean_1, expected_length, patient_col, pl, time_col):
    # On garde les encounters de longueur exacte 
    valid_ids = df_clean_1.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
    df_clean_2 = df_clean_1.join(valid_ids, on=patient_col, how='inner')
    if df_clean_2.is_empty():
        raise ValueError("Aucun patient n'a exactement la longueur attendue.")
    df_clean_2 = df_clean_2.sort(patient_col, time_col)
    return (df_clean_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Transformation des variables qualitatives en One Hot Encoding
    """)
    return


@app.cell
def _(config_models, cs, df_clean_2, keep_features, modex, pl, target_col):
    # On passe le type d'admission en one hot encoding avec polars 
    df_clean_3 = (
        df_clean_2
        .with_columns(pl.col("admission_type").fill_null("Unknown"))
        .to_dummies(columns=["admission_type"])
    )
    if modex.value in ["Mode Custom", "Mode All Without pmsi", "Mode All"]:
        df_clean_3 = (
            df_clean_3.with_columns(pl.col("gender").fill_null("Unknown")).to_dummies(columns = ["gender"])
        )
        cols_gender = [c for c in df_clean_3.columns if c.startswith("gender_")]
        if "gender" in keep_features:
            keep_features.remove("gender") 
            keep_features.extend(cols_gender)


    df_clean_3 = df_clean_3.with_columns(
        cs.numeric().cast(pl.Float64),
        cs.boolean().cast(pl.Float64)
    )
    cols_admission = [c for c in df_clean_3.columns if c.startswith("admission_type_")]
    if "admission_type" in keep_features:
        keep_features.remove("admission_type") 
        keep_features.extend(cols_admission)

    final_features = list(dict.fromkeys(keep_features))

    assert target_col not in final_features, f"Alerte Leakage : {target_col} est présente dans les features !"
    print(f"Nombre de features envoyées au {config_models.models_name} : {len(final_features)} : {final_features}")

    X_init = df_clean_3.select(final_features).to_numpy()
    y_init = df_clean_3[target_col].to_numpy()
    return X_init, df_clean_3, final_features, y_init


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Split Train/Test, Extraction de features/Feature Selection et Scaling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On garde d'abord 20% pour tester le modèle :
    """)
    return


@app.cell
def _(
    StratifiedGroupKFold,
    X_init,
    config_models,
    df_clean_3,
    exp,
    extract,
    extract_feat,
    extract_tsfel,
    final_features,
    np,
    os,
    patient_col,
    pl,
    seed,
    target_col,
    time_col,
    type_donnees,
    y_init,
):
    if config_models.extraction_type == "TSFEL":
        filename_global_brut = exp.get_tsfel_parquet_path()
        if extract_tsfel.value or not os.path.exists(filename_global_brut):
            print("Lancement de l'extraction TSFEL globale sur tous les patients")
            # On enlève les features statiques
            static_feats = ["admission_type_Medical", "admission_type_Scheduled Surgery", "admission_type_Unknown", "admission_type_Unscheduled Surgery", "score_glasgow", "age"]
            tsfel_features = [c for c in final_features if c not in static_feats]

            TSFEL_global_df = extract_feat.extract_tsfel_per_patient(df_clean_3, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
            static_global = df_clean_3.select([extract.ID_COL, *static_feats]).unique()

            df_tsfel_complet = TSFEL_global_df.join(static_global, on = extract.ID_COL, how = "inner")
            df_tsfel_complet.write_parquet(filename_global_brut)
            print("Extraction globale sauvegardée")
        else:
            df_tsfel_complet = pl.read_parquet(filename_global_brut)
        keepVariableList_0 = df_tsfel_complet.columns
        parent_folder = filename_global_brut.parent
        np.save(parent_folder / "keepVariableList_0.npy", keepVariableList_0)

    groups_init = df_clean_3[patient_col].to_numpy()
    sgkf_init = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    train_init_idx, test_init_idx = next(sgkf_init.split(X=X_init, y=y_init, groups = groups_init))

    train_init_patients = df_clean_3[train_init_idx].select(patient_col).unique()
    test_init_patients = df_clean_3[test_init_idx].select(patient_col).unique()



    if config_models.extraction_type == "TSFEL" : 
        # On filtre notre gros DataFrame TSFEL pré-calculé pour ce fold (unique)
        train_init_tsfel = df_tsfel_complet.join(train_init_patients, on=patient_col, how="inner").sort(patient_col)
        test_holdout_tsfel = df_tsfel_complet.join(test_init_patients, on=patient_col, how="inner").sort(patient_col)

        X = train_init_tsfel
        y = train_init_tsfel[target_col].to_numpy()
        groups = train_init_tsfel[patient_col].to_numpy()

    elif config_models.extraction_type == "time" or type_donnees.value == "score":
        train_init_df = df_clean_3[train_init_idx].sort([patient_col, time_col])
        test_holdout_df = df_clean_3[test_init_idx].sort([patient_col, time_col])

        X = train_init_df
        y = train_init_df[target_col].to_numpy()
        groups = train_init_df[patient_col].to_numpy()
    return X, groups, train_init_df, train_init_tsfel, y


@app.cell
def _(X, df_clean, pl, type_donnees):
    if type_donnees.value == "score":
        df_clean_saps2 = (
            X
            .join(
                df_clean.select(["sapsii_prob", "encounterId"]).cast(pl.Float64), 
                on="encounterId", 
                how="inner"
            )
            .filter(pl.col("sapsii_prob").is_not_null())
            .group_by("encounterId")
            .first()
            .sort(by="encounterId")
        )
        df_clean_saps2.describe()
        saps2_pred = df_clean_saps2["sapsii_prob"].to_numpy()
        saps2_true = df_clean_saps2["isDeceased_lt_28d"].to_numpy()
    return saps2_pred, saps2_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Split Train/Test
    cette cellule ne fonctionne que parce qu'on a des fenêtres qui sont toutes de 24h => Le poids entre les étiquettes est le même peu importe le patient qu'on prend ce qui permet un bon équilibrage train/test
    """)
    return


@app.cell
def _(
    StratifiedGroupKFold,
    X,
    config_balance,
    config_boruta,
    config_models,
    exp,
    expected_length,
    extract,
    final_features,
    groups,
    np,
    preproc,
    seed,
    target_col,
    train_init_df,
    train_init_tsfel,
    y,
):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # Initialisation des conteneurs
    folds_X_train, folds_X_test = [], []
    folds_y_train, folds_y_test = [], []
    folds_groups = []
    folds_sampling_stats = []
    pipeline_config = {
        "patient_col": extract.ID_COL,
        "time_col": extract.TIME_COL,
        "target_col": target_col,
        "boruta_filter": config_boruta,
        "balance_method": config_balance.balance_method,
        "expected_length": expected_length,
        "final_features": final_features,
        "exp": exp
    }

    if "train_init_df" in locals() and train_init_df is not None:
        pipeline_config["train_init"] = train_init_df
    else:
        pipeline_config["train_init"] = train_init_tsfel

    # Exécution de la Cross-Validation
    for fold_idxx, (train_idx, test_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
        print(f"\n─────────────────── Traitement du Fold {fold_idxx + 1}/5 ───────────────────")


        if config_models.extraction_type == "TSFEL":
            X_tr, X_te, y_tr, y_te, grp, stats = preproc.process_tsfel_fold(fold_idxx, train_idx, test_idx, X, y, groups, seed, **pipeline_config)
        elif config_models.extraction_type == "time":
            X_tr, X_te, y_tr, y_te, grp, stats = preproc.process_time_fold(fold_idxx, train_idx, test_idx, seed, **pipeline_config)
        else:
            raise ValueError(f"Type d'extraction inconnu ou non implémenté : {config_models.extraction_type}")

        # Accumulation des données nettoyées du fold
        folds_X_train.append(X_tr)
        folds_X_test.append(X_te)
        folds_y_train.append(y_tr)
        folds_y_test.append(y_te)
        folds_groups.append(grp)
        folds_sampling_stats.append(stats)

    # Sauvegarde globale finale
    np.save(exp.get_var_path(), final_features)
    print("Les 5 folds ont été calculés avec succès !")
    print(folds_groups)
    return (
        folds_X_test,
        folds_X_train,
        folds_groups,
        folds_sampling_stats,
        folds_y_test,
        folds_y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Attention : ici, train_df contient toutes les features et pas seulement celles qui ont été conservées. C'est normal et n'impacte pas les résultats du modèle puisque lui ne reçoit que les bonnes features (en théorie)
    """)
    return


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ## Training sur {config_models.models_name}
    """)
    return


@app.cell
def _(config_cleaning, config_models, config_y, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        mo.md(f"### Entraînement avec les paramètres suivants : \n - Modèle utilisé = {config_models.models_name} \n - Nettoyage des Surveillances Continues = {config_cleaning.clean} \n - Cible à prédire = {config_y.target_name} \n "),
        mo.md(mo_utils.config_end)
    ])
    return


@app.cell
def _(class_weight_choice, config_balance, config_keep_pop, config_models):
    # on créé un nom unique de modèle
    str_pop = ""
    if config_keep_pop.keep_population != "all_diseases":
        str_pop = "_"+config_keep_pop.keep_population

    str_balance_method = ""
    if config_balance.balance_method:
        str_balance_method = config_balance.balance_method + "_"
    extension = ".joblib" if config_models.extraction_type == "TSFEL" else ".pt"

    DEFAULT_PARAMS = {
        # --- Deep Learning (Tes valeurs par défaut) ---
        "InceptionTimeModified": {
            "epochs": 100,
            "patience": 30,
            "lr": 1e-3,
        },
        "LstmTimeModified": {
            "epochs": 100,
            "patience": 30,
            "lr": 1e-3,
        },

        # --- Machine Learning Classique (TSFEL) ---
        "RandomForest TSFEL": {
            "n_estimators": 200,
            "max_depth": 12,          # Évite le surapprentissage par rapport à un max_depth infini
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight" : class_weight_choice.value[1:]
        },
        "RandomForest Imbalanced TSFEL": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight" : class_weight_choice.value[1:]
        },
        "XGBoost TSFEL": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,    # Un poil plus bas pour une meilleure convergence
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "SVC TSFEL": {
            "C": 1.0,                 # Paramètre de régularisation standard
            "gamma": "scale",
            "kernel": "rbf",
            "class_weight" : class_weight_choice.value[1:]
        }
    }
    return DEFAULT_PARAMS, extension, str_balance_method, str_pop


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ## Optimisation des hyperparamètres de {config_models.models_name} avec Optuna
    """)
    return


@app.cell
def _(mo, mo_utils, run_optuna):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run_optuna,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(config_models, exp):
    # --- CHARGEMENT DES PARAMÈTRES VIA EXPERIMENT ---
    model_name = config_models.models_name
    # Récupération du dossier de sortie de l'expérience et définition du fichier JSON
    output_direc = exp.get_output_path(model_name)
    HYPERPARAMS_FILE = output_direc / "best_hyperparameters.json"
    return HYPERPARAMS_FILE, model_name


@app.cell
def _(
    HYPERPARAMS_FILE,
    config_models,
    exp,
    folds_X_train,
    folds_y_train,
    json,
    mo,
    run_optuna,
):
    mo.stop(not run_optuna.value, "Clique pour lancer")
    print("Recherche lancée")

    print("[OPTUNA] Début de la recherche...")

    # 1. Charger l'historique existant s'il existe
    if HYPERPARAMS_FILE.exists():
        with open(HYPERPARAMS_FILE, "r") as fil:
            saved_configs = json.load(fil)
    else:
        saved_configs = {}


    save_optuna_name = config_models.models_name + "_" + exp.shortdirname()

    print(f"[DISK-SAVE] Les meilleurs paramètres seront sauvegardés dans '{HYPERPARAMS_FILE}'.")

    # 2. Exécuter l'optimisation selon le modèle sélectionné
    if config_models.models_name == "InceptionTimeModified":
        from utilitaries.optuna.optuna_inception_utils import run_stage1_search
        study = run_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
        bp = study.best_params
        bp["out_channels"] = 2 ** bp.pop("out_channels_exp")
        bp["bottleneck_channels"] = 2 ** bp.pop("bottleneck_channels_exp")
        bp["batch_size"] = 2 ** bp.pop("batch_size_exp")
        saved_configs[config_models.models_name] = bp

    elif config_models.models_name == "LstmTimeModified":
        from utilitaries.optuna.optuna_lstm_utils import run_lstm_stage1_search
        study = run_lstm_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
        bp = study.best_params
        bp["hidden_size"] = 2 ** bp.pop("hidden_size_exp")
        bp["batch_size"] = 2 ** bp.pop("batch_size_exp")
        if bp.get("clip_grad") == 0.0:
            bp["clip_grad"] = None
        saved_configs[config_models.models_name] = bp

    elif config_models.models_name == "XGBoost TSFEL":
        from utilitaries.optuna.optuna_xgb_utils import run_xgb_stage1_search
        study = run_xgb_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
        saved_configs[config_models.models_name] = study.best_params

    elif config_models.models_name == "SVC TSFEL":
        from utilitaries.optuna.optuna_svc_utils import run_svc_stage1_search
        study = run_svc_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
        saved_configs[config_models.models_name] = study.best_params
    elif config_models.models_name == "RandomForest TSFEL":
        from utilitaries.optuna.optuna_rf_utils import run_rf_stage1_search
        study = run_rf_stage1_search(folds_X_train[0], folds_y_train[0], study_name=save_optuna_name)
        bp = study.best_params
        if bp.get("max_depth") == 0:
            bp["max_depth"] = None
        saved_configs[config_models.models_name] = bp
    # 3. Écriture sur le disque dans le fichier DYNAMIQUE
    with open(HYPERPARAMS_FILE, "w") as fil:
        json.dump(saved_configs, fil, indent=4)

    print(f"[DISK-SAVE] Meilleurs paramètres sauvegardés dans '{HYPERPARAMS_FILE}' pour {config_models.models_name}.")
    return


@app.cell
def _(mo, mo_utils, run):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(
    DEFAULT_PARAMS,
    HYPERPARAMS_FILE,
    config_models,
    folds_X_train,
    json,
    mo,
    model_name,
    run,
    use_optuna,
):
    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")

    parameters = {}
    config_optuna = False

    if use_optuna.value and HYPERPARAMS_FILE.exists():
        with open(HYPERPARAMS_FILE, "r") as fileh:
            all_configs = json.load(fileh)

        if model_name in all_configs:
            print(f"[LOAD] Configuration Optuna trouvée dans {HYPERPARAMS_FILE} pour {model_name} !")
            parameters = all_configs[model_name]
            config_optuna = True
        else:
            print(f"[LOAD] Aucune config pour {model_name} dans ce fichier. Valeurs PAR DÉFAUT.")
            parameters = DEFAULT_PARAMS.get(model_name, {})
            config_optuna = False
    else:
        print(f"[WARNING] Aucun fichier d'hyperparamètres trouvé à : {HYPERPARAMS_FILE}. Valeurs PAR DÉFAUT.")
        parameters = DEFAULT_PARAMS.get(model_name, {})
        config_optuna = False

    print(f"--> Paramètres appliqués : {parameters}\n")

    if len(folds_X_train) == 0:
        raise ValueError("Les listes de folds sont vides")

    # Sécurité
    is_dl_model = config_models.extraction_type == "time"
    n_dims = len(folds_X_train[0].shape) if hasattr(folds_X_train[0], "shape") else 0
    if is_dl_model and n_dims != 3:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice 3D [patients, temps, features], mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'time' ?")
    elif not is_dl_model and n_dims != 2:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice tabulaire 2D, mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'TSFEL' ?")
    return config_optuna, is_dl_model, parameters


@app.cell
def _(parameters):
    parameters
    return


@app.cell
def _(
    StratifiedGroupKFold,
    calibration,
    calibration_mode,
    config_balance,
    config_models,
    config_optuna,
    exp,
    extension,
    folds_X_test,
    folds_X_train,
    folds_groups,
    folds_sampling_stats,
    folds_y_test,
    folds_y_train,
    is_dl_model,
    joblib,
    np,
    os,
    parameters,
    seed,
    training,
):
    folds_X_fit_exact = []
    folds_y_fit_exact = []

    # Choix des paliers (ex: 5 paliers pour ne pas surcharger le temps de calcul)
    paliers_lc = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Matrices de stockage : [5 folds, 5 paliers]
    lc_train_scores = np.zeros((5, len(paliers_lc)))
    lc_val_scores = np.zeros((5, len(paliers_lc)))
    lc_sample_sizes = [] # Pour stocker les tailles réelles en nombre de lignes

    for fold_idx_2 in range(5):

        print(f"\n─────────────────── Entraînement du Fold {fold_idx_2 + 1}/5 ───────────────────")

        # Extraction des données spécifiques à ce fold (Noms d'origine restaurés)
        X_train_fold_2 = folds_X_train[fold_idx_2]
        y_train_fold_2 = folds_y_train[fold_idx_2]
        groups_fold_2 = folds_groups[fold_idx_2]

        # Génération d'un chemin STRICT et DÉTERMINISTE
        model_path_fold = exp.get_model_path(config_models.models_name, fold_idx_2, extension, config_optuna = config_optuna)
        file_X_exact = exp.get_lasso_path("X", fold_idx_2, "parquet")
        file_y_exact = exp.get_lasso_path("y", fold_idx_2, "npy")

        if os.path.exists(model_path_fold) and os.path.getsize(model_path_fold) > 0:
            print(f"--> Modèle déjà entraîné trouvé à : {model_path_fold} (Passage au fold suivant)")
            continue

        print(f"\n[DEBUG TRAIN - Fold {fold_idx_2 + 1}] Shape de X_train_fold_2: {X_train_fold_2.shape}")

        X_train_final_fold, y_train_final = X_train_fold_2, y_train_fold_2
        groups_final_fold = groups_fold_2
        X_calib, y_calib = None, None

        if calibration.value:
            if config_balance.balance_method != "":
                print(f"    [INFO] Calibration Prior (analytique). Utilisation de 100% du fold pour l'entraînement...")

                # Le modèle de base utilise TOUT le fold
                X_train_final_fold = X_train_fold_2
                y_train_final = y_train_fold_2
                groups_final_fold = groups_fold_2

                # Pas besoin de jeu held-out pour une formule mathématique
                X_calib = None
                y_calib = None
            else:
                print(f"    [INFO] Calibration activée ({calibration_mode.value}) Séparation du fold via StratifiedGroupKFold...")

                # Initialisation du splitter interne (identique pour ML et DL)
                skf_calib = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

                # Le split fonctionne nativement sur le 3D comme sur le 2D
                train_idx_calib, calib_idx = next(skf_calib.split(X_train_fold_2, y_train_fold_2, groups=groups_fold_2))

                # Sous-jeu pour l'entraînement du modèle de base
                X_train_final_fold = X_train_fold_2[train_idx_calib]
                y_train_final = y_train_fold_2[train_idx_calib]
                groups_final_fold = np.asarray(groups_fold_2)[train_idx_calib]

                # Sous-jeu "held-out" pour la calibration
                X_calib = X_train_fold_2[calib_idx]
                y_calib = y_train_fold_2[calib_idx]

                print(f"    [DEBUG] Shapes - Train Base: {X_train_final_fold.shape}, Calib: {X_calib.shape}")

        print(f"    [LEARNING CURVE] Lancement de la boucle de paliers unifiée...")

        X_test_fold_lc = folds_X_test[fold_idx_2]
        X_val_np = X_test_fold_lc.to_numpy() if hasattr(X_test_fold_lc, "to_numpy") else np.asarray(X_test_fold_lc)
        y_val_np = np.asarray(folds_y_test[fold_idx_2])

        # ─── BOUCLE UNIQUE SUR TOUS LES PALIERS ───────────────────────────────────
        for p_idx, p in enumerate(paliers_lc):
            is_final_palier = (p == 1.0)

            # Extraction propre du chunk via la fonction isolée (Renvoie tes tableaux numpy d'origine)
            X_chunk_np, y_chunk_np, mask_chunk = training.get_learning_curve_chunk(
                X_train_final_fold, y_train_final, groups_final_fold, p, is_dl_model, seed
            )

            if fold_idx_2 == 0:
                lc_sample_sizes.append(len(X_chunk_np))

            print(f"      -> Palier {int(p*100)}% ({len(X_chunk_np)} patients)" + (" [ENTRAÎNEMENT FINAL & DISK-SAVE]" if is_final_palier else " [ÉPHÉMÈRE]"))

            if len(np.unique(y_chunk_np)) < 2:
                print(f"          [WARNING] Une seule classe présente, saut de ce palier.")
                lc_train_scores[fold_idx_2, p_idx] = np.nan
                lc_val_scores[fold_idx_2, p_idx] = np.nan
                continue

            current_save_path = model_path_fold if is_final_palier else None

            # Préparation des arguments pour le cas spécifique de la Régression Logistique Lasso
            lasso_args = {
                'X_raw': X_train_final_fold, 
                'y_raw': y_train_final, 
                'groups_mask': groups_final_fold[mask_chunk] if mask_chunk is not None else None, 
                'file_X': file_X_exact, 
                'file_y': file_y_exact
            } if config_models.models_name == "Logistic Regression Lasso TSFEL" else None

            FC_UNITS_MAP = {
                "none": None,
                "64": (64,),
                "128": (128,),
                "256": (256,),
                "128_64": (128, 64),
                "256_128": (256, 128),
            }

            #On convertit la chaîne "fc_units" en sa vraie valeur attendue par PyTorch
            if "fc_units" in parameters and isinstance(parameters["fc_units"], str):
                parameters["fc_units"] = FC_UNITS_MAP[parameters["fc_units"]]

            # Entraînement délégué à la fonction Usine (Factory)
            final_model_to_save, train_auc, val_auc = training.fit_model_by_name(
                model_name=config_models.models_name,
                X_train=X_chunk_np,
                y_train=y_chunk_np,
                X_val=X_val_np,
                y_val=y_val_np,
                seed=seed,
                is_final_palier=is_final_palier,
                save_path=current_save_path,
                lasso_args=lasso_args,
                **parameters
            )

            # Stockage dans tes matrices d'origine
            lc_train_scores[fold_idx_2, p_idx] = train_auc
            lc_val_scores[fold_idx_2, p_idx] = val_auc
            # Sauvegarde et calibration au dernier palier
            if is_final_palier and not is_dl_model and final_model_to_save is not None:
                if calibration.value:
                    if config_balance.balance_method != "":
                        statsX = folds_sampling_stats[fold_idx_2]
                        top = statsX["n_malades_avant"] * statsX["n_sains_apres"]
                        bottom = statsX["n_sains_avant"] * statsX["n_malades_apres"]
                        if bottom > 0 and top > 0:
                            beta = top / bottom
                            final_model_to_save = training.apply_prior_calibration(final_model_to_save, beta)
                            print(f"    [DEBUG] Prior Calibration OK (beta = {beta:.4f})")
                        else:
                            print(f"    [WARNING] Impossible de calculer beta, classe manquante.")
                    else:
                        print(f"    [INFO] Application de la calibration {calibration_mode.value} sur le jeu held-out...")
                        final_model_to_save = training.apply_model_calibration(
                            final_model_to_save, X_calib, y_calib, calibration_mode.value, seed
                        )

                joblib.dump(final_model_to_save, model_path_fold)
                print(f"--> Modèle final enregistré à : {model_path_fold}")

        folds_X_fit_exact.append(X_train_final_fold)
        folds_y_fit_exact.append(y_train_final)

    print("Cross Validation terminée ! 5 modèles ont été enregistrés avec succès.")
    return lc_sample_sizes, lc_train_scores, lc_val_scores


@app.cell
def _(
    config_models,
    config_transparent,
    output_dir,
    saps2_pred,
    saps2_true,
    save_figure,
    sfu,
    type_donnees,
):
    if type_donnees.value == "score":
        auc_final_score, fpr_score, tpr_score, th_score, brier_score_score, best_f1_score, best_t_score, y_pred_score, mcc_score, non_overlap_area_score, asymetric_incertitude_score, mean_risk_diff_score, mean_p1_score = sfu.plot_all_figs(saps2_pred, saps2_true, config_models, False, save_figure.value, output_dir, config_transparent)
    return (
        asymetric_incertitude_score,
        auc_final_score,
        best_f1_score,
        brier_score_score,
        mcc_score,
        mean_p1_score,
        mean_risk_diff_score,
        non_overlap_area_score,
        y_pred_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation des résultats obtenus
    """)
    return


@app.cell
def _(config_models, config_optuna, exp):
    output_dir = exp.get_output_path(config_models.models_name, config_optuna = config_optuna)
    print("Dossier de sortie prêt :", output_dir)
    return (output_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistiques générales
    """)
    return


@app.cell
def _(
    config_models,
    config_transparent,
    lc_sample_sizes,
    lc_train_scores,
    lc_val_scores,
    output_dir,
    save_figure,
    sfu,
):
    sfu.plot_collected_learning_curve(lc_sample_sizes, lc_train_scores, lc_val_scores, config_models.models_name, savefig = save_figure.value, folder = output_dir, transparent = config_transparent)
    return


@app.cell
def _(
    calibration,
    calibration_mode,
    classification_report,
    config_models,
    config_optuna,
    config_transparent,
    evaluate,
    exp,
    extension,
    folds_X_test,
    folds_X_train,
    folds_y_test,
    folds_y_train,
    np,
    output_dir,
    save_figure,
    sfu,
):
    # Initialisation des listes de scores
    all_test_scores, all_train_scores = [], []
    all_auc_scores, all_brier_scores = [], []
    all_y_true_report, all_y_pred_report = [], []
    all_y_test_global, all_probas_uncalib, all_probas_calib = [], [], []

    # Boucle d'évaluation sur les 5 folds
    for fold_idx in range(5):
        print(f"\n─────────────────── Évaluation du Fold {fold_idx + 1}/5 ───────────────────")

        X_train, X_test = folds_X_train[fold_idx], folds_X_test[fold_idx]
        y_train, y_test = folds_y_train[fold_idx], folds_y_test[fold_idx]

        loaded_model = exp.get_model_path(config_models.models_name, fold_idx, extension, config_optuna = config_optuna)
        print("Modèle chargé :", loaded_model)

        # Dispatcher vers le bon pipeline de calcul
        if config_models.models_name == "InceptionTimeModified":
            res = evaluate.evaluate_inception_fold(X_test, y_test, loaded_model)
            all_auc_scores.append(res["auc"])
            all_brier_scores.append(res["brier"])

        elif config_models.models_name == "LstmTimeModified":
            res = evaluate.evaluate_lstm_fold(fold_idx, X_test, y_test, loaded_model)
            all_auc_scores.append(res["auc"])
            all_brier_scores.append(res["brier"])

        elif config_models.extraction_type == "TSFEL":
            res = evaluate.evaluate_tsfel_fold(fold_idx, X_train, X_test, y_train, y_test, loaded_model, calibration.value)
            all_test_scores.append(res["test_score"])
            all_train_scores.append(res["train_score"])
            all_y_true_report.extend(res["y_test"])
            all_y_pred_report.extend(res["y_pred_test"])
        else:
            raise ValueError(f"Modèle ou type d'extraction non pris en compte : {config_models.models_name}")

        # Données communes collectées par tous les modèles
        all_y_test_global.extend(res["y_test"])
        all_probas_uncalib.extend(res["probas_uncalib"])
        all_probas_calib.extend(res["probas_calib"])


    # ──────────────────────────────────────────────────────────────────────────────
    # BILAN GLOBAL DE LA CROSS-VALIDATION
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*20 + " BILAN GLOBAL DE LA CROSS-VALIDATION " + "="*20)

    all_y_test_global = np.array(all_y_test_global)
    all_probas_uncalib = np.array(all_probas_uncalib)
    all_probas_calib = np.array(all_probas_calib)

    if config_models.extraction_type == "TSFEL":
        mean_acc = np.mean(all_test_scores)
        std_acc = np.std(all_test_scores)
        print(f"Score moyen (Accuracy) : {mean_acc:.4f} (± {std_acc:.4f})")
        print("\nRapport de classification cumulé (sur l'ensemble des 5 folds) :")
        print(classification_report(all_y_true_report, all_y_pred_report, target_names=["Alive", "Deceased"], zero_division=0))
    else:
        mean_auc, std_auc = np.mean(all_auc_scores), np.std(all_auc_scores)
        mean_brier, std_brier = np.mean(all_brier_scores), np.std(all_brier_scores)
        print(f"AUC moyenne   : {mean_auc:.4f} (± {std_auc:.4f})")
        print(f"Brier moyenne : {mean_brier:.4f} (± {std_brier:.4f})")
    print("="*79)

    print("\nGénération de la courbe de calibration poolée...")
    print(f"DEBUG SIZES -> y_true: {len(all_y_test_global)}, uncalib: {len(all_probas_uncalib)}, calib: {len(all_probas_calib)}")

    sfu.calibration_curve_homemade(
        all_probas_uncalib, all_probas_calib, all_y_test_global, 
        config_models.models_name, config_models.extraction_type, 
        calibration.value, save_figure.value, output_dir, 
        config_transparent, calibration_mode.value
    )

    sfu.calibration_curve_homemade(
        all_probas_uncalib, all_probas_calib, all_y_test_global, 
        config_models.models_name, config_models.extraction_type, 
        calibration.value, save_figure.value, output_dir, 
        config_transparent, calibration_mode.value
    )

    sfu.calibration_curve_advanced(
        all_probas_uncalib, all_probas_calib, all_y_test_global, 
        config_models.models_name, config_models.extraction_type, 
        calibration.value, save_figure.value, output_dir, 
        config_transparent, calibration_mode.value
    )

    probas = all_probas_calib
    y_test = all_y_test_global
    return probas, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Si on est en mode Logistic Regression Lasso TSFEL, on montre le logistic_regression_path
    """)
    return


@app.cell
def _(mo, mo_utils, run_lasso):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run_lasso,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(
    LogisticRegression,
    Parallel,
    Path,
    config_models,
    config_transparent,
    delayed,
    exp,
    extension,
    joblib,
    mo,
    np,
    output_dir,
    pl,
    plt,
    run_lasso,
    save_figure,
    seed,
):
    mo.stop(not run_lasso.value, "Clique pour lancer")
    print("Lasso Path lancé")
    if config_models.models_name == "Logistic Regression Lasso TSFEL":
        print("Génération ultra-rapide des Lasso Paths (Warm Start + Multi-processing)...")

        Cs_grid = np.logspace(-4, 4, 100) 

        # Fonction isolée pour traiter un fold (permet la parallélisation)
        def process_single_fold(idx):
            # Chargement des données
            file_X = exp.get_lasso_path("X", idx, "parquet")
            file_y = exp.get_lasso_path("y", idx, "npy")
            X_pure_fit_np = pl.read_parquet(file_X).to_numpy()
            y_pure_fit = np.load(file_y)

            # Chargement du C optimal
            model_path = exp.get_model_path("Logistic Regression Lasso TSFEL", idx, extension)
            model_L1 = joblib.load(model_path)
            best_C2 = model_L1.C

            lr_path_model = LogisticRegression(
                l1_ratio=1.0,
                solver='saga',
                max_iter=100001,
                random_state=seed,
                warm_start=True
            )

            coefs_list = []
            # Crucial : On parcourt la grille du plus petit C (gros Lasso) au plus grand C (Lasso faible)
            # C'est dans ce sens que le warm start est le plus efficace géométriquement
            sorted_Cs = np.sort(Cs_grid) 

            for c_val in sorted_Cs:
                lr_path_model.set_params(C=c_val)
                lr_path_model.fit(X_pure_fit_np, y_pure_fit)
                coefs_list.append(lr_path_model.coef_[0].copy())

            return sorted_Cs, np.array(coefs_list), best_C2, X_pure_fit_np.shape[1]

        # Lancement des 5 folds en parallèle
        results = Parallel(n_jobs=-1)(delayed(process_single_fold)(f_idx) for f_idx in range(5))

        # Récupération et affichage des graphiques (ultra rapide car les calculs sont déjà faits)
        for fold_idx_L1, (sorted_Cs, coefs_path, best_C2, n_features) in enumerate(results):
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_Cs, coefs_path, alpha=0.7)
            plt.axvline(x=best_C2, color='black', linestyle='--', linewidth=2, 
                        label=f'C optimal (Fold {fold_idx_L1 + 1}) = {best_C2:.4f}')
            plt.xscale('log')
            plt.xlabel('Paramètre de régularisation C (Log Scale)')
            plt.ylabel(f'Coefficients ({n_features} features)')
            plt.title(f'L1 Regularization Path - Fold {fold_idx_L1 + 1}\nOptimisé (Warm Start)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()

            if save_figure.value:
                filename = f"L1_Log_path_fold_{fold_idx_L1 + 1}.png"
                plt.savefig(output_dir / Path(filename), dpi=300, bbox_inches="tight", transparent=config_transparent)
            plt.show()
    return


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Courbe ROC du modèle {config_models.models_name}
    """)
    return


@app.cell
def _(
    config_models,
    config_transparent,
    output_dir,
    probas,
    save_figure,
    sfu,
    y_test,
):
    auc_final, fpr, tpr, thresholds_roc = sfu.roc_curve_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)
    return (auc_final,)


@app.cell
def _(
    config_models,
    config_transparent,
    output_dir,
    probas,
    save_figure,
    sfu,
    y_test,
):
    auprc_final, precision, recall, thresholds = sfu.prc_curve_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)
    return


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Distribution des scores (KDE) de {config_models.models_name}
    """)
    return


@app.cell
def _(
    config_models,
    config_transparent,
    output_dir,
    probas,
    save_figure,
    sfu,
    y_test,
):
    non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1 = sfu.kde_plot_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)
    return asymetric_incertitude, mean_p1, mean_risk_diff, non_overlap_area


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Choix du threshold en fonction du F1 score pour {config_models.models_name}
    """)
    return


@app.cell
def _(
    config_models,
    config_transparent,
    output_dir,
    probas,
    save_figure,
    sfu,
    y_test,
):
    best_f1, best_t = sfu.f1_score_evolution(probas, y_test, config_models.models_name, save_figure.value, output_dir, config_transparent)
    return best_f1, best_t


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Matrice de confusion de {config_models.models_name}
    """)
    return


@app.cell
def _(
    best_t,
    config_models,
    config_transparent,
    output_dir,
    probas,
    save_figure,
    sfu,
    y_test,
):
    y_pred, mcc = sfu.confusion_matrix_homemade(probas, y_test, best_t, config_models.models_name, save_figure.value, output_dir, config_transparent)
    return mcc, y_pred


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Evolution du score de Brier de {config_models.models_name}
    """)
    return


@app.cell
def _(config_transparent, output_dir, probas, save_figure, sfu, y_test):
    global_brier = sfu.brier_evolution(probas, y_test, save_figure.value, output_dir, transparent=config_transparent, )
    return (global_brier,)


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Enregistrement des scores obtenus de {config_models.models_name}
    """)
    return


@app.cell
def _(
    asymetric_incertitude,
    asymetric_incertitude_score,
    auc_final,
    auc_final_score,
    best_f1,
    best_f1_score,
    brier_score_score,
    global_brier,
    joblib,
    mcc,
    mcc_score,
    mean_p1,
    mean_p1_score,
    mean_risk_diff,
    mean_risk_diff_score,
    non_overlap_area,
    non_overlap_area_score,
    output_dir,
    probas,
    saps2_pred,
    saps2_true,
    type_donnees,
    y_pred,
    y_pred_score,
    y_test,
):
    # Résumé des scores obtenus
    if type_donnees.value == "modèle":
        all_res = {
                'y_true' : y_test,
                'probas': probas,
                'preds': y_pred,       
                'f1_score': best_f1, 
                'mcc': mcc,
                'auc': auc_final,
                'brier': global_brier,
            "non_overlap_area" : non_overlap_area, 
            "asymetric_incertitude" : asymetric_incertitude,
            "mean_risk_diff" : mean_risk_diff,
            "mean_deaths_prediction" : mean_p1
        }
    else:
        all_res = {
                'y_true' : saps2_true,
                'probas': saps2_pred,
                'preds': y_pred_score,       
                'f1_score': best_f1_score, 
                'mcc': mcc_score,
                'auc': auc_final_score,
                'brier': brier_score_score,
            "non_overlap_area" : non_overlap_area_score, 
            "asymetric_incertitude" : asymetric_incertitude_score,
            "mean_risk_diff" : mean_risk_diff_score,
            "mean_deaths_prediction" : mean_p1_score
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_res, output_dir / "all_res.joblib")

    resu = joblib.load(output_dir / "all_res.joblib")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Mesure d'importance de nos variables TSFEL + statiques
    """)
    return


@app.cell(disabled=True)
def _(
    config_models,
    config_optuna,
    exp,
    joblib,
    np,
    patient_col,
    pl,
    sfu,
    target_col,
    torch,
):
    show_shap = {}
    model_name_shap = config_models.models_name

    for f_idx in range(5):
        output_dir_shap = exp.get_output_path(model_name_shap, config_optuna = config_optuna) / f"fold_{f_idx}"
        output_dir_shap.mkdir(parents = True, exist_ok = True)
        if config_models.extraction_type == "TSFEL":
            X_train_fold_shap = exp.get_tsfel_boruta("train", f_idx)
            keepVariableList_shap = X_train_fold_shap.parent / f"keepVariableList_2_fold_{f_idx}.npy"
            loaded_model_shap = exp.get_model_path(model_name_shap, f_idx, ".joblib",  "", config_optuna = config_optuna)
            clf_shap = joblib.load(loaded_model_shap)
            sfu.mesureImportance_tsfel(clf_shap, pl.read_parquet(X_train_fold_shap).select(pl.exclude(patient_col, target_col)), np.load(keepVariableList_shap), top_n=20, class_labels=["Survivors", "Deaths"], savefig = True, transparent = False, folder = output_dir_shap)

        else:
            loaded_model_shap = exp.get_model_path(model_name_shap, f_idx, ".pt",  "", config_optuna = config_optuna)
            checkpoint = torch.load(loaded_model_shap, weights_only = False)
            if config_models.models_name == "InceptionTimeModified":
                from utilitaries.models.inceptionTimeModified import InceptionModel
                varnames_3d = np.load(exp.get_var_path())
                X_train_3d = np.load(exp.get_time_path(mode="train", fold_idx = f_idx))
                clf_shap = InceptionModel(**checkpoint['init_args'])
                clf_shap.load_state_dict(checkpoint['state_dict'])
                clf_shap.eval()
                print("Le modèle InceptionModel a été instancié et chargé avec succès !")
            sfu.mesureImportance_tsfel(clf_shap, X_train_3d, varnames_3d, top_n=20, class_labels=["Survivors", "Deaths"], savefig = True, transparent = False, folder = output_dir_shap)
        show_shap[f"XGB_{f_idx}"] = output_dir_shap
    return


@app.cell
def _(exp):
    show_shap2 = {}
    model_name_shap_show = "XGBoost TSFEL"

    for f_idx2 in range(5):
        output_dir_shap_show = exp.get_output_path(model_name_shap_show) / f"fold_{f_idx2}"
        output_dir_shap_show.mkdir(parents = True, exist_ok = True)
        show_shap2[f"XGB_{f_idx2}"] = output_dir_shap_show
    return model_name_shap_show, show_shap2


@app.cell(disabled=True)
def _(exp, model_name_shap_show, sfu, show_shap2):
    sfu.compare_models_figure("feature_importance.png",savefig=True, folder = exp.get_output_path(model_name_shap_show), **show_shap2)
    return


@app.cell(disabled=True)
def _(exp, model_name_shap_show, sfu, show_shap2):
    sfu.compare_models_figure("global_feature_importance_mdi.png",savefig=True, folder = exp.get_output_path(model_name_shap_show), **show_shap2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparaison des modèles
    """)
    return


@app.cell
def _(mo, mo_utils, run_test):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run_test,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(mo, run_test):
    mo.stop(not run_test.value, "Clique pour lancer")
    print("Comparaison lancée")
    return


@app.cell
def _():
    return


@app.cell
def _(exp, générer_rapport_comparatif):
    comparaisons = [exp.load_model("InceptionTimeModified"), exp.load_model("LstmTimeModified"), exp.load_model("RandomForest TSFEL", ""), exp.load_model("XGBoost TSFEL"), exp.load_model("SVC TSFEL"), exp.load_model("Logistic Regression Lasso TSFEL", ""),
    exp.load_model("IGS2")]

    comparaisons

    générer_rapport_comparatif(comparaisons, save_dir="Comparaison ALl", table_format='fancy_grid')
    return


@app.cell(disabled=True)
def _(exp, sfu, y_test):
    comparaisons_time = [exp.load_model("InceptionTimeModified"), exp.load_model("LstmTimeModified")]

    sfu.générer_rapport_comparatif(y_test, comparaisons_time, save_dir="Comparaison Time", table_format='fancy_grid')
    return


@app.cell(disabled=True)
def _(exp, sfu, y_test):
    comparaisons_ml = [exp.load_model("RandomForest TSFEL", "_balanced"), exp.load_model("XGBoost TSFEL"), exp.load_model("SVC TSFEL")]
    sfu.générer_rapport_comparatif(y_test, comparaisons_ml, save_dir="Comparaison TSFEL", table_format='fancy_grid')
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("kde_plot.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("calibration_curve.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("roc_curve.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("threshold_evolution.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("confusion_matrix.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("calibPerDec.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    sfu.compare_models_figure("brierPerTrancheRisk.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell(disabled=True)
def _(exp, sfu):
    figure_compare = ["kde_plot", "brier_per_decile", "brier_per_risk_bracket", "calib_per_decile", "calib_per_risk_bracket", "calibration_curve", "confusion_matrix", "learning_curve", "roc_curve", "threshold_evolution"]

    for f in figure_compare:
        sfu.compare_models_figure(f + ".png",
        InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
        LSTMT = exp.get_output_path("LstmTimeModified", ""),
        RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", ""),
        XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
        SVC = exp.get_output_path("SVC TSFEL", ""),
        LR_lasso = exp.get_output_path("Logistic Regression Lasso TSFEL", ""),
        IGS2 = exp.get_output_path("IGS2", ""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sidebar
    """)
    return


@app.cell
def _(
    custom_features,
    keep_feats,
    mo,
    models,
    run,
    run_optuna,
    seed,
    spl,
    str_keep_feats,
    ui_tsfel,
    uid,
):
    sidebar_items = spl.render_sidebar(uid, models, ui_tsfel, keep_feats, str_keep_feats, run, run_optuna, seed, custom_features)
    mo.sidebar(
        mo.vstack(sidebar_items), 
        width="550px"
    )
    return


if __name__ == "__main__":
    app.run()
