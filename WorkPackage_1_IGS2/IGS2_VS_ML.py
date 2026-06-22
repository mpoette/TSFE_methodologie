import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Work Package 1 : Prédiction de la survie à J28 en réanimation : Comparaison avec IGS2
    """)
    return


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
    import seaborn as sns
    import tsfel
    from sklearn.metrics import (
        brier_score_loss,
        classification_report,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

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
        Path,
        StratifiedGroupKFold,
        classification_report,
        create_merged_dataset,
        evaluate_lstm_on_test,
        evaluate_on_test,
        extract,
        extract_feat,
        joblib,
        json,
        load_lstm_from_checkpoint,
        load_model_from_checkpoint,
        mo,
        mo_utils,
        np,
        os,
        path_utils,
        pl,
        plt,
        predict_proba,
        predict_proba_lstm,
        preproc,
        seed,
        sfu,
        sys,
        train_inception_time,
        train_lstm_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Widgets Marimo utilisés dans ce notebook
    """)
    return


@app.cell
def _(mo):
    transparent = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Non",
                                label = "Rendre les figures transparentes")
    return (transparent,)


@app.cell
def _(plt, transparent):
    if not transparent.value:
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams['axes.facecolor'] = "white"
        plt.rcParams['savefig.facecolor'] = "white"
    return


@app.cell
def _(mo, mo_utils):
    cleaning = mo.ui.dropdown(
        options=mo_utils.CLEAN,
        value = "Enlever Surveillance Continue",
        label = "Nettoyage des patients en SC",
    )
    return (cleaning,)


@app.cell
def _(mo, mo_utils):
    mode = mo.ui.dropdown(
        options=mo_utils.MODES,
        value="24h début réanimation sans remplissage",
        label="Mode de fenêtrage",
    )
    return (mode,)


@app.cell
def _(mo, mo_utils):
    models = mo.ui.dropdown(
        options=mo_utils.MODELS,
        value="InceptionTimeModified",
        label="Modèle utilisé",
    )
    return (models,)


@app.cell
def _(mo, mo_utils):
    modex = mo.ui.dropdown(
        options = list(mo_utils.FEAT.keys()),
        value = "Mode IGS2",
        label = "Choix des features gardées",
    )
    return (modex,)


@app.cell
def _(mo, mo_utils):
    balance = mo.ui.dropdown(
        options = mo_utils.BALANCE,
        value = "Aucune Méthode",
        label = "Méthode pour équilibrer les charges"
    )
    return (balance,)


@app.cell
def _(mo, mo_utils):
    y_dd =  mo.ui.dropdown(
        options=mo_utils.Y,
        value="Survie à 28 jours",
        label="Cible (y) à prédire",
    )
    return (y_dd,)


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
    save_figure = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Oui",
                                label = "Sauvegarder les figures")
    return (save_figure,)


@app.cell
def _(config_models, mo):
    if config_models.extraction_type == "TSFEL" and config_models.models_type != "calibrated":
        bool_calib_str = "Oui"
    else:
        bool_calib_str = "Non"
    calibration = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = bool_calib_str,
                                label = "Activer la calibration du modèle")
    return (calibration,)


@app.cell
def _(mo):
    boruta_filter = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Oui",
                                label = "Filtre Boruta")
    return (boruta_filter,)


@app.cell
def _(mo, mo_utils):
    keep_pop = mo.ui.dropdown(
        options = mo_utils.POPULATION,
        value = "Tout",
        label = "Type de patients que l'on veut garder (ICU_DP filter)")
    return (keep_pop,)


@app.cell
def _(boruta_filter, calibration, config_models, mo):
    if config_models.extraction_type == "TSFEL" :
        extract_tsfel = mo.ui.dropdown(
            options = {"Oui" : True, "Non" : False},
            value = "Non",
            label = "Extraire les données TSFEL"
        )
        class_weight_choice = mo.ui.dropdown(options = {"balanced" : "_balanced", "balanced_subsample" : "_balanced_subsample"},
                                value = "balanced",
                                label = "class_weight")

        ui_tsfel = mo.vstack([extract_tsfel, boruta_filter, calibration, class_weight_choice])
        if config_models.models_type == "calibrated":
            ui_tsfel = mo.vstack([extract_tsfel, boruta_filter, class_weight_choice])
    else :
        class_weight_choice = mo.ui.dropdown(options = {"" : ""},
                                value = "")
        extract_tsfel = None
        ui_tsfel = mo.md("")
    return class_weight_choice, extract_tsfel, ui_tsfel


@app.cell
def _(mode):
    config_mode = mode.value
    return (config_mode,)


@app.cell
def _(models):
    config_models = models.value
    return (config_models,)


@app.cell
def _(cleaning):
    config_cleaning = cleaning.value
    return (config_cleaning,)


@app.cell
def _(y_dd):
    config_y = y_dd.value
    return (config_y,)


@app.cell
def _(keep_pop):
    config_keep_pop = keep_pop.value
    return (config_keep_pop,)


@app.cell
def _(balance):
    config_balance = balance.value
    return (config_balance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration dynamique des chemins utilisés
    """)
    return


@app.cell
def _(
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
    exp = path_utils.Experiment(config_mode, config_cleaning, config_y, str_balance_method, modex, class_weight_choice, str_pop, seed)
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
def _(cleaning, keep_pop, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        keep_pop,
        cleaning,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(df_merged_1, extract, pl):
    df_clean = extract.prepare_data(df_merged_1, hour_offset = 0, random = False, max_hour = 6, strict_mode = True, target_col = "isDeceased_lt_28d", show_fig = True)


    # TODO : feature engineering à mettre à la bonne place
    df_clean = df_clean.with_columns(
        (pl.col("is_ventilated").fill_null(pl.lit(False))).alias("is_ventilated"),
        (pl.col("is_prone").fill_null(pl.lit(False))).alias("is_prone"),
        (pl.col("is_conscious").fill_null(pl.lit(False))).alias("is_conscious")
    )
    df_clean = df_clean.filter(pl.col("taille").is_not_null() & pl.col("poids_admission").is_not_null())
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO : vérifier le nombre de données qu'on filtre ici !
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtrage du dataset/Prétraitement
    """)
    return


@app.cell
def _(df_clean, pl):
    feat = "score_glasgow"

    # On compte le nombre de NULL par patient
    df_check_nulls = (
        df_clean
        .group_by("encounterId")
        .agg(
            # .is_null().sum() compte le nombre de True (donc le nombre de nulls)
            pl.col(feat).is_null().sum().alias("nb_nulls_en_24h")
        )
        # Optionnel : on ne garde que les patients qui ont AU MOINS un null
        .filter(pl.col("nb_nulls_en_24h") > 0)
    )

    df_check_nulls["nb_nulls_en_24h"].value_counts()
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
    dustbin = ["endotracheal_tube","tracheo", "installation", "eer", "hx_comorbidité_majeure", "imc", "neuro_status", "ecmo_all", "prone"]
    useless = ["arret_therapeutique", "limitation_therapeutique", "hematocrit", "peak_pressure"]
    icu_useless = ["icu_DP", "icu_actes", "icu_mode_entree", "icu_mode_sortie", "icu_DA", "hosp_admissionMode", "hosp_primaryDiagnosis", "hosp_primaryDiagnosisCode"]
    icu_useful = ["icu_ghm", "icu_DP_code"]
    broken = ["urine_rate"]
    cheat = ["category", "heure_entiere", "encounterId", "delta_hour", "heure_calibree", "heure_entiere", "year_inTime", target_col, "deces_datediff_days", "isDeceased", "adm_unit", "out_unit", "transition_units", "los", "adm_year", "hosp_los", "hosp_dischargeMode", "deces_hosp"]
    df_clean_keep = df_clean.drop([*score, *dustbin, *useless, *cheat, *icu_useless, *broken])
    df_clean_keep = df_clean_keep.select([pl.col(c) for c in df_clean_keep.columns if not c.endswith("_detected_term")])
    return (df_clean_keep,)


@app.cell
def _(df_clean_keep, mo):
    custom_features = mo.ui.multiselect(
        options=df_clean_keep.columns,
        value= df_clean_keep.columns,
        label="(features sélectionnables)",
    )
    return (custom_features,)


@app.cell
def _(custom_features, df_clean_keep, json, mo, mo_utils, modex):
    with open ("../../Preprocessing_pipeline/preprocessing-pipelines/json/dynamic_features.json", "r") as file:
        json_feat = json.load(file)

    import polars.selectors as cs

    if modex.value == "Mode All":
        keep_feats = df_clean_keep.columns

    elif modex.value == "Mode All Without pmsi":
        # On sélectionne TOUT, SAUF ce qui commence par "hx_" OU "icu_"
        keep_feats = df_clean_keep.select(
            ~cs.starts_with("hx_") & ~cs.starts_with("icu_")
        ).columns

    elif modex.value == "Mode Commonly Used Without pmsi":
        keep_feats = df_clean_keep.select(
            ["score_glasgow", "heart_rate", "creat", "pao2", "is_ventilated", "fio2_corr", "age", "temp", "diurese" ]
        )
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
    return cs, keep_feats, str_keep_feats


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
def _(cs, df_clean_2, keep_features, modex, pl, target_col):
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
    print(final_features)
    X_init = df_clean_3.select(final_features).to_numpy()
    y_init = df_clean_3[target_col].to_numpy()
    return X_init, df_clean_3, final_features, y_init


@app.cell
def _(balance, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        balance,
        mo.md(mo_utils.config_end)])
    return


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

    elif config_models.extraction_type == "time":
        train_init_df = df_clean_3[train_init_idx].sort([patient_col, time_col])
        test_holdout_df = df_clean_3[test_init_idx].sort([patient_col, time_col])

        X = train_init_df
        y = train_init_df[target_col].to_numpy()
        groups = train_init_df[patient_col].to_numpy()
    return X, groups, train_init_df, train_init_tsfel, y


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
    boruta_filter,
    config_balance,
    config_models,
    df_clean_3,
    exp,
    expected_length,
    extract,
    extract_feat,
    final_features,
    groups,
    np,
    os,
    patient_col,
    pl,
    preproc,
    seed,
    target_col,
    time_col,
    train_init_df,
    train_init_tsfel,
    y,
):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # On stocke les données de tous les folds
    folds_X_train = []
    folds_X_test = []
    folds_y_train = []
    folds_y_test = []
    folds_groups = []

    for fold_idx, (train_idx, test_idx) in enumerate (sgkf.split(X=X, y=y, groups = groups)):
        print(f"\n─────────────────── Traitement du Fold {fold_idx + 1}/5 ───────────────────")

        # Récupération des IDs patients correspondants au split de ce fold
        train_patients = df_clean_3[train_idx].select(patient_col).unique()
        test_patients = df_clean_3[test_idx].select(patient_col).unique()
        if config_models.extraction_type == "TSFEL" :
            # On filtre notre gros DataFrame train pré-calculé pour ce fold
            train_fold_tsfel = train_init_tsfel.join(train_patients, on=patient_col, how="inner").sort(patient_col)
            test_fold_tsfel = train_init_tsfel.join(test_patients, on=patient_col, how="inner").sort(patient_col)
            # Filtrage corrélation/variance
            train_clean, test_clean, keepVariableList_1 = extract_feat.filtrage_corr_var(train_fold_tsfel, test_fold_tsfel, patient_col, target_col)
            if boruta_filter.value:
                # Construction du nom de fichier unique intégrant le Fold et la Graine (seed)
                filename_train_boruta = exp.get_tsfel_boruta("train", fold_idx)
                filename_test_boruta = exp.get_tsfel_boruta("test", fold_idx)

                if os.path.exists(filename_train_boruta) and os.path.exists(filename_test_boruta):
                    print(f"Lecture des fichiers Boruta existants pour le fold {fold_idx} (Graine {seed}).")
                    train_clean = pl.read_parquet(filename_train_boruta)
                    test_clean = pl.read_parquet(filename_test_boruta)
                else:
                    train_clean, test_clean, keepVariableList_2 = extract_feat.filtrage_boruta(train_clean, test_clean, patient_col, target_col, max_iter = 100, seed = seed)

                    train_clean.write_parquet(filename_train_boruta)
                    test_clean.write_parquet(filename_test_boruta)
                    print(f"Save de Boruta pour le fold {fold_idx} (Graine {seed}).")

                    parent_folder2 = filename_train_boruta.parent
                    np.save(parent_folder2 / f"keepVariableList_1_fold_{fold_idx}.npy", keepVariableList_1)

                    np.save(parent_folder2 / f"keepVariableList_2_fold_{fold_idx}.npy", keepVariableList_2)
            # Tri au cas-où
            train_clean = train_clean.sort(patient_col)
            test_clean = test_clean.sort(patient_col)

            # Equilibrage
            train_clean = preproc.equilibrer_dataset_tabulaire(train_clean, extract.ID_COL, target_col, method = config_balance.balance_method, seed = seed)

            # Test d'intégrité
            assert train_clean.height == train_clean[extract.ID_COL].n_unique(), f"Erreur d'alignement Train TSFEL Fold {fold_idx}"
            assert test_clean.height == test_clean[extract.ID_COL].n_unique(), f"Erreur d'alignement Test TSFEL Fold {fold_idx}"

            y_train_fold = train_clean[target_col].to_numpy()
            y_test_fold = test_clean[target_col].to_numpy()
            groups_fold = train_clean[patient_col].to_numpy()
            train_clean = train_clean.select(pl.exclude(patient_col, target_col))
            test_clean = test_clean.select(pl.exclude(patient_col, target_col))

            folds_groups.append(groups_fold)

            # Scaling final
            X_train_fold, X_test_fold = preproc.scaling(train_clean, test_clean)

        elif config_models.extraction_type == "time" :
            train_df = train_init_df[train_idx].sort([patient_col, time_col])
            test_df = train_init_df[test_idx].sort([patient_col, time_col])

            # Gestion exclusive de l'équilibrage homemade (avec polars)
            if config_balance.balance_method in ["downsampling_homemade", ""]:
                train_df = preproc.equilibrer_dataset_tabulaire(train_df, extract.ID_COL, target_col, method = config_balance.balance_method, seed = seed)

            # On prépare le jeu d'entraînement

            # Scaling
            (train_df, test_df) = preproc.scaling(train_df, test_df)

            # Transformation en 3D Array
            (X_train_fold, y_train_fold) = preproc.build_sequences(train_df, patient_col, target_col, expected_length, final_features)  # grouper en fonction d'un individu
            (X_test_fold, y_test_fold) = preproc.build_sequences(test_df, patient_col, target_col, expected_length, final_features)

            # Gestion de l'équilibrage avec imblearn (avec un 3D Array directement)
            if config_balance.balance_method not in ["downsampling_homemade", ""]:
                # On applatit le 3D Array en 2D Array
                n_samples, n_timesteps, n_feats = X_train_fold.shape
                X_train_fold_2d = X_train_fold.reshape(n_samples, n_timesteps * n_feats)

                # On applique la méthode d'équilibrage imblearn
                if config_balance.balance_method == "downsampling_50-50":
                    from imblearn.under_sampling import RandomUnderSampler
                    rs = RandomUnderSampler(random_state=seed)
                elif config_balance.balance_method == "upsampling_50-50":
                    from imblearn.over_sampling import RandomOverSampler
                    rs = RandomOverSampler(random_state=seed)
                else:
                    raise ValueError("Cet équilibrage n'a pas encore été implémenté")

                X_res_2d, y_train_fold = rs.fit_resample(X_train_fold_2d, y_train_fold)

                # On redonne sa forme 3D d'origine au tenseur équilibré
                X_train_fold = X_res_2d.reshape(-1, n_timesteps, n_feats)

            # TODO : rajouter une gestion des NaN (appel à fonction de utils.py)

            # TODO : externaliser la gestion des NaN
            # On enlève les NaN après extraction de features
            total_nan = np.isnan(X_train_fold).sum()
            # Compte les NaN pour chaque feature
            nan_par_feature = np.isnan(X_train_fold).sum(axis=(0, 1))
            # for i, feat_name in enumerate(final_features):
                # print(f"Feature '{feat_name}' : {nan_par_feature[i]} NaN")
            print(f"Nombre total de valeurs NaN : {total_nan}")

            X_train_fold = np.nan_to_num(X_train_fold, nan=0.0)
            X_test_fold = np.nan_to_num(X_test_fold, nan = 0.0)
        else:
            raise ValueError("Modèle inexistant/Pas implémenté")
        # On accumule les données nettoyées du fold en cours
        folds_X_train.append(X_train_fold)
        folds_X_test.append(X_test_fold)
        folds_y_train.append(y_train_fold)
        folds_y_test.append(y_test_fold)
    print("Les 5 folds ont été calculé avec succès !")
    return (
        folds_X_test,
        folds_X_train,
        folds_groups,
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
def _(mo, mo_utils, run):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(config_balance, config_keep_pop, config_models):
    # on créé un nom unique de modèle
    str_pop = ""
    if config_keep_pop.keep_population != "all_diseases":
        str_pop = "_"+config_keep_pop.keep_population

    str_balance_method = ""
    if config_balance.balance_method:
        str_balance_method = config_balance.balance_method + "_"
    extension = ".joblib" if config_models.extraction_type == "TSFEL" else ".pt"

    default_params = {
            "epochs" : 100,
            "patience" : 10,
        }

    parameters = default_params
    return extension, parameters, str_balance_method, str_pop


@app.cell
def _(
    CalibratedClassifierCV,
    LogisticRegression,
    StratifiedGroupKFold,
    calibration,
    class_weight_choice,
    config_models,
    exp,
    extension,
    folds_X_train,
    folds_groups,
    folds_y_train,
    joblib,
    mo,
    np,
    os,
    parameters,
    run,
    seed,
    train_inception_time,
    train_lstm_model,
):
    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")

    if len(folds_X_train) == 0:
        raise ValueError("Les listes de folds sont vides")

    # --- BARRIÈRE DE SÉCURITÉ GÉOMÉTRIQUE ---
    is_dl_model = config_models.models_name in ["InceptionTimeModified", "LstmTimeModified"]
    n_dims = len(folds_X_train[0].shape) if hasattr(folds_X_train[0], "shape") else 0
    if is_dl_model and n_dims != 3:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice 3D [patients, temps, features], mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'time' ?")
    elif not is_dl_model and n_dims != 2:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice tabulaire 2D, mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'TSFEL' ?")

    folds_X_fit_exact = []
    folds_y_fit_exact = []
    for fold_idx_2 in range(5):

        print(f"\n─────────────────── Entraînement du Fold {fold_idx_2 + 1}/5 ───────────────────")
        # Extraction des données spécifiques à ce fold
        X_train_fold_2 = folds_X_train[fold_idx_2]
        y_train_fold_2 = folds_y_train[fold_idx_2]
        groups_fold_2 = folds_groups[fold_idx_2]

        # Génération d'un chemin STRICT et DÉTERMINISTE unique par fold et par graine
        model_path_fold = exp.get_model_path(config_models.models_name, fold_idx_2, extension)

        file_X_exact = exp.get_lasso_path("X", fold_idx_2, "parquet")
        file_y_exact = exp.get_lasso_path("y", fold_idx_2, "npy")

        if os.path.exists(model_path_fold) and os.path.getsize(model_path_fold) > 0:
            print(f"--> Modèle déjà entraîné trouvé à : {model_path_fold} (Passage au fold suivant)")
            continue  # On passe directement au fold suivant sans réentraîner
        print(f"\n[DEBUG TRAIN - Fold {fold_idx_2 + 1}] Shape de X_train_fold_2: {X_train_fold_2.shape}")
        X_train_final_fold, y_train_final = X_train_fold_2, y_train_fold_2
        X_calib, y_calib = None, None

        if calibration.value:
            print(f"    [INFO] Calibration activée. Séparation du fold en sous-jeux d'entraînement et de calibration (respect des groupes)...")
            skf_calib = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            train_idx_calib, calib_idx = next(skf_calib.split(X_train_fold_2, y_train_fold_2, groups=groups_fold_2))

            # Sous-jeu pour l'entraînement du modèle de base
            X_train_final_fold = X_train_fold_2[train_idx_calib]
            y_train_final = y_train_fold_2[train_idx_calib]

            # Sous-jeu "held-out" pour la calibration
            X_calib = X_train_fold_2[calib_idx]
            y_calib = y_train_fold_2[calib_idx]
            print(f"    [DEBUG] Shapes - Train Base: {X_train_final_fold.shape}, Calib: {X_calib.shape}")

        final_model_to_save = None

        if config_models.models_name == "InceptionTimeModified":
            print(parameters)
            model, T, history, splits = train_inception_time(
                X_train_final_fold, y_train_final,
                save_best_path=model_path_fold,
                seed = seed,
                **parameters
                )

        elif config_models.models_name == "LstmTimeModified":

            model, T, history, splits = train_lstm_model(
                X_train_final_fold, y_train_final,
                epochs=100,
                patience=10,
                save_best_path=model_path_fold,
                seed = seed,
                )

        elif config_models.models_name == "RandomForest TSFEL":
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(class_weight=class_weight_choice.value[1:], random_state=seed)
            rf.fit(X_train_final_fold, y_train_final)
            final_model_to_save = rf
        elif config_models.models_name == "RandomForest Imbalanced TSFEL":
            from imblearn.ensemble import BalancedRandomForestClassifier
            brf = BalancedRandomForestClassifier(class_weight = class_weight_choice.value[1:], random_state = seed)
            brf.fit(X_train_final_fold, y_train_final)
            final_model_to_save = brf
        elif config_models.models_name == "XGBoost TSFEL":
            from xgboost import XGBClassifier
            X_train_base_numpy = X_train_final_fold.to_numpy()
            y_train_base_numpy = np.asarray(y_train_final).astype(int)

            n_pos = np.sum(y_train_base_numpy == 1)
            n_neg = np.sum(y_train_base_numpy == 0)

            if n_pos == 0 or n_neg == 0:
                raise ValueError(
                    f"XGBoost nécessite les deux classes. "
                    # Correction de la variable ici
                    f"Classes trouvées: {np.unique(y_train_base_numpy, return_counts=True)}"
                )

            ratio = n_neg / n_pos

            xgb_base = XGBClassifier(
                scale_pos_weight=ratio,
                random_state=seed,
                eval_metric="logloss",
                missing=np.nan,
                n_jobs = -1
                # n_jobs = 1 => Si on veut une reproductibilité complète mais plus long donc non pour l'instant
            )

            xgb_base.fit(X_train_base_numpy, y_train_base_numpy)
            final_model_to_save = xgb_base

        elif config_models.models_name == "SVC TSFEL" : 
            from sklearn.svm import SVC
            svc_base = SVC(kernel="rbf", C=1.0, random_state=seed, class_weight=class_weight_choice.value[1:], probability=False)
            svc_base.fit(X_train_final_fold, y_train_final)
            final_model_to_save = svc_base
        elif config_models.models_name == "Logistic Regression Lasso TSFEL" :
            from sklearn.model_selection import GridSearchCV
            # Si X_train_final_fold est déjà standardisé ou stocké en DataFrame Polars :
            X_train_final_fold.write_parquet(file_X_exact)
            np.save(file_y_exact, y_train_final)
            print(f"    [DISK-SAVE] X et y exacts enregistrés pour le Lasso Path.")

            print(f"    [INFO] Entraînement LogisticRegression avec Lasso via GridSearchCV...")

            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

            lr_base = LogisticRegression(
                l1_ratio = 1.0,
                solver='saga',
                max_iter=10000,
                random_state=seed
            )

            # On teste 10 valeurs de C sur une échelle logarithmique
            param_grid = {'C': np.logspace(-4, 4, 10)}

            # 5. On enveloppe le tout dans un GridSearchCV
            lasso_cv = GridSearchCV(
                estimator=lr_base,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            groups_for_fit = groups_fold_2

            if calibration.value:
                 groups_for_fit = groups_fold_2[train_idx_calib]

            # On utilise groups_for_fit pour éviter les problèmes d'indices
            lasso_cv.fit(X_train_final_fold, y_train_final, groups=groups_for_fit)

            # On récupère le meilleur C trouvé
            best_C = lasso_cv.best_params_['C']
            print(f"    [INFO] Meilleur C trouvé pour le Fold {fold_idx_2 + 1} : {best_C:.4f}")

            # On sauvegarde le meilleur modèle estimé (.best_estimator_)
            final_model_to_save = lasso_cv.best_estimator_
        else :
            print(f"oups tu t'es trompé : {config_models.models_name}")

        if calibration.value and final_model_to_save is not None:
            print(f"    [INFO] Application de la calibration Isotonique (prefit) sur le jeu held-out...")
            # cv='prefit' indique d'utiliser le modèle déjà entraîné et de calibrer sur (X_calib, y_calib)
            calibrated_clf = CalibratedClassifierCV(estimator=final_model_to_save, method='isotonic', cv='prefit')

            # Gestion spécifique pour XGBoost qui n'aime pas les DataFrames Polars/Pandas dans CalibratedClassifierCV
            X_calib_final, y_calib_final = X_calib, y_calib
            if "XGB" in type(final_model_to_save).__name__:
                X_calib_final = X_calib.to_numpy()
                y_calib_final = np.asarray(y_calib).astype(int)

            calibrated_clf.fit(X_calib_final, y_calib_final)
            final_model_to_save = calibrated_clf

        # --- SAUVEGARDE FINALE DU MODÈLE (BASE OU CALIBRÉ) ---
        if final_model_to_save is not None:
            joblib.dump(final_model_to_save, model_path_fold)
            print(f"--> Modèle final enregistré à : {model_path_fold}")
        folds_X_fit_exact.append(X_train_final_fold)
        folds_y_fit_exact.append(y_train_final)


    print("Cross Validation terminée ! 5 modèles ont été enregistrés avec succès.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation des résultats obtenus
    """)
    return


@app.cell
def _(config_models, exp):
    output_dir = exp.get_output_path(config_models.models_name)
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
    calibration,
    classification_report,
    config_models,
    evaluate_lstm_on_test,
    evaluate_on_test,
    exp,
    extension,
    folds_X_test,
    folds_X_train,
    folds_y_test,
    folds_y_train,
    joblib,
    load_lstm_from_checkpoint,
    load_model_from_checkpoint,
    np,
    output_dir,
    pl,
    predict_proba,
    predict_proba_lstm,
    save_figure,
    sfu,
    transparent,
):
    # --- LISTES D'ACCUMULATION POUR LES MÉTRIQUES TEXTES ---
    all_test_scores = []
    all_train_scores = []
    all_auc_scores = []
    all_brier_scores = []

    # --- LISTES D'ACCUMULATION POUR LES GRAPHIQUES ---
    all_y_true_report = []  # Pour le classification report cumulé (TSFEL)
    all_y_pred_report = []  # Pour le classification report cumulé (TSFEL)

    all_y_test_global = []      # Cibles réelles pour la courbe de calibration
    all_probas_uncalib = []     # Probas brutes pour la courbe de calibration
    all_probas_calib = []       # Probas calibrées pour la courbe de calibration


    for fold_idx_bis in range(5):
        print(f"\n─────────────────── Évaluation du Fold {fold_idx_bis + 1}/5 ───────────────────")
        # Extraction des matrices propres à ce fold
        X_train = folds_X_train[fold_idx_bis]
        X_test = folds_X_test[fold_idx_bis]
        y_train = folds_y_train[fold_idx_bis]
        y_test = folds_y_test[fold_idx_bis]

        # Reconstruction du fichier à charger
        loaded_model = exp.get_model_path(config_models.models_name, fold_idx_bis, extension)
        print("Modèle chargé :", loaded_model)

        if config_models.models_name == "InceptionTimeModified":
            X_train_final = X_train
            X_test_final = X_test

            (auc, brier, T_1) = evaluate_on_test(X_test_final, y_test, loaded_model)
            (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)

            probas_fold = predict_proba(model_1, X_test_final, T = T_1)

            all_auc_scores.append(auc)
            all_brier_scores.append(brier)
            all_y_test_global.extend(y_test)
            all_probas_uncalib.extend(probas_fold)
            all_probas_calib.extend(probas_fold)

        elif config_models.models_name == "LstmTimeModified":
            X_train_final = X_train
            X_test_final = X_test
            # 🔍 DEBUG ENTRÉE ÉVALUATION
            print(f"\n[DEBUG EVAL - Fold {fold_idx_bis + 1}] Shape de X_test_final: {X_test_final.shape}")
            (auc, brier, T_1) = evaluate_lstm_on_test(X_test_final, y_test, loaded_model)
            (model_1, _, T_1) = load_lstm_from_checkpoint(loaded_model)
            probas_fold = predict_proba_lstm(model_1, X_test_final, T = T_1)

            all_auc_scores.append(auc)
            all_brier_scores.append(brier)
            all_y_test_global.extend(y_test)
            all_probas_uncalib.extend(probas_fold)
            all_probas_calib.extend(probas_fold)

        elif config_models.extraction_type == "TSFEL":
            from sklearn.calibration import CalibratedClassifierCV
            clf = joblib.load(loaded_model)

            # 1. Extraction universelle des features
            expected_features = None
            if hasattr(clf, "feature_names_in_"):
                expected_features = list(clf.feature_names_in_)
            # Cas spécifique où CalibratedClassifierCV enveloppe un modèle avec feature_names_in_
            elif hasattr(clf, "estimator") and hasattr(clf.estimator, "feature_names_in_"):
                expected_features = list(clf.estimator.feature_names_in_)
            elif hasattr(clf, "get_booster"):
                expected_features = clf.get_booster().feature_names

            # 2. Alignement conditionnel
            if expected_features is not None:
                if expected_features and expected_features[0].startswith('f') and expected_features[0][1:].isdigit():
                    print("XGBoost utilise des indices génériques. Utilisation des matrices brutes.")
                    X_train_final = X_train.to_numpy()
                    X_test_final = X_test.to_numpy()
                else:
                    missing_cols = [c for c in expected_features if c not in X_train.columns]
                    if missing_cols:
                        print(f"Ajout de {len(missing_cols)} colonnes manquantes (0.0)")
                        padding_expr = [pl.lit(0.0).alias(c) for c in missing_cols]
                        X_train_final = X_train.with_columns(padding_expr).select(expected_features)
                        X_test_final = X_test.with_columns(padding_expr).select(expected_features)
                    else:
                        X_train_final = X_train.select(expected_features)
                        X_test_final = X_test.select(expected_features)

                    if "XGB" in type(clf).__name__:
                        X_train_final = X_train_final.to_pandas()
                        X_test_final = X_test_final.to_pandas()
            else:
                print("Aucun nom de feature trouvé dans le modèle. Passage en matrices NumPy brutes.")
                X_train_final = X_train.to_numpy()
                X_test_final = X_test.to_numpy()

            # 3. Prédictions et Scores
            y_pred_nb_train = clf.predict(X_train_final)
            y_pred_nb_test = clf.predict(X_test_final)

            train_score = clf.score(X_train_final, y_train)
            test_score = clf.score(X_test_final, y_test)

            all_test_scores.append(test_score)
            all_train_scores.append(train_score)
            all_y_true_report.extend(y_test)
            all_y_pred_report.extend(y_pred_nb_test)

            # 4. RÉCUPÉRATION DES PROBABILITÉS
            prob_uncalib_fold = None
            prob_calib_fold = None
            # Gestion spécifique pour XGBoost numpy vs pandas
            X_test_final_numpy = X_test_final
            if hasattr(X_test_final, "to_numpy"):
                 X_test_final_numpy = X_test_final.to_numpy()

            if isinstance(clf, CalibratedClassifierCV):
                # Le modèle EST calibré (ex: SVC, XGBoost)
                print("    [INFO] Modèle calibré détecté. Extraction des probabilités brutes et calibrées...")

                # A. Probabilités CALIBRÉES (via le CalibratedClassifierCV)
                all_probas_calib_fold = clf.predict_proba(X_test_final)
                # On assume que la classe positive (1) est à l'indice 1
                prob_calib_fold = all_probas_calib_fold[:, 1]

                # B. Probabilités NON-CALIBRÉES (via l'estimateur de base brut)
                base_estimator = clf.estimator

                # XGBoost base estimator nécessite souvent du numpy pur
                if "XGB" in type(base_estimator).__name__:
                     all_probas_uncalib_fold = base_estimator.predict_proba(X_test_final_numpy)
                else:
                     all_probas_uncalib_fold = base_estimator.predict_proba(X_test_final)

                prob_uncalib_fold = all_probas_uncalib_fold[:, 1]

            else:
                # Le modèle N'EST PAS calibré extraordinairement (Lasso, RF)
                # calibration.value était False lors de l'entraînement
                print("    [INFO] Modèle non-calibré (ou naturellement calibré) détecté.")
                all_probas_fold = clf.predict_proba(X_test_final)
                prob_fold = all_probas_fold[:, 1]

                # Les probabilités sont les mêmes avant/après
                prob_uncalib_fold = prob_fold
                prob_calib_fold = prob_fold

            # Accumulation dans les listes globales
            all_probas_uncalib.extend(prob_uncalib_fold)
            all_probas_calib.extend(prob_calib_fold)
            all_y_test_global.extend(y_test)
            print(f"Le score (Accuracy) sur le fold {fold_idx_bis + 1} est : {test_score:.4f}")

    print("\n" + "="*20 + " BILAN GLOBAL DE LA CROSS-VALIDATION " + "="*20)

    all_y_test_global = np.array(all_y_test_global)
    all_probas_uncalib = np.array(all_probas_uncalib)
    all_probas_calib = np.array(all_probas_calib)

    print("\n" + "="*20 + " BILAN GLOBAL DE LA CROSS-VALIDATION " + "="*20)

    if config_models.extraction_type == "TSFEL":
        mean_acc = np.mean(all_test_scores)
        std_acc = np.std(all_test_scores)
        print(f"Score moyen (Accuracy) : {mean_acc:.4f} (± {std_acc:.4f})")
        print("\nRapport de classification cumulé (sur l'ensemble des 5 folds mis en commun) :")
        print(classification_report(all_y_true_report, all_y_pred_report, target_names=["Alive", "Deceased"], zero_division=0))
    else:
        mean_auc = np.mean(all_auc_scores)
        std_auc = np.std(all_auc_scores)
        mean_brier = np.mean(all_brier_scores)
        std_brier = np.std(all_brier_scores)
        print(f"AUC moyenne  : {mean_auc:.4f} (± {std_auc:.4f})")
        print(f"Brier moyenne : {mean_brier:.4f} (± {std_brier:.4f})")
    print("="*79)

    print("\nGénération de la courbe de calibration poolée...")
    print(f"DEBUG SIZES -> y_true: {len(all_y_test_global)}, uncalib: {len(all_probas_uncalib)}, calib: {len(all_probas_calib)}")
    sfu.calibration_curve_homemade(all_probas_uncalib, all_probas_calib, all_y_test_global, config_models.models_name, config_models.extraction_type, calibration.value, save_figure.value, output_dir, transparent.value)

    # --- RE-MAPPING DES ÉTATS GLOBAUX POUR LES CELLULES SUIVANTES (COURBE ROC / MATRICE) ---
    probas = all_probas_calib
    y_test = all_y_test_global
    return CalibratedClassifierCV, probas, y_test


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
    Path,
    config_models,
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
    transparent,
):
    mo.stop(not run_lasso.value, "Clique pour lancer")
    print("Lasso Path lancé")
    if config_models.models_name == "Logistic Regression Lasso TSFEL":
        from joblib import Parallel, delayed
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
                max_iter=1000,
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

        # Lancement des 5 folds en parallèle sur tous tes cœurs CPU (-1)
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
                plt.savefig(output_dir / Path(filename), dpi=300, bbox_inches="tight", transparent=transparent.value)
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
    output_dir,
    probas,
    save_figure,
    sfu,
    transparent,
    y_test,
):
    auc_final, fpr, tpr, thresholds_roc = sfu.roc_curve_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, transparent.value)
    return (auc_final,)


@app.cell
def _(df_clean_3, folds_y_test, patient_col):
    total_predictions = sum(len(f) for f in folds_y_test)
    total_patients_uniques = df_clean_3[patient_col].n_unique()

    print(f"Nombre total de patients uniques dans la cohorte : {total_patients_uniques}")
    print(f"Somme des tailles des 5 jeux de test cumulés : {total_predictions}")
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
    output_dir,
    probas,
    save_figure,
    sfu,
    transparent,
    y_test,
):
    non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1 = sfu.kde_plot_homemade(probas, y_test, config_models.models_name, save_figure.value, output_dir, transparent.value)
    return asymetric_incertitude, mean_p1, mean_risk_diff, non_overlap_area


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("kde_plot.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""))
    return


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Choix du threshold en fonction du F1 score pour {config_models.models_name}
    """)
    return


@app.cell
def _(
    config_models,
    output_dir,
    probas,
    save_figure,
    sfu,
    transparent,
    y_test,
):
    best_f1, best_t = sfu.f1_score_evolution(probas, y_test, config_models.models_name, save_figure.value, output_dir, transparent.value)
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
    output_dir,
    probas,
    save_figure,
    sfu,
    transparent,
    y_test,
):
    y_pred, mcc = sfu.confusion_matrix_homemade(probas, y_test, best_t, config_models.models_name, save_figure.value, output_dir, transparent.value)
    return mcc, y_pred


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Evolution du score de Brier de {config_models.models_name}
    """)
    return


@app.cell
def _(output_dir, probas, save_figure, sfu, transparent, y_test):
    global_brier = sfu.brier_evolution(probas, y_test, save_figure.value, output_dir, transparent=transparent.value, )
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
    auc_final,
    best_f1,
    global_brier,
    joblib,
    mcc,
    mean_p1,
    mean_risk_diff,
    non_overlap_area,
    output_dir,
    probas,
    y_pred,
):
    # Résumé des scores obtenus
    all_res = {
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
def _(exp, joblib, np, patient_col, pl, sfu, target_col):
    show_shap = {}
    model_name_shap = "XGBoost TSFEL"

    for f_idx in range(5):
        output_dir_shap = exp.get_output_path(model_name_shap) / f"fold_{f_idx}"
        output_dir_shap.mkdir(parents = True, exist_ok = True)
        loaded_model_shap = exp.get_model_path(model_name_shap, f_idx, ".joblib",  "",)
        X_train_fold_shap = exp.get_tsfel_boruta("train", f_idx)
        keepVariableList_shap = X_train_fold_shap.parent / f"keepVariableList_2_fold_{f_idx}.npy"
        clf_shap = joblib.load(loaded_model_shap)
        sfu.mesureImportance_tsfel(clf_shap, pl.read_parquet(X_train_fold_shap).select(pl.exclude(patient_col, target_col)), np.load(keepVariableList_shap), top_n=20, class_labels=["Survivors", "Deaths"], savefig = True, transparent = False, folder = output_dir_shap)
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


@app.cell
def _(exp, model_name_shap_show, sfu, show_shap2):
    sfu.compare_models_figure("feature_importance.png",savefig=True, folder = exp.get_output_path(model_name_shap_show), **show_shap2)
    return


@app.cell
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
def _(df_clean, df_clean_3, pl):
    df_clean_saps2 = (
        df_clean_3
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


@app.cell
def _(mo, run_test):
    mo.stop(not run_test.value, "Clique pour lancer")
    print("Comparaison lancée")
    return


@app.cell
def _(exp, saps2_pred, saps2_true, sfu, y_test):
    comparaisons = [exp.load_model("InceptionTimeModified"), exp.load_model("LstmTimeModified"), exp.load_model("RandomForest TSFEL", "_balanced"), exp.load_model("XGBoost TSFEL"), exp.load_model("SVC TSFEL")]

    comparaisons

    sfu.générer_rapport_comparatif(y_test, comparaisons, save_dir="Comparaison ALl", table_format='fancy_grid', saps2_pred = saps2_pred, saps2_true = saps2_true)
    return


@app.cell
def _(exp, sfu, y_test):
    comparaisons_time = [exp.load_model("InceptionTimeModified"), exp.load_model("LstmTimeModified")]

    sfu.générer_rapport_comparatif(y_test, comparaisons_time, save_dir="Comparaison Time", table_format='fancy_grid')
    return


@app.cell
def _(exp, sfu, y_test):
    comparaisons_ml = [exp.load_model("RandomForest TSFEL", "_balanced"), exp.load_model("XGBoost TSFEL"), exp.load_model("SVC TSFEL")]
    sfu.générer_rapport_comparatif(y_test, comparaisons_ml, save_dir="Comparaison TSFEL", table_format='fancy_grid')
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("KDE.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("Calibration_curve.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("Courbe_ROC.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("threshold.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("confusion_matrix.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("calibPerDec.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("brierPerTrancheRisk.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(exp, sfu):
    sfu.compare_models_figure("brierPerDec.png", RandomForestTSFEL= exp.get_output_path("RandomForest TSFEL", "_balanced"),
    InceptionTime = exp.get_output_path("InceptionTimeModified", ""),
    LSTMT = exp.get_output_path("LstmTimeModified", ""),
    XGBoost = exp.get_output_path("XGBoost TSFEL", ""),
    SVC = exp.get_output_path("SVC TSFEL", ""))
    return


@app.cell
def _(
    balance,
    cleaning,
    custom_features,
    keep_feats,
    keep_pop,
    mo,
    mo_utils,
    mode,
    models,
    modex,
    run,
    save_figure,
    seed,
    str_keep_feats,
    sys,
    transparent,
    ui_tsfel,
    y_dd,
):
    mo.sidebar(
    mo.vstack([
        mo.md(mo_utils.config_sidebar),
        mo.md(f"<U>Seed utilisée pour l'ensemble du code : **{seed}**</U>"),
        mo.md(f"Version de python : {sys.version}"),
        mo.md("-------------------------------"),
        mode,
        mo.md("-------------------------------"),
        balance,
        models,
        ui_tsfel,
        cleaning,
        y_dd,
        keep_pop,
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français (dynamic feature only):** \n{str_keep_feats}"),
        save_figure,
        transparent,
        mo.md("-------------------------------"),
        run,
        mo.md("-------------------------------"),
        mo.md(mo_utils.config_end)]),
    width = "550px")
    return


if __name__ == "__main__":
    app.run()
