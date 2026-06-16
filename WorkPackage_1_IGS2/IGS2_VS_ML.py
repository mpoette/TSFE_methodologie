import marimo

__generated_with = "0.23.2"
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
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    # 5. Imports locaux (modules customs)
    import utilitaries.create_merged_dataset as create_merged_dataset
    import utilitaries.extract_data_utils as extract
    import utilitaries.features_extraction_utils as extract_feat
    import utilitaries.marimo_utils as mo_utils
    import utilitaries.optuna.optuna_utils as optuna_utils
    import utilitaries.preprocessing_utils as preproc
    import utilitaries.preprocessing_utils as ui
    import utilitaries.utils as utils
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
        Path,
        StratifiedGroupKFold,
        brier_score_loss,
        calibration_curve,
        classification_report,
        confusion_matrix,
        create_merged_dataset,
        evaluate_lstm_on_test,
        evaluate_on_test,
        extract,
        extract_feat,
        f1_score,
        joblib,
        json,
        load_lstm_from_checkpoint,
        load_model_from_checkpoint,
        matthews_corrcoef,
        mo,
        mo_utils,
        np,
        os,
        pd,
        pl,
        plt,
        predict_proba,
        predict_proba_lstm,
        preproc,
        roc_auc_score,
        roc_curve,
        seed,
        sns,
        train_inception_time,
        train_lstm_model,
        utils,
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
        ui_tsfel = mo.vstack([extract_tsfel, boruta_filter, calibration])
        if config_models.models_type == "calibrated":
            ui_tsfel = mo.vstack([extract_tsfel, boruta_filter])
    else :
        extract_tsfel = None
        ui_tsfel = mo.md("")
    return extract_tsfel, ui_tsfel


@app.cell
def _(config_models, mo):
    if config_models.models_name == "LstmTimeModified":
        metric_options = ["val_loss", "val_auc"]
    else:
        metric_options = ["val_loss"]

    # Dropdown
    metric_name = mo.ui.dropdown(
        options=metric_options,
        value=metric_options[0],
        label="Métrique choisie pour l'optimisation optuna"
    )
    return (metric_name,)


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


@app.cell
def _(metric_name, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        metric_name,
        mo.md(mo_utils.config_end)
    ])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### On sélectionne les features inutiles/donnant trop d'informations
    """)
    return


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
    X = df_clean_3.select(final_features).to_numpy()
    y = df_clean_3[target_col].to_numpy()
    return X, df_clean_3, final_features, y


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
    #### Split Train/Test
    cette cellule ne fonctionne que parce qu'on a des fenêtres qui sont toutes de 24h => Le poids entre les étiquettes est le même peu importe le patient qu'on prend ce qui permet un bon équilibrage train/test
    """)
    return


@app.cell
def _(config_balance):
    underscore = "_"
    if config_balance.balance_method == "":
        underscore = ""
    return (underscore,)


@app.cell
def _(
    StratifiedGroupKFold,
    X,
    boruta_filter,
    config_balance,
    config_cleaning,
    config_mode,
    config_models,
    config_y,
    df_clean_3,
    expected_length,
    extract,
    extract_feat,
    extract_tsfel,
    final_features,
    modex,
    np,
    patient_col,
    pl,
    preproc,
    seed,
    target_col,
    time_col,
    y,
):
    groups = df_clean_3[patient_col].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    (train_idx, test_idx) = next(sgkf.split(X=X, y=y, groups=groups))

    # Tri obligatoire (même si en théorie il est déjà fait)
    train_df = df_clean_3[train_idx].sort([patient_col, time_col])
    test_df = df_clean_3[test_idx].sort([patient_col, time_col])

    if config_models.extraction_type == "TSFEL" : 

         # On enlève les features statiques
        static_feats = ["admission_type_Medical", "admission_type_Scheduled Surgery", "admission_type_Unknown", "admission_type_Unscheduled Surgery", "score_glasgow", "age"]

        # Gestion des noms de fichier Boruta
        str_boruta = "_Boruta" if boruta_filter.value else ""
        filename_train = f"tsfel_train_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}{str_boruta}.parquet"
        filename_test = f"tsfel_test_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}{str_boruta}.parquet"

        if extract_tsfel.value :
            # Extraction TSFEL brute
            tsfel_features = [c for c in final_features if c not in static_feats]
            TSFEL_train_df = extract_feat.extract_tsfel_per_patient(train_df, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
            TSFEL_test_df = extract_feat.extract_tsfel_per_patient(test_df, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)

            # Filtrage corrélation/variance
            TSFEL_train_clean, TSFEL_test_clean, keepVariableList = extract_feat.filtrage_corr_var(TSFEL_train_df, TSFEL_test_df, patient_col, target_col)

            # Jointure avec données statiques
            static_train = train_df.select([extract.ID_COL, *static_feats]).unique()
            static_test = test_df.select([extract.ID_COL, *static_feats]).unique()
            new_train_df = TSFEL_train_clean.join(static_train, on=extract.ID_COL, how="inner")
            new_test_df = TSFEL_test_clean.join(static_test, on=extract.ID_COL, how="inner")

            # Save 1 : Dataset complet sans Boruta
            new_train_df.write_parquet(f"tsfel_train_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")
            new_test_df.write_parquet(f"tsfel_test_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")

            if boruta_filter.value:
                # Calcul de Boruta
                new_train_df, new_test_df, keepVariableList = extract_feat.filtrage_boruta( new_train_df, new_test_df, patient_col, target_col, max_iter = 100, seed = seed)

                # Save 2 : Dataset complet avec Boruta
                new_train_df.write_parquet(filename_train)
                new_test_df.write_parquet(filename_test)
        else:
            # Lecture des Dataset soit avec Boruta soit sans
            new_train_df = pl.read_parquet(filename_train)
            new_test_df = pl.read_parquet(filename_test)
            keepVariableList = [c for c in new_train_df.columns if c not in [extract.ID_COL, target_col]]

        # Equilibrage
        new_train_df = preproc.equilibrer_dataset_tabulaire(new_train_df, extract.ID_COL, target_col, method = config_balance.balance_method, seed = seed)

        # Tri (obligatoire pour comparabilité)
        new_train_df = new_train_df.sort(patient_col)
        new_test_df = new_test_df.sort(patient_col)

        # Sécurité reproductibilité/intégrité : On s'assure qu'il n'y a qu'UNE seule ligne par patient
        assert new_train_df.height == new_train_df[extract.ID_COL].n_unique(), "Erreur d'alignement Train TSFEL"
        assert new_test_df.height == new_test_df[extract.ID_COL].n_unique(), "Erreur d'alignement Test TSFEL"
        y_train = new_train_df[target_col].to_numpy()
        y_test = new_test_df[target_col].to_numpy()
        new_train_df = new_train_df.select(pl.exclude(patient_col, target_col))
        new_test_df = new_test_df.select(pl.exclude(patient_col, target_col))

        # Scaling
        (X_train, X_test) = preproc.scaling(new_train_df, new_test_df)

    elif config_models.extraction_type == "time" :

        # Gestion exclusive de l'équilibrage homemade (avec polars)
        if config_balance.balance_method in ["downsampling_homemade", ""]:
            train_df = preproc.equilibrer_dataset_tabulaire(train_df, extract.ID_COL, target_col, method = config_balance.balance_method, seed = seed)

        # On prépare le jeu d'entraînement

        # Scaling
        (train_df, test_df) = preproc.scaling(train_df, test_df)

        # Transformation en 3D Array
        (X_train, y_train) = preproc.build_sequences(train_df, patient_col, target_col, expected_length, final_features)  # grouper en fonction d'un individu
        (X_test, y_test) = preproc.build_sequences(test_df, patient_col, target_col, expected_length, final_features)

        # Gestion de l'équilibrage avec imblearn (avec un 3D Array directement)
        if config_balance.balance_method not in ["downsampling_homemade", ""]:
            # On applatit le 3D Array en 2D Array
            n_samples, n_timesteps, n_feats = X_train.shape
            X_train_2d = X_train.reshape(n_samples, n_timesteps * n_feats)

            # On applique la méthode d'équilibrage imblearn
            if config_balance.balance_method == "downsampling_50-50":
                from imblearn.under_sampling import RandomUnderSampler
                rs = RandomUnderSampler(random_state=seed)
            elif config_balance.balance_method == "upsampling_50-50":
                from imblearn.over_sampling import RandomOverSampler
                rs = RandomOverSampler(random_state=seed)
            else:
                raise ValueError("Cet équilibrage n'a pas encore été implémenté")

            X_res_2d, y_train = rs.fit_resample(X_train_2d, y_train)

            # On redonne sa forme 3D d'origine au tenseur équilibré
            X_train = X_res_2d.reshape(-1, n_timesteps, n_feats)

        # TODO : externaliser la gestion des NaN
        # On enlève les NaN après extraction de features
        total_nan = np.isnan(X_train).sum()
        # Compte les NaN pour chaque feature
        nan_par_feature = np.isnan(X_train).sum(axis=(0, 1))
        # for i, feat_name in enumerate(final_features):
            # print(f"Feature '{feat_name}' : {nan_par_feature[i]} NaN")
        print(f"Nombre total de valeurs NaN : {total_nan}")

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan = 0.0)

    else:
        raise ValueError("Modèle inexistant/Pas implémenté")
    return X_test, X_train, keepVariableList, test_df, y_test, y_train


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
def _(
    config_balance,
    config_cleaning,
    config_keep_pop,
    config_models,
    config_y,
    modex,
    underscore,
    utils,
):
    # on créé un nom unique de modèle
    str_pop = ""
    if config_keep_pop.keep_population != "all_diseases":
        str_pop = "_"+config_keep_pop.keep_population

    extension = ".joblib" if config_models.extraction_type == "TSFEL" else ".pt"

    model_path = utils.get_unique_path(
    f"models/{config_models.models_name}/{config_cleaning.clean}_{config_y.target_name}_"
        f"{config_balance.balance_method}{underscore}{modex.value}{str_pop}{extension}"
    )

    default_params = {
            "epochs" : 100,
            "patience" : 10,
        }

    parameters = default_params
    return extension, model_path, parameters, str_pop


@app.cell
def _(
    X_train,
    config_models,
    joblib,
    mo,
    model_path,
    np,
    parameters,
    run,
    seed,
    train_inception_time,
    train_lstm_model,
    y_train,
):
    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")

    # --- BARRIÈRE DE SÉCURITÉ GÉOMÉTRIQUE ---
    is_dl_model = config_models.models_name in ["InceptionTimeModified", "LstmTimeModified"]
    n_dims = len(X_train.shape) if hasattr(X_train, "shape") else 0
    if is_dl_model and n_dims != 3:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice 3D [patients, temps, features], mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'time' ?")
    elif not is_dl_model and n_dims != 2:
        raise ValueError(f"Mismatch : Le modèle {config_models.models_name} attend une matrice tabulaire 2D, mais X_train a {n_dims} dimension(s). As-tu configuré le pipeline en mode 'TSFEL' ?")


    if config_models.models_name == "InceptionTimeModified":
        print(parameters)
        model, T, history, splits = train_inception_time(
            X_train, y_train,
            save_best_path=model_path,
            seed = seed,
            **parameters
            )

    elif config_models.models_name == "LstmTimeModified":
        model, T, history, splits = train_lstm_model(
            X_train, y_train,
            epochs=100,
            patience=10,
            save_best_path=model_path
            )

    elif config_models.models_name == "RandomForest TSFEL":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(class_weight='balanced', random_state=seed)
        rf.fit(X_train, y_train)
        joblib.dump(rf, model_path)
    elif config_models.models_name == "XGBoost TSFEL":
        from xgboost import XGBClassifier
        # TODO : rajouter n_jobs = 1 ou n_threads = 1 pour éviter le random dans le multiprocessing si jamais on a des petites variations
        X_train_tsfel = X_train.to_numpy()
        y_train_tsfel = np.asarray(y_train).astype(int)

        n_pos = np.sum(y_train_tsfel == 1)
        n_neg = np.sum(y_train_tsfel == 0)

        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"XGBoost nécessite les deux classes. "
                f"Classes trouvées: {np.unique(y_train_tsfel, return_counts=True)}"
            )

        ratio = n_neg / n_pos

        xgb = XGBClassifier(
            scale_pos_weight=ratio,
            random_state=seed,
            eval_metric="logloss",
            missing=np.nan,
        )

        xgb.fit(X_train_tsfel, y_train_tsfel)
        joblib.dump(xgb, model_path)

    elif config_models.models_name == "SVC TSFEL" : 
        from sklearn.svm import SVC
        svc = SVC(kernel = "rbf", C = 1.0, random_state = seed, class_weight = "balanced", probability = True)
        svc.fit(X_train, y_train)
        joblib.dump(svc, model_path)
    else :
        print("oups tu t'es trompé")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistiques générales
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    classification_report,
    config_balance,
    config_cleaning,
    config_models,
    config_y,
    evaluate_lstm_on_test,
    evaluate_on_test,
    extension,
    joblib,
    load_lstm_from_checkpoint,
    load_model_from_checkpoint,
    modex,
    pl,
    str_pop,
    underscore,
    utils,
    y_test,
    y_train,
):
    base_pattern = f"models/{config_models.models_name}/{config_cleaning.clean}_{config_y.target_name}_{config_balance.balance_method}{underscore}{modex.value}{str_pop}_*{extension}"  

    loaded_model = utils.get_latest_model_path(base_pattern, extension)

    print("Modèle chargé :", loaded_model)

    if config_models.models_name == "InceptionTimeModified":
        X_train_final = X_train
        X_test_final = X_test
        (_auc, brier, T_1) = evaluate_on_test(X_test_final, y_test, loaded_model)
        (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)

    elif config_models.models_name == "LstmTimeModified":
        X_train_final = X_train
        X_test_final = X_test
        (_auc, brier, T_1) = evaluate_lstm_on_test(X_test_final, y_test, loaded_model)
        (model_1, _, T_1) = load_lstm_from_checkpoint(loaded_model)

    elif config_models.extraction_type == "TSFEL":
        clf = joblib.load(loaded_model)

        # 1. Extraction universelle des features
        expected_features = None
        if hasattr(clf, "feature_names_in_"):
            expected_features = list(clf.feature_names_in_)
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

        print(f"Le score sur les données de test est {test_score:.4f}")
        print(classification_report(y_test, y_pred_nb_test, target_names=["Alive", "Deceased"], zero_division=0))
    return T_1, X_test_final, X_train_final, clf, loaded_model, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Création du dossier Output
    """)
    return


@app.cell
def _(Path, config_models, loaded_model):
    # construire le dossier output correspondant
    output_dir = Path("outputs") / Path(config_models.models_name) / Path(loaded_model).stem

    # créer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calcul des prédictions probabilistes des modèles
    """)
    return


@app.cell
def _(
    Path,
    T_1,
    X_test_final,
    X_train_final,
    calibration,
    calibration_curve,
    clf,
    config_models,
    model_1,
    output_dir,
    plt,
    predict_proba,
    predict_proba_lstm,
    save_figure,
    transparent,
    y_test,
    y_train,
):
    # c'est la même fonction pour les 2 modèles donc c'est ok
    if config_models.models_name == "InceptionTimeModified":
        probas = predict_proba(model_1, X_test_final, T=T_1)

    elif config_models.models_name == "LstmTimeModified":
        probas = predict_proba_lstm(model_1, X_test_final, T=T_1)

    elif config_models.extraction_type == "TSFEL":
        all_probas = clf.predict_proba(X_test_final.to_numpy())
        classes = list(clf.classes_)
        positive_idx = classes.index(1)
        probas = all_probas[:, positive_idx]


    if calibration.value:
        from sklearn.calibration import CalibratedClassifierCV
        iso_calibrator = CalibratedClassifierCV(estimator=clf, method='isotonic', cv=5)

        iso_calibrator.fit(X_train_final, y_train)

        prob_calibrated = iso_calibrator.predict_proba(X_test_final)[:, 1]

        # Évaluation et Visualisation via une courbe de calibration
        fraction_of_positives_uncalib, mean_predicted_value_uncalib = calibration_curve(y_test, probas, n_bins=10)
        fraction_of_positives_calib, mean_predicted_value_calib = calibration_curve(y_test, prob_calibrated, n_bins=10)

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], "k:", label="Calibration parfaite")
        plt.plot(mean_predicted_value_uncalib, fraction_of_positives_uncalib, "s-", label="Avant calibration (Random Forest)")
        plt.plot(mean_predicted_value_calib, fraction_of_positives_calib, "s-", label="Après calibration (Isotonique)")

        plt.ylabel("Fraction réelle de positifs")
        plt.xlabel("Probabilité moyenne prédite")
        plt.title("Effet de la Calibration Isotonique")
        plt.legend(loc="lower right")
        plt.grid(True)
        if save_figure.value:
            plt.savefig(output_dir / Path("Courbe_ROC"), dpi = 300, bbox_inches="tight", transparent=transparent)
        plt.show()
        prob_uncalibrated = probas
        probas = prob_calibrated
    return prob_uncalibrated, probas


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Courbe ROC du modèle {config_models.models_name}
    """)
    return


@app.cell
def _(
    Path,
    calibration,
    config_models,
    output_dir,
    plt,
    prob_uncalibrated,
    probas,
    roc_auc_score,
    roc_curve,
    save_figure,
    transparent,
    y_test,
):
    (fpr, tpr, _thresholds) = roc_curve(y_test, probas)
    # TODO : là si l'AUC est différente entre le modèle LSTM et ici c'est parce que pour le modèle elle est calculée par rapport à 20% des données de train (validation) alors que là c'est par rapport à test.
    auc = roc_auc_score(y_test, probas)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC {config_models.models_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color = "green")

    if calibration.value:
        (fpr_unc, tpr_unc, _thresholds) = roc_curve(y_test, prob_uncalibrated)
        auc_unc = roc_auc_score(y_test, prob_uncalibrated)
        plt.plot(fpr_unc, tpr_unc, label=f'ROC {config_models.models_name} (uncalibrated) (AUC = {auc_unc:.3f})')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbe ROC du modèle {config_models.models_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_figure.value :
        plt.savefig(output_dir / Path("Courbe_ROC"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    
    return (auc,)


@app.cell
def _(
    Path,
    config_models,
    output_dir,
    plt,
    probas,
    save_figure,
    sns,
    transparent,
    y_test,
):
    plt.figure()

    sns.kdeplot(probas[y_test == 0], label="Survivants", fill=True)
    sns.kdeplot(probas[y_test == 1], label="Décès", fill=True)

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Densité")
    plt.title(f"Distribution des scores (KDE) de {config_models.models_name}")
    plt.legend()
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("KDE"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    return


@app.cell
def _(
    Path,
    calibration_curve,
    config_models,
    output_dir,
    plt,
    probas,
    save_figure,
    transparent,
    y_test,
):
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label=f"{config_models.models_name}")
    plt.plot([0, 1], [0, 1], "--", label="Calibration idéale")

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title(f"Calibration curve de {config_models.models_name} ")
    plt.legend()
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("Calibration_curve"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    return


@app.cell
def _(
    Path,
    config_models,
    f1_score,
    np,
    output_dir,
    plt,
    probas,
    save_figure,
    transparent,
    y_test,
):
    _thresholds = np.linspace(0.1, 0.9, 50)
    f1s = []
    best_f1 = 0
    for t in _thresholds:
        _y_pred = (probas >= t).astype(int)
        f1s.append(f1_score(y_test, _y_pred))
        f1 = f1_score(y_test, _y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    plt.plot(_thresholds, f1s)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title(f"Evolution du F1 score en fonction du Threshold pour le modèle {config_models.models_name}")
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("threshold"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    print(f'Le meilleur f1 score de{best_f1: .2f} est atteint lorsque le threshold est égal à{best_t: .2f}')
    return best_f1, best_t


@app.cell
def _(
    Path,
    best_t,
    config_models,
    confusion_matrix,
    matthews_corrcoef,
    output_dir,
    plt,
    probas,
    save_figure,
    sns,
    transparent,
    y_test,
):
    y_pred = (probas >= best_t).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title(f'Confusion matrix du modèle {config_models.models_name}  (threshold={ best_t: .2f}, MCC = {mcc})')
    if save_figure.value :
        plt.savefig(output_dir / Path("confusion_matrix"), dpi = 300, bbox_inches="tight", transparent=transparent, facecolor = "white")
    plt.show()
    return mcc, y_pred


@app.cell
def _(
    Path,
    brier_score_loss,
    output_dir,
    pl,
    plt,
    probas,
    save_figure,
    transparent,
    y_test,
):
    df_brier = pl.DataFrame({"y" : y_test, "pred" : probas})

    df_brier = df_brier.with_columns(
        ((pl.col("pred") - pl.col("y")) ** 2).alias("brier")
    )

    # score global de brier
    global_brier = brier_score_loss(df_brier["y"], df_brier["pred"])
    print("score de brier : ",global_brier)
    # bins fixes de risque
    df_brier_fixed = (
        df_brier.with_columns(
            (
                pl.col("pred")
                .clip(0, 0.999999)
                .mul(10)
                .floor()
                .cast(pl.Int64)
            ).alias("bin_fixed")
        )
        .group_by("bin_fixed")
        .agg([
            pl.col("brier").mean().alias("brier_mean"),
            pl.len().alias("n"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        ])
        .sort("bin_fixed")
        .with_columns(
            ((pl.col("bin_fixed") + 0.5) / 10).alias("x")
        )
    )

    # déciles de patients 
    n_total = df_brier.height

    df_brier_dec = (
        df_brier.sort("pred")
        .with_row_count("row_idx")
        .with_columns(
            (
                (pl.col("row_idx") * 10 / n_total)
                .floor()
                .clip(upper_bound=9)
                .cast(pl.Int64)
            ).alias("decile")
        )
        .group_by("decile")
        .agg([
            pl.col("brier").mean().alias("brier_mean"),
            pl.len().alias("n"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        ])
        .sort("decile")
        .with_columns(
            (pl.col("decile") + 1).alias("x")
        )
    )

    fixed_pd = df_brier_fixed.to_pandas()
    dec_pd = df_brier_dec.to_pandas()

    # Pour avoir 2 plots au même endroit, on utilise twinx
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(fixed_pd["x"], fixed_pd["n"], width=0.08, alpha=0.3)
    ax1.set_xlabel("Risque prédit")
    ax1.set_ylabel("Nombre de patients")

    ax2 = ax1.twinx()
    ax2.plot(fixed_pd["x"], fixed_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Brier moyen")

    plt.title(f"Brier par tranches fixes de risque ")
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("brierPerTrancheRisk"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3)
    ax1.set_xlabel("Décile de patients")
    ax1.set_ylabel("Nombre de patients")

    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Brier moyen")

    plt.title(f"Brier par déciles de patients ")
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("brierPerDec"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients
    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Décile de patients")
    ax1.set_ylabel("Nombre de patients")

    # Courbe de calibration
    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["obs_rate"], marker="o", label="Risque prédit", color="black")
    ax2.plot(dec_pd["x"], dec_pd["pred_mean"], marker="s", label="Mortalité observée", color="black", linestyle="--")
    ax2.set_ylabel("Mortalité (Taux observé vs Risque prédit)")

    plt.title(f"Courbe de Calibration par déciles de patients ")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("calibPerDec"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients (Tranches fixes)
    ax1.bar(fixed_pd["x"], fixed_pd["n"], width=0.08, alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Risque prédit (Tranches de 10%)")
    ax1.set_ylabel("Nombre de patients")
    ax1.set_xlim(0, 1)

    # Courbe de calibration (axe Y droit)
    ax2 = ax1.twinx()
    ax2.plot(fixed_pd["x"], fixed_pd["obs_rate"], marker="o", label="Risque moyen prédit", color="black", linestyle="-")
    ax2.plot(fixed_pd["x"], fixed_pd["pred_mean"], marker="s", label="Mortalité observée", color="black", linestyle="--")
    ax2.set_ylabel("Mortalité (Taux observé vs Risque prédit)")
    ax2.set_ylim(0, 1) 

    plt.title(f"Courbe de Calibration par tranches fixes de risque ")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("calibPerTrancheRisk"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    return (global_brier,)


@app.cell
def _(auc, best_f1, global_brier, joblib, mcc, output_dir, probas, y_pred):
    # Résumé des scores obtenus
    all_res = {
            'probas': probas,
            'preds': y_pred,       
            'f1_score': best_f1, 
            'mcc': mcc,
            'auc': auc,
            'brier': global_brier,
    }
    joblib.dump(all_res, output_dir / "all_res.joblib")

    resu = joblib.load(output_dir / "all_res.joblib")
    return


@app.cell(disabled=True)
def _(X_train_final, clf, config_models, extract_feat, keepVariableList):
    if config_models.extraction_type == "TSFEL":
        extract_feat.mesureImportance_tsfel(clf, X_train_final, keepVariableList, top_n=20, class_labels=["Survie", "Décès"], savefig = True, transparent = False)
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
def _(pd):
    # 1. Vos données (Modèles en lignes, Métriques en colonnes)
    data = {
        'Accuracy': [0.92, 0.95, 0.65],
        'F1-Score': [0.88, 0.91, 0.45],
        'MCC': [0.79, 0.86, 0.00]  # Votre nouveau favori !
    }
    model_names = ['Random Forest', 'XGBoost', 'Baseline (Dummy)']
    df = pd.DataFrame(data, index=model_names)

    from tabulate import tabulate

    # Export au format LaTeX (booktabs est le standard des publications scientifiques)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".3f"))
    return (tabulate,)


@app.cell
def _(df_clean, pl, test_df):
    df_clean_saps2 = (
        test_df
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
def _(Path, pd, plt, roc_auc_score, roc_curve, tabulate):
    def générer_rapport_comparatif(y_true, configurations, save_dir=None, table_format='fancy_grid', saps2_pred = None, saps2_true = None):
        """
        Génère un tableau comparatif et une courbe ROC unique à partir de scores et 
        de prédictions déjà calculés.

        y_true : tableau des vraies étiquettes (ex: y_test ou y_news)
        configurations : dictionnaire contenant les scores, prédictions et couleurs pour chaque modèle
        """
        results = {}

        # Configuration de la figure ROC
        plt.figure(figsize=(8, 8))
        for name, config in configurations:
            # Extraction des vecteurs précalculés
            probas = config['probas']
            y_pred = config['preds']

            # Calcul des métriques

            f1 = config['f1_score']
            mcc = config['mcc']
            auc = config['auc']
            brier = config ['brier']

            # Stockage pour le tableau
            results[name] = {
                'F1-Score': f1,
                'MCC': mcc,
                'AUC': auc,
                "brier" : brier
            }

            # Ajout à la courbe ROC collective
            fpr, tpr, _ = roc_curve(y_true, probas)
            color = config.get('color', None)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

        # 1. Génération du tableau avec tabulate
        df_results = pd.DataFrame(results).T
        print("\n=== TABLEAU COMPARATIF DES PERFORMANCES ===")
        print(tabulate(df_results, headers='keys', tablefmt=table_format, floatfmt=".3f"))
        # Calcul de l'AUC de IGS2 : 
        if saps2_pred is not None and saps2_true is not None:
            fpr_saps2, tpr_saps2, _ = roc_curve(saps2_true, saps2_pred)
            auc_saps2 = roc_auc_score(saps2_true, saps2_pred)
            name_saps2 = "Score IGS2"
            plt.plot(fpr_saps2, tpr_saps2, label=f'{name_saps2} (AUC = {auc_saps2:.3f})', lw=2, linestyle='-.')
        # 2. Finalisation de la courbe ROC
        plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color='gray')
        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.title('Comparaison des Courbes ROC')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        # Sauvegarde automatique des artefacts pour votre publi
        if save_dir:
            output_path = Path(save_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Sauvegarde de l'image
            plt.savefig(output_path / "courbe_roc_collective.png", dpi=300, bbox_inches="tight")

            # Sauvegarde du tableau au format LaTeX booktabs pour Overleaf
            with open(output_path / "tableau_resultats.tex", "w") as f:
                f.write(tabulate(df_results, headers='keys', tablefmt='latex_booktabs', floatfmt=".3f"))

        plt.show()

        return df_results

    return (générer_rapport_comparatif,)


@app.cell
def _(mo, run_test):
    mo.stop(not run_test.value, "Clique pour lancer")
    print("Comparaison lancée")
    return


@app.cell
def _(
    Path,
    config_balance,
    config_cleaning,
    config_y,
    générer_rapport_comparatif,
    joblib,
    modex,
    os,
    saps2_pred,
    saps2_true,
    str_pop,
    underscore,
    utils,
    y_test,
):
    def load_model(model_name):
        extension_hm = ".joblib" if "TSFEL" in model_name else ".pt" 
        base_res_pattern = f"models/{model_name}/{config_cleaning.clean}_{config_y.target_name}_{config_balance.balance_method}{underscore}{modex.value}{str_pop}_*{extension_hm}"
        loaded_output = utils.get_latest_model_path(base_res_pattern, extension_hm)
        output_directory = Path("outputs") / Path(model_name) / Path(loaded_output).stem

        return model_name, joblib.load(os.path.join(output_directory, "all_res.joblib"))

    comparaisons = [load_model("InceptionTimeModified"), load_model("LstmTimeModified"), load_model("RandomForest TSFEL"), load_model("XGBoost TSFEL"), load_model("SVC TSFEL")]

    comparaisons

    générer_rapport_comparatif(y_test, comparaisons, save_dir="Comparaison ALl", table_format='fancy_grid', saps2_pred = saps2_pred, saps2_true = saps2_true )
    return (load_model,)


@app.cell
def _(générer_rapport_comparatif, load_model, y_test):
    comparaisons_time = [load_model("InceptionTimeModified"), load_model("LstmTimeModified")]

    générer_rapport_comparatif(y_test, comparaisons_time, save_dir="Comparaison Time", table_format='fancy_grid')
    return


@app.cell
def _(générer_rapport_comparatif, load_model, y_test):
    comparaisons_ml = [load_model("RandomForest TSFEL"), load_model("XGBoost TSFEL"), load_model("SVC TSFEL")]
    générer_rapport_comparatif(y_test, comparaisons_ml, save_dir="Comparaison TSFEL", table_format='fancy_grid')
    return


@app.cell
def _():
    return


@app.cell
def _(
    balance,
    cleaning,
    custom_features,
    keep_feats,
    keep_pop,
    metric_name,
    mo,
    mo_utils,
    mode,
    models,
    modex,
    run,
    save_figure,
    seed,
    str_keep_feats,
    transparent,
    ui_tsfel,
    y_dd,
):
    mo.sidebar(
    mo.vstack([
        mo.md(mo_utils.config_sidebar),
        mo.md(f"<U>Seed utilisée pour l'ensemble du code : **{seed}**</U>"),
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
        metric_name,
        mo.md("-------------------------------"),
        run,
        mo.md("-------------------------------"),
        mo.md(mo_utils.config_end)]),
    width = "550px")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
