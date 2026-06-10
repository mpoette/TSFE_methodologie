import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Importation des bibliothèques
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import time
    import os
    import sys
    import json
    import tsfel
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna
    import math
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
        brier_score_loss,
        classification_report,
        matthews_corrcoef
    )
    from pathlib import Path
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    sys.path.append(os.path.abspath(".."))

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

    import utilitaries.preprocessing_utils as ui
    import utilitaries.marimo_utils as mo_utils
    import utilitaries.extract_data_utils as extract
    import utilitaries.preprocessing_utils as preproc
    import utilitaries.features_extraction_utils as extract_feat
    import utilitaries.optuna.optuna_utils as optuna_utils
    import utilitaries.utils as utils
    import utilitaries.create_merged_dataset as create_merged_dataset

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
        optuna,
        os,
        pd,
        pl,
        plt,
        predict_proba,
        predict_proba_lstm,
        preproc,
        roc_auc_score,
        roc_curve,
        sns,
        train_inception_time,
        train_lstm_model,
        utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Widgets Marimo utilisés dans ce notebook
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
def _(config_models, mo):
    run_search = mo.ui.run_button(
        label=f"Recherche d'hyperparamètres du modèle {config_models.models_name}"
    )
    return (run_search,)


@app.cell
def _(mo):
    save_figure = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Oui",
                                label = "Sauvegarder les figures")
    return (save_figure,)


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
def _(boruta_filter, config_models, mo):
    if config_models.extraction_type == "TSFEL" :
        extract_tsfel = mo.ui.dropdown(
            options = {"Oui" : True, "Non" : False},
            value = "Non",
            label = "Extraire les données TSFEL"
        )
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
def _(mo, mo_utils, save_figure, transparent):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        save_figure,
        mo.md("---"),
        transparent,
        mo.md(mo_utils.config_end)
    ])
    return


@app.cell
def _(save_figure):
    save_figure.value
    return


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
    ### Extraction du dataset
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
    ##### Chargement Dataframes et merge de statique et dynamique
    """)
    return


@app.cell
def _(dataset_path, os, pl):
    _path = os.path.join(dataset_path, "df_static_ano_clean.parquet")
    df_static = pl.scan_parquet(_path)

    df_static = df_static.with_columns(((1 - pl.col("sapsii_prob")).truediv(100)).alias("sapsii_prob"))
    return (df_static,)


@app.cell
def _(df_static):
    df_static.columns
    return


@app.cell
def _(dataset_path, extract, os):
    _path = os.path.join(dataset_path, 'df_dynamic_full_clean.parquet')
    df_dynamic = extract.extract_data_survie(_path)
    return (df_dynamic,)


@app.cell
def _(df_dynamic):
    df_dynamic.collect().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Dataframe dynamique
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


@app.cell
def _(df_clean):
    df_clean
    return


@app.cell
def _(df_clean):
    df_clean["score_glasgow"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO : vérifier le nombre de données qu'on filtre ici !
    """)
    return


@app.cell
def _(df_clean):
    df_clean
    return


@app.cell
def _(df_clean):
    df_clean["is_conscious"].value_counts()
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
def _(df_clean):
    df_clean.describe()
    return


@app.cell
def _(df_clean):
    df_clean.columns
    return


@app.cell
def _(dataset_path, os, pl):
    _path = os.path.join(dataset_path, "df_static_full_clean.parquet")
    df_static_full = pl.read_parquet(_path)
    df_static_full = df_static_full.with_columns(((1 - pl.col("sapsii_prob")).truediv(100)).alias("sapsii_prob"))
    return (df_static_full,)


@app.cell
def _(df_static_full):
    df_static_full.select('encounterNumber', "sapsii_prob")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transformation_Prétraitement
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Split train/test + scaling + reshape
    """)
    return


@app.cell
def _(balance, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        balance,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(keep_feats):
    keep_features = keep_feats.copy()
    return (keep_features,)


@app.cell
def _(extract):
    patient_col = extract.ID_COL
    time_col = extract.TIME_COL
    target_col = "isDeceased_lt_28d"
    expected_length = extract.WINDOW_SIZE
    return expected_length, patient_col, target_col, time_col


@app.cell
def _(df_clean_1, expected_length, patient_col, pl, time_col):
    # On garde les encounters de longueur exacte 
    valid_ids = df_clean_1.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
    df_clean_2 = df_clean_1.join(valid_ids, on=patient_col, how='inner')
    if df_clean_2.is_empty():
        raise ValueError("Aucun patient n'a exactement la longueur attendue.")
    df_clean_2 = df_clean_2.sort(patient_col, time_col)
    return (df_clean_2,)


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
    keep_features[:] = list(dict.fromkeys(keep_features))
    print(keep_features)
    X = df_clean_3.select(keep_features).to_numpy()
    y = df_clean_3[target_col].to_numpy()
    return X, df_clean_3, y


@app.cell
def _(df_clean_3):
    df_clean_3
    return


@app.cell
def _(df_clean_3):
    df_clean_3.columns
    return


@app.cell
def _(pl):
    pl.Config.set_tbl_cols(-1)
    return


@app.cell
def _(config_balance):
    underscore = "_"
    if config_balance.balance_method == "":
        underscore = ""
    return (underscore,)


@app.cell
def _(df_clean_3):
    (df_clean_3.describe())
    return


@app.cell
def _():
    # feat_temp = "urine_rate"
    # df_clean[feat_temp].value_counts().sort(by = feat_temp)
    return


@app.cell
def _():
    # # On filtre sur les valeurs aberrantes (ex: 0) et on compte combien de fois ça arrive par encounterId
    # df_clean_3.filter(pl.col(feat_temp) < 0).group_by("encounterId").len().sort(by="len", descending=True)
    return


@app.cell
def _():
    # _path = os.path.join(dataset_path, "dynamic_pivot_table.parquet")
    # df_dynamic_raw = pl.read_parquet(_path)
    return


@app.cell
def _():
    # df_dynamic_raw.describe()
    return


@app.cell
def _():
    # # 1. On isole et on renomme pour éviter les conflits de noms si besoin
    # df1 = df_clean_3.filter(pl.col("encounterId") == 58312).select(["urine_rate", "encounterId", "delta_hour"])
    # df2 = df_dynamic_raw.filter(pl.col("encounterId") == 58312).select(["urine_output", "encounterId", "delta_hour"]).cast(pl.Float64)

    # df1.join(df2, on = ["encounterId", "delta_hour"], how = "inner").sort(by = "delta_hour").select(["encounterId", "delta_hour", "urine_rate", "urine_output"])
    return


@app.cell
def _():
    # df_static_full2.collect().filter(pl.col("encounterNumber") == 536327814).select("encounterId")
    return


@app.cell
def _():
    # df_clean.filter(pl.col("encounterId") == 14937)
    return


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
    keep_features,
    modex,
    np,
    patient_col,
    pl,
    preproc,
    target_col,
    time_col,
    y,
):
    groups = df_clean_3[patient_col].to_numpy()
    # TODO cette ligne ne fonctionne que parce qu'on a des fenêtres qui sont toutes de 24h => Le poids entre les étiquettes est le même peu importe le patient qu'on prend ce qui permet un bon équilibrage train/test
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    (train_idx, test_idx) = next(sgkf.split(X=X, y=y, groups=groups))

    # Tri obligatoire (même si en théorie il est déjà fait)
    train_df = df_clean_3[train_idx].sort([patient_col, time_col])
    test_df = df_clean_3[test_idx].sort([patient_col, time_col])

    # DownSampling
    print("NB lignes train_df initial :", len(train_df))
    print("Distribution y initial :", train_df[target_col].value_counts())
    if config_models.extraction_type == "TSFEL" : 
        y_test = []
        y_train = []
        for subdf in test_df.partition_by(patient_col, maintain_order=True):
            y_test.append(subdf[target_col][0])  # un seul label par séquence

        for subdf in train_df.partition_by(patient_col, maintain_order=True):
            y_train.append(subdf[target_col][0])  # un seul label par séquence
         # On enlève les features non temporelles (y'en a qu'une je pense vu que les autres (même is_conscious) peuvent varier avec le temps)
        print(keep_features)
        train_df.describe()
        static_feats = ["admission_type_Medical", "admission_type_Scheduled Surgery", "admission_type_Unknown", "admission_type_Unscheduled Surgery", "score_glasgow", "age"]
        if extract_tsfel.value :
            tsfel_features = [c for c in keep_features if c not in static_feats]
            TSFEL_train_df = extract_feat.extract_tsfel_per_patient(train_df, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
            TSFEL_train_df.write_parquet(f"tsfel_train_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")
            TSFEL_test_df = extract_feat.extract_tsfel_per_patient(test_df, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col)
            TSFEL_test_df.write_parquet(f"tsfel_test_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")

        TSFEL_train_df = pl.read_parquet(f"tsfel_train_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")
        TSFEL_test_df = pl.read_parquet(f"tsfel_test_df_{config_mode.name}_{config_cleaning.clean}_{config_y.target_name}_{modex.value}.parquet")

        TSFEL_train_clean, TSFEL_test_clean, keepVariableList = extract_feat.filtrage_corr_var(TSFEL_train_df, TSFEL_test_df, patient_col, target_col)
        static_train = train_df.select([extract.ID_COL, *static_feats]).unique()
        static_test = test_df.select([extract.ID_COL, *static_feats]).unique()
        new_train_df = TSFEL_train_clean.join(static_train, on=extract.ID_COL, how="inner")
        new_test_df = TSFEL_test_clean.join(static_test, on=extract.ID_COL, how="inner")
        if boruta_filter.value:
            new_train_df, new_test_df, keepVariableList = extract_feat.filtrage_boruta( new_train_df, new_test_df, patient_col, target_col, max_iter = 100)
        new_train_df = preproc.equilibrer_dataset_tabulaire(new_train_df, extract.ID_COL, target_col, method = config_balance.balance_method )
        y_train = new_train_df[target_col].to_numpy()
        y_test = new_test_df[target_col].to_numpy()
        new_train_df = new_train_df.select(pl.exclude(patient_col, target_col))
        new_test_df = new_test_df.select(pl.exclude(patient_col, target_col))
        # On prépare le jeu d'entraînement
        (new_train_df_scale, new_test_df_scale) = preproc.scaling(new_train_df, new_test_df)

    elif config_models.extraction_type == "time" :
        # 1. Gestion exclusive du mode homemade si configuré
        if config_balance.balance_method in ["downsampling_homemade", ""]:
            train_df = preproc.equilibrer_dataset_tabulaire(train_df, extract.ID_COL, target_col, method = config_balance.balance_method)
        # On prépare le jeu d'entraînement
        (train_df, test_df) = preproc.scaling(train_df, test_df)
        (X_train, y_train) = preproc.build_sequences(train_df, patient_col, target_col, expected_length, keep_features)  # grouper en fonction d'un individu
        (X_test_3d, y_test) = preproc.build_sequences(test_df, patient_col, target_col, expected_length, keep_features)

        if config_balance.balance_method not in ["downsampling_homemade", ""]:
            # Comme X_train a 3 dimensions (N, 24, F), imblearn ne sait pas le lire.
            # Astuce : On l'aplatit temporairement en 2D (N, 24 * F)
            n_samples, n_timesteps, n_feats = X_train.shape
            X_train_2d = X_train.reshape(n_samples, n_timesteps * n_feats)

            if config_balance.balance_method == "downsampling_50-50":
                from imblearn.under_sampling import RandomUnderSampler
                rs = RandomUnderSampler(random_state=42)
            elif config_balance.balance_method == "upsampling_50-50":
                from imblearn.over_sampling import RandomOverSampler
                rs = RandomOverSampler(random_state=42)
            else:
                raise ValueError("Cet équilibrage n'a pas encore été implémenté")

            X_res_2d, y_train = rs.fit_resample(X_train_2d, y_train)

            # On redonne sa forme 3D d'origine au tenseur équilibré
            X_train = X_res_2d.reshape(-1, n_timesteps, n_feats)



        total_nan = np.isnan(X_train).sum()
        # Compte les NaN pour chaque feature (axe 2)
        nan_par_feature = np.isnan(X_train).sum(axis=(0, 1))

        # for i, feat_name in enumerate(keep_features):
            # print(f"Feature '{feat_name}' : {nan_par_feature[i]} NaN")
        print(f"Nombre total de valeurs NaN : {total_nan}")
        X_train = np.nan_to_num(X_train, nan=0.0)


    else:
        raise ValueError("Modèle inexistant/Pas implémenté")
    return (
        X_test_3d,
        X_train,
        keepVariableList,
        new_test_df_scale,
        new_train_df_scale,
        y_test,
        y_train,
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
    ### Training sur {config_models.models_name}
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
def _():
    return


@app.cell
def _(mo, mo_utils, run):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(config_keep_pop, config_models):

    # on créé un nom unique de modèle
    str_pop = ""
    if config_keep_pop.keep_population != "all_diseases":
        str_pop = "_"+config_keep_pop.keep_population

    extension = ".joblib" if config_models.extraction_type == "TSFEL" else ".pt"

    # optuna_stage_str = ""
    # optuna_str = ""

    # optuna_study_name = f"{config2.models_name}_{config.name}_{config3.clean}_{config4.target_name}_{config6.balance_method}{underscore}{modex.value}{str_pop}_{metric_name.value}"

    # optuna_storage = f"sqlite:///optuna_{config_models.models_name}.db"
    # optuna_stage = ["stage2", "stage1"]
    # default_params = {
    #             "epochs" : 100,
    #             "patience" : 10,
    #         }
    # continu = True

    # for optu_s in optuna_stage:
    #     if continu :
    #         parameters, continu = optuna_utils.get_params(f"{optuna_study_name}_{optu_s}", default_params, optuna_storage)
    #     else :
    #         optuna_stage_str = optu_s

    # if parameters != default_params:
    #     print("le modèle utilisé a été optimisé avec optuna !")
    #     optuna_str = f"optuna_{optuna_stage_str}"
    # else:
    print("le modèle utilisé n'a pas été optimisé par optuna... \n Application des paramètres par défaut")
    return extension, str_pop


@app.cell
def _(
    X_train,
    config_balance,
    config_cleaning,
    config_models,
    config_y,
    extension,
    joblib,
    mo,
    modex,
    new_train_df_scale,
    np,
    run,
    str_pop,
    train_inception_time,
    train_lstm_model,
    underscore,
    utils,
    y_train,
):
    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")

    model_path = utils.get_unique_path(
        f"models/{config_models.models_name}/{config_cleaning.clean}_{config_y.target_name}_"
        f"{config_balance.balance_method}{underscore}{modex.value}{str_pop}{extension}"
    )

    default_params = {
            "epochs" : 100,
            "patience" : 10,
        }

    parameters = default_params

    # print(f"{optuna_study_name}_{optu_s}")

    if config_models.models_name == "InceptionTimeModified":
        print(parameters)
        model, T, history, splits = train_inception_time(
            X_train, y_train,
            save_best_path=model_path,
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
        seed = 42
        rf = RandomForestClassifier(class_weight='balanced', random_state=seed)
        rf.fit(new_train_df_scale, y_train)
        joblib.dump(rf, model_path)
    elif config_models.models_name == "XGBoost TSFEL":
        from xgboost import XGBClassifier


        X_train_tsfel = new_train_df_scale.to_numpy()
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
            random_state=42,
            eval_metric="logloss",
            missing=np.nan,
        )

        xgb.fit(X_train_tsfel, y_train_tsfel)
        joblib.dump(xgb, model_path)

    elif config_models.models_name == "SVC TSFEL" : 
        from sklearn.svm import SVC
        svc = SVC(kernel = "rbf", C = 1.0, random_state = 42, class_weight = "balanced", probability = True)
        svc.fit(new_train_df_scale, y_train)
        joblib.dump(svc, model_path)
    else :
        print("oups tu t'es trompé")
    return (model_path,)


@app.cell
def _(model_path):
    print(model_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluation des résultats obtenus
    """)
    return


@app.cell
def _(
    X_test_3d,
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
    new_test_df_scale,
    new_train_df_scale,
    str_pop,
    underscore,
    utils,
    y_test,
    y_train,
):
    base_pattern = f"models/{config_models.models_name}/{config_cleaning.clean}_{config_y.target_name}_{config_balance.balance_method}{underscore}{modex.value}{str_pop}_*{extension}"  
    print(base_pattern)

    print(base_pattern, extension)

    loaded_model = utils.get_latest_model_path(base_pattern, extension)
    # loaded_model = model_path

    print("Modèle chargé :", loaded_model)

    if config_models.models_name == "InceptionTimeModified":
        (_auc, brier, T_1) = evaluate_on_test(X_test_3d, y_test, loaded_model)
        (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)

    elif config_models.models_name == "LstmTimeModified":
        (_auc, brier, T_1) = evaluate_lstm_on_test(X_test_3d, y_test, loaded_model)
        (model_1, _, T_1) = load_lstm_from_checkpoint(loaded_model)
    elif config_models.extraction_type == "TSFEL":
        clf = joblib.load(loaded_model)
        y_pred_nb_train = clf.predict(new_train_df_scale)
        y_pred_nb_test = clf.predict(new_test_df_scale)
        train_score=clf.score(new_train_df_scale,y_train)
        test_score=clf.score(new_test_df_scale,y_test)
        print(f"Le score sur les données de test est {test_score}")
        print(classification_report(y_test, y_pred_nb_test, target_names=["Alive", "Deceased"], zero_division=0))
    else :
        print("erreur de choix de modèle")
    return T_1, clf, loaded_model, model_1


@app.cell
def _(Path, config_models, loaded_model):
    # construire le dossier output correspondant
    output_dir = Path("outputs") / Path(config_models.models_name) / Path(loaded_model).stem

    # créer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _(Path, loaded_model):
    print(Path(loaded_model).stem)
    return


@app.cell
def _(
    T_1,
    X_test_3d,
    clf,
    config_models,
    model_1,
    new_test_df_scale,
    predict_proba,
    predict_proba_lstm,
):
    # c'est la même fonction pour les 2 modèles donc c'est ok
    if config_models.models_name == "InceptionTimeModified":
        probas = predict_proba(model_1, X_test_3d, T=T_1)
    elif config_models.models_name == "LstmTimeModified":
        probas = predict_proba_lstm(model_1, X_test_3d, T=T_1)
    elif config_models.extraction_type == "TSFEL":
        all_probas = clf.predict_proba(new_test_df_scale.to_numpy())
        classes = list(clf.classes_)
        positive_idx = classes.index(1)
        probas = all_probas[:, positive_idx]
    return (probas,)


@app.cell
def _(
    Path,
    config_models,
    df_clean,
    output_dir,
    plt,
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
    df_temp =  df_clean.group_by("encounterId")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC {config_models.models_name} (AUC = {auc:.3f})')
    df_news_patient = (
        df_clean
        .sort(["encounterId", "heure_calibree"])
        .group_by("encounterId")
        .last()
    )

    # Extraire y_true et NEWS
    # y_news = df_news_patient[target_col].to_numpy()   

    # news_score = df_news_patient["news"].to_numpy()
    # news2_score = df_news_patient["news2"].to_numpy()
    # fpr_news, tpr_news, _ = roc_curve(y_news, news_score)
    # auc_news = roc_auc_score(y_news, news_score)
    # plt.plot(fpr_news, tpr_news, label=f'ROC NEWS (AUC = {auc_news:.3f})', color = "orange")
    # fpr_news2, tpr_news2, _ = roc_curve(y_news, news2_score)
    # auc_news2 = roc_auc_score(y_news, news2_score)
    # plt.plot(fpr_news2, tpr_news2, label=f'ROC NEWS2 (AUC = {auc_news2:.3f})', color = "#E07604")

    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color = "green")
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

    # 3. Ça marche directement comme un dictionnaire Python standard !
    print(resu)
    print(f"Mon MCC est de : {resu['mcc']:.4f}")
    return


@app.cell(hide_code=True)
def _(config_models, mo):
    mo.md(rf"""
    ### Recherche des meilleurs hyperparamètres de {config_models.models_name} avec Optuna
    """)
    return


@app.cell
def _(mo, mo_utils, run_search):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run_search,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(mo, run_search):
    mo.stop(not run_search.value, "Clique pour lancer")
    res = 3
    return (res,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Création de l'étude Optuna
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    config_balance,
    config_cleaning,
    config_models,
    config_modename,
    config_y,
    metric_name,
    modex,
    optuna,
    res,
    str_pop,
    underscore,
):
    res2 = res + 1
    study_name = f"{config_models.models_name}_{config_modename}_{config_cleaning.clean}_{config_y.target_name}_{config_balance.balance_method}{underscore}{modex.value}{str_pop}_{metric_name.value}"
    storage = f"sqlite:///optuna_{config_models.models_name}.db"
    stage = ["stage2", "stage1"]
    study = optuna.create_study(
        study_name="test_optuna",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stage 1 : recherche large des meilleurs hyperparamètres, en en fixant tout de même certains
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # if config_models.models_name == "InceptionTimeModified" :
    #     import utilitaries.optuna.optuna_inception_utils as optuna_inception
    #     fixed_params = {
    #     "val_ratio": 0.2,
    #     "epochs": 100,
    #     "patience": 10,
    #     "min_delta": 0.0,
    #     "calibrate": False,
    #     "device": "cuda",
    #     }


    #     study_stage1 = optuna_inception.run_stage1_search(
    #         X_train,
    #         y_train,
    #         n_trials=40,
    #         study_name=study_name + "_" + stage[1], 
    #         storage=storage,
    #         metric_name="val_loss",
    #         fixed_params=fixed_params,
    #     )

    # elif config_models.models_name == "LstmTimeModified" :
    #     import utilitaries.optuna.optuna_lstm_utils as optuna_lstm
    #     fixed_params = {
    #     "val_ratio": 0.2,
    #     "epochs": 100,
    #     "patience": 10,
    #     "min_delta": 0.0,
    #     "calibrate": False,
    #     "device": "cuda",
    #     }
    #     study_stage1 = optuna_lstm.run_lstm_stage1_search(
    #         X_train,
    #         y_train,
    #         n_trials = 40,
    #         study_name = study_name + "_" + stage[1],
    #         storage = storage,
    #         metric_name = metric_name.value,
    #         fixed_params = fixed_params,
    #     )

    # else :
    #     pass
    return


@app.cell
def _():
    #### Stage 2 : recherche plus fine autour des valeurs déjà trouvées
    return


@app.cell
def _():
    # study_stage1_load = optuna.load_study(
    #         study_name= study_name + "_" + stage[1],
    #         storage=storage
    #     )
    # if config_models.models_name == "InceptionTimeModified" :
    #     study_stage2 = optuna_inception.run_stage2_search(
    #         X_train,
    #         y_train,
    #         study_stage1_load,
    #         n_trials = 25,
    #         study_name = study_name + "_" + stage[0],
    #         storage=storage,
    #         metric_name = metric_name.value,
    #         fixed_params = fixed_params
    #         )

    # elif config_models.models_name == "LstmTimeModified" :
    #     study_stage2 = optuna_lstm.run_lstm_stage2_search(
    #         X_train,
    #         y_train,
    #         study_stage1_load,
    #         n_trials = 25,
    #         study_name = study_name + "_" + stage[0],
    #         storage=storage,
    #         metric_name =  metric_name.value,
    #         fixed_params = fixed_params
    #         )

    # else :
    #     pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### On regarde les features les plus déterminantes
    """)
    return


@app.cell(disabled=True)
def _(clf, config_models, extract_feat, keepVariableList, new_train_df_scale):
    if config_models.extraction_type == "TSFEL":
        extract_feat.mesureImportance_tsfel(clf, new_train_df_scale, keepVariableList, top_n=20, class_labels=["Survie", "Mort"], savefig = True, transparent = False)
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
def _(Path, pd, plt, roc_curve, tabulate):
    def générer_rapport_comparatif(y_true, configurations, save_dir=None, table_format='fancy_grid'):
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
    joblib,
    modex,
    os,
    str_pop,
    underscore,
    utils,
):

    model_name = "RandomForest TSFEL"
    model_name = "InceptionTimeModified"
    def load_model(model_name):
        extension_hm = ".joblib" if "TSFEL" in model_name else ".pt" 
        base_res_pattern = f"models/{model_name}/{config_cleaning.clean}_{config_y.target_name}_{config_balance.balance_method}{underscore}{modex.value}{str_pop}_*{extension_hm}"
        loaded_output = utils.get_latest_model_path(base_res_pattern, extension_hm)
        output_directory = Path("outputs") / Path(model_name) / Path(loaded_output).stem

        return model_name, joblib.load(os.path.join(output_directory, "all_res.joblib"))

    comparaisons = [load_model("InceptionTimeModified"), load_model("LstmTimeModified"), load_model("RandomForest TSFEL"), load_model("XGBoost TSFEL"), load_model("SVC TSFEL")]

    comparaisons

    return (comparaisons,)


@app.cell
def _(comparaisons, générer_rapport_comparatif, y_test):
    générer_rapport_comparatif(y_test, comparaisons, save_dir="", table_format='fancy_grid')
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
    run_search,
    save_figure,
    str_keep_feats,
    transparent,
    ui_tsfel,
    y_dd,
):
    mo.sidebar(
    mo.vstack([
        mo.md(mo_utils.config_sidebar),
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
        # mo.md(f" \n \n **Modification Thesaurus** : Afin d'avoir des valeurs cohérentes, avec une bonne imputation notamment, j'ai rajouté la pression artérielle systolique ainsi que la fréquence respiratoire. Il faudra voir aussi si on laisse les valeurs par défaut à 0 ou non. J'ai pris le parti pris pour la pas et fr de mettre en valeur par défaut une valeur qui fait un score de 0 sur news, sinon ça augmenterait le score juste parce qu'on a pas l'info ce qui n'est pas optimal... J'ai donc 130 pour pas en imputation method ffill_bfill et 16 pour fr en ffill_bfill aussi. Je me suis rendu compte que la valeur par défaut de heart_rate et spo2 était aussi de 0. Cela classe donc instantanément le patient en grave, alors qu'on a juste pas l'information... j'ai mis pour heart_rate une valeur par défaut de 60 et un spo2 de 96%. Je pense qu'il faudra qu'on fasse un point sur les valeurs par défaut du thesaurus car la majorité sont à 0, ce qui peut poser problème"),
        # mo.md("-------------------------------"),
        run_search,
        mo.md(mo_utils.config_end)]),
    width = "550px")
    return


if __name__ == "__main__":
    app.run()
