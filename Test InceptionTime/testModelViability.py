import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    config_dropdown_color = "<div style='background:#F0FFD4;padding:8px;border-radius:6px'>"

    config_run_button = "<div style='background:#FFFA7A;padding:8px;border-radius:6px'>"
    return config_dropdown_color, config_run_button


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Widgets Marimo
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ConfigFenetrage:
        name : str
        hour_offset: int
        random: bool
        max_hour: int
        strict_mode: bool
        used_distribution : str

    MODES = {
        "24h début réanimation avec remplissage": ConfigFenetrage(
            name = "24h_debut_rea_fill",
            hour_offset = 0,
            random = False,
            max_hour = 12,
            strict_mode = False,
            used_distribution = "uniform",
        ),
        "24h début réanimation sans remplissage": ConfigFenetrage(
            name = "24h_debut_rea_no-fill",
            hour_offset = 0,
            random = False,
            max_hour = 12,
            strict_mode = True,
            used_distribution = "uniform",
        ),
        "24h aléatoire 'real' avec remplissage": ConfigFenetrage(
            name = "24h_alea_real_fill",
            hour_offset = 0, # pas utilisé en pratique
            random = True,
            max_hour = 12,
            strict_mode = False,
            used_distribution = "real",
        ),
        "24h aléatoire 'real' sans remplissage": ConfigFenetrage(
            name = "24h_alea_real_no-fill",
            hour_offset = 0,
            random = True,
            max_hour = 12,
            strict_mode = True,
            used_distribution = "real",
        ),
    }

    @dataclass(frozen=True)
    class ConfigModels:
        models_name : str
    MODELS = {
        "InceptionTimeModified" : ConfigModels(
            models_name = "InceptionTimeModified"
        )
    }

    @dataclass(frozen=True)
    class ConfigCleaning:
        clean : bool
    CLEAN = {
        "Enlever Surveillance Continue" : ConfigCleaning(
            clean = True
        ),
        "Garder le dataset intact" : ConfigCleaning(
            clean = False
        ),
    }


    @dataclass(frozen=True)
    class ConfigY:
        target_name : str
    Y = {
        "Survie à 24 heures" : ConfigY(
            target_name = "isDeceased_lt_24h"
        ),
        "Survie à 7 jours" : ConfigY(
            target_name = "isDeceased_lt_7d"
        ),
        "Survie à 28 jours" : ConfigY(
            target_name = "isDeceased_lt_28d"
        ),
        "Survie à 3 mois" : ConfigY(
            target_name = "isDeceased_lt_3m"
        ),
        "Survie (isDeceased)" : ConfigY(
            target_name = "isDeceased")

    }

    all_features = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf', 'fr', 'pas']


    @dataclass(frozen=True)
    class ConfigFeatures:
        keep_feats : list
    FEAT = {
        "Mode classique" : ConfigFeatures(
            keep_feats = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf']
        ),
        "Mode NEWS" : ConfigFeatures(
            keep_feats = ["fio2_corr", "fr", "spo2", "temp", "is_conscious", "pas", "heart_rate"]
        ),
        "Mode Custom" : ConfigFeatures(
            keep_feats= ['heure_calibree', 'pam', 'pad', 'heart_rate']
        )
    }



    dico_terme = {
        "fr" : "Fréquence Respiratoire",
        "pas" : "Pression Artérielle Systolique",
        "pam" : "Pression Artérielle Moyenne",
        "heure_calibree" : "Heure relative à la fin du séjour",
        "pad" : "Pression Artérielle Diastolique",
        "heart_rate" : "Fréquence Cardiaque",
        "spo2" : "Saturation en Oxygène",
        "temp" : "Température",
        "fio2_corr" : "Fraction Inspirée en Oxygène (calculée)",
        "glyc_cap" : "Glycémie Capilaire",
        "nad_dose_poids" : "Dosage de nicotinamide adénine di-nucléotide",
        "is_ventilated" : "Patient avec ventilation invasive ou non",
        "is_conscious" : "Conscient ou non",
        "is_sedated" : "Sédaté ou non",
        "is_not_alert" : "Alerte ou non",
        "age" : "Age",
        "creat" : "quantité de créatine",
        "num_plq" : "numération plaquettaire", 
        "bili_tot" : "Billirubine totale",
        "tp" : "Taux de prothrombine",
        "abs_dialyse" : "Dialyse ou non",
        "dialyse_hdi" : "Dialyse HDI ou non",
        "dialyse_cvvhf" : "Dialyse CVVHF ou non",
    }

    return CLEAN, FEAT, MODELS, MODES, Y, all_features, dico_terme


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Importation des bibliothèques
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import polars as pl
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
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    from Extraction import extract
    from Transformation_Pretraitement import preprocessing_polars
    from inceptionTimeModified import (
        evaluate_on_test,
        load_model_from_checkpoint,
        predict_proba,
        train_inception_time,
    )
    import utils_inception as ui
    import old_utils_inception as oui

    return (
        StratifiedGroupKFold,
        calibration_curve,
        confusion_matrix,
        evaluate_on_test,
        extract,
        f1_score,
        load_model_from_checkpoint,
        np,
        pd,
        pl,
        plt,
        predict_proba,
        preprocessing_polars,
        roc_auc_score,
        roc_curve,
        sns,
        train_inception_time,
        ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Extraction
    """)
    return


@app.cell
def _(pl):
    pl.Config.set_tbl_cols(-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Dataframe statique
    """)
    return


@app.cell
def _(pl, ui):
    _path = '../Datasets/clean_full_static_ano.parquet'
    df_static = pl.read_parquet(_path)
    df_static = df_static.with_columns(pl.col(ui.patient_col).cast(pl.Int32))
    # df_static = df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]))
    return (df_static,)


@app.cell
def _(df_static):
    print(df_static.height)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A partir d'ici, le notebook devient intéractif. On peut :

    - Sélectionner le type de calcul pour la fenêtre
    - Sélectionner si on veut enlever ou non la surveillance continue
    - Selectionner le modèle utilisé
    """)
    return


@app.cell
def _(CLEAN, config_dropdown_color, mo):
    cleaning = mo.ui.dropdown(
        options=CLEAN,
        value = "Enlever Surveillance Continue",
        label = "Nettoyage ou non des patients en surveillance continue",
    )
    mo.vstack([
        mo.md(config_dropdown_color),
        cleaning,
        mo.md("</div>")])
    return (cleaning,)


@app.cell
def _(cleaning):
    config3 = cleaning.value
    return (config3,)


@app.cell
def _(config3, df_static, pl):
    if config3.clean :
        print(df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]).not_()).height)

        df_static_bis = df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]))

        print(df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	])).height)

    else : 
        df_static_bis = df_static
    return (df_static_bis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On enlève les patients qui viennent de services avec peu de décès, afin d'équilibrer le dataset un peu mieux
    """)
    return


@app.cell
def _(df_static_bis, pl):
    df_static_1 = df_static_bis.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days')).filter((pl.col('deces_datediff_days') >= 0) | pl.col('deces_datediff_days').is_null())
    return (df_static_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO : trouver un moyen d'appliquer le filtre sur <-1 sachant que 3 outliers n'en sont pas réellement.
    """)
    return


@app.cell
def _(df_static_1, pl):
    df_static_2 = df_static_1.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days'))
    return (df_static_2,)


@app.cell
def _(df_static_2):
    df_static_2['isDeceased'].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Dataframe dynamique
    """)
    return


@app.cell
def _(extract):
    _path = '../Datasets/df_with_calculated_features.parquet'
    df_test = extract.extract_data_survie(_path)
    return (df_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Transformation_Prétraitement
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il faudra changer hour_offset pour pouvoir prendre une date fixe et non juste un temps en arrière
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ajout de la colonne age qui est dans le thesaurus
    """)
    return


@app.cell
def _(df_static_2, df_test, ui):
    df_test_1 = df_test.join(df_static_2[[ui.patient_col, 'age']], on=ui.patient_col, how='left')
    return (df_test_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On utilise ici le preprocessing de Gabrielle mais avec l'optimisation polars réalisée par mes soins, puisque l'ancien code mettait beaucoup trop de temps à tourner.
    """)
    return


@app.cell
def _(MODES, config_dropdown_color, mo):
    mode = mo.ui.dropdown(
        options=MODES,
        value="24h début réanimation avec remplissage",
        label="Mode de fonctionnement du fenêtrage",
    )
    mo.vstack([
        mo.md(config_dropdown_color),
        mode,
        mo.md("</div>")])
    return (mode,)


@app.cell
def _(mode):
    config = mode.value
    return (config,)


@app.cell
def _(config, df_test_1, preprocessing_polars):
    df_clean = preprocessing_polars.prepare_data(df_test_1, 
                                                 hour_offset = config.hour_offset, 
                                                 random = config.random, 
                                                 max_hour = config.max_hour,
                                                 used_distribution = config.used_distribution, 
                                                 strict_mode = config.strict_mode)
    df_clean
    return (df_clean,)


@app.cell
def _(df_clean, pl):
    df_with_idx = df_clean.with_row_index("idx")

    idx = (

        df_with_idx

        .filter(pl.col("fio2_corr").is_null())

        .select("idx")

    )

    print(len(idx))

    df_clean.filter(pl.col("fio2_corr").is_null())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il faut rajouter isDeceased sinon on n'a pas de Y
    """)
    return


@app.cell
def _(df_clean, df_static_2, ui):
    df_clean_1 = df_clean.join(df_static_2[[ui.patient_col, 'isDeceased', 'deces_datediff_days']], on=ui.patient_col, how='inner')
    return (df_clean_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On rajoute 3 features qui valent 1 si :

    - Pour isDeceased_lt_24h : 1 si décès inférieur à 24h après la fin de la fenêtre et 0 sinon

    - Pour isDeceased_lt_28d : 1 si décès inférieur à 28 jours après la fin de la fenêtre et 0 sinon

    - Pour isDeaceased_lt_3m : 1 si décès inférieur à 3 mois après la fin de la fenêtre et 0 sinon

    - Si les 3 features précédentes sont falses, alors le patient est vivant ou mort après 3 mois
    """)
    return


@app.cell
def _(df_clean_1, pl, ui):
    df_clean_24h = df_clean_1.with_columns(pl.when((pl.col('deces_datediff_days') * 24 + pl.col('heure_calibree').min().over(ui.patient_col)) < 24).then(pl.lit(True)).otherwise(pl.lit(False)).alias("isDeceased_lt_24h"))
    return (df_clean_24h,)


@app.cell
def _(df_clean_24h, pl, ui):
    df_clean_28d = df_clean_24h.with_columns(pl.when((pl.col('deces_datediff_days') * 24 + pl.col('heure_calibree').min().over(ui.patient_col)) < 672).then(pl.lit(True)).otherwise(pl.lit(False)).alias("isDeceased_lt_28d"))
    return (df_clean_28d,)


@app.cell
def _(df_clean_28d, pl, ui):
    df_clean_2m = df_clean_28d.with_columns(pl.when((pl.col('deces_datediff_days') * 24 + pl.col('heure_calibree').min().over(ui.patient_col)) < 2190).then(pl.lit(True)).otherwise(pl.lit(False)).alias("isDeceased_lt_3m"))
    return (df_clean_2m,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ajout d'une métrique pour regarder le décès inférieur à 7 jours afin de pouvoir être comparé à NEWS et NEWS2
    """)
    return


@app.cell
def _(df_clean_2m, pl, ui):
    df_clean_7d = df_clean_2m.with_columns(pl.when((pl.col('deces_datediff_days') * 24 + pl.col('heure_calibree').min().over(ui.patient_col)) < 168).then(pl.lit(True)).otherwise(pl.lit(False)).alias("isDeceased_lt_7d"))
    return (df_clean_7d,)


@app.cell
def _(df_clean_7d, pl, ui):
    df_clean_2 = df_clean_7d.with_columns(pl.when((pl.col('deces_datediff_days') * 24 + pl.col('heure_calibree').min().over(ui.patient_col)) >= 2190).then(pl.lit(True)).otherwise(pl.lit(False)).alias("isDeceased_gt_3m"))
    return (df_clean_2,)


@app.cell
def _(df_clean_2):
    df_clean_2['isDeceased_lt_24h'].describe()
    return


@app.cell
def _(df_clean_2):
    df_clean_2['isDeceased_lt_28d'].describe()
    return


@app.cell
def _(df_clean_2):
    df_clean_2['isDeceased_lt_3m'].describe()
    return


@app.cell
def _(df_clean_2):
    df_clean_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    à titre indicatif :
    """)
    return


@app.cell
def _(df_clean_2, pl, ui):
    print("nombre d'enregistrement de patients vivants (isDeceased)", df_clean_2.filter(pl.col('isDeceased') == False).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients vivants (DeceasedTimeType)", df_clean_2.filter((pl.col('isDeceased_lt_24h') == False)
                                                                                                 & (pl.col("isDeceased_lt_28d") == False)
                                                                                                 & (pl.col("isDeceased_lt_3m") == False)
                                                                                             & (pl.col("isDeceased_gt_3m") == False)).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts (isDeceased)", df_clean_2.filter(pl.col('isDeceased') == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts (4 features)", df_clean_2.filter((pl.col('isDeceased_lt_24h') == True)
                                                                                                 | (pl.col("isDeceased_lt_28d") == True)
                                                                                                 | (pl.col("isDeceased_lt_3m") == True)
                                                                                                 | (pl.col("isDeceased_gt_3m") == True)).select(pl.col(ui.patient_col).n_unique()).item())

    print("nombre d'enregistrement de patients morts moins de 24 heures après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_24h') == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 28 jours après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_28d") == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_3m") == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts plus de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_gt_3m') == True).select(pl.col(ui.patient_col).n_unique()).item())
    return


@app.cell
def _(MODELS, config_dropdown_color, mo):
    models = mo.ui.dropdown(
        options=MODELS,
        value="InceptionTimeModified",
        label="Modèle utilisé",
    )
    mo.vstack([
        mo.md(config_dropdown_color),
        models,
        mo.md("</div>")])
    return (models,)


@app.cell
def _(models):
    config2 = models.value
    return (config2,)


@app.cell
def _(df_clean_2):
    print(df_clean_2.columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Calcul de NEWS et NEWS2 : nos baselines

    En soi je peux faire un score par heure donc 24 scores.
    """)
    return


@app.cell
def _(df_clean_2, df_test_1, ui):
    df_clean_3 = df_clean_2.join(df_test_1[["encounterId","fr", "pas"]], on = ui.patient_col, how = "inner")
    return (df_clean_3,)


@app.cell
def _(df_clean_3, pl):
    df_clean_48 = df_clean_3.with_columns(
        # Supplémentation en oxygène
        (pl.when(pl.col("fio2_corr") != 21).then(2).otherwise(0)
         +
        # Fréquence respiratoire
        pl.when((pl.col("fr") <= 8) |
               (pl.col("fr") >=25)).then(3)
            .when((pl.col("fr") >= 21)
                  & (pl.col("fr") <= 24)).then(2)
            .when((pl.col("fr")) <= 20 
                  & (pl.col("fr") >= 12)).then(0)
            .otherwise(1)
        +
        # Saturation en oxygène
        pl.when((pl.col("spo2") <= 91.0)).then(3)
        .when((pl.col("spo2") >= 92.0)
            & (pl.col("spo2") <= 93.0)).then(2)
        .when((pl.col("spo2") >= 94.0)
             & (pl.col("spo2") <= 95)).then(1)
        .otherwise(0)
        +
        # Température
        pl.when((pl.col("temp") <= 35.0)).then(3)
        .when(((pl.col('temp') >= 35.1)
             & (pl.col("temp") <= 36.0))
             |
             ((pl.col("temp") <= 39.0)
             & (pl.col("temp") >= 38.1))).then(1)
        .when((pl.col("temp") >= 39.1)).then(2)
         .otherwise(0)
         +
         # Consience
         pl.when((pl.col("is_conscious")) == 0).then(0)
         .otherwise(3)
         +
         # Pression Artérielle Systolique
         pl.when((pl.col("pas") <= 90.0)
                | (pl.col('pas') >= 220.0)).then(3)
         .when((pl.col("pas") <= 110.0)
              & (pl.col("pas") >= 101.0)).then(1)
         .when((pl.col("pas") <= 219.0)
              & (pl.col("pas") >= 111)).then(0)
         .otherwise(2)
         +
         # Fréquence cardiaque
         pl.when((pl.col("heart_rate") <= 40)
                |(pl.col("heart_rate") >= 131)).then(3)
         .when((pl.col("heart_rate") <= 90) 
              & (pl.col("heart_rate") >= 51)).then(0)
         .when((pl.col("heart_rate") >= 111)
              & (pl.col("heart_rate") <= 130)).then(2)
         .otherwise(1)
        ).alias("news"))
    return


@app.cell
def _():
    # df_clean_3 = df_clean2.with_columns(
    #     pl.when(pl.col("fio2_corr") != 21 then )
    # )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Préparation pour InceptionTime
    """)
    return


@app.cell
def _(FEAT, mo):
    modex = mo.ui.dropdown(
        options = list(FEAT.keys()),
        value = "Mode classique",
        label = "Choix du mode",
    )
    return (modex,)


@app.cell
def _(FEAT, all_features, mo):
    custom_features = mo.ui.multiselect(
        options=all_features,
        value= FEAT["Mode Custom"].keep_feats,
        label="(features sélectionnables)",
    )

    return (custom_features,)


@app.cell
def _(FEAT, config_dropdown_color, custom_features, dico_terme, mo, modex):

    if modex.value =="Mode classique":
        keep_feats = FEAT["Mode classique"].keep_feats
    elif modex.value == "Mode NEWS":
        keep_feats = FEAT["Mode NEWS"].keep_feats
    else:
        keep_feats = custom_features.value

    str_keep_feats = ""

    for kf in keep_feats:
        str_keep_feats += f"- {dico_terme[kf]} \n"
    mo.vstack([
        mo.md(config_dropdown_color),
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français :** \n{str_keep_feats}"),
        mo.md("</div>")
    ])
    return (keep_feats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Split train/test + scaling + reshape
    """)
    return


@app.cell
def _(Y, config_dropdown_color, mo):
    y_dd =  mo.ui.dropdown(
        options=Y,
        value="Survie (isDeceased)",
        label="Cible (y) que le modèle doit prédire",
    )
    mo.vstack([
        mo.md(config_dropdown_color),
        y_dd,
        mo.md("</div>")])
    return (y_dd,)


@app.cell
def _(y_dd):
    config4 = y_dd.value
    return (config4,)


@app.cell
def _(config4):
    print(config4.target_name)
    return


@app.cell
def _(df_clean_2, target_col):
    print(df_clean_2[target_col].describe())
    return


@app.cell
def _(test_df):
    test_df
    return


@app.cell
def _(StratifiedGroupKFold, config4, df_clean_2, keep_feats, pl, ui):
    keep_features = keep_feats
    patient_col = ui.patient_col
    time_col = ui.time_col
    target_col = config4.target_name
    expected_length = 24
    valid_ids = df_clean_2.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
    # On garde les encounters de longueur exacte 
    df_clean_4 = df_clean_2.join(valid_ids, on=patient_col, how='inner')
    if df_clean_4.is_empty():
        raise ValueError("Aucun patient n'a exactement la longueur attendue.")
    df_clean_4 = df_clean_4.sort(patient_col, time_col)
    X = df_clean_4.select(keep_features).to_numpy()
    y = df_clean_4[target_col].to_numpy()
    groups = df_clean_4[patient_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    (train_idx, test_idx) = next(sgkf.split(X=X, y=y, groups=groups))
    # Tri obligatoire (même si en théorie il est déjà fait)
    train_df = df_clean_4[train_idx].sort([patient_col, time_col])
    test_df = df_clean_4[test_idx].sort([patient_col, time_col])
    # On prépare le jeu d'entraînement
    (train_pd, test_pd) = ui.scaling(train_df, test_df, target_col)
    (train_df, test_df) = (pl.from_pandas(train_pd), pl.from_pandas(test_pd))
    (X_train_3d, y_train_seq) = ui.build_sequences(train_df, patient_col, target_col, expected_length, keep_features)  # grouper en fonction d'un individu
    # On prend un split (comme train/test mais adapté aux individus)
    # Normalement pas besoin de sort mais soyons prudents...
    # Ok maintenant, on applique le scaler sur le dataframe
    # On retransforme en df polars
    # On construit la séquence attendue (N, T, F) à partir des deux dataframes train/test

    (X_test_3d, y_test_seq) = ui.build_sequences(test_df, patient_col, target_col, expected_length, keep_features)

    print(y_test_seq)
    return (
        X_test_3d,
        X_train_3d,
        df_clean_4,
        patient_col,
        target_col,
        test_df,
        train_df,
        y_test_seq,
        y_train_seq,
    )


@app.cell
def _(df_clean_4, patient_col, pl, target_col):
    df_clean_4.select(pl.exclude(target_col, patient_col, "DeceasedTimeType", "isDeceased_lt_24h", "isDeceased_lt_28d", "isDeceased_lt_7d", "isDeceased_lt_3m", "isDeceased_gt_3m", "deces_datediff_days")).columns
    return


@app.cell
def _(train_df, ui):
    train_df.group_by(ui.patient_col).len().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Attention : ici, train_df contient toutes les features et pas seulement celles qui ont été conservées. C'est normal et n'impacte pas les résultats du modèle puisque lui ne reçoit que les bonnes features (en théorie)
    """)
    return


@app.cell
def _(train_df):
    train_df.describe()
    #
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Training sur InceptionTime
    """)
    return


@app.cell
def _(config, config2, config3, config4, config_dropdown_color, mo, modex):
    mo.vstack([
        mo.md(config_dropdown_color),
        mo.md(f"### Entraînement avec les paramètres suivants : \n - typeFenêtrage = {config.name} \n - Modèle utilisé = {config2.models_name} \n - Nettoyage des Surveillances Continues = {config3.clean} \n - Cible à prédire = {config4.target_name} \n - Mode de features = {modex.value}"),
        mo.md("</div>")
    ])
    return


@app.cell
def _(config2, config_run_button, mo):
    import time
    run = mo.ui.run_button(
        label=f"Lancer l'entraînement du modèle {config2.models_name}"
    )
    mo.vstack([
        mo.md(config_run_button),
        run,
        mo.md("</div>")])
    return (run,)


@app.cell
def _(
    X_train_3d,
    config,
    config2,
    config3,
    config4,
    mo,
    modex,
    run,
    train_inception_time,
    y_train_seq,
):
    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")
    if config2.models_name == "InceptionTimeModified":
        model, T, history, splits = train_inception_time(
            X_train_3d, y_train_seq,
            num_blocks = 6,
            out_channels = 32,
            bottleneck_channels = 8,
            kernel_sizes = 21,
            batch_size = 16,
            lr = 0.0009572131781501278,
            weight_decay = 1.216426840149487e-06,
            clip_grad = 0.5,
            use_scheduler = False,
            epochs=100,
            patience=10,
            save_best_path=f"models/{config.name}_{config2.models_name}_{config3.clean}_{config4.target_name}_{modex.value}.pt"
        )
    else :
        print("oups tu t'es trompé")
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
    config,
    config2,
    config3,
    config4,
    evaluate_on_test,
    load_model_from_checkpoint,
    modex,
    y_test_seq,
):
    loaded_model = f"models/{config.name}_{config2.models_name}_{config3.clean}_{config4.target_name}_{modex.value}.pt"
    (_auc, brier, T_1) = evaluate_on_test(X_test_3d, y_test_seq, loaded_model)
    (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)
    return T_1, model_1


@app.cell
def _(T_1, X_test_3d, model_1, predict_proba):
    probas = predict_proba(model_1, X_test_3d, T=T_1)
    print(probas)
    return (probas,)


@app.cell
def _(plt, probas, roc_auc_score, roc_curve, y_test_seq):
    (fpr, tpr, _thresholds) = roc_curve(y_test_seq, probas)
    _auc = roc_auc_score(y_test_seq, probas)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(plt, probas, sns, y_test_seq):
    plt.figure()

    sns.kdeplot(probas[y_test_seq == 0], label="Survivants", fill=True)
    sns.kdeplot(probas[y_test_seq == 1], label="Décès", fill=True)

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Densité")
    plt.title("Distribution des scores (KDE)")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(plt, probas, sns, y_test_seq):
    y_test = y_test_seq

    plt.figure()

    # Survivants
    data_0 = probas[y_test == 0]
    sns.kdeplot(data_0, color="lightblue")
    x0, y0 = plt.gca().lines[-1].get_data()
    y0 = y0 * len(data_0)  # conversion densité → counts
    plt.plot(x0, y0, color="blue", label="Survivants")
    plt.fill_between(x0, y0, alpha=0.3, color="lightblue")

    # Décès
    data_1 = probas[y_test == 1]
    sns.kdeplot(data_1, color="orange")
    x1, y1 = plt.gca().lines[-1].get_data()
    y1 = y1 * len(data_1)
    plt.plot(x1, y1, color="orange", label="Décès")
    plt.fill_between(x1, y1, alpha=0.3, color="orange")

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Nombre de patients")
    plt.title("Distribution des scores")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.style.use("seaborn-v0_8")
    plt.show()
    return (y_test,)


@app.cell
def _(calibration_curve, plt, probas, y_test):
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Modèle")
    plt.plot([0, 1], [0, 1], "--", label="Calibration idéale")

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(f1_score, np, plt, probas, y_test):
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
    plt.title('F1 vs Threshold')
    plt.grid()
    plt.show()
    print(f'Le meilleur f1 score de{best_f1: .2f} est atteint lorsque le threshold est égal à{best_t: .2f}')
    return


@app.cell
def _(confusion_matrix, plt, probas, sns, y_test):
    threshold = 0.48
    y_pred = (probas >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title(f'Confusion matrix (threshold={threshold})')
    plt.show()
    return (y_pred,)


@app.cell
def _(pd, plt, y_pred, y_test):
    df = pd.DataFrame({"y_true" : y_test,
                       "y_pred" : y_pred})

    df["decile"] = pd.qcut(df["y_pred"], 10, labels = False, duplicates="drop") + 1

    calib = (
        df.groupby("decile", as_index=False)
          .agg(
              n=("y_true", "size"),
              pred_mean=("y_pred", "mean"),
              obs_rate=("y_true", "mean"),
          )
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # fond: effectifs
    ax2.bar(calib["decile"], calib["n"], alpha=0.25)
    ax2.set_ylabel("Nombre d'individus")

    # premier plan: calibration
    ax1.plot(calib["decile"], calib["pred_mean"], marker="o", label="Prédit")
    ax1.plot(calib["decile"], calib["obs_rate"], marker="o", label="Observé")
    ax1.set_xlabel("Décile de probabilité prédite")
    ax1.set_ylabel("Probabilité / taux observé")
    ax1.legend()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
