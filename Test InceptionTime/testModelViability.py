import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Importation des bibliothèques
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import time
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
        brier_score_loss
    )
    from pathlib import Path
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

    return (
        Path,
        StratifiedGroupKFold,
        brier_score_loss,
        calibration_curve,
        confusion_matrix,
        dataclass,
        evaluate_on_test,
        extract,
        f1_score,
        load_model_from_checkpoint,
        mo,
        np,
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
    #### Widgets Marimo
    """)
    return


@app.cell
def _():
    config_dropdown_color = "<div style='background:#F0FFD4;padding:8px;border-radius:6px'>"

    config_run_button = "<div style='background:#FFFA7A;padding:8px;border-radius:6px'>"

    config_msg = "<div style='background:#FFABAB;padding:8px;border-radius:6px>"

    config_sidebar = "<div style='background:#F9F9F9;padding:8px;border-radius:6px'>"
    return config_dropdown_color, config_run_button, config_sidebar


@app.cell
def _(dataclass):
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
            used_distribution = "uniform", # pas utilisé
        ),
        "24h fin réanimation avec remplissage" : ConfigFenetrage(
            name = "24h_fin_rea-fill",
            hour_offset = -1,
            max_hour = 12,
            strict_mode = False,
            random = False,
            used_distribution = "uniform", # pas utilisé
        ),
         "24h fin réanimation sans remplissage" : ConfigFenetrage(
            name = "24h_fin_rea_no-fill",
            hour_offset = -1,
            max_hour = 12,
            strict_mode = True,
            random = False,
            used_distribution = "uniform", # pas utilisé
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
    return (MODES,)


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class ConfigPopulation:
        keep_population : str
    POPULATION = {
        "Tout" : ConfigPopulation(
            keep_population = "all_diseases"
        ),
        "Sepsis" : ConfigPopulation(
            keep_population = "sepsis"
        ),
    }
    return (POPULATION,)


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class ConfigModels:
        models_name : str
    MODELS = {
        "InceptionTimeModified" : ConfigModels(
            models_name = "InceptionTimeModified"
        )
    }
    return (MODELS,)


@app.cell
def _(dataclass):
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
    return (CLEAN,)


@app.cell
def _(dataclass):
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
    return (Y,)


@app.cell
def _(dataclass):
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
    return (FEAT,)


@app.cell
def _(CLEAN, mo):
    cleaning = mo.ui.dropdown(
        options=CLEAN,
        value = "Enlever Surveillance Continue",
        label = "Nettoyage des patients en SC",
    )
    return (cleaning,)


@app.cell
def _(MODES, mo):
    mode = mo.ui.dropdown(
        options=MODES,
        value="24h début réanimation avec remplissage",
        label="Mode de fenêtrage",
    )
    return (mode,)


@app.cell
def _(MODELS, mo):
    models = mo.ui.dropdown(
        options=MODELS,
        value="InceptionTimeModified",
        label="Modèle utilisé",
    )
    return (models,)


@app.cell
def _(FEAT, mo):
    modex = mo.ui.dropdown(
        options = list(FEAT.keys()),
        value = "Mode classique",
        label = "Choix des features gardées",
    )
    return (modex,)


@app.cell
def _(Y, mo):
    y_dd =  mo.ui.dropdown(
        options=Y,
        value="Survie (isDeceased)",
        label="Cible (y) à prédire",
    )
    return (y_dd,)


@app.cell
def _(config2, mo):
    run = mo.ui.run_button(
        label=f"Lancer l'entraînement du modèle {config2.models_name}"
    )
    return (run,)


@app.cell
def _(mo):
    save_figure = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Oui",
                                label = "Sauvegarde des figures")
    return (save_figure,)


@app.cell
def _(POPULATION, mo):
    keep_pop = mo.ui.dropdown(
        options = POPULATION,
        value = "Tout",
        label = "Type de patients que l'on veut garder (ICU_DP filter)")
    return (keep_pop,)


@app.cell
def _(config_dropdown_color, mo, save_figure):
    mo.vstack([
        mo.md(config_dropdown_color),
        save_figure,
        mo.md("</div>")
    ])
    return


@app.cell
def _(save_figure):
    save_figure.value
    return


@app.cell
def _(mode):
    config = mode.value
    return (config,)


@app.cell
def _(models):
    config2 = models.value
    return (config2,)


@app.cell
def _(cleaning):
    config3 = cleaning.value
    return (config3,)


@app.cell
def _(y_dd):
    config4 = y_dd.value
    return (config4,)


@app.cell
def _(keep_pop):
    config5 = keep_pop.value
    return (config5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Définitions utiles
    """)
    return


@app.cell
def _():
    all_features = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf', 'fr', 'pas']
    return (all_features,)


@app.cell
def _():
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
    return (dico_terme,)


@app.cell
def _(Path):
    def get_unique_path(path):
        path = Path(path)

        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        i = 1
        new_path = parent / f"{stem}_{i}{suffix}"

        while new_path.exists():
            i += 1
            new_path = parent / f"{stem}_{i}{suffix}"

        return new_path


    return (get_unique_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extraction
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
    df_static_0 = df_static_bis.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days')).filter((pl.col('deces_datediff_days') >= 0) | pl.col('deces_datediff_days').is_null())
    return (df_static_0,)


@app.cell
def _(df_static_0):
    df_static_0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut aussi trier en fonction du diagnostic principal envisagé
    """)
    return


@app.cell
def _(config_dropdown_color, keep_pop, mo):
    mo.vstack([
        mo.md(config_dropdown_color),
        keep_pop,
        mo.md("</div>")])
    return


@app.cell
def _(config5, df_static_0, pl):
    patterns = {

        "Oncology": r"tumeur maligne|cancer|carcinome|lymphome|leucémie|métastase|néoplasie|sarcome",

        "Neurological": r"cerveau|méninges|cérébrale|sous-durale|sous-arachnoïdienne|intracrânienne|intracérébrale|coma|épilepsie|neurologique|avc|encéphalopathie|carotide|nerfs crâniens|vaisseaux cérébraux|grand mal|épileptique|hydrocéphalie|encéphale",

        "Sepsis_Infection": r"septique|sepsis|septicémie|infection|choc septique|endocardite|péritonite|pyonéphrose|abcès|prostatite",

        "Respiratory": r"respiratoire|covid-19|pneumonie|pneumopathie|poumon|broncho|asthme|pleurale|détresse respiratoire|pneumothorax|obstructive chronique|asphyxie|hémoptysie|asthmatique|épanchement pleural|fibrose",

        "Cardiovascular": r"myocarde|cardiaque|aortique|aorte|mitrale|valvule|ischémique|infarctus|cœur|coronaire|arythmie|embolie|thrombose|artère|cardiogénique|ventriculaire|rupture d'une artère|choc|syncope|collapsus|péricarde|cardiopulmonaire",

        "Trauma_Toxicology": r"traumatique|fracture|accident|brûlure|plaie|contusion|intoxication|overdose|substances|bêta-bloquants|benzodiazépines|toxique|monoxyde",

        "Gastro_Renal_Metabolic": r"rénale|rein|hépatique|foie|pancréatite|estomac|intestin|gastrique|œsophage|diabète|acidocétose|varices oesophagiennes|hématémèse|ulcère|néphrite|hypokaliémie|hémopéritoine",

        "Surgical_Procedures": r"dispositif|sutures|pansements|chirurgicaux|soins|examen|greffe"

    }



    df_static_1 = df_static_0.with_columns(

        pl.col("icu_DP").str.to_lowercase().alias("temp_lower"),
    )

    condition = pl.when(pl.col("temp_lower").is_null()).then(pl.lit("Unknown"))

    for cat_name, regex in patterns.items():

        condition = condition.when(pl.col("temp_lower").str.contains(regex)).then(pl.lit(cat_name))

    df_static_1 = df_static_1.with_columns(

        condition.otherwise(pl.lit("Other")).alias("category")

    ).to_dummies("category").cast(

        {'encounterId' : pl.Int32}

    )

    if config5.keep_population == "sepsis":
        df_static_1 = df_static_1.filter(pl.col("category_Sepsis_Infection") == 1)
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
    ### Transformation_Prétraitement
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
def _(config_dropdown_color, mo, mode):
    mo.vstack([
        mo.md(config_dropdown_color),
        mode,
        mo.md("</div>")])
    return


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
    print("nombre d'enregistrement de patients morts moins de 7 jours après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_7d') == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 28 jours après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_28d") == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_3m") == True).select(pl.col(ui.patient_col).n_unique()).item())
    print("nombre d'enregistrement de patients morts plus de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_gt_3m') == True).select(pl.col(ui.patient_col).n_unique()).item())
    return


@app.cell
def _(config_dropdown_color, mo, models):
    mo.vstack([
        mo.md(config_dropdown_color),
        models,
        mo.md("</div>")])
    return


@app.cell
def _():
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(df_clean_2, pl):
    # si on a pas la pression artérielle systolique, 
    df_clean_3 = df_clean_2.with_columns(
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
    return (df_clean_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Préparation pour InceptionTime
    """)
    return


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
    return keep_feats, str_keep_feats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Split train/test + scaling + reshape
    """)
    return


@app.cell
def _(config_dropdown_color, mo, y_dd):
    mo.vstack([
        mo.md(config_dropdown_color),
        y_dd,
        mo.md("</div>")])
    return


@app.cell
def _():
    return


@app.cell
def _(test_df):
    test_df
    return


@app.cell
def _(StratifiedGroupKFold, config4, df_clean_3, keep_feats, pl, ui):
    keep_features = keep_feats
    patient_col = ui.patient_col
    time_col = ui.time_col
    target_col = config4.target_name
    expected_length = 24
    valid_ids = df_clean_3.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
    # On garde les encounters de longueur exacte 
    df_clean_4 = df_clean_3.join(valid_ids, on=patient_col, how='inner')
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
    (train_pd, test_pd) = ui.scaling(train_df, test_df)
    (train_df, test_df) = (pl.from_pandas(train_pd), pl.from_pandas(test_pd))
    (X_train_3d, y_train_seq) = ui.build_sequences(train_df, patient_col, target_col, expected_length, keep_features)  # grouper en fonction d'un individu
    # On prend un split (comme train/test mais adapté aux individus)
    # Normalement pas besoin de sort mais soyons prudents...
    # Ok maintenant, on applique le scaler sur le dataframe
    # On retransforme en df polars
    # On construit la séquence attendue (N, T, F) à partir des deux dataframes train/test

    (X_test_3d, y_test_seq) = ui.build_sequences(test_df, patient_col, target_col, expected_length, keep_features)
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
def _():
    # print(train_df.group_by('encounterId').max().select('isDeceased_lt_24h').mean())

    # print(test_df.group_by('encounterId').max().select('isDeceased_lt_24h').mean())

    # print(df_clean_4.group_by('encounterId').max().select('isDeceased_lt_24h').mean())
    return


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
    ### Training sur InceptionTime
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
def _(config_run_button, mo, run):
    mo.vstack([
        mo.md(config_run_button),
        run,
        mo.md("</div>")])
    return


@app.cell
def _(
    X_train_3d,
    config,
    config2,
    config3,
    config4,
    config5,
    get_unique_path,
    mo,
    modex,
    run,
    train_inception_time,
    y_train_seq,
):
    # on créé un nom unique de modèle
    str_pop = ""
    if config5.keep_population != "all_diseases":
        str_pop = "_"+config5.keep_population
    model_path = get_unique_path(f"models/{config2.models_name}/{config.name}_{config2.models_name}_{config3.clean}_{config4.target_name}_{modex.value}_{config5.keep_population}{str_pop}.pt")

    mo.stop(not run.value, "Clique pour lancer")
    print("Entraînement lancé")
    if config2.models_name == "InceptionTimeModified":
        # model, T, history, splits = train_inception_time(
        #     X_train_3d, y_train_seq,
        #     num_blocks = 6,
        #     out_channels = 32,
        #     bottleneck_channels = 8,
        #     kernel_sizes = 21,
        #     batch_size = 16,
        #     lr = 0.0009572131781501278,
        #     weight_decay = 1.216426840149487e-06,
        #     clip_grad = 0.5,
        #     use_scheduler = False,
        #     epochs=100,
        #     patience=10,
        #     save_best_path=path)
        model, T, history, splits = train_inception_time(
            X_train_3d, y_train_seq,
            epochs=100,
            patience=10,
            save_best_path=model_path
            )

    else :
        print("oups tu t'es trompé")
    return (str_pop,)


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
    str_pop,
    y_test_seq,
):
    loaded_model = f"models/{config2.models_name}/{config.name}_{config2.models_name}_{config3.clean}_{config4.target_name}_{modex.value}{str_pop}.pt"
    (_auc, brier, T_1) = evaluate_on_test(X_test_3d, y_test_seq, loaded_model)
    (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)
    return T_1, loaded_model, model_1


@app.cell
def _(Path, loaded_model):
    # construire le dossier output correspondant
    output_dir = Path("outputs") / Path(loaded_model).stem

    # créer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _(T_1, X_test_3d, model_1, predict_proba):
    probas = predict_proba(model_1, X_test_3d, T=T_1)
    print(probas)
    return (probas,)


@app.cell
def _(
    Path,
    df_clean_3,
    modex,
    output_dir,
    plt,
    probas,
    roc_auc_score,
    roc_curve,
    save_figure,
    target_col,
    y_test_seq,
):
    (fpr, tpr, _thresholds) = roc_curve(y_test_seq, probas)
    auc = roc_auc_score(y_test_seq, probas)
    df_temp =  df_clean_3.group_by("encounterId")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC trainedModel (AUC = {auc:.3f})')
    if modex.value == "Mode NEWS" :
        # Une ligne par patient = dernière heure disponible
        df_news_patient = (
            df_clean_3
            .sort(["encounterId", "heure_calibree"])
            .group_by("encounterId")
            .last()
        )

        # Extraire y_true et NEWS
        y_news = df_news_patient[target_col].to_numpy()   

        news_score = df_news_patient["news"].to_numpy()
        fpr_news, tpr_news, _ = roc_curve(y_news, news_score)
        auc_news = roc_auc_score(y_news, news_score)
        plt.plot(fpr_news, tpr_news, label=f'ROC NEWS (AUC = {auc_news:.3f})', color = "orange")
    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color = "green")
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_figure.value :
        plt.savefig(output_dir / Path("Courbe_ROC"))
    plt.show()
    return (auc,)


@app.cell
def _(Path, output_dir, plt, probas, save_figure, sns, y_test_seq):
    plt.figure()

    sns.kdeplot(probas[y_test_seq == 0], label="Survivants", fill=True)
    sns.kdeplot(probas[y_test_seq == 1], label="Décès", fill=True)

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Densité")
    plt.title("Distribution des scores (KDE)")
    plt.legend()
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("KDE"))
    plt.show()
    return


@app.cell
def _(
    Path,
    calibration_curve,
    output_dir,
    plt,
    probas,
    save_figure,
    y_test_seq,
):
    y_test = y_test_seq
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Modèle")
    plt.plot([0, 1], [0, 1], "--", label="Calibration idéale")

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("Calibration_curve"))
    plt.show()
    return (y_test,)


@app.cell
def _(Path, f1_score, np, output_dir, plt, probas, save_figure, y_test):
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
    if save_figure.value :
        plt.savefig(output_dir / Path("threshold"))
    plt.show()
    print(f'Le meilleur f1 score de{best_f1: .2f} est atteint lorsque le threshold est égal à{best_t: .2f}')
    return (best_t,)


@app.cell
def _(
    Path,
    best_t,
    confusion_matrix,
    output_dir,
    plt,
    probas,
    save_figure,
    sns,
    y_test,
):
    y_pred = (probas >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title(f'Confusion matrix (threshold={ best_t: .2f})')
    if save_figure.value :
        plt.savefig(output_dir / Path("confusion_matrix"))
    plt.show()
    return


@app.cell
def _(
    Path,
    brier_score_loss,
    output_dir,
    pl,
    plt,
    probas,
    save_figure,
    y_test,
):
    df_brier = pl.DataFrame({"y" : y_test, "pred" : probas})

    df_brier = df_brier.with_columns(
        ((pl.col("pred") - pl.col("y")) ** 2).alias("brier")
    )

    # score global de brier
    global_brier = brier_score_loss(df_brier["y"], df_brier["pred"])

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

    plt.title("Brier par tranches fixes de risque")
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("brierPerTrancheRisk"))
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3)
    ax1.set_xlabel("Décile de patients")
    ax1.set_ylabel("Nombre de patients")

    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Brier moyen")

    plt.title("Brier par déciles de patients")
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("brierPerDec"))
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients
    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Décile de patients")
    ax1.set_ylabel("Nombre de patients")

    # Courbe de calibration
    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["obs_rate"], marker="o", label="Mortalité observée", color="black")
    ax2.plot(dec_pd["x"], dec_pd["pred_mean"], marker="s", label="Risque prédit", color="black", linestyle="--")
    ax2.set_ylabel("Mortalité (Taux observé vs Risque prédit)")

    plt.title("Courbe de Calibration par déciles de patients")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("calibPerDec"))
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients (Tranches fixes)
    ax1.bar(fixed_pd["x"], fixed_pd["n"], width=0.08, alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Risque prédit (Tranches de 10%)")
    ax1.set_ylabel("Nombre de patients")
    ax1.set_xlim(0, 1)

    # Courbe de calibration (axe Y droit)
    ax2 = ax1.twinx()
    ax2.plot(fixed_pd["x"], fixed_pd["obs_rate"], marker="o", label="Mortalité observée", color="black", linestyle="-")
    ax2.plot(fixed_pd["x"], fixed_pd["pred_mean"], marker="s", label="Risque moyen prédit", color="black", linestyle="--")
    ax2.set_ylabel("Mortalité (Taux observé vs Risque prédit)")
    ax2.set_ylim(0, 1) 

    plt.title("Courbe de Calibration par tranches fixes de risque")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure.value:
        plt.savefig(output_dir / Path("calibPerTrancheRisk"))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Progression de NEWS moyenne sur 24h
    """)
    return


@app.cell
def _(Path, df_clean_3, output_dir, pl, plt, save_figure, ui):
    # D'abord, on met tous les patients sur le même temps pour pouvoir les merge
    df_modif = df_clean_3.with_columns(
        pl.arange(0,pl.count()).over(ui.patient_col).alias("hour_local")
    )
    df_modif_agg = (
    df_modif.group_by("hour_local")
    .agg([
        pl.col("news").mean().alias("mean"),
        pl.col("news").std().alias("std"),
        pl.count().alias("n")
    ])
    .sort("hour_local")
    )
    df_modif_agg = df_modif_agg.with_columns([
            (pl.col("mean") - 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("lower"),
            (pl.col("mean") + 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("upper")
        ])
    plt.plot(df_modif_agg["hour_local"], df_modif_agg["mean"], color="purple")
    plt.fill_between(
        df_modif_agg["hour_local"],
        df_modif_agg["lower"],
        df_modif_agg["upper"],
        color="purple",
        alpha=0.2
    )

    plt.xlabel("Temps (heures)")
    plt.ylabel("NEWS score")
    plt.title("Évolution du score NEWS sur 24 heures")
    if save_figure.value :
        plt.savefig(output_dir / Path("NEWS_24H"))
    plt.show()
    return (df_modif,)


@app.cell
def _(
    Path,
    auc,
    df_modif,
    np,
    output_dir,
    pl,
    plt,
    roc_auc_score,
    save_figure,
    target_col,
):
    rows = []
    for h in sorted(df_modif["hour_local"].unique().to_list()):
        df_h = df_modif.filter(pl.col("hour_local") == h)
        y_true = df_h[target_col].to_numpy()
        y_score = df_h["news"].to_numpy()

        auc_news_h = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
        rows.append({"hour_local": h, "auc": auc_news_h})

    df_auc = pl.DataFrame(rows).sort("hour_local")


    plt.plot(df_auc["hour_local"], df_auc["auc"], color="purple", label = 'AUC par heure de NEWS')
    plt.axhline(y = auc, label = "AUC globale du modèle")
    plt.xlabel("Temps (heures)")
    plt.ylabel("AUC de NEWS")
    plt.title("Évolution de l'AUC de NEWS sur 24 heures")
    plt.legend()
    if save_figure.value :
        plt.savefig(output_dir / Path("AUC_NEWS_24H"))
    plt.show()
    return


@app.cell
def _(
    cleaning,
    config_sidebar,
    custom_features,
    keep_feats,
    keep_pop,
    mo,
    mode,
    models,
    modex,
    run,
    save_figure,
    str_keep_feats,
    y_dd,
):
    mo.sidebar(
    mo.vstack([
        mo.md(config_sidebar),
        mode,
        models,
        cleaning,
        y_dd,
        keep_pop,
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français :** \n{str_keep_feats}"),
        save_figure,
        mo.md("==============================================="),
        run,
        mo.md("==============================================="),
        mo.md(f" \n \n **Modification Thesaurus** : Afin d'avoir des valeurs cohérentes, avec une bonne imputation notamment, j'ai rajouté la pression artérielle systolique ainsi que la fréquence respiratoire. Il faudra voir aussi si on laisse les valeurs par défaut à 0 ou non. J'ai pris le parti pris pour la pas et fr de mettre en valeur par défaut une valeur qui fait un score de 0 sur news, sinon ça augmenterait le score juste parce qu'on a pas l'info ce qui n'est pas optimal... J'ai donc 130 pour pas en imputation method ffill_bfill et 16 pour fr en ffill_bfill aussi. Je me suis rendu compte que la valeur par défaut de heart_rate et spo2 était aussi de 0. Cela classe donc instantanément le patient en grave, alors qu'on a juste pas l'information... j'ai mis pour heart_rate une valeur par défaut de 60 et un spo2 de 96%. Je pense qu'il faudra qu'on fasse un point sur les valeurs par défaut du thesaurus car la majorité sont à 0, ce qui peut poser problème"),
        mo.md("</div>")]),
    width = "550px")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
