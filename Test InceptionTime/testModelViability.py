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

    return (
        Path,
        StratifiedGroupKFold,
        brier_score_loss,
        calibration_curve,
        confusion_matrix,
        evaluate_lstm_on_test,
        evaluate_on_test,
        extract,
        f1_score,
        load_lstm_from_checkpoint,
        load_model_from_checkpoint,
        mo,
        mo_utils,
        np,
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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Widgets Marimo utilisés dans ce notebook
    """)
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
        value="24h aléatoire 'flexible' sans remplissage",
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
        value = "Mode classique",
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
        value="Survie à 24 heures",
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
    confirm = mo.ui.run_button(
        label=f"Confirmer les paramètres actuels"
    )
    return (confirm,)


@app.cell
def _(mo):
    save_figure = mo.ui.dropdown(options = {"Oui" : True, "Non" : False},
                                value = "Oui",
                                label = "Sauvegarde des figures")
    return (save_figure,)


@app.cell
def _(mo, mo_utils):
    keep_pop = mo.ui.dropdown(
        options = mo_utils.POPULATION,
        value = "Tout",
        label = "Type de patients que l'on veut garder (ICU_DP filter)")
    return (keep_pop,)


@app.cell
def _(mo):
    get_max_hour, set_max_hour = mo.state(6)
    gap_start, gap_end = 0, 12
    return gap_end, gap_start, get_max_hour, set_max_hour


@app.cell
def _(gap_end, gap_start, get_max_hour, mo, set_max_hour):
    slider_max_hour = mo.ui.slider(gap_start, gap_end, value=get_max_hour(), on_change=set_max_hour)
    return (slider_max_hour,)


@app.cell
def _(gap_end, gap_start, get_max_hour, mo, set_max_hour):
    number_max_hour = mo.ui.number(gap_start, gap_end, value=get_max_hour(), on_change=set_max_hour)
    return (number_max_hour,)


@app.cell
def _(mo):
    get_marge, set_marge = mo.state(0)
    gap2_start, gap2_end = 0, 24
    return gap2_end, gap2_start, get_marge, set_marge


@app.cell
def _(gap2_end, gap2_start, get_marge, mo, set_marge):
    slider_marge = mo.ui.slider(gap2_start, gap2_end, value=get_marge(), on_change=set_marge)
    return (slider_marge,)


@app.cell
def _(gap2_end, gap2_start, get_marge, mo, set_marge):
    number_marge = mo.ui.number(gap2_start, gap2_end, value=get_marge(), on_change=set_marge)
    return (number_marge,)


@app.cell
def _(mo, mo_utils, save_figure):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        save_figure,
        mo.md(mo_utils.config_end)
    ])
    return


@app.cell
def _():
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


@app.cell
def _(balance):
    config6 = balance.value
    return (config6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Définitions utiles
    """)
    return


@app.cell
def _():
    all_features = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf', 'fr', 'pas', "hx_respi_chronique"]
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
        "hx_respi_chronique" : "Antécédents de problème de respiration chronique ou non"
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
def _(extract, pl):
    _path = '../Datasets/clean_full_static_ano.parquet'
    df_static = pl.read_parquet(_path)
    df_static = df_static.with_columns(pl.col(extract.ID_COL).cast(pl.Int32))
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
def _(keep_pop, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        keep_pop,
        mo.md(mo_utils.config_end)])
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


@app.cell
def _(df_static_2):
    df_static_2["hx_respi_chronique"].describe()
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
    ajout de la colonne age qui est dans le thesaurus et de deces_datediff_days qui nous permet de calculer les étiquettes
    """)
    return


@app.cell
def _(df_static_2, df_test, extract):
    df_test_1 = df_test.join(df_static_2[[extract.ID_COL, 'age', 'deces_datediff_days', "hx_respi_chronique"]], on=extract.ID_COL, how='left')
    return (df_test_1,)


@app.cell
def _(df_test_1):
    df_test_1["deces_datediff_days"].describe()
    return


@app.cell
def _(df_test_1):
    df_test_1["encounterId"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On rajoute 4 features qui valent 1 si : (enfin True et False)

    - Pour isDeceased_lt_24h : 1 si décès inférieur à 24h après la fin de la fenêtre et 0 sinon

    - Pour isDeceased_lt_7d : 1 si décès inférieur à 7 jours après la fin de la fenêtre et 0 sinon

    - Pour isDeceased_lt_28d : 1 si décès inférieur à 28 jours après la fin de la fenêtre et 0 sinon

    - Pour isDeaceased_lt_3m : 1 si décès inférieur à 3 mois après la fin de la fenêtre et 0 sinon

    - Si les 4 features précédentes sont falses, alors le patient est vivant ou mort après 3 mois

    On ajoute à cela une version étendue qui prend en compte une marge d'erreur, afin de favoriser l'aléatoire de la fenêtre, au détriment d'un étiquetage dégradé (dégradation qui augmente à mesure que la marge augmente)
    """)
    return


@app.cell
def _(
    confirm,
    mo,
    mo_utils,
    mode,
    number_marge,
    number_max_hour,
    slider_marge,
    slider_max_hour,
):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        mode,
        mo.md(mo_utils.config_end)])

    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        mo.md("Durée sanctuarisée avant la sortie du patient"),
        slider_max_hour,
        number_max_hour,
        mo.md("-------------------------------"),
        mo.md("Marge d'erreur de l'étiquetage pour permettre un plus grand aléatoire"),
        slider_marge,
        number_marge,
        mo.md("-------------------------------"),
        confirm,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(confirm, df_test_1, get_marge, mo, pl):
    marge = get_marge()
    mo.stop(not confirm.value, "Clique pour lancer")
    df_test_2 = df_test_1.with_columns([
            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_24h"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_24h_EXTENDED"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_28d"),


            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_28d_EXTENDED"),

           pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 168)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_7d"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 168 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_7d_EXTENDED"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 2190)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_3m"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 2190 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_3m_EXTENDED"),
            pl.when(pl.col("deces_datediff_days") * 24 > pl.col("delta_hour") + 2190)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_gt_3m"),

            pl.when(pl.col("deces_datediff_days") * 24 > pl.col("delta_hour") + 2190 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_gt_3m_EXTENDED"),
        ])
    return (df_test_2,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ajout d'une métrique pour regarder le décès inférieur à 7 jours afin de pouvoir être comparé à NEWS et NEWS2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On utilise ici le preprocessing de Gabrielle mais avec l'optimisation polars réalisée par mes soins, puisque l'ancien code mettait beaucoup trop de temps à tourner.
    """)
    return


@app.cell
def _(config, config4, df_test_2, extract, get_max_hour):
    target_col = config4.target_name
    other_cols = ["isDeceased_lt_24h","isDeceased_lt_7d", "isDeceased_lt_28d", "isDeceased_lt_3m", "isDeceased_lt_24h_EXTENDED","isDeceased_lt_7d_EXTENDED", "isDeceased_lt_28d_EXTENDED", "isDeceased_lt_3m_EXTENDED", "isDeceased_gt_3m", "isDeceased_gt_3m_EXTENDED"]
    if target_col != "isDeceased" :
        target_col += "_EXTENDED"

    other_cols.remove(target_col)

    df_clean = extract.prepare_data(df_test_2, 
                                                 hour_offset = config.hour_offset, 
                                                 random = config.random, 
                                                 max_hour = get_max_hour(),
                                                 used_distribution = config.used_distribution, 
                                                 strict_mode = config.strict_mode,
                                                 target_col = target_col, other_cols = other_cols)
    df_clean.columns
    return df_clean, target_col


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


@app.cell
def _(df_clean):
    df_clean["hx_respi_chronique"].describe()
    return


@app.cell
def _(df_clean):
    df_clean["pas"].describe()
    return


@app.cell
def _(df_clean):
    df_clean["is_ventilated"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il faut rajouter isDeceased sinon on n'a pas de Y
    """)
    return


@app.cell
def _(df_clean):
    # df_clean_1 = df_clean.join(df_static_2[[extract.ID_COL, 'isDeceased']], on=extract.ID_COL, how='inner')
    df_clean_2 = df_clean
    return (df_clean_2,)


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
def _(df_clean_2, extract, pl):
    # print("nombre d'enregistrement de patients vivants (isDeceased)", df_clean_2.filter(pl.col('isDeceased') == False).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients vivants (DeceasedTimeType)", df_clean_2.filter((pl.col('isDeceased_lt_24h') == False)
                                                                                                 & (pl.col("isDeceased_lt_28d") == False)
                                                                                                 & (pl.col("isDeceased_lt_3m") == False)
                                                                                                 & (pl.col("isDeceased_lt_7d") == False)
                                                                                                 & (pl.col("isDeceased_gt_3m") == False)).select(pl.col(extract.ID_COL).n_unique()).item())
    # print("nombre d'enregistrement de patients morts (isDeceased)", df_clean_2.filter(pl.col('isDeceased') == True).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients morts (4 features)", df_clean_2.filter((pl.col('isDeceased_lt_24h') == True)
           | (pl.col("isDeceased_lt_7d") == True)                                                | (pl.col("isDeceased_lt_28d") == True)
                                                                                                 | (pl.col("isDeceased_lt_3m") == True)
                                                                                                 | (pl.col("isDeceased_gt_3m") == True)).select(pl.col(extract.ID_COL).n_unique()).item())

    print("nombre d'enregistrement de patients morts moins de 24 heures après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_24h') == True).select(pl.col(extract.ID_COL).n_unique()).item())

    print("nombre d'enregistrement de patients morts moins de 24 heures + marge après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_24h_EXTENDED') == True).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 7 jours après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_7d') == True).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 28 jours après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_28d") == True).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients morts moins de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col("isDeceased_lt_3m") == True).select(pl.col(extract.ID_COL).n_unique()).item())
    print("nombre d'enregistrement de patients morts plus de 3 mois après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_gt_3m') == True).select(pl.col(extract.ID_COL).n_unique()).item())
    return


@app.cell
def _(df_clean_2, extract, pl):
    print("nombre d'enregistrement de patients morts moins de 24 heures après la fin de la fenêtre", df_clean_2.filter(pl.col('isDeceased_lt_24h_EXTENDED') == 1).select(pl.col(extract.ID_COL).n_unique()).item())


    print("pourcentage de 0 dans ces patients devant être étiquetés 1 :", df_clean_2.filter(pl.col('isDeceased_lt_24h_EXTENDED') == 1).select(pl.col("isDeceased_lt_24h")).mean().item())
    return


@app.cell
def _(mo, mo_utils, models):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        models,
        mo.md(mo_utils.config_end)])
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


@app.cell
def _(df_clean_2, pl):
    # Supplémentation en oxygène
    sup_oxy = (pl.when(pl.col("fio2_corr") != 21).then(2).otherwise(0))

    # Fréquence respiratoire
    fr = pl.when((pl.col("fr") <= 8) |
               (pl.col("fr") >=25)).then(3).when((pl.col("fr") >= 21)
                  & (pl.col("fr") <= 24)).then(2).when((pl.col("fr")) <= 20 
                  & (pl.col("fr") >= 12)).then(0).otherwise(1)

    # Saturation en oxygène
    spo2_classic = pl.when((pl.col("spo2") <= 91.0)).then(3).when((pl.col("spo2") >= 92.0)
            & (pl.col("spo2") <= 93.0)).then(2).when((pl.col("spo2") >= 94.0)
             & (pl.col("spo2") <= 95)).then(1).otherwise(0)

    # Température
    temp = pl.when((pl.col("temp") <= 35.0)).then(3).when(((pl.col('temp') >= 35.1)
             & (pl.col("temp") <= 36.0))
             |
             ((pl.col("temp") <= 39.0)
             & (pl.col("temp") >= 38.1))).then(1).when((pl.col("temp") >= 39.1)).then(2).otherwise(0)

    # Conscience
    conscience = pl.when((pl.col("is_conscious")) == 0).then(0).otherwise(3)

    # Pression Artérielle Systolique
    pas = pl.when((pl.col("pas") <= 90.0)
                | (pl.col('pas') >= 220.0)).then(3).when((pl.col("pas") <= 110.0)
              & (pl.col("pas") >= 101.0)).then(1).when((pl.col("pas") <= 219.0)
              & (pl.col("pas") >= 111)).then(0).otherwise(2)

    # Fréquence cardiaque
    hr = pl.when((pl.col("heart_rate") <= 40)
                |(pl.col("heart_rate") >= 131)).then(3).when((pl.col("heart_rate") <= 90) 
              & (pl.col("heart_rate") >= 51)).then(0).when((pl.col("heart_rate") >= 111)
              & (pl.col("heart_rate") <= 130)).then(2).otherwise(1)

    df_clean_3 = df_clean_2.with_columns((
        sup_oxy + fr + spo2_classic + temp
        + conscience + pas + hr).alias("news"))

    # Saturation en oxygène NEWS2 
    # TODO ATTENTION LA ON UTILISE hx_respi_chronique et pas hx_hypercapnie (qui n'existe pas)
    spo2_NEWS2 = (
        pl.when(pl.col("hx_respi_chronique") == 1)
        .then(
            pl.when(pl.col("spo2") <= 83).then(3)
            .when((pl.col("spo2") >= 84) & (pl.col("spo2") <= 85)).then(2)
            .when((pl.col("spo2") >= 86) & (pl.col("spo2") <= 87)).then(1)
            .when((pl.col("spo2") >= 88) & (pl.col("spo2") <= 92)).then(0)
            .when(
                (pl.col("spo2") >= 93) & (pl.col("spo2") <= 94) &
                (pl.col("fio2_corr") != 21)
            ).then(1)
            .when(
                (pl.col("spo2") >= 95) & (pl.col("spo2") <= 96) &
                (pl.col("fio2_corr") != 21)
            ).then(2)
            .when(
                (pl.col("spo2") >= 97) &
                (pl.col("fio2_corr") != 21)
            ).then(3)
            .otherwise(0)
        )
        .otherwise(spo2_classic)
    )


    df_clean_3 = df_clean_3.with_columns((
        sup_oxy + fr + spo2_NEWS2 + temp
        + conscience + pas + hr).alias("news2"))
    return (df_clean_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Préparation pour InceptionTime
    """)
    return


@app.cell
def _(all_features, mo, mo_utils):
    custom_features = mo.ui.multiselect(
        options=all_features,
        value= mo_utils.FEAT["Mode Custom"].keep_feats,
        label="(features sélectionnables)",
    )
    return (custom_features,)


@app.cell
def _(all_features, custom_features, dico_terme, mo, mo_utils, modex):
    if modex.value == "Mode All":
        keep_feats = all_features
    elif modex.value == "Mode Custom":
        keep_feats = custom_features.value
    else:
        keep_feats = mo_utils.FEAT[modex.value].keep_feats

    str_keep_feats = ""

    for kf in keep_feats:
        str_keep_feats += f"- {dico_terme[kf]} \n"
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français :** \n{str_keep_feats}"),
        mo.md(mo_utils.config_end)
    ])
    return keep_feats, str_keep_feats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Split train/test + scaling + reshape
    """)
    return


@app.cell
def _(mo, mo_utils, y_dd):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        y_dd,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(balance, mo, mo_utils):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        balance,
        mo.md(mo_utils.config_end)])
    return


@app.cell
def _(
    StratifiedGroupKFold,
    config6,
    df_clean_3,
    extract,
    keep_feats,
    pl,
    preproc,
    target_col,
):
    keep_features = keep_feats
    patient_col = extract.ID_COL
    time_col = extract.TIME_COL2
    expected_length = 24
    valid_ids = df_clean_3.group_by(patient_col).len().filter(pl.col('len') == expected_length).select(patient_col)
    # On garde les encounters de longueur exacte 
    df_clean_4 = df_clean_3.join(valid_ids, on=patient_col, how='inner')
    if df_clean_4.is_empty():
        raise ValueError("Aucun patient n'a exactement la longueur attendue.")
    df_clean_4 = df_clean_4.sort(patient_col, time_col)

    # DownSampling
    # Je veux :
    # - un groupe isDeceased_lt_24h
    # - un groupe isDeceased_lt_28d qui ne contient pas les lt_24h
    # - un groupe sain
    # que l'addition des 2 derniers groupes fasse le total de lt_24h.
    X = df_clean_4.select(keep_features).to_numpy()
    y = df_clean_4[target_col].to_numpy()
    print(y)
    groups = df_clean_4[patient_col].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    (train_idx, test_idx) = next(sgkf.split(X=X, y=y, groups=groups))

    # Tri obligatoire (même si en théorie il est déjà fait)
    train_df = df_clean_4[train_idx].sort([patient_col, time_col])
    test_df = df_clean_4[test_idx].sort([patient_col, time_col])

    if config6.balance_method == "downsampling_50-50":
        # Downsampling uniquement sur le train
        train_df = preproc.downsample_train_patients(
            train_df=train_df,
            patient_col=patient_col,
            col_24h="isDeceased_lt_24h",
            col_28d="isDeceased_lt_28d",
        )

    # On prépare le jeu d'entraînement
    (train_df, test_df) = preproc.scaling(train_df, test_df)
    # (train_df, test_df) = (pl.from_pandas(train_pd), pl.from_pandas(test_pd))
    (X_train_3d, y_train_seq) = preproc.build_sequences(train_df, patient_col, target_col, expected_length, keep_features)  # grouper en fonction d'un individu
    # On prend un split (comme train/test mais adapté aux individus)
    # Normalement pas besoin de sort mais soyons prudents...
    # Ok maintenant, on applique le scaler sur le dataframe
    # On retransforme en df polars
    # On construit la séquence attendue (N, T, F) à partir des deux dataframes train/test

    (X_test_3d, y_test_seq) = preproc.build_sequences(test_df, patient_col, target_col, expected_length, keep_features)
    return (
        X_test_3d,
        X_train_3d,
        df_clean_4,
        patient_col,
        test_df,
        train_df,
        y_test_seq,
        y_train_seq,
    )


@app.cell
def _(df_clean_4, test_df, train_df):
    print(train_df.group_by('encounterId').max().select('isDeceased_lt_24h').mean())

    print(test_df.group_by('encounterId').max().select('isDeceased_lt_24h').mean())

    print(df_clean_4.group_by('encounterId').max().select('isDeceased_lt_24h').mean())
    return


@app.cell
def _(df_clean_4, patient_col, pl, target_col):
    df_clean_4.select(pl.exclude(target_col, patient_col, "DeceasedTimeType", "isDeceased_lt_24h", "isDeceased_lt_28d", "isDeceased_lt_7d", "isDeceased_lt_3m", "isDeceased_gt_3m", "deces_datediff_days")).columns
    return


@app.cell
def _(train_df):
    train_df.head()
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
def _(config, config2, config3, config4, mo, mo_utils, modex):
    mo.vstack([
        mo.md(mo_utils.config_dropdown_color),
        mo.md(f"### Entraînement avec les paramètres suivants : \n - typeFenêtrage = {config.name} \n - Modèle utilisé = {config2.models_name} \n - Nettoyage des Surveillances Continues = {config3.clean} \n - Cible à prédire = {config4.target_name} \n - Mode de features = {modex.value}"),
        mo.md(mo_utils.config_end)
    ])
    return


@app.cell
def _(config6, mo, mo_utils, run):
    mo.vstack([
        mo.md(mo_utils.config_run_button),
        run,
        mo.md(mo_utils.config_end)])

    underscore = "_"
    if config6.balance_method == "":
        underscore = ""
    return (underscore,)


@app.cell
def _(y_train_seq):
    print(y_train_seq)
    return


@app.cell
def _(
    X_train_3d,
    config,
    config2,
    config3,
    config4,
    config5,
    config6,
    get_unique_path,
    mo,
    modex,
    run,
    train_inception_time,
    train_lstm_model,
    underscore,
    y_train_seq,
):
    # on créé un nom unique de modèle
    str_pop = ""
    if config5.keep_population != "all_diseases":
        str_pop = "_"+config5.keep_population
    model_path = get_unique_path(f"models/{config2.models_name}/{config.name}_{config3.clean}_{config4.target_name}_{config6.balance_method}{underscore}{modex.value}{str_pop}.pt")

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
    elif config2.models_name == "LstmTimeModified":
        model, T, history, splits = train_lstm_model(
            X_train_3d, y_train_seq,
            epochs=100,
            patience=10,
            save_best_path=model_path
            )
    else :
        print("oups tu t'es trompé")
    return model_path, str_pop


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
    config,
    config2,
    config3,
    config4,
    config6,
    evaluate_lstm_on_test,
    evaluate_on_test,
    load_lstm_from_checkpoint,
    load_model_from_checkpoint,
    modex,
    str_pop,
    underscore,
    y_test_seq,
):
    loaded_model = f"models/{config2.models_name}/{config.name}_{config3.clean}_{config4.target_name}_{config6.balance_method}{underscore}{modex.value}{str_pop}.pt"
    if config2.models_name == "InceptionTimeModified":
        (_auc, brier, T_1) = evaluate_on_test(X_test_3d, y_test_seq, loaded_model)
        (model_1, _, T_1) = load_model_from_checkpoint(loaded_model)

    elif config2.models_name == "LstmTimeModified":
        (_auc, brier, T_1) = evaluate_lstm_on_test(X_test_3d, y_test_seq, loaded_model)
        (model_1, _, T_1) = load_lstm_from_checkpoint(loaded_model)
    else :
        print("erreur de choix de modèle")
    return T_1, loaded_model, model_1


@app.cell
def _(Path, config2, loaded_model):
    # construire le dossier output correspondant
    output_dir = Path("outputs") / Path(config2.models_name) / Path(loaded_model).stem

    # créer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _(Path, loaded_model):
    print(Path(loaded_model).stem)
    return


@app.cell
def _(T_1, X_test_3d, config2, model_1, predict_proba, predict_proba_lstm):
    # c'est la même fonction pour les 2 modèles donc c'est ok
    if config2.models_name == "InceptionTimeModified":
        probas = predict_proba(model_1, X_test_3d, T=T_1)
    elif config2.models_name == "LstmTimeModified":
        probas = predict_proba_lstm(model_1, X_test_3d, T=T_1)
    return (probas,)


@app.cell
def _(
    Path,
    config2,
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
    plt.plot(fpr, tpr, label=f'ROC {config2.models_name} (AUC = {auc:.3f})')
    print(modex.value, modex.value == "Mode NEWS2", modex.value in ["Mode NEWS", "Mode NEWS2"])
    if modex.value in ["Mode NEWS", "Mode NEWS2"] :
        df_news_patient = (
            df_clean_3
            .sort(["encounterId", "heure_calibree"])
            .group_by("encounterId")
            .last()
        )

        # Extraire y_true et NEWS
        y_news = df_news_patient[target_col].to_numpy()   

        news_score = df_news_patient["news"].to_numpy()
        news2_score = df_news_patient["news2"].to_numpy()
        fpr_news, tpr_news, _ = roc_curve(y_news, news_score)
        auc_news = roc_auc_score(y_news, news_score)
        plt.plot(fpr_news, tpr_news, label=f'ROC NEWS (AUC = {auc_news:.3f})', color = "orange")
        fpr_news2, tpr_news2, _ = roc_curve(y_news, news2_score)
        auc_news2 = roc_auc_score(y_news, news2_score)
        plt.plot(fpr_news2, tpr_news2, label=f'ROC NEWS2 (AUC = {auc_news2:.3f})', color = "#E07604")

    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color = "green")
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbe ROC du modèle {config2.models_name} pour les features du {modex.value}')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_figure.value :
        plt.savefig(output_dir / Path("Courbe_ROC"))
    plt.show()
    return (auc,)


@app.cell
def _(
    Path,
    config2,
    modex,
    output_dir,
    plt,
    probas,
    save_figure,
    sns,
    y_test_seq,
):
    plt.figure()

    sns.kdeplot(probas[y_test_seq == 0], label="Survivants", fill=True)
    sns.kdeplot(probas[y_test_seq == 1], label="Décès", fill=True)

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Densité")
    plt.title(f"Distribution des scores (KDE) de {config2.models_name} pour les features du {modex.value}")
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
    config2,
    modex,
    output_dir,
    plt,
    probas,
    save_figure,
    y_test_seq,
):
    y_test = y_test_seq
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label=f"{config2.models_name}")
    plt.plot([0, 1], [0, 1], "--", label="Calibration idéale")

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title(f"Calibration curve de {config2.models_name} pour le {modex.value}")
    plt.legend()
    plt.grid()
    if save_figure.value :
        plt.savefig(output_dir / Path("Calibration_curve"))
    plt.show()
    return (y_test,)


@app.cell
def _(
    Path,
    config2,
    f1_score,
    modex,
    np,
    output_dir,
    plt,
    probas,
    save_figure,
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
    plt.title(f"Evolution du F1 score en fonction du Threshold pour le modèle {config2.models_name} avec le {modex.value}")
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
    config2,
    confusion_matrix,
    modex,
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
    plt.title(f'Confusion matrix du modèle {config2.models_name} sur {modex.value} (threshold={ best_t: .2f})')
    if save_figure.value :
        plt.savefig(output_dir / Path("confusion_matrix"))
    plt.show()
    return


@app.cell
def _(
    Path,
    brier_score_loss,
    modex,
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

    plt.title(f"Brier par tranches fixes de risque pour le {modex.value}")
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

    plt.title(f"Brier par déciles de patients pour le {modex.value}")
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

    plt.title(f"Courbe de Calibration par déciles de patients pour le {modex.value}")
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

    plt.title(f"Courbe de Calibration par tranches fixes de risque pour le {modex.value}")
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
def _(Path, df_clean_3, extract, modex, output_dir, pl, plt, save_figure):
    # D'abord, on met tous les patients sur le même temps pour pouvoir les merge
    df_modif = df_clean_3.with_columns(
        pl.arange(0,pl.count()).over(extract.ID_COL).alias("hour_local")
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

    df_modif_agg_news2 = (
    df_modif.group_by("hour_local")
    .agg([
        pl.col("news2").mean().alias("mean"),
        pl.col("news2").std().alias("std"),
        pl.count().alias("n")
    ])
    .sort("hour_local")
    )
    df_modif_agg_news2 = df_modif_agg_news2.with_columns([
            (pl.col("mean") - 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("lower"),
            (pl.col("mean") + 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("upper")
        ])
    plt.plot(df_modif_agg_news2["hour_local"], df_modif_agg_news2["mean"], color="pink")
    plt.fill_between(
        df_modif_agg_news2["hour_local"],
        df_modif_agg_news2["lower"],
        df_modif_agg_news2["upper"],
        color="pink",
        alpha=0.3
    )

    plt.xlabel("Temps (heures)")
    plt.ylabel("NEWS score")
    plt.title(f"Évolution du score NEWS sur 24 heures pour les features du {modex.value}")
    if save_figure.value :
        plt.savefig(output_dir / Path("NEWS_24H"))
    plt.show()
    return (df_modif,)


@app.cell
def _(
    Path,
    auc,
    config2,
    df_modif,
    modex,
    np,
    output_dir,
    pl,
    plt,
    roc_auc_score,
    save_figure,
    target_col,
):
    rows = []
    rows2 = []
    for h in sorted(df_modif["hour_local"].unique().to_list()):
        df_h = df_modif.filter(pl.col("hour_local") == h)
        y_true = df_h[target_col].to_numpy()
        y_score = df_h["news"].to_numpy()
        y_score_news2 = df_h["news2"].to_numpy()

        auc_news_h = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
        auc_news2_h = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score_news2)
        rows.append({"hour_local": h, "auc": auc_news_h})
        rows2.append({"hour_local": h, "auc": auc_news2_h})

    df_auc = pl.DataFrame(rows).sort("hour_local")
    df_auc_news2 = pl.DataFrame(rows2).sort("hour_local")


    plt.plot(df_auc["hour_local"], df_auc["auc"], color="purple", label = 'AUC par heure de NEWS')
    plt.plot(df_auc_news2["hour_local"], df_auc_news2["auc"], color="pink", label = 'AUC par heure de NEWS2')
    plt.axhline(y = auc, label = f"AUC globale de {config2.models_name}")
    plt.xlabel("Temps (heures)")
    plt.ylabel("AU")
    plt.title(f"Évolution de l'AUC de NEWS sur 24 heures pour les features du {modex.value}")
    plt.legend()
    if save_figure.value :
        plt.savefig(output_dir / Path("AUC_NEWS_24H"))
    plt.show()
    return


@app.cell
def _(
    balance,
    cleaning,
    confirm,
    custom_features,
    keep_feats,
    keep_pop,
    mo,
    mo_utils,
    mode,
    models,
    modex,
    number_marge,
    number_max_hour,
    run,
    save_figure,
    slider_marge,
    slider_max_hour,
    str_keep_feats,
    y_dd,
):
    mo.sidebar(
    mo.vstack([
        mo.md(mo_utils.config_sidebar),
        mode,
        mo.md("Durée sanctuarisée avant la sortie du patient"),
        slider_max_hour,
        number_max_hour,
        mo.md("-------------------------------"),
        mo.md("Marge d'erreur de l'étiquetage pour permettre un plus grand aléatoire"),
        slider_marge,
        number_marge,
        mo.md("-------------------------------"),
        confirm,
        mo.md("-------------------------------"),
        balance,
        models,
        cleaning,
        y_dd,
        keep_pop,
        modex,
        custom_features if modex.value == "Mode Custom" else "(features fixe)",
        mo.md(f"**Features gardées :** `{keep_feats}`"),
        mo.md(f"**Soit en Français :** \n{str_keep_feats}"),
        save_figure,
        mo.md("-------------------------------"),
        run,
        mo.md("-------------------------------"),
        mo.md(f" \n \n **Modification Thesaurus** : Afin d'avoir des valeurs cohérentes, avec une bonne imputation notamment, j'ai rajouté la pression artérielle systolique ainsi que la fréquence respiratoire. Il faudra voir aussi si on laisse les valeurs par défaut à 0 ou non. J'ai pris le parti pris pour la pas et fr de mettre en valeur par défaut une valeur qui fait un score de 0 sur news, sinon ça augmenterait le score juste parce qu'on a pas l'info ce qui n'est pas optimal... J'ai donc 130 pour pas en imputation method ffill_bfill et 16 pour fr en ffill_bfill aussi. Je me suis rendu compte que la valeur par défaut de heart_rate et spo2 était aussi de 0. Cela classe donc instantanément le patient en grave, alors qu'on a juste pas l'information... j'ai mis pour heart_rate une valeur par défaut de 60 et un spo2 de 96%. Je pense qu'il faudra qu'on fasse un point sur les valeurs par défaut du thesaurus car la majorité sont à 0, ce qui peut poser problème"),
        mo.md(mo_utils.config_end)]),
    width = "550px")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
