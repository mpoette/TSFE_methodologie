import polars as pl
import os
import utilitaries.extract_data_utils as extract

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

ICU_unit = ["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]


def create_merged_dataset(df_static, df_dynamic, remove_continuous_monitoring, main_diagnosis = "all_diseases", save = False, folder = ""):
    if isinstance(df_static, pl.DataFrame):
        df_static = df_static.lazy()
    if isinstance(df_dynamic, pl.DataFrame):
        df_dynamic = df_dynamic.lazy()
    # Dataframe statique
    df_static = df_static.with_columns(pl.col(extract.ID_COL).cast(pl.Int32))

    # On enlève les patients qui viennent de services avec peu de décès, afin d'équilibrer le dataset un peu mieux
    if remove_continuous_monitoring:
        df_static = df_static.filter(pl.col("adm_unit").is_in(ICU_unit))
    else : 
        df_static = df_static
    
    df_static = (df_static
    .with_columns(pl.when(
        pl.col('deces_datediff_days').is_between(-1, 0)).then(0)
        .otherwise(pl.col('deces_datediff_days'))
    .alias('deces_datediff_days'))
    .filter((pl.col('deces_datediff_days') >= 0) 
    | pl.col('deces_datediff_days').is_null()))

    df_static = df_static.with_columns(
    pl.col("icu_DP").str.to_lowercase().alias("temp_lower"),
    )

    expressions_categories = []
    
    for cat_name, regex in patterns.items():
        expressions_categories.append(
            pl.when(pl.col("temp_lower").str.contains(regex)).then(pl.lit(cat_name))
        )
    
    expressions_categories.append(
        pl.when(pl.col("temp_lower").is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.lit("Other"))
    )
    df_static = df_static.with_columns(
        pl.coalesce(expressions_categories)
        .alias("category")
    )
    # On peut aussi trier en fonction du diagnostic principal envisagé
    if main_diagnosis == "sepsis" or main_diagnosis == "Sepsis_Infection":
        df_static = df_static.filter(pl.col("category") == "Sepsis_Infection")
    elif main_diagnosis != "all_diseases":
        raise ValueError(f"ce diagnostic ({main_diagnosis}) n'est pour l'instant pas pris en charge")
    
    # Nettoyage des colonnes temporaires avant le join
    df_static = df_static.drop(["temp_lower"])
    # Join entre les 2
    df_merged = df_dynamic.join(df_static, on=extract.ID_COL, how='inner')

    # Sécurité d'ordre
    df_merged = df_merged.sort([extract.ID_COL, "delta_hour"])
    if save:
        df_merged.sink_parquet(os.path.join(folder, "merged_static_ano_and_dynamic.parquet"))
    return df_merged