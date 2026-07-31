import os

import polars as pl

import utilitaries.extract_data_utils as extract


patterns = {
    "Oncology": (
        r"tumeur maligne|cancer|carcinome|lymphome|leucémie|métastase|"
        r"néoplasie|sarcome"
    ),
    "Neurological": (
        r"cerveau|méninges|cérébrale|sous-durale|sous-arachnoïdienne|"
        r"intracrânienne|intracérébrale|coma|épilepsie|neurologique|avc|"
        r"encéphalopathie|carotide|nerfs crâniens|vaisseaux cérébraux|"
        r"grand mal|épileptique|hydrocéphalie|encéphale"
    ),
    "Sepsis_Infection": (
        r"septique|sepsis|septicémie|infection|choc septique|endocardite|"
        r"péritonite|pyonéphrose|abcès|prostatite"
    ),
    "Respiratory": (
        r"respiratoire|covid-19|pneumonie|pneumopathie|poumon|broncho|"
        r"asthme|pleurale|détresse respiratoire|pneumothorax|"
        r"obstructive chronique|asphyxie|hémoptysie|asthmatique|"
        r"épanchement pleural|fibrose"
    ),
    "Cardiovascular": (
        r"myocarde|cardiaque|aortique|aorte|mitrale|valvule|ischémique|"
        r"infarctus|cœur|coronaire|arythmie|embolie|thrombose|artère|"
        r"cardiogénique|ventriculaire|rupture d'une artère|choc|syncope|"
        r"collapsus|péricarde|cardiopulmonaire"
    ),
    "Trauma_Toxicology": (
        r"traumatique|fracture|accident|brûlure|plaie|contusion|"
        r"intoxication|overdose|substances|bêta-bloquants|"
        r"benzodiazépines|toxique|monoxyde"
    ),
    "Gastro_Renal_Metabolic": (
        r"rénale|rein|hépatique|foie|pancréatite|estomac|intestin|"
        r"gastrique|œsophage|diabète|acidocétose|varices oesophagiennes|"
        r"hématémèse|ulcère|néphrite|hypokaliémie|hémopéritoine"
    ),
    "Surgical_Procedures": (
        r"dispositif|sutures|pansements|chirurgicaux|soins|examen|greffe"
    ),
}


ICU_UNITS = [
    "RANGUEIL DECHO. REA.",
    "NEURO-CHIR REA",
    "PURPAN DECHO. REA.",
    "RANGUEIL REA. POLY.",
    "PURPAN REA. POLY.",
]


def create_merged_dataset(
    df_static: pl.DataFrame | pl.LazyFrame,
    df_dynamic: pl.DataFrame | pl.LazyFrame,
    restrict_to_icu_units: bool,
    main_diagnosis: str = "all_diseases",
    save: bool = False,
    folder: str = "",
) -> pl.LazyFrame:
    """Merge static and time-series patient data.

    The function preprocesses static patient data, optionally restricts the
    dataset to selected ICU units, categorizes patients according to their
    primary diagnosis, and merges the resulting data with the dynamic
    time-series dataset.

    Slightly negative death delays between -1 and 0 days are replaced with
    zero, while rows containing lower invalid values are removed.

    Args:
        df_static:
            Static patient data. It must contain the patient identifier,
            admission unit, primary diagnosis, and death-delay columns.
        df_dynamic:
            Time-series patient data. It must contain the patient identifier
            and the ``delta_hour`` column.
        restrict_to_icu_units:
            Whether to retain only patients admitted to the units listed in
            ``ICU_UNITS``.
        main_diagnosis:
            Diagnosis category to retain. Supported values are
            ``"all_diseases"``, ``"sepsis"``, and
            ``"Sepsis_Infection"``.
        save:
            Whether to save the merged dataset as a Parquet file.
        folder:
            Destination directory used when ``save`` is enabled.

    Returns:
        The merged dataset, sorted by patient identifier and ``delta_hour``.

    Raises:
        ValueError:
            If the requested diagnosis category is not supported.
    """
    # Convert eager DataFrames to LazyFrames.
    if isinstance(df_static, pl.DataFrame):
        df_static = df_static.lazy()

    if isinstance(df_dynamic, pl.DataFrame):
        df_dynamic = df_dynamic.lazy()

    # Cast the patient identifier to a consistent integer type.
    df_static = df_static.with_columns(
        pl.col(extract.ID_COL).cast(pl.Int32)
    )

    # Retain only patients admitted to the selected ICU units.
    if restrict_to_icu_units:
        df_static = df_static.filter(
            pl.col("adm_unit").is_in(ICU_UNITS)
        )

    # Replace slightly negative death delays with zero and remove invalid rows.
    df_static = (
        df_static
        .with_columns(
            pl.when(
                pl.col("deces_datediff_days").is_between(-1, 0)
            )
            .then(0)
            .otherwise(pl.col("deces_datediff_days"))
            .alias("deces_datediff_days")
        )
        .filter(
            (pl.col("deces_datediff_days") >= 0)
            | pl.col("deces_datediff_days").is_null()
        )
    )

    # Create a lowercase temporary column for case-insensitive matching.
    df_static = df_static.with_columns(
        pl.col("icu_DP")
        .str.to_lowercase()
        .alias("temp_lower")
    )

    # Build one conditional expression for each diagnosis category.
    category_expressions = []

    for category_name, regex in patterns.items():
        category_expressions.append(
            pl.when(
                pl.col("temp_lower").str.contains(regex)
            ).then(
                pl.lit(category_name)
            )
        )

    # Assign a fallback category when no pattern matches.
    category_expressions.append(
        pl.when(pl.col("temp_lower").is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.lit("Other"))
    )

    # Assign the first matching diagnosis category to each patient.
    df_static = df_static.with_columns(
        pl.coalesce(category_expressions).alias("category")
    )

    # Filter patients according to the requested primary diagnosis.
    if main_diagnosis in {"sepsis", "Sepsis_Infection"}:
        df_static = df_static.filter(
            pl.col("category") == "Sepsis_Infection"
        )
    elif main_diagnosis != "all_diseases":
        raise ValueError(
            f"Diagnosis category {main_diagnosis!r} is not currently supported."
        )

    # Remove the temporary column before merging the datasets.
    df_static = df_static.drop("temp_lower")

    # Retain only patients with at least one valid spo2 measurement.
    patients_with_spo2 = (
        df_dynamic
        .filter((pl.col("spo2").is_not_null()) & (pl.col("spo2") != 0))
        .select(extract.ID_COL)
        .unique()
    )
    
    df_dynamic = df_dynamic.join(
        patients_with_spo2,
        on=extract.ID_COL,
        how="inner",
    )

    # Retain only patient with at least 24h of stay
    patients_with_gt_24h = (
        df_dynamic
        .group_by("encounterId")
        .agg(pl.col("delta_hour").max().alias("delta_max"))
        .filter(pl.col("delta_max") >= 23)
        .select("encounterId")
)

    df_dynamic = df_dynamic.join(
        patients_with_gt_24h,
        on = extract.ID_COL,
        how = "inner",
    )

    # Merge static and time-series data using the patient identifier.
    df_merged = df_dynamic.join(
        df_static,
        on=extract.ID_COL,
        how="inner",
    )

    # Ensure a consistent patient and chronological order.
    df_merged = df_merged.sort(
        [extract.ID_COL, "delta_hour"]
    )

    # Optionally save the merged LazyFrame directly as a Parquet file.
    if save:
        output_path = os.path.join(
            folder,
            "merged_static_ano_and_dynamic.parquet",
        )
        df_merged.sink_parquet(output_path)

    return df_merged