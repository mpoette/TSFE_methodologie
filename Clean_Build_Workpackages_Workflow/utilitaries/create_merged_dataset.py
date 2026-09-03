"""Dataset merging utilities for combining static and dynamic patient data."""

import os

import polars as pl

import utilitaries.extract_data_utils as extract


# ============================================================================
# DATASET DIAGNOSTICS
# ============================================================================

def _count_unique_encounters(df: pl.LazyFrame) -> int:
    """Return the number of unique encounterIds in a LazyFrame.

    Args:
        df: LazyFrame containing patient data with an encounter identifier
            column (``ID_COL``).

    Returns:
        The count of unique encounter identifiers.
    """
    return df.select(extract.ID_COL).unique().collect().shape[0]


# Regex patterns for categorizing patients by primary diagnosis.
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


# List of ICU unit identifiers used when filtering by admission unit.
ICU_UNITS = [
    "RANGUEIL DECHO. REA.",
    "NEURO-CHIR REA",
    "PURPAN DECHO. REA.",
    "RANGUEIL REA. POLY.",
    "PURPAN REA. POLY.",
]


# ============================================================================
# PUBLIC DATASET ASSEMBLY
# ============================================================================

def create_merged_dataset(
    df_static: pl.DataFrame | pl.LazyFrame,
    df_dynamic: pl.DataFrame | pl.LazyFrame,
    restrict_to_icu_units: bool,
    main_diagnosis: str = "all_diseases",
    save: bool = False,
    folder: str = "",
    keep_duplicates: bool = False,
    mode_duplicates: str = "prio_first",
    df_static_full_clean: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Merge static and time-series patient data.

    The function preprocesses static patient data, optionally restricts the
    dataset to selected ICU units, categorizes patients according to their
    primary diagnosis, and merges the resulting data with the dynamic
    time-series dataset.

    Slightly negative death delays between -1 and 0 days are replaced with
    zero, while rows containing lower invalid values are removed.

    When ``keep_duplicates`` is False, patient duplicates are removed after
    all filters have been applied, using ``df_static_full_clean`` to identify
    which encounters to retain.

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
        keep_duplicates:
            Whether to keep duplicate patient encounters. When False,
            duplicates are removed using the strategy defined by
            ``mode_duplicates``.
        mode_duplicates:
            Duplicate removal strategy. ``"prio_first"`` retains the first
            encounter, ``"prio_last"`` retains the last encounter.
        df_static_full_clean:
            Optional static dataset without anomalies, used to identify
            which patient encounters to retain when removing duplicates.

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

    # Log initial encounter count.
    initial_static = _count_unique_encounters(df_static)
    initial_dynamic = _count_unique_encounters(df_dynamic)
    print(f"[LOG] Initial encounters - static: {initial_static}, dynamic: {initial_dynamic}")

    # Replace slightly negative death delays with zero (transformation only).
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

    # Remove the temporary column before merging.
    df_static = df_static.drop("temp_lower")

    # Merge static and time-series data using the patient identifier.
    static_before_merge = _count_unique_encounters(df_static)
    dynamic_before_merge = _count_unique_encounters(df_dynamic)
    df_merged = df_dynamic.join(
        df_static,
        on=extract.ID_COL,
        how="inner",
    )
    after_merge = _count_unique_encounters(df_merged)
    print(f"[LOG] Inner merge - static: {static_before_merge}, dynamic: {dynamic_before_merge}, after merge: {after_merge}")

    # ---- All filters applied on the merged dataset for traceability ----

    # Retain only patients admitted to the selected ICU units.
    if restrict_to_icu_units:
        before_filter = _count_unique_encounters(df_merged)
        df_merged = df_merged.filter(
            pl.col("adm_unit").is_in(ICU_UNITS)
        )
        after_filter = _count_unique_encounters(df_merged)
        print(f"[LOG] ICU unit filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

    # Remove patients with invalid death delays.
    before_filter = _count_unique_encounters(df_merged)
    df_merged = df_merged.filter(
        (pl.col("deces_datediff_days") >= 0)
        | pl.col("deces_datediff_days").is_null()
    )
    after_filter = _count_unique_encounters(df_merged)
    print(f"[LOG] Death delay filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

    # Filter patients according to the requested primary diagnosis.
    if main_diagnosis in {"sepsis", "Sepsis_Infection"}:
        before_filter = _count_unique_encounters(df_merged)
        df_merged = df_merged.filter(
            pl.col("category") == "Sepsis_Infection"
        )
        after_filter = _count_unique_encounters(df_merged)
        print(f"[LOG] Diagnosis filter (Sepsis_Infection) - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")
    elif main_diagnosis != "all_diseases":
        raise ValueError(
            f"Diagnosis category {main_diagnosis!r} is not currently supported."
        )

    # Retain only patients with a valid Glasgow score.
    before_filter = _count_unique_encounters(df_merged)
    df_merged = df_merged.filter(
        pl.col("score_glasgow").is_between(3, 15)
    )
    after_filter = _count_unique_encounters(df_merged)
    print(f"[LOG] Glasgow score filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

    # Retain only patients with valid gender and non-PIE entry mode.
    before_filter = _count_unique_encounters(df_merged)
    df_merged = df_merged.filter(
        pl.col("gender").is_not_null()
        & (pl.col("gender") != "Inconnu")
        & (pl.col("gender") != "None")
        & ~pl.col("icu_mode_entree").is_in(["PIE"]).fill_null(False)
    )
    after_filter = _count_unique_encounters(df_merged)
    print(f"[LOG] Gender/entry mode filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

    # Retain only patients with at least one valid SpO2 measurement.
    before_filter = _count_unique_encounters(df_merged)
    patients_with_spo2 = (
        df_merged
        .filter((pl.col("spo2").is_not_null()) & (pl.col("spo2") != 0))
        .select(extract.ID_COL)
        .unique()
    )
    df_merged = df_merged.join(
        patients_with_spo2,
        on=extract.ID_COL,
        how="inner",
    )
    after_filter = _count_unique_encounters(df_merged)
    print(f"[LOG] SpO2 filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

    # Retain only patients with at least 24 hours of stay.
    before_filter = _count_unique_encounters(df_merged)
    patients_with_gt_24h = (
        df_merged
        .group_by(extract.ID_COL)
        .agg(pl.col("delta_hour").max().alias("delta_max"))
        .filter(pl.col("delta_max") >= 23)
        .select(extract.ID_COL)
    )
    df_merged = df_merged.join(
        patients_with_gt_24h,
        on=extract.ID_COL,
        how="inner",
    )
    after_filter = _count_unique_encounters(df_merged)
    print(f"[LOG] 24h stay filter - before: {before_filter}, after: {after_filter}, dropped: {before_filter - after_filter}")

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