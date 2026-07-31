from collections.abc import Sequence
from pathlib import Path
import re

import pandas as pd
import polars as pl
import polars.selectors as cs
from tableone import TableOne


def encode_categorical_features(
    dataframe: pl.DataFrame,
    feature_mode: str,
    features_to_keep: Sequence[str],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Encode categorical model features.

    Admission type is always one-hot encoded. GHM prefixes and ICU entry
    modes are encoded in ``Mode Commonly Used``. Gender is encoded in the
    custom and complete feature modes.

    Args:
        dataframe: Input patient time-series dataframe.
        feature_mode: Selected feature configuration.
        features_to_keep: Features selected before categorical encoding.

    Returns:
        A tuple containing:
            - The transformed dataframe.
            - The updated feature list.
            - The names of all generated dummy columns.
    """
    processed_dataframe = dataframe
    updated_features = list(features_to_keep)
    generated_dummy_columns: list[str] = []

    processed_dataframe = (
        processed_dataframe
        .with_columns(
            pl.col("admission_type").fill_null("Unknown"),
        )
        .to_dummies(columns=["admission_type"])
    )

    admission_dummy_columns = [
        column
        for column in processed_dataframe.columns
        if column.startswith("admission_type_")
    ]
    generated_dummy_columns.extend(admission_dummy_columns)

    if "admission_type" in updated_features:
        updated_features.remove("admission_type")
        updated_features.extend(admission_dummy_columns)

    if feature_mode == "Mode Commonly Used":
        processed_dataframe = processed_dataframe.with_columns(
            pl.col("icu_ghm")
            .list.first()
            .cast(pl.String)
            .str.head(3)
            .fill_null("Unknown")
            .alias("icu_ghm_f3"),
        )

        ghm_dummies = (
            processed_dataframe
            .select("icu_ghm_f3")
            .to_dummies(columns=["icu_ghm_f3"])
        )

        unknown_ghm_column = "icu_ghm_f3_Unknown"
        if unknown_ghm_column in ghm_dummies.columns:
            ghm_dummies = ghm_dummies.drop(unknown_ghm_column)

        top_entry_modes = [
            "Mutation MCO",
            "Domicile",
            "Urgence",
            "Transfert MCO",
        ]

        entry_mode_dummies = (
            processed_dataframe
            .select(
                pl.when(
                    pl.col("icu_mode_entree").is_in(top_entry_modes),
                )
                .then(pl.col("icu_mode_entree"))
                .otherwise(pl.lit("Other"))
                .alias("icu_mode_entree"),
            )
            .to_dummies(columns=["icu_mode_entree"])
        )

        new_dummy_columns = [
            *ghm_dummies.columns,
            *entry_mode_dummies.columns,
        ]
        generated_dummy_columns.extend(new_dummy_columns)

        source_columns = {
            "icu_ghm",
            "icu_ghm_f3",
            "icu_mode_entree",
        }

        updated_features = [
            feature
            for feature in updated_features
            if feature not in source_columns
        ]
        updated_features.extend(new_dummy_columns)

        existing_source_columns = [
            column
            for column in source_columns
            if column in processed_dataframe.columns
        ]

        processed_dataframe = pl.concat(
            [
                processed_dataframe.drop(existing_source_columns),
                ghm_dummies,
                entry_mode_dummies,
            ],
            how="horizontal",
        )

    gender_feature_modes = {
        "Mode Custom",
        "Mode All Without pmsi",
        "Mode All",
    }

    if feature_mode in gender_feature_modes:
        processed_dataframe = (
            processed_dataframe
            .with_columns(
                pl.col("gender").fill_null("Unknown"),
            )
            .to_dummies(columns=["gender"])
        )

        gender_dummy_columns = [
            column
            for column in processed_dataframe.columns
            if column.startswith("gender_")
        ]
        generated_dummy_columns.extend(gender_dummy_columns)

        if "gender" in updated_features:
            updated_features.remove("gender")
            updated_features.extend(gender_dummy_columns)

    processed_dataframe = processed_dataframe.with_columns(
        cs.numeric().cast(pl.Float64),
        cs.boolean().cast(pl.Float64),
    )

    return (
        processed_dataframe,
        list(dict.fromkeys(updated_features)),
        list(dict.fromkeys(generated_dummy_columns)),
    )


def build_static_feature_list(
    dataframe: pl.DataFrame,
    generated_dummy_columns: Sequence[str],
    config_mode_name: str,
    final_features: Sequence[str] | None = None,
) -> list[str]:
    """Build the list of patient-level static features.

    Static features are excluded from time-series feature extraction and are
    described directly in the cohort summary table.

    Args:
        dataframe: Fully preprocessed time-series dataframe.
        generated_dummy_columns: Dummy columns created during categorical
            encoding.
        config_mode_name: Windowing or resampling configuration name.
        final_features: Optional final feature list used to filter the result.

    Returns:
        Static feature names available in the requested feature set.

    Raises:
        ValueError: If an expected static feature is absent from the dataframe
            when no final feature list is provided.
    """
    static_features = [
        *generated_dummy_columns,
        "score_glasgow",
        "age",
    ]

    if config_mode_name == "resampling_X_points":
        static_features.append("real_time_hours")
    elif (
        config_mode_name
        == "resampling_x_points_alea_lomax_prio_24h_no-fill"
    ):
        static_features.append("observed_duration")

    static_features = list(dict.fromkeys(static_features))

    if final_features is not None:
        return [
            feature
            for feature in static_features
            if feature in final_features
        ]

    missing_features = [
        feature
        for feature in static_features
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Static features are missing from the preprocessed dataframe: "
            f"{missing_features}"
        )

    return static_features


def add_variable_type_sections(
    table_dataframe: pd.DataFrame,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Add continuous and categorical section headers to a TableOne result.

    The summary statistics are left unchanged. Only the row layout is
    reorganized to make the distinction between continuous and categorical
    variables explicit.

    Args:
        table_dataframe: DataFrame produced by ``TableOne.tableone``.
        continuous_features: Variables displayed as continuous.
        categorical_features: Variables displayed as categorical.

    Returns:
        A copy of the TableOne dataframe containing explicit section rows.

    Raises:
        ValueError: If the TableOne dataframe does not use the expected
            two-level row index.
    """
    if (
        not isinstance(table_dataframe.index, pd.MultiIndex)
        or table_dataframe.index.nlevels != 2
    ):
        raise ValueError(
            "The TableOne result must have a two-level MultiIndex."
        )

    table_dataframe = table_dataframe.copy()
    first_index_level = (
        table_dataframe.index
        .get_level_values(0)
        .astype(str)
    )

    def create_section_row(title: str) -> pd.DataFrame:
        """Create an empty row used as a section header."""
        section_index = pd.MultiIndex.from_tuples(
            [(title, "")],
            names=table_dataframe.index.names,
        )

        return pd.DataFrame(
            "",
            index=section_index,
            columns=table_dataframe.columns,
        )

    def select_feature_rows(features: Sequence[str]) -> list[pd.DataFrame]:
        """Select complete TableOne row blocks for the requested features."""
        selected_blocks: list[pd.DataFrame] = []

        for feature in features:
            pattern = rf"^{re.escape(feature)}(?:,|$)"
            matching_rows = first_index_level.str.match(pattern)

            if matching_rows.any():
                selected_blocks.append(table_dataframe.loc[matching_rows])

        return selected_blocks

    output_blocks: list[pd.DataFrame] = []

    sample_size_rows = first_index_level == "n"
    if sample_size_rows.any():
        output_blocks.append(table_dataframe.loc[sample_size_rows])

    continuous_blocks = select_feature_rows(continuous_features)
    if continuous_blocks:
        output_blocks.append(create_section_row("Continuous variables"))
        output_blocks.extend(continuous_blocks)

    categorical_blocks = select_feature_rows(categorical_features)
    if categorical_blocks:
        output_blocks.append(create_section_row("Categorical variables"))
        output_blocks.extend(categorical_blocks)

    return pd.concat(output_blocks)


def build_tableone(
    df_clean: pl.DataFrame,
    categorical_data: pl.DataFrame,
    patient_col: str,
    target_col: str,
    final_features: list[str],
    generated_dummy_columns: list[str],
    categorical_source_columns: list[str],
    config_mode_name: str,
    output_dir: Path,
) -> TableOne:
    """Build and export a cohort TableOne with explicit variable sections.

    The input dataframe is reduced to one observation per patient. Original
    categorical columns are merged back after one-hot encoded columns are
    removed. The exported table separates continuous and categorical
    variables into visually distinct sections.

    Args:
        df_clean: Fully preprocessed dataframe, potentially containing several
            rows per patient.
        categorical_data: Original categorical variables collected before
            one-hot encoding.
        patient_col: Patient identifier column.
        target_col: Binary outcome column.
        final_features: Features retained after preprocessing.
        generated_dummy_columns: Dummy columns generated during categorical
            encoding.
        categorical_source_columns: Original categorical variables to display
            in the descriptive table.
        config_mode_name: Windowing or resampling configuration name.
        output_dir: Directory in which the table files are written.

    Returns:
        The original TableOne object containing the computed statistics.
        The exported HTML, CSV, and LaTeX files contain the additional
        continuous and categorical section headers.

    Raises:
        ValueError: If no static feature is available for description.
    """
    selected_columns = list(
        dict.fromkeys(
            [
                patient_col,
                target_col,
                *final_features,
            ]
        )
    )

    df_static = (
        df_clean
        .select(selected_columns)
        .unique(subset=[patient_col], keep="first")
    )

    static_features = build_static_feature_list(
        dataframe=df_static,
        generated_dummy_columns=generated_dummy_columns,
        config_mode_name=config_mode_name,
        final_features=final_features,
    )

    pd_static = df_static.to_pandas()

    dummy_columns = [
        column
        for column in generated_dummy_columns
        if column in pd_static.columns
    ]
    pd_static = pd_static.drop(columns=dummy_columns)

    pd_categorical = (
        categorical_data
        .unique(subset=[patient_col], keep="first")
        .to_pandas()
    )

    pd_static = pd_static.merge(
        pd_categorical,
        on=patient_col,
        how="left",
        suffixes=("", "_categorical"),
    )

    continuous_features = [
        feature
        for feature in [
            "age",
            "real_time_hours",
            "observed_duration",
        ]
        if feature in static_features
        and feature in pd_static.columns
    ]

    for feature in continuous_features:
        pd_static[feature] = pd.to_numeric(
            pd_static[feature],
            errors="coerce",
        )

    categorical_features = [
        feature
        for feature in categorical_source_columns
        if feature in pd_static.columns
        and feature not in continuous_features
    ]

    for feature in ["score_glasgow", "gender", target_col]:
        if (
            feature in pd_static.columns
            and feature not in categorical_features
            and feature not in continuous_features
        ):
            categorical_features.append(feature)

    describe_columns = [
        *continuous_features,
        *categorical_features,
    ]
    describe_columns = list(dict.fromkeys(describe_columns))

    if not describe_columns:
        raise ValueError(
            "No static features are available for the descriptive table."
        )

    table = TableOne(
        data=pd_static,
        columns=describe_columns,
        categorical=categorical_features,
        continuous=continuous_features,
        pval=False,
        decimals=2,
        missing=False,
    )

    formatted_table = add_variable_type_sections(
        table_dataframe=table.tableone,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_table.to_html(
        output_dir / "tableone_static_features.html",
    )
    formatted_table.to_csv(
        output_dir / "tableone_static_features.csv",
    )
    formatted_table.to_latex(
        output_dir / "tableone_static_features.tex",
    )

    return table