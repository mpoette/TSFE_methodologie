from collections.abc import Sequence

import polars as pl
import polars.selectors as cs


def encode_categorical_features(
    dataframe: pl.DataFrame,
    feature_mode: str,
    features_to_keep: Sequence[str],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Encode categorical model features and update the feature list.

    The function preserves the existing pipeline behavior:

    - admission type is always one-hot encoded;
    - GHM prefixes are encoded only in ``Mode Commonly Used``;
    - gender is encoded only in the custom/all feature modes;
    - the generated dummy-column names are returned for later static-feature
      detection.

    Args:
        dataframe:
            Input patient time-series dataframe.
        feature_mode:
            Selected feature configuration name.
        features_to_keep:
            Feature names selected before categorical encoding.

    Returns:
        A tuple containing:

        - the transformed dataframe;
        - the updated feature list;
        - every generated dummy-column name.
    """
    processed_dataframe = dataframe
    updated_features = list(features_to_keep)
    generated_dummy_columns: list[str] = []

    # ---------------------------------------------------------
    # Admission type
    # ---------------------------------------------------------
    processed_dataframe = (
        processed_dataframe
        .with_columns(
            pl.col("admission_type")
            .fill_null("Unknown")
        )
        .to_dummies(
            columns=["admission_type"]
        )
    )

    admission_dummy_columns = [
        column
        for column in processed_dataframe.columns
        if column.startswith("admission_type_")
    ]

    generated_dummy_columns.extend(
        admission_dummy_columns
    )

    if "admission_type" in updated_features:
        updated_features.remove("admission_type")
        updated_features.extend(
            admission_dummy_columns
        )

    # ---------------------------------------------------------
    # GHM categories
    # ---------------------------------------------------------
    if feature_mode == "Mode Commonly Used":
        processed_dataframe = (
            processed_dataframe
            .with_columns(
                pl.col("icu_ghm")
                .list.first()
                .cast(pl.String)
                .str.head(3)
                .fill_null("Unknown")
                .alias("icu_ghm_f3")
            )
        )

        ghm_dummies = (
            processed_dataframe
            .select("icu_ghm_f3")
            .to_dummies(
                columns=["icu_ghm_f3"]
            )
        )

        unknown_ghm_column = "icu_ghm_f3_Unknown"

        if unknown_ghm_column in ghm_dummies.columns:
            ghm_dummies = ghm_dummies.drop(
                unknown_ghm_column
            )

        ghm_dummy_columns = ghm_dummies.columns

        generated_dummy_columns.extend(
            ghm_dummy_columns
        )

        ghm_source_columns = {
            "icu_ghm",
            "icu_ghm_f3",
        }

        updated_features = [
            column
            for column in updated_features
            if column not in ghm_source_columns
        ]

        updated_features.extend(
            ghm_dummy_columns
        )

        existing_ghm_source_columns = [
            column
            for column in ghm_source_columns
            if column in processed_dataframe.columns
        ]

        processed_dataframe = pl.concat(
            [
                processed_dataframe.drop(
                    existing_ghm_source_columns
                ),
                ghm_dummies,
            ],
            how="horizontal",
        )

    # ---------------------------------------------------------
    # Gender
    # ---------------------------------------------------------
    gender_feature_modes = {
        "Mode Custom",
        "Mode All Without pmsi",
        "Mode All",
    }

    if feature_mode in gender_feature_modes:
        processed_dataframe = (
            processed_dataframe
            .with_columns(
                pl.col("gender")
                .fill_null("Unknown")
            )
            .to_dummies(
                columns=["gender"]
            )
        )

        gender_dummy_columns = [
            column
            for column in processed_dataframe.columns
            if column.startswith("gender_")
        ]

        generated_dummy_columns.extend(
            gender_dummy_columns
        )

        if "gender" in updated_features:
            updated_features.remove("gender")
            updated_features.extend(
                gender_dummy_columns
            )

    # Preserve the existing numeric conversion.
    processed_dataframe = (
        processed_dataframe
        .with_columns(
            cs.numeric().cast(pl.Float64),
            cs.boolean().cast(pl.Float64),
        )
    )

    updated_features = list(
        dict.fromkeys(updated_features)
    )

    generated_dummy_columns = list(
        dict.fromkeys(generated_dummy_columns)
    )

    return (
        processed_dataframe,
        updated_features,
        generated_dummy_columns,
    )



def build_static_feature_list(
    dataframe: pl.DataFrame,
    generated_dummy_columns: Sequence[str],
    config_mode_name: str,
) -> list[str]:
    """Build the list of patient-level features excluded from TSFEL.

    Args:
        dataframe:
            Fully preprocessed time-series dataframe.
        generated_dummy_columns:
            Dummy-column names created during categorical encoding.
        config_mode_name:
            Internal name of the windowing or resampling configuration.

    Returns:
        Existing dataframe columns that must be treated as static.
    """
    static_features = [
        *generated_dummy_columns,
        "score_glasgow",
        "age",
    ]

    if config_mode_name == "resampling_X_points":
        static_features.append(
            "real_time_hours"
        )

    elif (
        config_mode_name
        == "resampling_x_points_alea_lomax_prio_24h_no-fill"
    ):
        static_features.append(
            "observed_duration"
        )

    static_features = list(
        dict.fromkeys(static_features)
    )

    missing_static_features = [
        column
        for column in static_features
        if column not in dataframe.columns
    ]

    assert not missing_static_features, (
        "Some static features are missing from the preprocessed dataframe: "
        f"{missing_static_features}"
    )

    return static_features