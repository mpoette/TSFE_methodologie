from collections.abc import Sequence
from typing import TypeAlias

import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import tsfel
import tsfel.feature_extraction.features as tsfel_features
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

PolarsFrame: TypeAlias = pl.DataFrame | pl.LazyFrame


# ---------------------------------------------------------------------
# TSFEL compatibility patches
# ---------------------------------------------------------------------

def _disabled_histogram_feature(
    signal: np.ndarray,
    nbins: int = 10,
) -> float:
    """Return a neutral value for disabled histogram-based TSFEL features.

    Args:
        signal:
            Input signal. It is intentionally ignored.
        nbins:
            Number of histogram bins. It is intentionally ignored.

    Returns:
        Always ``0.0``.
    """
    del signal, nbins
    return 0.0


# Disable histogram-based features that may fail on constant or invalid
# signals. This modifies TSFEL globally for the current Python process.
tsfel_features.hist_mode = _disabled_histogram_feature
tsfel_features.hist_entropy = _disabled_histogram_feature


def _ensure_dataframe(
    df: PolarsFrame,
) -> pl.DataFrame:
    """Return an eager Polars DataFrame.

    Args:
        df:
            Input Polars DataFrame or LazyFrame.

    Returns:
        The input data as an eager DataFrame.
    """
    if isinstance(df, pl.LazyFrame):
        return df.collect()

    return df


def _validate_columns(
    df: pl.DataFrame,
    required_columns: Sequence[str],
    context: str,
) -> None:
    """Validate that all required columns exist in a DataFrame.

    Args:
        df:
            DataFrame whose schema is checked.
        required_columns:
            Column names that must be present.
        context:
            Short description used in the error message.

    Raises:
        ValueError:
            If one or more required columns are missing.
    """
    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for {context}: "
            f"{sorted(missing_columns)}"
        )


def _sanitize_numeric_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean numerical train and test feature matrices.

    Infinite values are replaced with missing values. Missing values in both
    datasets are then imputed using medians computed exclusively from the
    training set.

    Features containing only missing or infinite values in the training set
    are removed because no training median can be computed for them.

    Args:
        train_df:
            Training feature matrix.
        test_df:
            Test feature matrix.

    Returns:
        Cleaned and column-aligned training and test feature matrices.

    Raises:
        ValueError:
            If no usable feature remains after cleaning.
    """
    train_clean = (
        train_df
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    test_clean = (
        test_df
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # Retain only features for which the training set contains at least one
    # usable value.
    valid_columns = train_clean.columns[
        train_clean.notna().any(axis=0)
    ]

    if valid_columns.empty:
        raise ValueError(
            "No usable numerical feature remains after cleaning."
        )

    train_clean = train_clean.loc[:, valid_columns]
    test_clean = test_clean.reindex(
        columns=valid_columns
    )

    train_medians = train_clean.median(axis=0)

    train_clean = train_clean.fillna(train_medians)
    test_clean = test_clean.fillna(train_medians)

    return train_clean, test_clean


def _process_single_patient(
    patient_df: pl.DataFrame,
    config: dict,
    feature_cols: list[str],
    patient_col: str,
    target_col: str,
) -> pl.DataFrame:
    """Extract TSFEL features for one patient or ICU stay.

    Args:
        patient_df:
            Time-series observations for one patient or stay.
        config:
            TSFEL feature extraction configuration.
        feature_cols:
            Dynamic columns from which features are extracted.
        patient_col:
            Patient or stay identifier column.
        target_col:
            Prediction target column.

    Returns:
        A one-row Polars DataFrame containing the extracted TSFEL features,
        patient identifier, and target value.

    Raises:
        ValueError:
            If the patient group is empty or does not contain exactly one
            patient identifier.
    """
    if patient_df.is_empty():
        raise ValueError(
            "Cannot extract TSFEL features from an empty patient group."
        )

    patient_ids = (
        patient_df[patient_col]
        .unique()
        .to_list()
    )

    if len(patient_ids) != 1:
        raise ValueError(
            "Each TSFEL group must contain exactly one patient identifier."
        )

    patient_id = patient_ids[0]

    # TSFEL currently expects a Pandas DataFrame.
    feature_data = (
        patient_df
        .select(feature_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Precision loss occurred in moment calculation.*"
            ),
            category=RuntimeWarning,
        )

        extracted_features = (
            tsfel.time_series_features_extractor(
                config,
                feature_data,
                fs=1,
                verbose=0,
            )
        )

    # The target is expected to be constant within each patient group.
    target_values = (
        patient_df[target_col]
        .drop_nulls()
        .unique()
        .to_list()
    )

    if len(target_values) != 1:
        raise ValueError(
            f"Target column {target_col!r} must contain exactly one "
            f"non-null value per patient. Patient: {patient_id!r}."
        )

    extracted_features[target_col] = target_values[0]
    extracted_features[patient_col] = patient_id

    feature_output_columns = [
        column
        for column in extracted_features.columns
        if column not in {
            patient_col,
            target_col,
        }
    ]

    extracted_features[feature_output_columns] = (
        extracted_features[feature_output_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype(float)
    )

    return pl.from_pandas(
        extracted_features,
        include_index=False,
    )


def extract_tsfel_per_patient(
    df: PolarsFrame,
    patient_col: str,
    time_col: str,
    feature_cols: list[str],
    target_col: str,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Extract TSFEL features independently for each patient or ICU stay.

    The input is sorted chronologically, partitioned by patient, and processed
    in parallel. Spectral TSFEL features are excluded.

    Args:
        df:
            Patient time-series dataset.
        patient_col:
            Patient or stay identifier column.
        time_col:
            Temporal ordering column.
        feature_cols:
            Dynamic variables used for feature extraction.
        target_col:
            Prediction target column.
        n_jobs:
            Number of parallel Joblib workers. ``-1`` uses all available
            logical CPU cores.

    Returns:
        A DataFrame containing one row per patient and one column per
        extracted TSFEL feature.

    Raises:
        ValueError:
            If the input is empty, required columns are missing, no feature
            column is provided, or no patient group is processed.
    """
    df = _ensure_dataframe(df)

    if df.is_empty():
        raise ValueError(
            "Cannot run TSFEL extraction on an empty DataFrame."
        )

    if not feature_cols:
        raise ValueError(
            "At least one dynamic feature column is required."
        )

    _validate_columns(
        df,
        [
            patient_col,
            time_col,
            target_col,
            *feature_cols,
        ],
        context="TSFEL extraction",
    )

    df = df.sort(
        [
            patient_col,
            time_col,
        ]
    )

    config = tsfel.get_features_by_domain()

    # Spectral features are intentionally excluded from this pipeline.
    config.pop("spectral", None)

    patient_groups = df.partition_by(
        patient_col,
        maintain_order=True,
    )

    print(
        "Starting parallel TSFEL extraction for "
        f"{len(patient_groups)} patients..."
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_patient)(
            patient_df=patient_group,
            config=config,
            feature_cols=feature_cols,
            patient_col=patient_col,
            target_col=target_col,
        )
        for patient_group in tqdm(
            patient_groups,
            desc="Parallel TSFEL extraction",
            unit="patient",
        )
    )

    if not results:
        raise ValueError(
            "No patient group was processed during TSFEL extraction."
        )

    return pl.concat(
        results,
        how="vertical",
    )


def filtrage_corr_var(
    Dataset_train: PolarsFrame,
    Dataset_test: PolarsFrame,
    patient_col: str,
    target_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Remove correlated and zero-variance TSFEL features.

    Correlation filtering and variance selection are fitted exclusively on
    the training set. The same selected columns and fitted variance selector
    are then applied to the test set.

    Infinite and missing values are imputed using medians computed from the
    training data.

    Args:
        Dataset_train:
            Training dataset containing metadata and TSFEL features.
        Dataset_test:
            Test dataset containing the same initial feature columns.
        patient_col:
            Patient or stay identifier column.
        target_col:
            Prediction target column.

    Returns:
        A tuple containing:

        - The filtered training DataFrame.
        - The filtered test DataFrame.
        - The selected feature names.

    Raises:
        ValueError:
            If required columns are missing, train and test schemas are
            incompatible, or no feature remains after filtering.
    """
    dataset_train = _ensure_dataframe(Dataset_train)
    dataset_test = _ensure_dataframe(Dataset_test)

    metadata_columns = [
        patient_col,
        target_col,
    ]

    _validate_columns(
        dataset_train,
        metadata_columns,
        context="training metadata",
    )
    _validate_columns(
        dataset_test,
        metadata_columns,
        context="test metadata",
    )

    train_metadata = (
        dataset_train
        .select(metadata_columns)
        .to_pandas()
        .reset_index(drop=True)
    )

    test_metadata = (
        dataset_test
        .select(metadata_columns)
        .to_pandas()
        .reset_index(drop=True)
    )

    feature_columns = sorted(
        column
        for column in dataset_train.columns
        if column not in metadata_columns
    )

    if not feature_columns:
        raise ValueError(
            "No feature is available for correlation filtering."
        )

    missing_test_features = (
        set(feature_columns) - set(dataset_test.columns)
    )

    if missing_test_features:
        raise ValueError(
            "The test set is missing training features: "
            f"{sorted(missing_test_features)}"
        )

    train_features = (
        dataset_train
        .select(feature_columns)
        .to_pandas()
    )

    test_features = (
        dataset_test
        .select(feature_columns)
        .to_pandas()
    )

    train_features, test_features = (
        _sanitize_numeric_dataframes(
            train_features,
            test_features,
        )
    )

    _, train_uncorrelated = tsfel.correlated_features(
        train_features,
        drop_correlated=True,
    )

    if train_uncorrelated.shape[1] == 0:
        raise ValueError(
            "No feature remains after correlation filtering."
        )

    test_uncorrelated = test_features.loc[
        :,
        train_uncorrelated.columns,
    ]

    variance_selector = VarianceThreshold()

    train_array = variance_selector.fit_transform(
        train_uncorrelated
    )
    test_array = variance_selector.transform(
        test_uncorrelated
    )

    selected_features = (
        train_uncorrelated.columns[
            variance_selector.get_support()
        ]
        .tolist()
    )

    if not selected_features:
        raise ValueError(
            "No feature remains after variance filtering."
        )

    train_selected = pd.DataFrame(
        train_array,
        columns=selected_features,
        index=train_metadata.index,
    )

    test_selected = pd.DataFrame(
        test_array,
        columns=selected_features,
        index=test_metadata.index,
    )

    train_final = pd.concat(
        [
            train_metadata,
            train_selected,
        ],
        axis=1,
    )

    test_final = pd.concat(
        [
            test_metadata,
            test_selected,
        ],
        axis=1,
    )

    print(
        "Shape after correlation filtering: "
        f"{train_uncorrelated.shape}"
    )
    print(
        "Shape after variance filtering: "
        f"{train_selected.shape}"
    )
    print(
        "Number of remaining features: "
        f"{len(selected_features)}"
    )

    return (
        pl.from_pandas(
            train_final,
            include_index=False,
        ),
        pl.from_pandas(
            test_final,
            include_index=False,
        ),
        selected_features,
    )


def filtrage_boruta(
    Dataset_train: PolarsFrame,
    Dataset_test: PolarsFrame,
    patient_col: str,
    target_col: str,
    max_iter: int = 100,
    seed: int = 42,
    include_tentative: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Select informative TSFEL features with the Boruta algorithm.

    Missing and infinite values are imputed with medians computed exclusively
    from the training set. Boruta is fitted only on the training data, and the
    resulting feature subset is then applied to the test data.

    Args:
        Dataset_train:
            Training dataset containing metadata and numerical features.
        Dataset_test:
            Test dataset containing the same initial feature columns.
        patient_col:
            Patient or stay identifier column.
        target_col:
            Prediction target column.
        max_iter:
            Maximum number of Boruta iterations.
        seed:
            Random seed used by the random forest and Boruta.
        include_tentative:
            Whether to retain tentative Boruta features in addition to
            confirmed features.

    Returns:
        A tuple containing:

        - The selected training DataFrame.
        - The selected test DataFrame.
        - The selected feature names.

    Raises:
        ValueError:
            If required columns are missing, no feature is available, target
            values are missing, or Boruta selects no feature.
    """
    dataset_train = _ensure_dataframe(Dataset_train)
    dataset_test = _ensure_dataframe(Dataset_test)

    metadata_columns = [
        patient_col,
        target_col,
    ]

    _validate_columns(
        dataset_train,
        metadata_columns,
        context="Boruta training data",
    )
    _validate_columns(
        dataset_test,
        metadata_columns,
        context="Boruta test data",
    )

    feature_columns = sorted(
        column
        for column in dataset_train.columns
        if column not in metadata_columns
    )

    if not feature_columns:
        raise ValueError(
            "No feature is available for Boruta selection."
        )

    missing_test_features = (
        set(feature_columns) - set(dataset_test.columns)
    )

    if missing_test_features:
        raise ValueError(
            "The test set is missing training features: "
            f"{sorted(missing_test_features)}"
        )

    train_features_pd = (
        dataset_train
        .select(feature_columns)
        .to_pandas()
    )

    test_features_pd = (
        dataset_test
        .select(feature_columns)
        .to_pandas()
    )

    train_features_pd, test_features_pd = (
        _sanitize_numeric_dataframes(
            train_features_pd,
            test_features_pd,
        )
    )

    target = (
        dataset_train[target_col]
        .to_numpy()
        .ravel()
    )

    if pd.isna(target).any():
        raise ValueError(
            "The training target contains missing values."
        )

    random_forest = RandomForestClassifier(
        n_jobs=-1,
        max_depth=8,
        class_weight="balanced",
        random_state=seed,
    )

    feature_selector = BorutaPy(
        estimator=random_forest,
        n_estimators="auto",
        verbose=2,
        alpha=0.05,
        perc=90,
        max_iter=max_iter,
        random_state=seed,
    )

    feature_selector.fit(
        train_features_pd.to_numpy(dtype=float),
        target,
    )

    selected_mask = feature_selector.support_.copy()

    if include_tentative:
        selected_mask |= feature_selector.support_weak_

    selected_features = [
        column
        for column, selected
        in zip(
            train_features_pd.columns,
            selected_mask,
            strict=True,
        )
        if selected
    ]

    if not selected_features:
        raise ValueError(
            "Boruta did not select any feature."
        )

    train_selected_features = pl.from_pandas(
        train_features_pd.loc[
            :,
            selected_features,
        ],
        include_index=False,
    )

    test_selected_features = pl.from_pandas(
        test_features_pd.loc[
            :,
            selected_features,
        ],
        include_index=False,
    )

    # Horizontal concatenation is explicit and preserves row alignment.
    train_final = pl.concat(
        [
            dataset_train.select(metadata_columns),
            train_selected_features,
        ],
        how="horizontal",
    )

    test_final = pl.concat(
        [
            dataset_test.select(metadata_columns),
            test_selected_features,
        ],
        how="horizontal",
    )

    print(
        "Boruta completed: "
        f"{len(selected_features)} features retained."
    )

    return (
        train_final,
        test_final,
        selected_features,
    )


def generer_suffixes_tsfel(
    include_spectral: bool = False,
) -> list[str]:
    """Generate the list of known TSFEL feature-name suffixes.

    Args:
        include_spectral:
            Whether to include spectral features. This should remain ``False``
            when spectral features are excluded during extraction.

    Returns:
        TSFEL feature names sorted from longest to shortest.
    """
    config = tsfel.get_features_by_domain()

    if not include_spectral:
        config.pop("spectral", None)

    suffixes = [
        feature_name
        for domain_features in config.values()
        for feature_name in domain_features
    ]

    # Longest suffixes are matched first to avoid partial matches.
    return sorted(
        set(suffixes),
        key=len,
        reverse=True,
    )


def extraire_racine(
    name: str,
    suffixes: Sequence[str],
) -> str | None:
    """Extract the original variable name from a TSFEL feature name.

    TSFEL commonly creates names following this pattern:

    ``<variable>_<feature>``

    Some features may additionally end with a numerical component such as
    ``_0`` or ``_1``.

    Args:
        name:
            Complete TSFEL-generated column name.
        suffixes:
            Known TSFEL feature suffixes, preferably generated with
            ``generer_suffixes_tsfel``.

    Returns:
        The original variable name when a known suffix is found. The complete
        input name is returned when no suffix matches.
    """
    for suffix in suffixes:
        escaped_suffix = re.escape(suffix)

        pattern = (
            rf"_{escaped_suffix}(?:_\d+)?$"
        )

        match = re.search(
            pattern,
            name,
            flags=re.IGNORECASE,
        )

        if match:
            return name[:match.start()]

    return None