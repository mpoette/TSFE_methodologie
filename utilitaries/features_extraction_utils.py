from collections.abc import Sequence
from typing import TypeAlias

import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tsfel
from tsfel.feature_extraction.calc_features import calc_window_features
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from tqdm.auto import tqdm


explainability_scores = {
    # --- TIER 1: Highly explainable (90 - 100) ---
    "max": 100,
    "min": 100,
    "mean": 95,
    "median": 95,
    "peak to peak distance": 90,
    "area under the curve": 85,

    # --- TIER 2: Moderately explainable (70 - 85) ---
    "average power": 80,
    "standard deviation": 80,
    "std": 80,
    "variance": 75,
    "root mean square": 75,
    "absolute energy": 75,
    "slope": 75,
    "zero crossing rate": 70,
    "positive turning points": 70,
    "negative turning points": 70,
    "mean absolute deviation": 70,
    "mean absolute diff": 70,
    "mean diff": 70,
    "median absolute deviation": 70,
    "median absolute diff": 70,
    "median diff": 70,
    "interquartile range": 70,
    "neighbourhood peaks": 70,
    "sum absolute diff": 70,

    # --- TIER 3: Poorly explainable (50 - 65) ---
    "distance": 65,
    "signal distance": 65,
    "skewness": 55,
    "median frequency": 55,
    "fundamental frequency": 55,
    "kurtosis": 50,
    "autocorrelation": 50,
    "centroid": 50,

    # --- TIER 4: Abstract / Informational (30 - 45) ---
    "entropy": 45,
    "lempel-ziv complexity": 40,
    "multiscale entropy": 40,
    "maximum frequency": 40,
    "max power spectrum": 35,
    "fft mean coeff": 35,
    "spectrogram mean coeff": 35,
    "human range energy": 30,

    # --- TIER 5: Purely spectral domain (10 - 25) ---
    "spectral centroid": 25,
    "spectral roll-off": 20,
    "spectral roll-on": 20,
    "spectral spread": 20,
    "spectral slope": 20,
    "spectral skewness": 15,
    "spectral kurtosis": 15,
    "spectral positive turning points": 15,
    "spectral positive turning": 15,
    "spectral decrease": 10,
    "spectral distance": 10,
    "spectral variation": 10,
    "spectral entropy": 10,
    "power bandwidth": 10,

    # --- TIER 6: Wavelets and others (0 - 5) ---
    "wavelet absolute mean": 5,
    "wavelet standard deviation": 5,
    "wavelet std": 5,
    "wavelet variance": 5,
    "wavelet energy": 5,
    "wavelet entropy": 5,
    "lpcc": 0,
}


def _get_explainability(tsfel_column_name: str) -> int:
    """Retrieve the explainability score of a TSFEL feature by column name.

    Args:
        tsfel_column_name:
            Name of the TSFEL feature column.

    Returns:
        Explainability score in [0, 100]. Returns -1 for unknown features.
    """
    col_lower = tsfel_column_name.lower()

    for feature_name, score in explainability_scores.items():
        if col_lower.endswith(feature_name):
            return score

    return -1


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

PolarsFrame: TypeAlias = pl.DataFrame | pl.LazyFrame


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


def assainir_et_filtrer_variance(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame | None = None,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Sanitize numeric features and apply variance threshold filtering.

    Missing and infinite values are first imputed using the median computed
    from the training set. A variance threshold selector is then fitted on the
    training data and applied to both train and optional test sets.

    Args:
        train_features:
            Pandas DataFrame containing training numerical features.
        test_features:
            Optional Pandas DataFrame containing test numerical features. Must
            have the same columns as train_features. Defaults to None.
        threshold:
            Features with a variance lower than or equal to this threshold
            will be removed. Defaults to 0.0 (removes constant features).

    Returns:
        A tuple containing:

        - The sanitized and variance-filtered training DataFrame.
        - The sanitized and variance-filtered test DataFrame (or None).
        - The list of selected feature names.

    Raises:
        ValueError:
            If input features are empty, schemas don't match, or no feature
            remains after variance filtering.
    """
    if train_features.empty:
        raise ValueError("Train features DataFrame cannot be empty.")

    if test_features is not None:
        if set(train_features.columns) != set(test_features.columns):
            raise ValueError(
                "Train and test features must have the exact same columns."
            )
        test_features_input = test_features[train_features.columns]
    else:
        test_features_input = pd.DataFrame(columns=train_features.columns)

    # 1. Sanitization (Imputation des Inf / NaN)
    try:
        clean_train, clean_test = _sanitize_numeric_dataframes(
            train_features,
            test_features_input,
        )
    except Exception:
        clean_train = train_features.replace([np.inf, -np.inf], np.nan)
        clean_train = clean_train.fillna(clean_train.median())

        if test_features is not None:
            clean_test = test_features.replace([np.inf, -np.inf], np.nan)
            clean_test = clean_test.fillna(train_features.median())
        else:
            clean_test = None

    # 2. Fit & Transform de la Variance
    selector = VarianceThreshold(threshold=threshold)
    train_array = selector.fit_transform(clean_train)

    selected_features = (
        clean_train.columns[selector.get_support()].tolist()
    )

    if not selected_features:
        raise ValueError("No feature remains after variance filtering.")

    train_filtered = pd.DataFrame(
        train_array,
        columns=selected_features,
        index=train_features.index,
    )

    if test_features is not None and clean_test is not None:
        test_array = selector.transform(clean_test)
        test_filtered = pd.DataFrame(
            test_array,
            columns=selected_features,
            index=test_features.index,
        )
    else:
        test_filtered = None

    return train_filtered, test_filtered, selected_features


def drop_correlated_by_explainability(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Remove correlated features, keeping the most explainable one.

    For each pair of features with absolute Pearson correlation greater than
    or equal to ``threshold``, the feature with the higher explainability
    score (from ``explainability_scores``) is retained. When scores are equal
    or both unknown (``-1``), the feature whose name sorts first alphabetically
    is kept to ensure deterministic behavior.

    Args:
        df:
            Pandas DataFrame containing numerical TSFEL features.
        threshold:
            Absolute correlation threshold above which a pair is considered
            redundant. Must be in ``(0, 1]``.

    Returns:
        The input DataFrame with the less explainable feature from each
        highly-correlated pair removed.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    corr_matrix = df.corr(method="pearson")
    columns = list(df.columns)
    drop_set: set[str] = set()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]

            # Skip if one of the pair is already marked for removal.
            if col_a in drop_set or col_b in drop_set:
                continue

            corr_value = corr_matrix.loc[col_a, col_b]

            if abs(corr_value) >= threshold:
                score_a = _get_explainability(col_a)
                score_b = _get_explainability(col_b)

                # Keep the feature with the higher explainability score.
                if score_a > score_b:
                    drop_set.add(col_b)
                elif score_b > score_a:
                    drop_set.add(col_a)
                else:
                    # Equal or both unknown - drop alphabetically last one.
                    if col_a >= col_b:
                        drop_set.add(col_a)
                    else:
                        drop_set.add(col_b)

    if drop_set:
        print(
            f"Dropped {len(drop_set)} correlated feature(s) "
            f"(threshold={threshold:.2f}, kept most explainable): "
            f"{sorted(drop_set)}"
        )

    return df.loc[:, [col for col in columns if col not in drop_set]]


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
        raise ValueError("Cannot extract TSFEL features from an empty patient group.")

    patient_ids = patient_df[patient_col].unique().to_list()
    if len(patient_ids) != 1:
        raise ValueError("Each TSFEL group must contain exactly one patient identifier.")

    patient_id = patient_ids[0]

    # 1. Deduplicate input columns
    unique_feature_cols = list(dict.fromkeys(feature_cols))

    feature_data = (
        patient_df
        .select(unique_feature_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    # 2. Temporary neutral aliasing (F0, F1...)
    alias_map = {col: f"F{i}" for i, col in enumerate(unique_feature_cols)}
    reverse_alias_map = {f"F{i}": col for i, col in enumerate(unique_feature_cols)}

    feature_data_tsfel = feature_data.rename(columns=alias_map)

    # 3. Use calc_window_features directly to BYPASS tsfel's broken .reindex() call
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )
        extracted_features = calc_window_features(
            config,
            feature_data_tsfel,
            fs=1,
            verbose=0,
            single_window=True,
        )

    # 4. FIX TSFEL BUG: Handle internal spectral duplicate column names if TSFEL produced any
    if extracted_features.columns.duplicated().any():
        # Disambiguate duplicate columns by appending _dup1, _dup2, etc.
        cols = pd.Series(extracted_features.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_mask = cols == dup
            dup_indices = cols[dup_mask].index
            for count, idx in enumerate(dup_indices[1:], start=1):
                cols.iloc[idx] = f"{dup}_dup{count}"
        extracted_features.columns = cols

    # 5. Restore original variable names safely
    new_columns = []
    for col in extracted_features.columns:
        parts = col.split("_", 1)
        if len(parts) == 2 and parts[0] in reverse_alias_map:
            orig_col = reverse_alias_map[parts[0]]
            feature_name = parts[1]
            new_columns.append(f"{orig_col}_{feature_name}")
        else:
            new_columns.append(col)

    extracted_features.columns = new_columns

    # 6. Attach metadata
    target_value = patient_df[target_col].drop_nulls().unique().last()
    extracted_features[target_col] = target_value
    extracted_features[patient_col] = patient_id

    feature_output_columns = [
        col for col in extracted_features.columns
        if col not in {patient_col, target_col}
    ]

    extracted_features[feature_output_columns] = (
        extracted_features[feature_output_columns]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )

    return pl.from_pandas(extracted_features, include_index=False)


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
    in parallel. Fractal features are excluded, along with Histogram mode and
    MFCC spectral features. All other spectral features are retained.

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

    # Fractal features are intentionally excluded from this pipeline.
    config.pop("fractal", None)

    # Histogram mode is excluded because it may crash on constant signals.
    config["statistical"].pop("Histogram mode", None)

    # MFCC is excluded because it reduces explainability.
    config["spectral"].pop("MFCC", None)

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
    corr_threshold: float = 0.9,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Remove correlated and zero-variance TSFEL features.

    Correlation filtering uses explainability scores to keep the most
    interpretable feature from each redundant pair. Variance selection and
    correlation filtering are fitted exclusively on the training set, and the
    same selected columns are then applied to the test set.

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
        corr_threshold:
            Absolute correlation threshold above which a feature pair is
            considered redundant and one member is dropped. Defaults to 0.9.

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

    # Step 1: Sanitize and filter zero-variance features.
    train_var, test_var, var_features = assainir_et_filtrer_variance(
        train_features=train_features,
        test_features=test_features,
        threshold=0.0,
    )

    # Step 2: Drop correlated features using explainability scores.
    train_uncorrelated = drop_correlated_by_explainability(
        train_var,
        threshold=corr_threshold,
    )

    if train_uncorrelated.shape[1] == 0:
        raise ValueError(
            "No feature remains after correlation filtering."
        )

    test_uncorrelated = test_var.loc[
        :,
        train_uncorrelated.columns,
    ]

    selected_features = train_uncorrelated.columns.tolist()

    if not selected_features:
        raise ValueError(
            "No feature remains after variance filtering."
        )

    train_selected = pd.DataFrame(
        train_uncorrelated,
        columns=selected_features,
        index=train_metadata.index,
    )

    test_selected = pd.DataFrame(
        test_uncorrelated,
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


def generer_suffixes_tsfel() -> list[str]:
    """Generate the list of known TSFEL feature-name suffixes.

    This function applies the same exclusions as ``extract_tsfel_per_patient``:

    - Fractal features are excluded entirely.
    - Histogram mode is excluded from statistical features.
    - MFCC is excluded from spectral features.

    Returns:
        TSFEL feature names sorted from longest to shortest.
    """
    config = tsfel.get_features_by_domain()

    # Apply the same exclusions as extract_tsfel_per_patient.
    config.pop("fractal", None)
    config["statistical"].pop("Histogram mode", None)
    config["spectral"].pop("MFCC", None)

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
        # TSFEL replaces spaces with underscores in output column names.
        escaped_suffix = re.escape(suffix.replace(" ", "_"))

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


def calculer_matrice_correlation(
    df: PolarsFrame,
    exclude_cols: Sequence[str] | None = None,
    threshold: float = 0.9,
) -> tuple[pl.DataFrame, list[tuple[str, str, float]]]:
    """Compute the correlation matrix and identify highly correlated pairs.

    Correlations are computed using Pearson's method on numeric columns.
    Missing values are excluded pairwise.

    Args:
        df:
            DataFrame containing numerical features.
        exclude_cols:
            Columns to exclude from the correlation computation (typically
            metadata columns such as patient ID or target).
        threshold:
            Absolute correlation threshold above which a pair is considered
            highly correlated. Must be in ``(0, 1]``.

    Returns:
        A tuple containing:

        - The full correlation matrix as a Polars DataFrame.
        - A list of ``(col_a, col_b, corr)`` tuples where the absolute
          correlation exceeds the threshold, sorted by descending absolute
          correlation.

    Raises:
        ValueError:
            If the input is empty, required columns are missing, or the
            threshold is out of range.
    """
    df = _ensure_dataframe(df)

    if df.is_empty():
        raise ValueError(
            "Cannot compute correlation on an empty DataFrame."
        )

    if not (0 < threshold <= 1):
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
        )
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            "At least two numeric columns are required for correlation."
        )

    # Convert to Pandas for correlation computation.
    numeric_df = (
        df.select(numeric_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    corr_matrix = numeric_df.corr(method="pearson")

    # Identify highly correlated pairs (upper triangle only).
    correlated_pairs: list[tuple[str, str, float]] = []

    for i, col_a in enumerate(numeric_cols):
        for j, col_b in enumerate(numeric_cols):
            if j <= i:
                continue

            corr_value = corr_matrix.loc[col_a, col_b]

            if pd.isna(corr_value):
                continue

            if abs(corr_value) >= threshold:
                correlated_pairs.append(
                    (col_a, col_b, float(corr_value))
                )

    # Sort by descending absolute correlation.
    correlated_pairs.sort(
        key=lambda pair: abs(pair[2]),
        reverse=True,
    )

    corr_pl = pl.from_pandas(corr_matrix, include_index=False)

    if correlated_pairs:
        print(
            f"Found {len(correlated_pairs)} feature pair(s) with "
            f"|correlation| >= {threshold}:"
        )
        for col_a, col_b, corr_value in correlated_pairs:
            print(f"  {col_a}  <->  {col_b}:  {corr_value:.4f}")
    else:
        print(
            f"No feature pair with |correlation| >= {threshold}."
        )

    return corr_pl, correlated_pairs


def afficher_correlation_par_blocs(
    df: PolarsFrame,
    exclude_cols: Sequence[str] | None = None,
    threshold: float = 0.5,
    figsize_per_block: tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    output_dir: str | None = "comparison_figs",
) -> None:
    """Display and save correlation matrix as block heatmaps by variable.

    Features are grouped by their original variable name (root). The full
    correlation matrix is displayed as a single heatmap with grid lines
    separating each block. For each pair of variable groups with at least
    one correlation above the threshold, a dedicated sub-heatmap is shown.

    All figures are saved to ``output_dir`` (default: ``comparison_figs/``).

    Args:
        df:
            DataFrame containing TSFEL features.
        exclude_cols:
            Columns to exclude from the visualization.
        threshold:
            Minimum absolute correlation to display a cross-block heatmap.
        figsize_per_block:
            Figure size (width, height) for each sub-heatmap.
        cmap:
            Matplotlib colormap name for the heatmaps.
        output_dir:
            Directory to save figures. Set to ``None`` to disable saving.

    Raises:
        ValueError:
            If the input is empty or contains fewer than two numeric columns.
    """
    df = _ensure_dataframe(df)

    if df.is_empty():
        raise ValueError(
            "Cannot display correlation blocks for an empty DataFrame."
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
        )
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    # Create output directory.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Extract roots and group columns by variable.
    suffixes = generer_suffixes_tsfel()

    root_map: dict[str, list[str]] = {}
    for col in numeric_cols:
        root = extraire_racine(col, suffixes)
        if root is None:
            root = col  # Fallback: feature is its own root
        root_map.setdefault(root, []).append(col)

    # Sort roots alphabetically for deterministic ordering.
    sorted_roots = sorted(root_map.keys())

    # Build ordered column list: all features of root 1, then root 2, etc.
    ordered_cols = [
        col
        for root in sorted_roots
        for col in root_map[root]
    ]

    # Compute correlation matrix on ordered columns.
    numeric_df = (
        df.select(ordered_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    corr_matrix = numeric_df.corr(method="pearson")

    # Build block boundaries for grid lines.
    block_sizes = [len(root_map[root]) for root in sorted_roots]
    boundaries = np.cumsum(block_sizes).tolist()

    # --- Full block heatmap ---
    # Limit DPI for large feature sets to avoid memory exhaustion.
    n_cols = len(ordered_cols)
    save_dpi = min(300, max(72, 15000 // n_cols))

    fig, ax = plt.subplots(
        figsize=(
            max(12, len(sorted_roots) * 1.5),
            max(10, n_cols // 3),
        )
    )

    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )

    # Draw block separator lines.
    for bound in boundaries:
        ax.axhline(y=bound, color="black", linewidth=2)
        ax.axvline(x=bound, color="black", linewidth=2)

    # Set tick labels at block midpoints (root names).
    mid_points = []
    cumulative = 0
    for i, root in enumerate(sorted_roots):
        cumulative += block_sizes[i]
        mid_points.append(cumulative)

    ax.set_xticks(
        [mp - block_sizes[i] // 2 for i, mp in enumerate(mid_points)]
    )
    ax.set_xticklabels(
        sorted_roots,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(
        [mp - block_sizes[i] // 2 for i, mp in enumerate(mid_points)]
    )
    ax.set_yticklabels(
        sorted_roots,
        fontsize=8,
    )

    ax.set_title(
        "Correlation Matrix (grouped by variable root)\n"
        f"{n_cols} features, {len(sorted_roots)} variable groups",
        fontsize=12,
    )

    plt.tight_layout()

    if output_dir is not None:
        path = os.path.join(output_dir, "corr_full_blocks.png")
        fig.savefig(path, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved: {path} (dpi={save_dpi})")

    plt.show()
    plt.close(fig)

    # --- Cross-block heatmaps for highly correlated pairs ---
    print(f"Cross-block heatmaps (|corr| >= {threshold}):")

    for i, root_a in enumerate(sorted_roots):
        cols_a = root_map[root_a]
        for j, root_b in enumerate(sorted_roots):
            if j <= i:
                continue

            cols_b = root_map[root_b]

            # Extract sub-matrix between the two groups.
            sub_corr = corr_matrix.loc[cols_a, cols_b]
            max_abs = float(sub_corr.abs().max().max())

            if max_abs < threshold:
                continue

            fig2, ax2 = plt.subplots(figsize=figsize_per_block)

            sns.heatmap(
                sub_corr,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={"shrink": 0.8},
                ax=ax2,
                annot=False,
                linewidths=0.3,
                linecolor="gray",
            )

            ax2.set_title(
                f"{root_a}  x  {root_b}\n"
                f"Max |corr| = {max_abs:.3f}",
                fontsize=11,
            )

            plt.tight_layout()

            if output_dir is not None:
                safe_a = root_a.replace(" ", "_").replace("/", "_")
                safe_b = root_b.replace(" ", "_").replace("/", "_")
                path = os.path.join(
                    output_dir,
                    f"cross_block_{safe_a}_x_{safe_b}.png",
                )
                fig2.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved: {path}")

            plt.show()
            plt.close(fig2)

    print("Done.")


def generer_map_correlation_complete(
    df: PolarsFrame,
    exclude_cols: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    output_path: str | None = "comparison_figs/corr_complete_map.png",
) -> pl.DataFrame:
    """Generate a single large correlation heatmap grouped by variable root.

    All features are ordered by their original variable name (root), and a
    single heatmap is displayed and optionally saved. Block separator lines
    and root labels make it easy to identify correlated groups.

    Args:
        df:
            DataFrame containing TSFEL features.
        exclude_cols:
            Columns to exclude from the visualization.
        cmap:
            Matplotlib colormap name for the heatmap.
        output_path:
            Path to save the figure. Set to ``None`` to disable saving.

    Returns:
        The correlation matrix as a Polars DataFrame.

    Raises:
        ValueError:
            If the input is empty or contains fewer than two numeric columns.
    """
    df = _ensure_dataframe(df)

    if df.is_empty():
        raise ValueError(
            "Cannot generate correlation map for an empty DataFrame."
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
        )
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    # Extract roots and group columns by variable.
    suffixes = generer_suffixes_tsfel()

    root_map: dict[str, list[str]] = {}
    for col in numeric_cols:
        root = extraire_racine(col, suffixes)
        if root is None:
            root = col
        root_map.setdefault(root, []).append(col)

    sorted_roots = sorted(root_map.keys())

    ordered_cols = [
        col
        for root in sorted_roots
        for col in root_map[root]
    ]

    # Compute correlation matrix.
    numeric_df = (
        df.select(ordered_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    corr_matrix = numeric_df.corr(method="pearson")

    block_sizes = [len(root_map[root]) for root in sorted_roots]
    boundaries = np.cumsum(block_sizes).tolist()

    # Determine figure size based on number of features.
    n_features = len(ordered_cols)
    fig_size = max(14, n_features // 4)

    # Limit DPI for large feature sets to avoid memory exhaustion.
    save_dpi = min(300, max(72, 15000 // n_features))
    dpi_threshold = 100  # Below this, the full map is considered unreadable.

    # Build index mapping: root -> row/col slice in corr_matrix.
    root_slices: list[tuple[str, slice]] = []
    cumulative = 0
    for root in sorted_roots:
        size = len(root_map[root])
        root_slices.append((root, slice(cumulative, cumulative + size)))
        cumulative += size

    if save_dpi < dpi_threshold:
        # DPI too low — generate individual block heatmaps instead.
        blocks_dir = os.path.join(
            os.path.dirname(output_path) if output_path else "comparison_figs",
            "all_correlations",
        )
        os.makedirs(blocks_dir, exist_ok=True)

        print(
            f"Full map DPI ({save_dpi}) is below {dpi_threshold}. "
            f"Generating individual block heatmaps to: {blocks_dir}/"
        )

        saved_count = 0
        for i, (root_a, sl_a) in enumerate(root_slices):
            for j, (root_b, sl_b) in enumerate(root_slices):
                if j < i:
                    continue

                sub_corr = corr_matrix.iloc[sl_a, sl_b]

                fig_b, ax_b = plt.subplots(
                    figsize=(max(6, len(root_b) * 0.6), max(5, len(root_a) * 0.6))
                )

                sns.heatmap(
                    sub_corr,
                    cmap=cmap,
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    ax=ax_b,
                    linewidths=0.3,
                    linecolor="gray",
                )

                if i == j:
                    ax_b.set_title(
                        f"{root_a}  (self-block)\n"
                        f"{len(root_map[root_a])} features",
                        fontsize=10,
                    )
                else:
                    max_abs = float(sub_corr.abs().max().max())
                    ax_b.set_title(
                        f"{root_a}  x  {root_b}\n"
                        f"Max |corr| = {max_abs:.3f}",
                        fontsize=10,
                    )

                plt.tight_layout()

                safe_a = root_a.replace(" ", "_").replace("/", "_")
                safe_b = root_b.replace(" ", "_").replace("/", "_")
                path = os.path.join(
                    blocks_dir,
                    f"block_{safe_a}_x_{safe_b}.png",
                )
                fig_b.savefig(path, dpi=200, bbox_inches="tight")
                saved_count += 1
                plt.close(fig_b)

        print(f"Saved {saved_count} block heatmap(s) to {blocks_dir}/")
    else:
        # DPI high enough — generate full heatmap normally.
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.6},
            ax=ax,
            linewidths=0.3,
            linecolor="lightgray",
            xticklabels=False,
            yticklabels=False,
        )

        # Draw block separator lines.
        for bound in boundaries:
            ax.axhline(y=bound, color="black", linewidth=2.5)
            ax.axvline(x=bound, color="black", linewidth=2.5)

        # Set root labels at block midpoints.
        mid_points = []
        cumulative = 0
        for i in range(len(sorted_roots)):
            cumulative += block_sizes[i]
            mid_points.append(cumulative - block_sizes[i] // 2)

        ax.set_xticks([mp + 0.5 for mp in mid_points])
        ax.set_xticklabels(
            sorted_roots,
            rotation=45,
            ha="right",
            fontsize=max(6, min(10, 200 // len(sorted_roots))),
        )
        ax.set_yticks([mp + 0.5 for mp in mid_points])
        ax.set_yticklabels(
            sorted_roots,
            fontsize=max(6, min(10, 200 // len(sorted_roots))),
        )

        ax.set_title(
            f"Complete Correlation Map\n"
            f"{n_features} features across {len(sorted_roots)} variable groups",
            fontsize=14,
            pad=15,
        )

        plt.tight_layout()

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
            print(f"Saved: {output_path} (dpi={save_dpi})")

        plt.show()
        plt.close(fig)

    corr_pl = pl.from_pandas(corr_matrix, include_index=False)

    return corr_pl


def _manual_elbow(
    thresholds: np.ndarray,
    values: list[int],
) -> float:
    """Compute the elbow point using the maximum distance to the chord.

    Args:
        thresholds:
            Array of threshold values (x-axis).
        values:
            List of corresponding y-values.

    Returns:
        The threshold value at the elbow point.
    """
    p1 = np.array([thresholds[0], values[0]])
    p2 = np.array([thresholds[-1], values[-1]])
    max_dist, elbow_idx = 0.0, 0

    for i, t in enumerate(thresholds):
        p = np.array([t, values[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return thresholds[elbow_idx]


def estimate_correlation_threshold(
    df: pd.DataFrame,
    threshold_step: float = 0.05,
    use_kneed: bool = True,
    save_path: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """Estimate the optimal correlation threshold using the elbow method.

    Generates a curve showing the number of remaining variables as a function
    of the correlation threshold, and automatically identifies the elbow point
    that maximizes the trade-off between the number of variables and the
    threshold value.

    Args:
        df:
            DataFrame containing numeric features.
        threshold_step:
            Step size between tested threshold values.
        use_kneed:
            Whether to use the kneed library for elbow detection. Falls back
            to a manual method if the library is not available.
        save_path:
            Optional path to save the figure as a PNG file.

    Returns:
        A tuple ``(optimal_threshold, results_df)`` where ``results_df``
        contains the columns ``'threshold'`` and ``'vars_remaining'``.
    """
    # Compute full Pearson correlation matrix.
    corr_matrix = df.corr(method="pearson")
    thresholds = np.arange(0.05, 1.0, threshold_step)
    vars_remaining = []


    for thresh in thresholds:
        # Identify pairs with |correlation| >= threshold.
        mask = (np.abs(corr_matrix) >= thresh).to_numpy().copy()
        np.fill_diagonal(mask, False)

        # A variable is "correlated" if it has at least one strong pair.
        correlated_vars = mask.any(axis=1)
        remaining = int((~correlated_vars).sum())
        vars_remaining.append(remaining)

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "vars_remaining": vars_remaining,
    })

    # Detect the elbow point.
    if use_kneed:
        try:
            from kneed import KneeLocator
            kl = KneeLocator(
                thresholds,
                vars_remaining,
                curve="convex",
                direction="decreasing",
            )
            elbow_threshold = kl.elbow or thresholds[len(thresholds) // 2]
        except ImportError:
            elbow_threshold = _manual_elbow(thresholds, vars_remaining)
    else:
        elbow_threshold = _manual_elbow(thresholds, vars_remaining)

    # Plot the elbow curve.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        thresholds, vars_remaining, "o-",
        linewidth=2, markersize=8, label="Curve",
    )
    ax.axvline(
        x=elbow_threshold, color="red", linestyle="--", alpha=0.7,
        label=f"Estimated elbow (t={elbow_threshold:.2f})",
    )
    elbow_idx = int(np.argmin(np.abs(thresholds - elbow_threshold)))
    ax.plot(
        elbow_threshold, vars_remaining[elbow_idx],
        "r*", markersize=20,
    )
    ax.set_xlabel("Correlation threshold", fontsize=12)
    ax.set_ylabel("Number of remaining variables", fontsize=12)
    ax.set_title(
        "Elbow curve — Correlation threshold selection", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thresholds)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)

    print(f"Estimated optimal threshold: {elbow_threshold:.2f}")
    print(f"Remaining variables at this threshold: {vars_remaining[elbow_idx]}")

    return elbow_threshold, results_df


def generate_correlation_analysis(
    df: pd.DataFrame,
    output_folder: str = "comparison_figs",
    threshold_step: float = 0.05,
    use_kneed: bool = True,
    corr_threshold: float | None = None,
) -> tuple[float, pd.DataFrame, pl.DataFrame]:
    """Run a full correlation analysis pipeline on a feature DataFrame.

    Combines threshold estimation, block heatmaps, and a complete correlation
    map into a single workflow. All figures are saved to ``output_folder``.

    If ``corr_threshold`` is provided, it overrides the automatically estimated
    threshold for the heatmap generation steps, while the elbow curve is still
    generated for reference.

    Args:
        df:
            DataFrame containing numeric TSFEL features.
        output_folder:
            Directory to save all generated figures.
        threshold_step:
            Step size between tested threshold values for elbow detection.
        use_kneed:
            Whether to use the kneed library for elbow detection.
        corr_threshold:
            Optional fixed correlation threshold to use for heatmap generation.
            When set to None (default), the elbow-estimated threshold is used.

    Returns:
        A tuple containing:

        - The correlation threshold used for heatmaps.
        - The elbow curve results DataFrame.
        - The complete correlation matrix as a Polars DataFrame.
    """
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 60)
    print("Step 1: Estimating optimal correlation threshold...")
    print("=" * 60)

    elbow_path = os.path.join(output_folder, "elbow_curve.png")
    estimated_threshold, elbow_results = estimate_correlation_threshold(
        df=df,
        threshold_step=threshold_step,
        use_kneed=use_kneed,
        save_path=elbow_path,
    )

    # Use manual threshold if provided, otherwise use estimated.
    final_threshold = corr_threshold if corr_threshold is not None else estimated_threshold

    if corr_threshold is not None:
        print()
        print(f"Note: Using manual threshold {corr_threshold:.2f} "
              f"(estimated: {estimated_threshold:.2f})")

    print()
    print("=" * 60)
    print(f"Step 2: Generating block heatmaps (threshold={final_threshold:.2f})...")
    print("=" * 60)

    # Convert to Polars for the visualization functions.
    df_pl = pl.from_pandas(df, include_index=True)

    afficher_correlation_par_blocs(
        df=df_pl,
        exclude_cols=None,
        threshold=final_threshold,
        output_dir=output_folder,
    )

    print()
    print("=" * 60)
    print("Step 3: Generating complete correlation map...")
    print("=" * 60)

    corr_map_path = os.path.join(output_folder, "corr_complete_map.png")
    corr_matrix = generer_map_correlation_complete(
        df=df_pl,
        exclude_cols=None,
        output_path=corr_map_path,
    )

    print()
    print("=" * 60)
    print("Analysis complete!")
    print(f"  Threshold used: {final_threshold:.2f}")
    if corr_threshold is not None:
        print(f"  (Manual override, estimated was: {estimated_threshold:.2f})")
    print(f"  Figures saved to: {output_folder}/")
    print("=" * 60)

    return final_threshold, elbow_results, corr_matrix
