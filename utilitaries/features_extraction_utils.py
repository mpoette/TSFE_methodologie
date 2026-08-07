"""Feature extraction, selection, and correlation analysis utilities."""

from collections.abc import Sequence
from typing import TypeAlias

import gc
import logging
import os
import re
import resource
import warnings

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tsfel
import pywt
from tsfel.feature_extraction.calc_features import calc_window_features
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from tqdm.auto import tqdm


# Explainability scores for TSFEL feature families, organized by tier.
# Higher scores indicate more clinically interpretable features.
explainability_scores = {
    # --- TIER 1: Highly explainable (90 - 100) ---
    "max": 100,
    "min": 100,
    "mean": 95,
    "median": 95,
    "peak to peak distance": 90,
    "area under the curve": 85,
    # ECDF percentiles are empirical quantiles and remain directly
    # interpretable in the original measurement unit.
    "ecdf percentile": 85,

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
    # Number of observations below an ECDF percentile threshold. This remains
    # interpretable, but depends more strongly on the window length.
    "ecdf percentile count": 75,

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
    # Keep both TSFEL spellings and historical aliases. TSFEL-generated
    # columns use "coefficient", not only the abbreviated "coeff".
    "fft mean coefficient": 35,
    "fft mean coeff": 35,
    "spectrogram mean coefficient": 35,
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


def _validate_regular_sampling(
    df: pl.DataFrame,
    patient_col: str,
    time_col: str,
    expected_interval_hours: float,
    tolerance: float = 1e-9,
) -> None:
    """Check that observations follow a regular temporal grid.

    Args:
        df: Patient time-series DataFrame.
        patient_col: Patient or stay identifier column.
        time_col: Temporal ordering column.
        expected_interval_hours: Expected time between consecutive rows, in
            hours.
        tolerance: Acceptable deviation from the expected interval. Defaults to
            1e-9.

    Raises:
        ValueError: If irregular sampling is detected.
    """

    time_diffs = (
        df.sort([patient_col, time_col])
        .with_columns(
            pl.col(time_col)
            .diff()
            .over(patient_col)
            .alias("__time_diff")
        )
        .filter(pl.col("__time_diff").is_not_null())
    )

    irregular = time_diffs.filter(
        (
            pl.col("__time_diff") - expected_interval_hours
        ).abs() > tolerance
    )

    if irregular.height > 0:
        examples = irregular.select(
            patient_col,
            time_col,
            "__time_diff",
        ).head(10)

        raise ValueError(
            "Irregular temporal sampling detected. "
            f"Expected {expected_interval_hours} hour(s) between rows.\n"
            f"Examples:\n{examples}"
        )
    
def _log_memory(label: str) -> None:
    """Print the process peak resident memory with immediate flushing.

    On Linux, ``ru_maxrss`` is reported in KiB. This helper deliberately uses
    only the standard library so it remains available in constrained jobs.

    Args:
        label: Short label prefixed in the log line.
    """
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"[MEMORY] {label}: pid={os.getpid()}, "
        f"peak_rss={max_rss_kib / 1024:.1f} MiB",
        flush=True,
    )


def _safe_save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 120,
) -> None:
    """Save a Matplotlib figure without the expensive tight-bbox pass.

    Args:
        fig: The Matplotlib figure to save.
        path: Output file path.
        dpi: Image resolution in dots per inch.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path} (dpi={dpi})", flush=True)


def _compute_numeric_correlation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sanitize numeric data and compute one reusable Pearson matrix.

    Args:
        df: DataFrame containing numerical features.

    Returns:
        A tuple of the sanitized numeric DataFrame and the Pearson correlation
        matrix.

    Raises:
        ValueError: If fewer than two usable numeric columns remain.
    """
    logger.debug("CORR: Sanitizing numeric data...")
    _log_memory("before numeric sanitization")

    numeric_df = (
        df
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    empty_columns = numeric_df.columns[~numeric_df.notna().any(axis=0)]
    if len(empty_columns):
        logger.debug(
            "CORR: Dropping %d all-NaN column(s).", len(empty_columns)
        )
        numeric_df = numeric_df.drop(columns=empty_columns)

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two usable numeric columns are required for correlation."
        )

    _log_memory("after numeric sanitization")
    logger.debug(
        "CORR: Computing Pearson matrix for %d rows x %d features...",
        numeric_df.shape[0],
        numeric_df.shape[1],
    )
    corr_matrix = numeric_df.corr(method="pearson")
    logger.debug("CORR: Matrix ready: %s", corr_matrix.shape)
    _log_memory("after correlation computation")
    return numeric_df, corr_matrix



_EXPLAINABILITY_ALIASES = {
    # Explicit aliases are normalized before matching. Keeping this registry
    # separate avoids silently falling back to generic tokens such as "mean".
    "spectrogram mean coefficient": "spectrogram mean coefficient",
    "spectrogram mean coeff": "spectrogram mean coefficient",
    "fft mean coefficient": "fft mean coefficient",
    "fft mean coeff": "fft mean coefficient",
    "ecdf percentile count": "ecdf percentile count",
    "ecdf percentile": "ecdf percentile",
}


def _normalize_feature_text(value: str) -> str:
    """Normalize TSFEL feature names while preserving word boundaries.

    Args:
        value: Raw feature name string.

    Returns:
        Lowercase, stripped string with normalized internal whitespace.
    """
    return re.sub(r"[_\s]+", " ", str(value).lower()).strip()


def _build_explainability_matchers() -> list[tuple[str, str, int]]:
    """Build deterministic longest-first aliases for explainability matching.

    Returns:
        Sorted list of ``(alias, canonical_family, score)`` tuples, ordered
        by descending alias length.
    """
    matchers: list[tuple[str, str, int]] = []

    for alias, canonical_name in _EXPLAINABILITY_ALIASES.items():
        if canonical_name not in explainability_scores:
            raise KeyError(
                "Explainability alias references an unknown canonical family: "
                f"{canonical_name!r}."
            )
        matchers.append(
            (
                _normalize_feature_text(alias),
                canonical_name,
                explainability_scores[canonical_name],
            )
        )

    # Every configured family is also its own alias.
    for family_name, score in explainability_scores.items():
        matchers.append(
            (
                _normalize_feature_text(family_name),
                family_name,
                score,
            )
        )

    # Deduplicate aliases, preferring the explicit registry entry above.
    unique_matchers: dict[str, tuple[str, str, int]] = {}
    for alias, family_name, score in matchers:
        unique_matchers.setdefault(alias, (alias, family_name, score))

    return sorted(
        unique_matchers.values(),
        key=lambda item: len(item[0]),
        reverse=True,
    )


_EXPLAINABILITY_MATCHERS = _build_explainability_matchers()


def _get_explainability_details(
    feature_column_name: str,
    static_features: Sequence[str] | None = None,
) -> tuple[int, str]:
    """Return an explainability score and the matched feature family.

    Matching is longest-first and uses explicit TSFEL aliases. This prevents
    generic words such as ``mean`` from stealing compound families such as
    ``Spectrogram mean coefficient`` or ``Wavelet absolute mean``.

    Raw static variables receive score 100 because they are not transformed.

    Args:
        feature_column_name: The TSFEL-generated feature column name.
        static_features: Optional list of static feature names.

    Returns:
        A tuple of ``(score, canonical_family)``.
    """
    column = str(feature_column_name)

    if static_features is not None and column in set(static_features):
        return 100, "raw/static feature"

    normalized_column = _normalize_feature_text(column)

    for alias, canonical_family, score in _EXPLAINABILITY_MATCHERS:
        pattern = (
            r"(?:^|[^a-z0-9])"
            + re.escape(alias)
            + r"(?:$|[^a-z0-9])"
        )
        if re.search(pattern, normalized_column):
            return score, canonical_family

    return -1, "unknown"

def _get_explainability(
    tsfel_column_name: str,
    static_features: Sequence[str] | None = None,
) -> int:
    """Return only the explainability score for backward compatibility.

    Args:
        tsfel_column_name: The TSFEL-generated feature column name.
        static_features: Optional list of static feature names.

    Returns:
        The explainability score.
    """
    score, _ = _get_explainability_details(
        feature_column_name=tsfel_column_name,
        static_features=static_features,
    )
    return score




def audit_explainability_recognition(
    columns: Sequence[str],
    static_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize how every feature name is classified.

    This is useful before correlation filtering to detect unexpected generic
    matches or unknown TSFEL families.

    Args:
        columns: Feature column names to classify.
        static_features: Optional list of static feature names.

    Returns:
        DataFrame with ``feature``, ``matched_family``, and ``score`` columns.
    """
    records = []
    for column in columns:
        score, family = _get_explainability_details(
            feature_column_name=str(column),
            static_features=static_features,
        )
        records.append({
            "feature": str(column),
            "matched_family": family,
            "score": score,
        })

    audit_df = pd.DataFrame(records)
    summary = (
        audit_df
        .groupby(["matched_family", "score"], dropna=False)
        .size()
        .reset_index(name="feature_count")
        .sort_values(
            ["score", "matched_family"],
            ascending=[False, True],
        )
    )

    logger.debug("EXPLAINABILITY AUDIT:")
    for row in summary.itertuples(index=False):
        logger.debug(
            f"  - {row.matched_family}: score={row.score}, "
            f"features={row.feature_count}",
            flush=True,
        )

    unknown = audit_df.loc[
        audit_df["matched_family"] == "unknown",
        "feature",
    ].tolist()
    if unknown:
        logger.debug(
            "EXPLAINABILITY AUDIT: %d unknown feature(s). Examples: %s",
            len(unknown),
            unknown[:20],
        )

    return audit_df


# Type aliases.
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

    # 1. Sanitize infinite and missing values.
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

    # 2. Fit and apply variance threshold selector.
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
    static_features: Sequence[str] | None = None,
    log_decisions: bool = True,
    decision_log_path: str | None = None,
) -> pd.DataFrame:
    """Remove correlated features while logging each keep/drop decision.

    For every pair with ``abs(correlation) >= threshold``, the most
    explainable feature is retained. Equal scores are resolved
    deterministically by alphabetical order. Already-dropped features are not
    reconsidered, preserving the original greedy behavior.

    Args:
        df: DataFrame containing numerical features.
        threshold: Absolute correlation threshold above which a feature pair is
            considered redundant. Defaults to 0.9.
        static_features: Optional list of static feature names. These receive a
            maximum explainability score during tie-breaking.
        log_decisions: Whether to print each correlation keep/drop decision.
        decision_log_path: Optional file path to persist the decision log.

    Returns:
        A DataFrame with correlated features removed.

    Raises:
        ValueError: If threshold is not in (0, 1].
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    corr_matrix = df.corr(method="pearson")
    columns = list(df.columns)
    drop_set: set[str] = set()
    decision_lines: list[str] = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]

            if col_a in drop_set or col_b in drop_set:
                continue

            corr_value = corr_matrix.loc[col_a, col_b]

            if pd.isna(corr_value) or abs(corr_value) < threshold:
                continue

            score_a, family_a = _get_explainability_details(
                col_a,
                static_features=static_features,
            )
            score_b, family_b = _get_explainability_details(
                col_b,
                static_features=static_features,
            )

            if score_a > score_b:
                kept, dropped = col_a, col_b
                kept_score, dropped_score = score_a, score_b
                kept_family, dropped_family = family_a, family_b
                reason = f"{score_a} > {score_b}"
            elif score_b > score_a:
                kept, dropped = col_b, col_a
                kept_score, dropped_score = score_b, score_a
                kept_family, dropped_family = family_b, family_a
                reason = f"{score_b} > {score_a}"
            else:
                kept = min(col_a, col_b)
                dropped = max(col_a, col_b)
                if kept == col_a:
                    kept_family, dropped_family = family_a, family_b
                else:
                    kept_family, dropped_family = family_b, family_a
                kept_score = dropped_score = score_a
                reason = (
                    f"equal scores ({score_a}); alphabetical tie-break"
                )

            drop_set.add(dropped)

            line = (
                "CORR DECISION |corr|=%.6f "
                "(corr=%+.6f) :: "
                f"KEEP '{kept}' [{kept_family}, score={kept_score}] "
                f"vs DROP '{dropped}' "
                f"[{dropped_family}, score={dropped_score}] "
                f"because {reason}"
            )
            decision_lines.append(line)

            if log_decisions:
                print(line, flush=True)

    if decision_log_path is not None:
        parent = os.path.dirname(decision_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(decision_log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(decision_lines))
            if decision_lines:
                handle.write("\n")
        logger.debug(
            "CORR DECISION: Saved %d decision(s) to %s",
            len(decision_lines),
            decision_log_path,
        )

    logger.debug(
        "Dropped %d correlated feature(s) at threshold=%.2f; logged %d decision(s).",
        len(drop_set),
        threshold,
        len(decision_lines),
    )

    return df.loc[:, [col for col in columns if col not in drop_set]]




# Regex to match TSFEL wavelet column names with lossy Hz frequency labels.
_WAVELET_COLUMN_RE = re.compile(
    r"^(?P<prefix>.+_Wavelet "
    r"(?:absolute mean|energy|standard deviation|variance))_"
    r"[^_]+Hz$",
    flags=re.IGNORECASE,
)


def _rename_wavelet_columns_in_cycles_per_hour(
    columns: Sequence[str],
    sampling_frequency_hz: float,
    wavelet: str = "mexh",
) -> list[str]:
    """Replace lossy TSFEL wavelet frequency labels with unique CPH labels.

    TSFEL rounds wavelet frequencies to two decimal places in hertz when it
    builds output names. For slowly sampled clinical series, several distinct
    wavelet scales therefore receive the same ``0.0Hz`` label.

    The values themselves remain ordered by wavelet width. This function
    reconstructs that width within each variable/feature group and names each
    output with both its scale and its frequency in cycles per hour.

    Args:
        columns: TSFEL output column names.
        sampling_frequency_hz: Sampling frequency in hertz.
        wavelet: Wavelet family used by PyWavelets. Defaults to ``"mexh"``.

    Returns:
        Renamed column list with CPH-based wavelet labels.
    """
    group_occurrences: dict[str, int] = {}
    renamed: list[str] = []
    samples_per_hour = sampling_frequency_hz * 3600.0

    for raw_column in columns:
        column = str(raw_column)
        match = _WAVELET_COLUMN_RE.match(column)

        if match is None:
            renamed.append(column)
            continue

        prefix = match.group("prefix")
        scale = group_occurrences.get(prefix, 0) + 1
        group_occurrences[prefix] = scale

        # PyWavelets returns a normalized frequency in cycles/sample.
        # Multiplying by samples/hour yields cycles/hour.
        frequency_cph = float(
            pywt.scale2frequency(wavelet, scale) * samples_per_hour
        )

        renamed.append(
            f"{prefix}_scale_{scale:02d}_{frequency_cph:.12g}cph"
        )

    return renamed



# Regex to match TSFEL spectrogram column names with lossy Hz frequency labels.
_SPECTROGRAM_COLUMN_RE = re.compile(
    r"^(?P<prefix>.+_Spectrogram mean coefficient)_"
    r"[^_]+Hz$",
    flags=re.IGNORECASE,
)


def _rename_spectrogram_columns_in_cycles_per_hour(
    columns: Sequence[str],
    sampling_frequency_hz: float,
) -> list[str]:
    """Replace lossy TSFEL spectrogram labels with exact CPH bin labels.

    TSFEL computes ``N`` one-sided spectrogram frequency bins and rounds their
    labels to two decimals in hertz. With hourly sampling, many distinct bins
    are consequently named ``0.0Hz``. For a group containing ``N`` outputs,
    bin ``i`` has frequency ``i * fs / (2 * (N - 1))``.

    A two-pass implementation is used so the total number of bins is known
    before labels are generated. The bin index is included to guarantee stable
    and unique names even at very small sampling frequencies.

    Args:
        columns: TSFEL output column names.
        sampling_frequency_hz: Sampling frequency in hertz.

    Returns:
        Renamed column list with CPH-based spectrogram labels.
    """
    raw_columns = [str(column) for column in columns]
    group_sizes: dict[str, int] = {}

    for column in raw_columns:
        match = _SPECTROGRAM_COLUMN_RE.match(column)
        if match is not None:
            prefix = match.group("prefix")
            group_sizes[prefix] = group_sizes.get(prefix, 0) + 1

    group_indices: dict[str, int] = {}
    renamed: list[str] = []
    samples_per_hour = sampling_frequency_hz * 3600.0

    for column in raw_columns:
        match = _SPECTROGRAM_COLUMN_RE.match(column)
        if match is None:
            renamed.append(column)
            continue

        prefix = match.group("prefix")
        bin_index = group_indices.get(prefix, 0)
        group_indices[prefix] = bin_index + 1
        n_bins = group_sizes[prefix]

        if n_bins <= 1:
            frequency_cph = 0.0
        else:
            frequency_cph = (
                bin_index * samples_per_hour / (2.0 * (n_bins - 1))
            )

        renamed.append(
            f"{prefix}_bin_{bin_index:02d}_{frequency_cph:.12g}cph"
        )

    return renamed

def _make_unique_column_names(
    columns: Sequence[str],
) -> list[str]:
    """Return deterministic unique column names.

    The first occurrence of a name is preserved. Later occurrences receive a
    ``_duplicate_<n>`` suffix, where ``n`` starts at 1.

    Args:
        columns:
            Column names to make unique.

    Returns:
        Unique column names in their original order.
    """
    occurrence_counts: dict[str, int] = {}
    unique_columns: list[str] = []

    for column in columns:
        occurrence = occurrence_counts.get(column, 0)

        if occurrence == 0:
            unique_columns.append(column)
        else:
            unique_columns.append(
                f"{column}_duplicate_{occurrence}"
            )

        occurrence_counts[column] = occurrence + 1

    return unique_columns


def _process_single_patient(
    patient_df: pl.DataFrame,
    config: dict,
    feature_cols: list[str],
    patient_col: str,
    target_col: str,
    sampling_frequency_hz: float,
) -> pl.DataFrame:
    """Extract TSFEL features for one patient or ICU stay.

    Column names are passed explicitly through ``header_names`` because
    ``calc_window_features`` converts the input window to a NumPy array and
    cannot infer DataFrame column names by itself. This preserves descriptive
    output names such as ``fio2_corr_Mean`` instead of ``0_Mean``.

    Duplicate names generated internally by TSFEL are made deterministic by
    appending a ``_duplicate_<n>`` suffix.

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
        sampling_frequency_hz:
            Sampling frequency passed to TSFEL, expressed in hertz. For one
            observation per hour, this value is ``1 / 3600``.

    Returns:
        A one-row Polars DataFrame containing the extracted TSFEL features,
        patient identifier, and target value.

    Raises:
        ValueError:
            If the patient group is empty, contains more than one patient
            identifier, or no dynamic feature is available.
    """
    if patient_df.is_empty():
        raise ValueError(
            "Cannot extract TSFEL features from an empty patient group."
        )

    patient_ids = patient_df[patient_col].unique().to_list()

    if len(patient_ids) != 1:
        raise ValueError(
            "Each TSFEL group must contain exactly one patient identifier."
        )

    unique_feature_cols = list(dict.fromkeys(feature_cols))

    if not unique_feature_cols:
        raise ValueError(
            "At least one dynamic feature is required for TSFEL extraction."
        )

    feature_data = (
        patient_df
        .select(unique_feature_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    # Diagnostic logging: detect constant (static-like) and all-NaN columns.
    variances = feature_data.var(axis=0)
    constant_cols = sorted(variances[variances == 0.0].index.tolist())
    all_nan_cols = sorted(
        col for col in unique_feature_cols if feature_data[col].isna().all()
    )
    valid_per_col = feature_data.count(axis=0)
    min_valid = int(valid_per_col.min())

    # Raise if insufficient valid data for TSFEL extraction.
    if min_valid < 2:
        raise ValueError(
            f"Patient '{patient_ids[0]}' has only {min_valid} valid sample(s) "
            f"across {len(unique_feature_cols)} feature(s) (need >= 2). "
            f"Constant columns ({len(constant_cols)}): {constant_cols}. "
            f"All-NaN columns ({len(all_nan_cols)}): {all_nan_cols}. "
            f"This likely indicates static features leaking into TSFEL input."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )

        extracted_features = calc_window_features(
            config=config,
            window=feature_data,
            fs=sampling_frequency_hz,
            verbose=0,
            single_window=True,
            header_names=unique_feature_cols,
        )

    # TSFEL rounds CWT frequencies to 2 decimals in Hz. With hourly data,
    # distinct scales collapse to labels such as ``0.0Hz``. Reconstruct the
    # scale order and express the names in cycles/hour before checking for
    # any remaining duplicates.
    extracted_features.columns = _rename_wavelet_columns_in_cycles_per_hour(
        extracted_features.columns,
        sampling_frequency_hz=sampling_frequency_hz,
        wavelet="mexh",
    )

    # TSFEL applies the same lossy 2-decimal Hz rounding to spectrogram
    # frequency labels. Reconstruct exact bin frequencies in cycles/hour.
    extracted_features.columns = (
        _rename_spectrogram_columns_in_cycles_per_hour(
            extracted_features.columns,
            sampling_frequency_hz=sampling_frequency_hz,
        )
    )

    duplicate_columns = (
        extracted_features.columns[
            extracted_features.columns.duplicated(keep=False)
        ]
        .unique()
        .tolist()
    )

    if duplicate_columns:
        warnings.warn(
            "Residual duplicate TSFEL names remained after wavelet and spectrogram renaming. "
            "Deterministic suffixes were added to: "
            f"{duplicate_columns}",
            RuntimeWarning,
            stacklevel=2,
        )

    extracted_features.columns = _make_unique_column_names(
        extracted_features.columns,
    )

    target_values = (
        patient_df[target_col]
        .drop_nulls()
        .unique()
        .to_list()
    )
    target_value = target_values[-1] if target_values else None

    extracted_features[patient_col] = patient_ids[0]
    extracted_features[target_col] = target_value

    feature_output_columns = [
        column
        for column in extracted_features.columns
        if column not in {patient_col, target_col}
    ]

    extracted_features[feature_output_columns] = (
        extracted_features[feature_output_columns]
        .apply(pd.to_numeric, errors="coerce")
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
    n_jobs: int = 8,
    sampling_interval_hours: float = 1.0,
) -> pl.DataFrame:
    """Extract TSFEL features independently for each patient or ICU stay.

    The input is sorted chronologically, partitioned by patient, and processed
    in parallel using Joblib generator mode to minimize peak memory consumption.
    Fractal features are excluded, along with Histogram mode and MFCC spectral
    features. All other spectral features are retained.

    Args:
        df: Patient time-series dataset.
        patient_col: Patient or stay identifier column.
        time_col: Temporal ordering column.
        feature_cols: Dynamic variables used for feature extraction.
        target_col: Prediction target column.
        n_jobs: Number of parallel Joblib workers. Defaults to 8 to avoid OOM
            spikes on systems with large core counts.
        sampling_interval_hours: Time between two consecutive observations, in
            hours. Defaults to 1.0 (one observation per hour). It is converted
            to hertz before being passed to TSFEL: ``fs = 1 / (hours * 3600)``.

    Returns:
        A DataFrame containing one row per patient and one column per
        extracted TSFEL feature.

    Raises:
        ValueError: If the input is empty, required columns are missing, no
            feature column is provided, or no patient group is processed.
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

    if not np.isfinite(sampling_interval_hours) or sampling_interval_hours <= 0:
        raise ValueError(
            "sampling_interval_hours must be a finite, strictly positive "
            f"number, got {sampling_interval_hours!r}."
        )

    sampling_interval_seconds = sampling_interval_hours * 3600.0
    sampling_frequency_hz = 1.0 / sampling_interval_seconds
    sampling_frequency_per_hour = sampling_frequency_hz * 3600.0

    logger.debug(
        "TSFEL sampling: interval=%.12g hour(s), fs=%.15g Hz, rate=%.12g sample/hour",
        sampling_interval_hours,
        sampling_frequency_hz,
        sampling_frequency_per_hour,
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

    _validate_regular_sampling(
        df=df,
        patient_col=patient_col,
        time_col=time_col,
        expected_interval_hours=sampling_interval_hours,
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

    # Log feature columns passed to TSFEL to detect static features leaking in.
    logger.info(
        "TSFEL feature_cols (%d columns): %s",
        len(feature_cols),
        feature_cols,
    )

    logger.debug(
        "Starting parallel TSFEL extraction for %d patients (n_jobs=%d)",
        len(patient_groups),
        n_jobs,
    )

    # Placing tqdm directly over patient_groups displays the progress bar immediately as tasks are submitted
    results_generator = Parallel(n_jobs=n_jobs, return_generator=True)(
        delayed(_process_single_patient)(
            patient_df=patient_group,
            config=config,
            feature_cols=feature_cols,
            patient_col=patient_col,
            target_col=target_col,
            sampling_frequency_hz=sampling_frequency_hz,
        )
        for patient_group in tqdm(
            patient_groups,
            desc="Parallel TSFEL extraction",
            unit="patient",
        )
    )

    # Materialize the generator directly into a list
    results = list(results_generator)

    if not results:
        raise ValueError(
            "No patient group was processed during TSFEL extraction."
        )

    # Single vertical concatenation of all processed patient DataFrames.
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
    static_features: Sequence[str] | None = None,
    log_correlation_decisions: bool = True,
    correlation_decision_log_path: str | None = None,
    keep_static: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Remove correlated and zero-variance TSFEL features.

    Correlation filtering uses explainability scores to keep the most
    interpretable feature from each redundant pair. Variance selection and
    correlation filtering are fitted exclusively on the training set, and the
    same selected columns are then applied to the test data.

    When ``keep_static`` is True, static features are excluded from the
    variance and correlation filtering pipeline and always retained in the
    output, regardless of statistical redundancy.

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
        static_features:
            Optional list of static feature names. These receive a maximum
            explainability score during correlation tie-breaking. When
            ``keep_static`` is True, they are preserved unconditionally.
        log_correlation_decisions:
            Whether to print each correlation keep/drop decision.
        correlation_decision_log_path:
            Optional file path to persist the decision log.
        keep_static:
            When True, static features are excluded from variance and
            correlation filtering and always kept in the final result.
            Defaults to True.

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

    static_set = set(static_features) if static_features else set()

    # Separate static features that should be preserved from filtering.
    if keep_static and static_set:
        static_in_features = sorted(static_set.intersection(feature_columns))
        dynamic_features = [
            col for col in feature_columns if col not in static_set
        ]
    else:
        static_in_features = []
        dynamic_features = list(feature_columns)

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

    # Extract static columns to preserve them aside.
    if static_in_features:
        train_static = train_features[static_in_features]
        test_static = test_features[static_in_features]
        train_dynamic = train_features[dynamic_features]
        test_dynamic = test_features[dynamic_features]
    else:
        train_static = None
        test_static = None
        train_dynamic = train_features
        test_dynamic = test_features

    # Step 1: Sanitize and filter zero-variance features (dynamic only).
    train_var, test_var, var_features = assainir_et_filtrer_variance(
        train_features=train_dynamic,
        test_features=test_dynamic,
        threshold=0.0,
    )

    # Step 2: Audit feature-family recognition before making decisions.
    audit_explainability_recognition(
        columns=train_var.columns,
        static_features=static_features,
    )

    # Step 3: Drop correlated features using explainability scores.
    train_uncorrelated = drop_correlated_by_explainability(
        train_var,
        threshold=corr_threshold,
        static_features=static_features,
        log_decisions=log_correlation_decisions,
        decision_log_path=correlation_decision_log_path,
    )

    if train_uncorrelated.shape[1] == 0:
        raise ValueError(
            "No feature remains after correlation filtering."
        )

    test_uncorrelated = test_var.loc[
        :,
        train_uncorrelated.columns,
    ]

    dynamic_selected = train_uncorrelated.columns.tolist()

    # Reinject preserved static features at the end.
    all_selected_features = [*static_in_features, *dynamic_selected]

    if not all_selected_features:
        raise ValueError(
            "No feature remains after variance filtering."
        )

    train_parts = []
    test_parts = []

    if static_in_features:
        train_parts.append(train_static[static_in_features])
        test_parts.append(test_static[static_in_features])

    if dynamic_selected:
        train_parts.append(train_uncorrelated[dynamic_selected])
        test_parts.append(test_uncorrelated[dynamic_selected])

    train_selected = pd.concat(
        train_parts,
        axis=1,
    )
    test_selected = pd.concat(
        test_parts,
        axis=1,
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

    logger.debug(
        "Shape after correlation filtering: %s", train_uncorrelated.shape
    )
    logger.debug(
        "Number of remaining features: %d", len(all_selected_features)
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
        all_selected_features,
    )


def filtrage_boruta(
    Dataset_train: PolarsFrame,
    Dataset_test: PolarsFrame,
    patient_col: str,
    target_col: str,
    max_iter: int = 100,
    seed: int = 42,
    include_tentative: bool = False,
    static_features: Sequence[str] | None = None,
    keep_static: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Select informative TSFEL features with the Boruta algorithm.

    Missing and infinite values are imputed with medians computed exclusively
    from the training set. Boruta is fitted only on the training data, and the
    resulting feature subset is then applied to the test data.

    When ``keep_static`` is True, static features are excluded from the Boruta
    selection and always retained in the output, regardless of the algorithm
    assessment.

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
        static_features:
            Optional list of static feature names. When ``keep_static`` is
            True, they are preserved unconditionally.
        keep_static:
            When True, static features are excluded from Boruta selection
            and always kept in the final result. Defaults to True.

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

    static_set = set(static_features) if static_features else set()

    # Separate static features that should be preserved from Boruta selection.
    if keep_static and static_set:
        static_in_features = sorted(static_set.intersection(feature_columns))
        dynamic_features = [
            col for col in feature_columns if col not in static_set
        ]
    else:
        static_in_features = []
        dynamic_features = list(feature_columns)

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

    # Extract static columns to preserve them aside.
    if static_in_features:
        train_static = train_features_pd[static_in_features]
        test_static = test_features_pd[static_in_features]
        train_dynamic = train_features_pd[dynamic_features]
        test_dynamic = test_features_pd[dynamic_features]
    else:
        train_static = None
        test_static = None
        train_dynamic = train_features_pd
        test_dynamic = test_features_pd

    train_dynamic, test_dynamic = (
        _sanitize_numeric_dataframes(
            train_dynamic,
            test_dynamic,
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
        train_dynamic.to_numpy(dtype=float),
        target,
    )

    selected_mask = feature_selector.support_.copy()

    if include_tentative:
        selected_mask |= feature_selector.support_weak_

    dynamic_selected = [
        column
        for column, selected
        in zip(
            train_dynamic.columns,
            selected_mask,
            strict=True,
        )
        if selected
    ]

    # Reinject preserved static features at the beginning.
    all_selected_features = [*static_in_features, *dynamic_selected]

    if not all_selected_features:
        raise ValueError(
            "Boruta did not select any feature and no static feature is available."
        )

    # Build the final feature DataFrames with static + dynamic columns.
    train_selected_features = pl.from_pandas(
        pd.DataFrame(
            index=dataset_train.select(patient_col).to_pandas().index,
        ),
        include_index=False,
    )
    test_selected_features = pl.from_pandas(
        pd.DataFrame(
            index=dataset_test.select(patient_col).to_pandas().index,
        ),
        include_index=False,
    )

    if static_in_features:
        train_selected_features = pl.from_pandas(
            train_static,
            include_index=False,
        )
        test_selected_features = pl.from_pandas(
            test_static,
            include_index=False,
        )

    if dynamic_selected:
        train_dyn_pl = pl.from_pandas(
            train_dynamic.loc[:, dynamic_selected],
            include_index=False,
        )
        test_dyn_pl = pl.from_pandas(
            test_dynamic.loc[:, dynamic_selected],
            include_index=False,
        )

        if static_in_features:
            train_selected_features = pl.concat(
                [train_selected_features, train_dyn_pl],
                how="horizontal",
            )
            test_selected_features = pl.concat(
                [test_selected_features, test_dyn_pl],
                how="horizontal",
            )
        else:
            train_selected_features = train_dyn_pl
            test_selected_features = test_dyn_pl

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

    logger.debug(
        "Boruta completed: %d features retained (%d static + %d dynamic).",
        len(all_selected_features),
        len(static_in_features),
        len(dynamic_selected),
    )

    return (
        train_final,
        test_final,
        all_selected_features,
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

        if suffix.lower().startswith("wavelet "):
            pattern = (
                rf"_{escaped_suffix}"
                rf"(?:_scale_\d+_[0-9.eE+-]+cph|_\d+)?$"
            )
        else:
            pattern = rf"_{escaped_suffix}(?:_\d+)?$"

        match = re.search(
            pattern,
            name,
            flags=re.IGNORECASE,
        )

        if match:
            return name[:match.start()]

    return None



def extraire_variable_source(
    column_name: str,
    source_features: Sequence[str],
) -> str | None:
    """Identify the original signal behind a TSFEL feature column.

    The longest matching source name is retained so overlapping names such as
    ``fio2`` and ``fio2_corr`` are handled deterministically.

    Args:
        column_name:
            Complete TSFEL-generated feature name.
        source_features:
            Original dynamic variables passed to TSFEL.

    Returns:
        The matching source variable, or ``None`` when no source matches.
    """
    candidates = [
        source
        for source in source_features
        if column_name == source
        or column_name.startswith(f"{source}_")
    ]

    if not candidates:
        return None

    return max(candidates, key=len)


def grouper_colonnes_par_variable_source(
    columns: Sequence[str],
    source_features: Sequence[str],
    strict: bool = True,
) -> dict[str, list[str]]:
    """Group TSFEL feature columns by their original dynamic variable.

    Args:
        columns:
            TSFEL feature column names.
        source_features:
            Original dynamic variables supplied to TSFEL.
        strict:
            Raise an error when a column cannot be assigned to a source.

    Returns:
        Mapping from original variable name to its derived TSFEL columns.
    """
    unique_sources = list(dict.fromkeys(source_features))

    if not unique_sources:
        raise ValueError(
            "source_features must contain at least one original variable."
        )

    root_map: dict[str, list[str]] = {}
    unmatched_columns: list[str] = []

    for column in columns:
        source = extraire_variable_source(
            column_name=column,
            source_features=unique_sources,
        )

        if source is None:
            unmatched_columns.append(column)
            continue

        root_map.setdefault(source, []).append(column)

    if unmatched_columns:
        message = (
            f"{len(unmatched_columns)} TSFEL column(s) could not be assigned "
            "to an original source variable. Examples: "
            f"{unmatched_columns[:10]}"
        )

        if strict:
            raise ValueError(message)

        print(f"[WARNING] {message}", flush=True)

    empty_sources = [
        source
        for source in unique_sources
        if source not in root_map
    ]

    if empty_sources:
        print(
            "[WARNING] No TSFEL output found for source feature(s): "
            f"{empty_sources}",
            flush=True,
        )

    return root_map


def _log_source_feature_groups(
    root_map: dict[str, list[str]],
    source_features: Sequence[str],
    total_columns: int,
) -> list[str]:
    """Log source-variable groups and return them in pipeline order.

    Args:
        root_map: Mapping from original variable name to TSFEL columns.
        source_features: Original dynamic variables supplied to TSFEL.
        total_columns: Total number of TSFEL feature columns.

    Returns:
        Ordered list of source variable names present in ``root_map``.
    """
    ordered_sources = [
        source
        for source in dict.fromkeys(source_features)
        if source in root_map
    ]

    print(
        "[BLOCK GROUPING] "
        f"{total_columns} TSFEL feature(s) grouped into "
        f"{len(ordered_sources)} source variable(s):",
        flush=True,
    )

    for source in ordered_sources:
        print(
            f"  - {source}: {len(root_map[source])} TSFEL feature(s)",
            flush=True,
        )

    return ordered_sources

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


def _compute_block_figure_size(
    n_rows: int,
    n_cols: int,
    min_size: float = 12.0,
    max_size: float = 26.0,
    inches_per_feature: float = 0.18,
) -> tuple[float, float]:
    """Return a large but bounded figure size for a correlation block.

    Args:
        n_rows: Number of rows in the correlation matrix.
        n_cols: Number of columns in the correlation matrix.
        min_size: Minimum figure dimension in inches.
        max_size: Maximum figure dimension in inches.
        inches_per_feature: Inches allocated per feature dimension.

    Returns:
        A ``(width, height)`` tuple in inches.
    """
    width = min(max_size, max(min_size, n_cols * inches_per_feature))
    height = min(max_size, max(min_size, n_rows * inches_per_feature))
    return width, height


def _short_feature_label(
    feature_name: str,
    source_name: str,
) -> str:
    """Remove the repeated source prefix from a derived feature label.

    Args:
        feature_name: Full TSFEL feature name.
        source_name: Original source variable name.

    Returns:
        The feature name with the source prefix removed.
    """
    prefix = f"{source_name}_"
    return (
        feature_name[len(prefix):]
        if feature_name.startswith(prefix)
        else feature_name
    )


def _render_cross_block(
    sub_corr: pd.DataFrame,
    path: str,
    cmap: str,
    dpi: int,
    source_a: str,
    source_b: str,
    source_a_is_static: bool,
    source_b_is_static: bool,
    threshold: float | None = None,
    show_colorbar: bool = True,
) -> None:
    """Render one cross-source correlation block.

    Dynamic x dynamic blocks remain label-free for readability. For a
    static x dynamic block, the singleton static axis displays its source
    name and the dynamic axis displays the associated TSFEL feature names.

    Args:
        sub_corr: Sub-matrix of correlations between two source groups.
        path: Output image file path.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.
        source_a: Name of the row source variable.
        source_b: Name of the column source variable.
        source_a_is_static: Whether ``source_a`` is a static feature.
        source_b_is_static: Whether ``source_b`` is a static feature.
        threshold: Optional absolute correlation threshold for masking.
        show_colorbar: Whether to include a colorbar.
    """
    matrix = sub_corr.to_numpy(dtype=np.float32, copy=True)

    if threshold is not None:
        matrix = np.ma.masked_where(
            ~np.isfinite(matrix) | (np.abs(matrix) < threshold),
            matrix,
        )
    else:
        matrix = np.ma.masked_invalid(matrix)

    if source_a_is_static or source_b_is_static:
        # A singleton static row/column benefits from a wide, shallow figure.
        width = min(34.0, max(16.0, sub_corr.shape[1] * 0.24))
        height = min(18.0, max(5.0, sub_corr.shape[0] * 0.24))
    else:
        width, height = _compute_block_figure_size(
            n_rows=sub_corr.shape[0],
            n_cols=sub_corr.shape[1],
        )

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )

    if source_a_is_static:
        ax.set_yticks(np.arange(sub_corr.shape[0]))
        ax.set_yticklabels([source_a] * sub_corr.shape[0], fontsize=11)
    else:
        ax.set_yticks([])

    if source_b_is_static:
        ax.set_xticks(np.arange(sub_corr.shape[1]))
        ax.set_xticklabels([source_b] * sub_corr.shape[1], fontsize=11)
    elif source_a_is_static:
        ax.set_xticks(np.arange(sub_corr.shape[1]))
        ax.set_xticklabels(
            [
                _short_feature_label(column, source_b)
                for column in sub_corr.columns
            ],
            rotation=90,
            fontsize=7,
        )
    else:
        ax.set_xticks([])

    if source_b_is_static and not source_a_is_static:
        ax.set_yticks(np.arange(sub_corr.shape[0]))
        ax.set_yticklabels(
            [
                _short_feature_label(index, source_a)
                for index in sub_corr.index
            ],
            fontsize=7,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        colorbar.set_label("Pearson correlation", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

    # Avoid tight_layout on dense labels; reserve explicit margins instead.
    if source_a_is_static:
        fig.subplots_adjust(
            left=0.08, right=0.93, bottom=0.38, top=0.98
        )
    elif source_b_is_static:
        fig.subplots_adjust(
            left=0.38, right=0.93, bottom=0.08, top=0.98
        )
    else:
        fig.subplots_adjust(
            left=0.01, right=0.93, bottom=0.01, top=0.99
        )

    _safe_save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    del fig, ax, image, matrix
    gc.collect()



def _build_static_temporal_summary(
    corr_matrix: pd.DataFrame,
    root_map: dict[str, list[str]],
    static_roots: Sequence[str],
    temporal_roots: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate static-to-temporal correlations by source variable.

    Each cell contains the signed correlation belonging to the TSFEL feature
    pair with the largest absolute correlation for the corresponding
    ``static source x temporal source`` pair. A detail table records the exact
    columns responsible for every displayed cell.

    Args:
        corr_matrix: Full Pearson correlation matrix.
        root_map: Mapping from source variable to TSFEL columns.
        static_roots: Static source variable names.
        temporal_roots: Temporal source variable names.

    Returns:
        A tuple of the summary matrix and the detail DataFrame.
    """
    summary = pd.DataFrame(
        index=list(static_roots),
        columns=list(temporal_roots),
        dtype=float,
    )
    details: list[dict[str, object]] = []

    for static_source in static_roots:
        for temporal_source in temporal_roots:
            sub_corr = corr_matrix.loc[
                root_map[static_source],
                root_map[temporal_source],
            ]
            values = sub_corr.to_numpy(dtype=float, copy=False)

            if values.size == 0 or np.isnan(values).all():
                summary.loc[static_source, temporal_source] = np.nan
                details.append({
                    "static_feature": static_source,
                    "temporal_feature": temporal_source,
                    "static_column": None,
                    "temporal_tsfel_column": None,
                    "correlation": np.nan,
                    "absolute_correlation": np.nan,
                })
                continue

            flat_index = int(np.nanargmax(np.abs(values)))
            row_index, column_index = np.unravel_index(
                flat_index, values.shape
            )
            correlation = float(values[row_index, column_index])
            static_column = str(sub_corr.index[row_index])
            temporal_column = str(sub_corr.columns[column_index])

            summary.loc[static_source, temporal_source] = correlation
            details.append({
                "static_feature": static_source,
                "temporal_feature": temporal_source,
                "static_column": static_column,
                "temporal_tsfel_column": temporal_column,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            })

    return summary, pd.DataFrame(details)


def _render_static_temporal_summary(
    summary: pd.DataFrame,
    path: str,
    cmap: str,
    dpi: int,
    threshold: float | None = None,
    show_colorbar: bool = True,
    annotate_values: bool = True,
) -> None:
    """Render one compact static-features x temporal-sources matrix.

    Args:
        summary: Aggregated static x temporal correlation matrix.
        path: Output image file path.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.
        threshold: Optional absolute correlation threshold for masking.
        show_colorbar: Whether to include a colorbar.
        annotate_values: Whether to overlay numeric values on cells.
    """
    matrix = summary.to_numpy(dtype=np.float32, copy=True)
    invalid_mask = ~np.isfinite(matrix)

    if threshold is not None:
        matrix = np.ma.masked_where(
            invalid_mask | (np.abs(matrix) < threshold),
            matrix,
        )
    else:
        matrix = np.ma.masked_where(invalid_mask, matrix)

    n_rows, n_cols = summary.shape
    width = min(30.0, max(10.0, 1.35 * n_cols + 3.0))
    height = min(20.0, max(5.0, 0.72 * n_rows + 3.0))

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        summary.columns.tolist(), rotation=45, ha="right", fontsize=10
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(summary.index.tolist(), fontsize=10)
    ax.set_xlabel("Temporal source features", fontsize=11)
    ax.set_ylabel("Static features", fontsize=11)

    if annotate_values:
        raw_values = summary.to_numpy(dtype=float, copy=False)
        for row in range(n_rows):
            for column in range(n_cols):
                value = raw_values[row, column]
                if not np.isfinite(value):
                    continue
                if threshold is not None and abs(value) < threshold:
                    continue
                ax.text(
                    column, row, f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.035,
            pad=0.03,
            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        colorbar.set_label("Pearson correlation", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(
        left=0.18, right=0.93, bottom=0.25, top=0.97
    )
    _safe_save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    del fig, ax, image, matrix
    gc.collect()

def afficher_correlation_par_blocs(
    df: PolarsFrame,
    source_features: Sequence[str],
    static_features: Sequence[str] | None = None,
    exclude_cols: Sequence[str] | None = None,
    threshold: float = 0.5,
    cmap: str = "RdBu_r",
    output_dir: str | None = "comparison_figs",
    corr_matrix: pd.DataFrame | None = None,
    max_cross_heatmaps: int | None = None,
    block_heatmap_dpi: int = 160,
    show_colorbar: bool = True,
    annotate_static_summary: bool = True,
) -> None:
    """Generate temporal cross-blocks and one static-temporal summary.

    Dynamic x dynamic source pairs are rendered as detailed TSFEL blocks.
    Static x static pairs are omitted. All static x temporal relationships are
    condensed into one matrix whose rows are the original static features and
    whose columns are the original temporal source features. Each summary cell
    is the signed correlation of the strongest underlying TSFEL pair.

    Args:
        df: DataFrame containing TSFEL feature columns.
        source_features: Original dynamic and static variable names.
        static_features: Optional list of static feature names.
        exclude_cols: Columns to exclude from correlation computation.
        threshold: Absolute correlation threshold for threshold-mode output.
        cmap: Matplotlib colormap name.
        output_dir: Base directory for output images.
        corr_matrix: Optional precomputed Pearson correlation matrix.
        max_cross_heatmaps: Maximum number of temporal cross-blocks to render.
        block_heatmap_dpi: Image resolution for block heatmaps.
        show_colorbar: Whether to include colorbars on heatmaps.
        annotate_static_summary: Whether to annotate the static summary cells.

    Raises:
        ValueError:
            If the input is empty, the threshold is invalid, or static
            features are not present in source features.
    """
    df = _ensure_dataframe(df)
    if df.is_empty():
        raise ValueError(
            "Cannot display correlation blocks for an empty DataFrame."
        )
    if not (0 < threshold <= 1):
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    static_set = set(static_features or [])
    unknown_static = static_set - set(source_features)
    if unknown_static:
        raise ValueError(
            "static_features contains values absent from source_features: "
            f"{sorted(unknown_static)}"
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32, pl.Float64, pl.Int32, pl.Int64
        )
    ]
    if len(numeric_cols) < 2:
        raise ValueError("At least two numeric columns are required.")

    root_map = grouper_colonnes_par_variable_source(
        columns=numeric_cols,
        source_features=source_features,
        strict=True,
    )
    sorted_roots = _log_source_feature_groups(
        root_map=root_map,
        source_features=source_features,
        total_columns=len(numeric_cols),
    )
    ordered_cols = [
        column
        for source in sorted_roots
        for column in root_map[source]
    ]

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(
            df.select(ordered_cols).to_pandas()
        )
    else:
        missing = set(ordered_cols) - set(corr_matrix.columns)
        if missing:
            raise ValueError(
                "Precomputed correlation matrix is missing columns: "
                f"{sorted(missing)}"
            )
        corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
        print(
            "[STEP 2.1] Reusing precomputed correlation matrix.",
            flush=True,
        )

    normal_dir = (
        os.path.join(output_dir, "Normal_Mode")
        if output_dir is not None else None
    )
    threshold_dir = (
        os.path.join(output_dir, "Threshold_Mode")
        if output_dir is not None else None
    )
    for directory in (normal_dir, threshold_dir):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

    static_roots = [
        source for source in sorted_roots if source in static_set
    ]
    temporal_roots = [
        source for source in sorted_roots if source not in static_set
    ]

    if static_roots and temporal_roots:
        static_summary, static_details = _build_static_temporal_summary(
            corr_matrix=corr_matrix,
            root_map=root_map,
            static_roots=static_roots,
            temporal_roots=temporal_roots,
        )
        if normal_dir is not None:
            _render_static_temporal_summary(
                summary=static_summary,
                path=os.path.join(
                    normal_dir, "static_vs_temporal_summary.png"
                ),
                cmap=cmap,
                dpi=block_heatmap_dpi,
                threshold=None,
                show_colorbar=show_colorbar,
                annotate_values=annotate_static_summary,
            )
            static_details.to_csv(
                os.path.join(
                    normal_dir, "static_vs_temporal_details.csv"
                ),
                index=False,
            )

        if threshold_dir is not None:
            _render_static_temporal_summary(
                summary=static_summary,
                path=os.path.join(
                    threshold_dir, "static_vs_temporal_summary.png"
                ),
                cmap=cmap,
                dpi=block_heatmap_dpi,
                threshold=threshold,
                show_colorbar=show_colorbar,
                annotate_values=annotate_static_summary,
            )
            static_details.loc[
                static_details["absolute_correlation"] >= threshold
            ].to_csv(
                os.path.join(
                    threshold_dir,
                    "static_vs_temporal_details.csv",
                ),
                index=False,
            )

        for detail in static_details.itertuples(index=False):
            print(
                "[STATIC CORR] "
                f"{detail.static_feature} x {detail.temporal_feature}: "
                f"corr={detail.correlation:+.6f}, "
                f"via {detail.temporal_tsfel_column!r}",
                flush=True,
            )

    # Individual images are now reserved for temporal x temporal blocks.
    candidates: list[tuple[float, str, str]] = []
    for index_a, source_a in enumerate(temporal_roots):
        for source_b in temporal_roots[index_a + 1:]:
            sub_corr = corr_matrix.loc[
                root_map[source_a],
                root_map[source_b],
            ]
            values = sub_corr.to_numpy(dtype=float, copy=False)
            max_abs = (
                float(np.nanmax(np.abs(values)))
                if values.size and not np.isnan(values).all()
                else float("nan")
            )
            candidates.append((max_abs, source_a, source_b))

    candidates.sort(
        key=lambda item: -np.nan_to_num(item[0], nan=-1.0)
    )
    if max_cross_heatmaps is not None:
        candidates = candidates[:max_cross_heatmaps]

    print(
        f"[STEP 2.2] Rendering {len(candidates)} temporal x temporal "
        "cross-block pair(s).",
        flush=True,
    )

    normal_count = 0
    threshold_count = 0
    for rank, (max_abs, source_a, source_b) in enumerate(
        candidates, start=1
    ):
        sub_corr = corr_matrix.loc[
            root_map[source_a], root_map[source_b]
        ]
        safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_a)
        safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_b)
        filename = (
            f"cross_block_{rank:03d}_{safe_a}_x_{safe_b}.png"
        )

        render_kwargs = {
            "sub_corr": sub_corr,
            "cmap": cmap,
            "dpi": block_heatmap_dpi,
            "source_a": source_a,
            "source_b": source_b,
            "source_a_is_static": False,
            "source_b_is_static": False,
            "show_colorbar": show_colorbar,
        }

        if normal_dir is not None:
            _render_cross_block(
                path=os.path.join(normal_dir, filename),
                threshold=None,
                **render_kwargs,
            )
        normal_count += 1

        if np.isfinite(max_abs) and max_abs >= threshold:
            if threshold_dir is not None:
                _render_cross_block(
                    path=os.path.join(threshold_dir, filename),
                    threshold=threshold,
                    **render_kwargs,
                )
            threshold_count += 1

        print(
            f"[BLOCK {rank}/{len(candidates)}] "
            f"{source_a} x {source_b}: "
            f"{len(root_map[source_a])} x "
            f"{len(root_map[source_b])}, "
            f"max |corr|={max_abs:.4f}",
            flush=True,
        )
        del sub_corr

    print(
        "[STEP 2] Correlation figures complete: "
        f"Normal temporal blocks={normal_count}, "
        f"Threshold temporal blocks={threshold_count}, "
        f"static summary={'generated' if static_roots and temporal_roots else 'not generated'}.",
        flush=True,
    )

def generer_map_correlation_complete(
    df: PolarsFrame,
    source_features: Sequence[str],
    exclude_cols: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    output_path: str | None = "comparison_figs/corr_complete_map.png",
    corr_matrix: pd.DataFrame | None = None,
    max_full_heatmap_features: int = 300,
    max_block_heatmaps: int = 100,
    full_heatmap_dpi: int = 120,
    block_heatmap_dpi: int = 150,
) -> pl.DataFrame:
    """Generate a bounded-memory complete map or a capped set of blocks.

    Args:
        df: DataFrame containing TSFEL feature columns.
        source_features: Original dynamic and static variable names.
        exclude_cols: Columns to exclude from correlation computation.
        cmap: Matplotlib colormap name.
        output_path: Output image file path for the complete map.
        corr_matrix: Optional precomputed Pearson correlation matrix.
        max_full_heatmap_features: Maximum feature count before switching to
            block mode.
        max_block_heatmaps: Maximum number of block maps to generate.
        full_heatmap_dpi: Image resolution for the complete map.
        block_heatmap_dpi: Image resolution for individual blocks.

    Returns:
        The correlation matrix as a Polars DataFrame.

    Raises:
        ValueError:
            If the input is empty or fewer than two numeric columns remain.
    """
    df = _ensure_dataframe(df)
    if df.is_empty():
        raise ValueError("Cannot generate correlation map for an empty DataFrame.")

    exclude_set = set(exclude_cols) if exclude_cols else set()
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    if len(numeric_cols) < 2:
        raise ValueError("At least two numeric columns are required.")

    root_map = grouper_colonnes_par_variable_source(
        columns=numeric_cols,
        source_features=source_features,
        strict=True,
    )
    sorted_roots = _log_source_feature_groups(
        root_map=root_map,
        source_features=source_features,
        total_columns=len(numeric_cols),
    )
    ordered_cols = [
        column
        for source in sorted_roots
        for column in root_map[source]
    ]

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(
            df.select(ordered_cols).to_pandas()
        )
    else:
        missing = set(ordered_cols) - set(corr_matrix.columns)
        if missing:
            raise ValueError(
                "Precomputed correlation matrix is missing columns: "
                f"{sorted(missing)}"
            )
        corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
        print("[STEP 3.1] Reusing precomputed correlation matrix.", flush=True)

    n_features = len(ordered_cols)
    if n_features <= max_full_heatmap_features:
        print(
            f"[STEP 3.2] Rendering complete map ({n_features} features)...",
            flush=True,
        )
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            cbar_kws={"shrink": 0.6},
            ax=ax,
            linewidths=0,
            xticklabels=False,
            yticklabels=False,
        )
        cumulative = 0
        for root in sorted_roots[:-1]:
            cumulative += len(root_map[root])
            ax.axhline(y=cumulative, color="black", linewidth=1.2)
            ax.axvline(x=cumulative, color="black", linewidth=1.2)
        ax.set_title(
            f"Complete Correlation Map\n"
            f"{n_features} features across {len(sorted_roots)} groups"
        )
        fig.tight_layout()
        if output_path is not None:
            _safe_save_figure(fig, output_path, dpi=full_heatmap_dpi)
        plt.close(fig)
        del fig, ax
        gc.collect()
    else:
        blocks_dir = os.path.join(
            os.path.dirname(output_path) if output_path else "comparison_figs",
            "all_correlations",
        )
        os.makedirs(blocks_dir, exist_ok=True)
        print(
            f"[STEP 3.2] Complete map skipped: {n_features} features exceed "
            f"the safe limit of {max_full_heatmap_features}. "
            f"Generating at most {max_block_heatmaps} block map(s).",
            flush=True,
        )

        candidates: list[tuple[float, str, str]] = []
        for i, root_a in enumerate(sorted_roots):
            for root_b in sorted_roots[i:]:
                sub_corr = corr_matrix.loc[root_map[root_a], root_map[root_b]]
                values = sub_corr.abs().to_numpy()
                max_abs = 1.0 if root_a == root_b else (
                    float(np.nanmax(values))
                    if values.size and not np.isnan(values).all()
                    else float("nan")
                )
                candidates.append((max_abs, root_a, root_b))

        candidates.sort(
            key=lambda item: -np.nan_to_num(item[0], nan=-1.0)
        )
        for rank, (max_abs, root_a, root_b) in enumerate(
            candidates[:max_block_heatmaps], start=1
        ):
            sub_corr = corr_matrix.loc[root_map[root_a], root_map[root_b]]
            width = min(16, max(6, 0.35 * len(root_map[root_b])))
            height = min(14, max(5, 0.35 * len(root_map[root_a])))
            fig_b, ax_b = plt.subplots(figsize=(width, height))
            sns.heatmap(
                sub_corr,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=False,
                cbar_kws={"shrink": 0.8},
                ax=ax_b,
                linewidths=0,
                xticklabels=False,
                yticklabels=False,
            )
            title_suffix = "self-block" if root_a == root_b else f"max |corr|={max_abs:.3f}"
            ax_b.set_title(f"{root_a} x {root_b}\n{title_suffix}")
            fig_b.tight_layout()
            safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_a)
            safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_b)
            _safe_save_figure(
                fig_b,
                os.path.join(
                    blocks_dir,
                    f"block_{rank:03d}_{safe_a}_x_{safe_b}.png",
                ),
                dpi=block_heatmap_dpi,
            )
            plt.close(fig_b)
            del fig_b, ax_b, sub_corr
            gc.collect()

    return pl.from_pandas(corr_matrix, include_index=False)


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
    corr_matrix: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    """Estimate a correlation threshold from one reusable correlation matrix.

    Args:
        df: DataFrame containing numerical features.
        threshold_step: Step size for the threshold grid search.
        use_kneed: Whether to use the kneed library for elbow detection when
            available.
        save_path: Optional output path for the elbow curve figure.
        corr_matrix: Optional precomputed Pearson correlation matrix.

    Returns:
        A tuple of the estimated threshold and the elbow curve DataFrame.

    Raises:
        ValueError: If ``threshold_step`` is not in (0, 1).
    """
    if not (0 < threshold_step < 1):
        raise ValueError("threshold_step must be in (0, 1).")

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(df)
    else:
        print("[STEP 1.1] Reusing precomputed correlation matrix.", flush=True)

    thresholds = np.arange(0.05, 1.0, threshold_step)
    vars_remaining: list[int] = []
    abs_corr = np.abs(corr_matrix.to_numpy(dtype=float))
    np.fill_diagonal(abs_corr, np.nan)

    for thresh in thresholds:
        correlated_vars = np.nan_to_num(abs_corr >= thresh, nan=False).any(axis=1)
        vars_remaining.append(int((~correlated_vars).sum()))

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "vars_remaining": vars_remaining,
    })

    if use_kneed:
        try:
            from kneed import KneeLocator
            kl = KneeLocator(
                thresholds,
                vars_remaining,
                curve="convex",
                direction="decreasing",
            )
            elbow_threshold = float(
                kl.elbow if kl.elbow is not None else thresholds[len(thresholds) // 2]
            )
        except ImportError:
            elbow_threshold = float(_manual_elbow(thresholds, vars_remaining))
    else:
        elbow_threshold = float(_manual_elbow(thresholds, vars_remaining))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, vars_remaining, "o-", linewidth=2, markersize=6, label="Curve")
    ax.axvline(
        x=elbow_threshold,
        linestyle="--",
        alpha=0.7,
        label=f"Estimated elbow (t={elbow_threshold:.2f})",
    )
    elbow_idx = int(np.argmin(np.abs(thresholds - elbow_threshold)))
    ax.plot(elbow_threshold, vars_remaining[elbow_idx], "*", markersize=16)
    ax.set_xlabel("Correlation threshold")
    ax.set_ylabel("Number of remaining variables")
    ax.set_title("Elbow curve — Correlation threshold selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _safe_save_figure(fig, save_path, dpi=150)
    plt.close(fig)
    del fig, ax
    gc.collect()

    print(f"Estimated optimal threshold: {elbow_threshold:.2f}", flush=True)
    print(
        f"Remaining variables at this threshold: {vars_remaining[elbow_idx]}",
        flush=True,
    )
    return elbow_threshold, results_df


def generate_correlation_analysis(
    df: pd.DataFrame,
    source_features: Sequence[str],
    static_features: Sequence[str] | None = None,
    output_folder: str = "comparison_figs",
    threshold_step: float = 0.05,
    use_kneed: bool = True,
    corr_threshold: float | None = None,
    max_cross_heatmaps: int | None = None,
    block_heatmap_dpi: int = 160,
    show_colorbar: bool = True,
    annotate_static_summary: bool = True,
    max_full_heatmap_features: int = 300,
    full_heatmap_dpi: int = 140,
) -> dict[str, object]:
    """Run dual-mode correlation analysis with static-aware block output.

    Args:
        df: DataFrame containing numerical features.
        source_features: Original dynamic and static variable names.
        static_features: Optional list of static feature names.
        output_folder: Directory for output figures.
        threshold_step: Step size for the elbow threshold estimation.
        use_kneed: Whether to use the kneed library for elbow detection.
        corr_threshold: Optional fixed correlation threshold. Estimated from
            data when None.
        max_cross_heatmaps: Maximum number of temporal cross-blocks.
        block_heatmap_dpi: Image resolution for block heatmaps.
        show_colorbar: Whether to include colorbars.
        annotate_static_summary: Whether to annotate the static summary.
        max_full_heatmap_features: Maximum feature count for the full map.
        full_heatmap_dpi: Image resolution for the complete map.

    Returns:
        Dictionary containing ``optimal_threshold``, ``elbow_results``, and
        ``corr_matrix``.

    Raises:
        ValueError: If the final correlation threshold is not in (0, 1].
    """
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 60, flush=True)
    print("Preparing reusable correlation matrix...", flush=True)
    print("=" * 60, flush=True)
    numeric_df, corr_matrix_pd = _compute_numeric_correlation(df)

    print("=" * 60, flush=True)
    print(
        "Step 1: Estimating optimal correlation threshold...",
        flush=True,
    )
    print("=" * 60, flush=True)
    estimated_threshold, elbow_results = estimate_correlation_threshold(
        df=numeric_df,
        threshold_step=threshold_step,
        use_kneed=use_kneed,
        save_path=os.path.join(output_folder, "elbow_curve.png"),
        corr_matrix=corr_matrix_pd,
    )

    final_threshold = (
        float(corr_threshold)
        if corr_threshold is not None
        else estimated_threshold
    )
    if not (0 < final_threshold <= 1):
        raise ValueError(
            "Correlation threshold must be in (0, 1], "
            f"got {final_threshold}."
        )

    print("\n" + "=" * 60, flush=True)
    print(
        "Step 2: Generating Normal_Mode and Threshold_Mode blocks...",
        flush=True,
    )
    print("=" * 60, flush=True)

    df_pl = pl.from_pandas(numeric_df, include_index=False)
    afficher_correlation_par_blocs(
        df=df_pl,
        source_features=source_features,
        static_features=static_features,
        threshold=final_threshold,
        output_dir=output_folder,
        corr_matrix=corr_matrix_pd,
        max_cross_heatmaps=max_cross_heatmaps,
        block_heatmap_dpi=block_heatmap_dpi,
        show_colorbar=show_colorbar,
        annotate_static_summary=annotate_static_summary,
    )

    # Keep the global map only when its size remains safe. Produce both an
    # unmasked and a threshold-masked version in the corresponding folders.
    n_features = corr_matrix_pd.shape[0]
    if n_features <= max_full_heatmap_features:
        print(
            f"[GLOBAL MAP] {n_features} features <= "
            f"{max_full_heatmap_features}: rendering both modes.",
            flush=True,
        )
        for mode, threshold_value in (
            ("Normal_Mode", None),
            ("Threshold_Mode", final_threshold),
        ):
            matrix = corr_matrix_pd.to_numpy(
                dtype=np.float32, copy=True
            )
            if threshold_value is not None:
                matrix = np.ma.masked_where(
                    ~np.isfinite(matrix)
                    | (np.abs(matrix) < threshold_value),
                    matrix,
                )
            else:
                matrix = np.ma.masked_invalid(matrix)

            fig, ax = plt.subplots(figsize=(18, 16))
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.03,
                pad=0.02,
                ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            )
            colorbar.set_label("Pearson correlation")
            fig.subplots_adjust(
                left=0.01, right=0.93, bottom=0.01, top=0.99
            )
            _safe_save_figure(
                fig,
                os.path.join(
                    output_folder, mode, "corr_complete_map.png"
                ),
                dpi=full_heatmap_dpi,
            )
            plt.close(fig)
            del fig, ax, image, matrix
            gc.collect()
    else:
        print(
            f"[GLOBAL MAP] Skipped: {n_features} > "
            f"{max_full_heatmap_features}.",
            flush=True,
        )

    print("\n" + "=" * 60, flush=True)
    print("Analysis complete!", flush=True)
    print(f"  Threshold used: {final_threshold:.2f}", flush=True)
    print(
        f"  Normal maps: {output_folder}/Normal_Mode/",
        flush=True,
    )
    print(
        f"  Threshold maps: {output_folder}/Threshold_Mode/",
        flush=True,
    )
    _log_memory("analysis complete")
    print("=" * 60, flush=True)

    return {
        "optimal_threshold": final_threshold,
        "elbow_results": elbow_results,
        "corr_matrix": pl.from_pandas(
            corr_matrix_pd, include_index=False
        ),
    }


def estimate_boruta_frequency_threshold(
    fold_feature_lists: list[list[str]],
    n_folds: int,
    output_folder: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """Estimate the optimal Boruta frequency threshold by elbow analysis.

    Given the list of Boruta-selected features for each fold, this function
    computes how often each feature appears across all folds, then finds the
    optimal frequency threshold using the elbow method (maximum distance to
    the chord between the first and last points).

    Args:
        fold_feature_lists:
            A list of length ``n_folds``, where each element is the list of
            feature names selected by Boruta for that fold.
        n_folds:
            Total number of folds used in cross-validation.
        output_folder:
            Directory in which to save the elbow curve figure. If None, no
            figure is saved.

    Returns:
        A tuple containing:

        - The optimal frequency threshold (value in [0, 1]).
        - A DataFrame with columns ``frequency``, ``vars_remaining``, and
          ``n_folds`` for each threshold step.
    """
    if len(fold_feature_lists) != n_folds:
        raise ValueError(
            f"Expected {n_folds} fold feature lists, "
            f"got {len(fold_feature_lists)}."
        )

    # Count how many folds each feature was selected in.
    feature_counts: dict[str, int] = {}
    for fold_features in fold_feature_lists:
        for feature in fold_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    if not feature_counts:
        raise ValueError(
            "No feature was selected by Boruta in any fold."
        )

    # Build frequency bins: thresholds from 1/n_folds to 1.0.
    threshold_step = 1.0 / n_folds
    thresholds = np.arange(threshold_step, 1.0 + threshold_step, threshold_step)

    # For each threshold, count how many features have frequency >= threshold.
    vars_remaining: list[int] = []
    for thresh in thresholds:
        count = sum(
            1 for freq in feature_counts.values() if freq / n_folds >= thresh
        )
        vars_remaining.append(count)

    results_df = pd.DataFrame({
        "frequency": thresholds,
        "vars_remaining": vars_remaining,
    })

    # Elbow detection: maximum perpendicular distance to the chord.
    p1 = np.array([thresholds[0], vars_remaining[0]])
    p2 = np.array([thresholds[-1], vars_remaining[-1]])
    max_dist, elbow_idx = 0.0, 0

    for i in range(len(thresholds)):
        p = np.array([thresholds[i], vars_remaining[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    elbow_threshold = float(thresholds[elbow_idx])

    # Save the elbow curve figure.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            thresholds, vars_remaining, "o-",
            linewidth=2, markersize=6, label="Curve",
        )
        ax.axvline(
            x=elbow_threshold,
            linestyle="--",
            alpha=0.7,
            label=f"Elbow (freq={elbow_threshold:.2f})",
        )
        ax.plot(
            elbow_threshold, vars_remaining[elbow_idx],
            "*", markersize=16,
        )
        ax.set_xlabel("Boruta frequency threshold")
        ax.set_ylabel("Number of remaining features")
        ax.set_title(
            "Elbow curve — Boruta cross-fold frequency selection"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_path = os.path.join(output_folder, "boruta_frequency_elbow.png")
        _safe_save_figure(fig, save_path, dpi=150)
        plt.close(fig)

    logger.debug(
        "BORUTA FREQ: Elbow threshold=%.2f (%.0f/%d folds), %d features retained.",
        elbow_threshold,
        elbow_threshold * n_folds,
        n_folds,
        vars_remaining[elbow_idx],
    )

    return elbow_threshold, results_df


def get_boruta_features_at_threshold(
    fold_feature_lists: list[list[str]],
    n_folds: int,
    frequency_threshold: float,
) -> list[str]:
    """Return features selected by Boruta in at least a given fraction of folds.

    Args:
        fold_feature_lists:
            A list of length ``n_folds``, where each element is the list of
            feature names selected by Boruta for that fold.
        n_folds:
            Total number of folds used in cross-validation.
        frequency_threshold:
            Minimum fraction of folds in which a feature must appear to be
            retained. Must be in ``(0, 1]``.

    Returns:
        Sorted list of feature names that meet the frequency threshold.
    """
    if not (0 < frequency_threshold <= 1):
        raise ValueError(
            f"frequency_threshold must be in (0, 1], got {frequency_threshold}."
        )

    feature_counts: dict[str, int] = {}
    for fold_features in fold_feature_lists:
        for feature in fold_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    min_folds = int(np.ceil(frequency_threshold * n_folds))

    selected = sorted(
        feature
        for feature, count in feature_counts.items()
        if count >= min_folds
    )

    logger.debug(
        "BORUTA FREQ: %d features selected at frequency >= %.2f (>= %d/%d folds).",
        len(selected),
        frequency_threshold,
        min_folds,
        n_folds,
    )

    return selected


def collect_boruta_features_for_fold(
    fold_idx,
    train_idx,
    test_idx,
    X,
    y,
    groups,
    seed,
    **kwargs,
):
    """Run Boruta for one fold and return the selected feature list.

    This helper is used in the first phase of cross-fold Boruta selection:
    each fold runs correlation/variance filtering followed by Boruta, and the
    resulting feature list is collected. After all folds have been processed,
    the frequency of each feature across folds is analysed to determine a
    unified feature set.

    Args:
        fold_idx:
            Zero-based fold index.
        train_idx:
            Training indices for the current fold.
        test_idx:
            Validation indices for the current fold.
        X:
            Dataset used to recover patient identifiers for the split.
        y:
            Target array associated with the complete dataset.
        groups:
            Group array associated with the complete dataset.
        seed:
            Random seed used for feature selection.
        **kwargs:
            Pipeline configuration, including ``corr_threshold``,
            ``static_feats``, and ``keep_static``.

    Returns:
        The list of feature names selected by Boruta for this fold (after
        correlation/variance filtering). Returns an empty list when Boruta
        is disabled.
    """
    patient_col = kwargs["patient_col"]
    target_col = kwargs["target_col"]
    train_init_tsfel = kwargs["train_init"]
    static_feats = kwargs.get("static_feats", [])
    keep_static = kwargs.get("keep_static", True)
    boruta_filter = kwargs.get("boruta_filter", False)

    # Retrieve patient identifiers for the current fold.
    train_patients = (
        X[train_idx]
        .select(patient_col)
        .unique()
    )

    test_patients = (
        X[test_idx]
        .select(patient_col)
        .unique()
    )

    train_fold_tsfel = (
        train_init_tsfel
        .join(train_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    test_fold_tsfel = (
        train_init_tsfel
        .join(test_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    # Data is already globally filtered by corr/var in the pipeline,
    # so we skip the per-fold corr/var step here.
    train_clean, test_clean = train_fold_tsfel, test_fold_tsfel

    if not boruta_filter:
        # No Boruta: return all features after corr/var filtering.
        feature_names = (
            train_clean
            .select(pl.exclude(patient_col, target_col))
            .columns
        )
        return list(feature_names)

    # Step 2: Boruta selection.
    _, _, boruta_features = filtrage_boruta(
        train_clean,
        test_clean,
        patient_col,
        target_col,
        max_iter=100,
        seed=seed,
        static_features=static_feats,
        keep_static=keep_static,
    )

    return list(boruta_features)


def resolve_boruta_crossfold_features(
    X,
    y,
    groups,
    seed,
    exp,
    sgkf,
    compare_figs_dir,
    **kwargs,
):
    """Load or compute the unified cross-fold Boruta feature set.

    When the cached feature file already exists, it is loaded directly.
    Otherwise, the function runs Phase 1 (per-fold Boruta collection),
    Phase 1.5 (elbow analysis), and saves the unified feature set.

    Args:
        X:
            Dataset used to recover patient identifiers for the split.
        y:
            Target array associated with the complete dataset.
        groups:
            Group array associated with the complete dataset.
        seed:
            Random seed used for feature selection.
        exp:
            Experiment path configuration object.
        sgkf:
            Fitted StratifiedGroupKFold splitter.
        compare_figs_dir:
            Output directory for elbow analysis figures and CSV.
        **kwargs:
            Pipeline configuration forwarded to each fold.

    Returns:
        The unified list of Boruta feature names selected at the elbow
        frequency threshold across all folds.
    """
    boruta_filter = kwargs.get("boruta_filter", False)

    boruta_crossfold_path = exp.get_boruta_crossfold_path()
    
    if boruta_crossfold_path.exists():
        features = list(
            np.load(boruta_crossfold_path, allow_pickle=True)
        )
        logger.debug(
            "BORUTA: Loaded %d features from cache.", len(features)
        )
        return features

    if not boruta_filter:
        return []

    # Phase 1: Collect Boruta features across 5 folds.
    logger.debug("BORUTA PHASE 1: Collecting features across 5 folds...")
    fold_boruta_feature_lists = []

    for _fold_idx, (_train_idx, _val_idx) in enumerate(
        sgkf.split(X=X, y=y, groups=groups)
    ):
        logger.debug(
            "BORUTA PHASE 1: Collecting fold %d/5...", _fold_idx + 1
        )
        features = collect_boruta_features_for_fold(
            _fold_idx,
            _train_idx,
            _val_idx,
            X,
            y,
            groups,
            seed,
            **kwargs,
        )
        fold_boruta_feature_lists.append(features)
        logger.debug(
            "BORUTA PHASE 1: Fold %d: %d features selected.",
            _fold_idx + 1,
            len(features),
        )

    # Phase 1.5: Elbow analysis.
    logger.debug("BORUTA PHASE 1.5: Analyzing frequency elbow...")
    boruta_freq_threshold, boruta_freq_df = (
        estimate_boruta_frequency_threshold(
            fold_feature_lists=fold_boruta_feature_lists,
            n_folds=5,
            output_folder=str(compare_figs_dir),
        )
    )

    # Save frequency results as CSV.
    boruta_freq_df.to_csv(
        compare_figs_dir / "boruta_frequency_results.csv",
        index=False,
    )

    # Get the unified feature set at the elbow threshold.
    boruta_crossfold_features = get_boruta_features_at_threshold(
        fold_feature_lists=fold_boruta_feature_lists,
        n_folds=5,
        frequency_threshold=boruta_freq_threshold,
    )

    # Save the unified feature list.
    np.save(boruta_crossfold_path, boruta_crossfold_features)

    logger.debug(
        "BORUTA CROSS-FOLD: Unified feature set: %d features at frequency >= %.2f.",
        len(boruta_crossfold_features),
        boruta_freq_threshold,
    )

    return list(boruta_crossfold_features)
