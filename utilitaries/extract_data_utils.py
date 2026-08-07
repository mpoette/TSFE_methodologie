"""Data extraction utilities for ICU survival prediction pipelines.

This module provides functions for loading, cleaning, and preparing ICU
patient time-series data from Parquet sources. It handles encounter-level
duplicate removal, time-axis recalibration, and the generation of either
fixed or random 24-hour observation windows for model training.
"""

from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

PolarsFrame: TypeAlias = pl.DataFrame | pl.LazyFrame
RandomSeed: TypeAlias = int | np.random.Generator


# ---------------------------------------------------------------------
# Technical column names and preprocessing constants
# ---------------------------------------------------------------------

COL_DATE_MESURE = "utcChartTime"
COL_DATE_ADMISSION = "utcInTime"

ID_COL = "encounterId"
TIME_COL = "delta_hour"
CALIBRATED_TIME_COL = "heure_calibree"

WINDOW_SIZE = 24

def _ensure_dataframe(df: PolarsFrame) -> pl.DataFrame:
    """Return an eager Polars DataFrame.

    Args:
        df:
            Polars DataFrame or LazyFrame.

    Returns:
        The input data as an eager DataFrame.
    """
    if isinstance(df, pl.LazyFrame):
        return df.collect()

    return df

def remove_duplicates(df_no_ano: PolarsFrame, df: PolarsFrame, remove_prio_last : bool) -> pl.DataFrame:
    """Removes patient duplicates by retaining either their most or least recent encounter.

    This function accepts either a Polars DataFrame or LazyFrame for both inputs.
    It ensures they are evaluated into DataFrames, sorts the unanonymized data
    by admission time (`utcInTime`) to identify the targeted encounter for each 
    unique patient (`lifeTimeNumber`), and filters the target dataframe using 
    an inner join.

    Args:
        df_no_ano:
            The unanonymized Polars DataFrame or LazyFrame containing at least
            `lifeTimeNumber`, `utcInTime`, and `encounterId` columns. Used to
            determine the chronological order of encounters per patient.
        df:
            The target Polars DataFrame or LazyFrame to filter, which must
            contain an `encounterId` column.
        remove_prio_last:
            Whether to prioritize removing the latest encounters. If ``True``, 
            the earliest (first) encounter is kept. If ``False``, the most recent 
            (latest) encounter is kept.

    Returns:
        A collected Polars DataFrame containing only the rows corresponding to
        each patient's selected encounter.
    """
    df_no_ano = _ensure_dataframe(df_no_ano)
    df = _ensure_dataframe(df)
    df_keep_eId = (
    df_no_ano.sort(["utcInTime"], descending=[remove_prio_last])
      .unique(subset=["lifeTimeNumber"], keep="first")
    ).select("encounterId")

    df = df.join(df_keep_eId, on = "encounterId", how = "inner")
    return df


def extract_data_survie(
    file_path: str | Path,
) -> pl.LazyFrame:
    """Load survival data from a Parquet file.

    The function lazily reads the Parquet file and normalizes the types of
    selected columns. In particular, ``ecmo_all`` is converted from its
    string representation to a Boolean value.

    Args:
        file_path:
            Path to the Parquet file.

    Returns:
        A LazyFrame containing the loaded and normalized data.

    Raises:
        FileNotFoundError:
            If the Parquet file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Parquet file does not exist: {file_path}"
        )

    return (
        pl.scan_parquet(file_path)
        .with_columns(
            pl.col(TIME_COL).cast(pl.Float64, strict=False),
            pl.col("pam").cast(pl.Float64, strict=False),
            # Normalize the string representation of the ECMO indicator.
            pl.col("ecmo_all")
            .cast(pl.String, strict=False)
            .str.to_lowercase()
            .str.strip_chars()
            .eq("true")
            .alias("ecmo_all"),
        )
    )


def prepare_base_data(
    df: PolarsFrame,
    target_col: str,
    other_cols: list[str],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Clean the input data and recalibrate its time axis.

    Time is recalibrated independently for each ICU stay so that hour zero
    corresponds to the last available observation. Earlier observations are
    represented by negative values.

    Args:
        df:
            Patient time-series data.
        target_col:
            Name of the main target column.
        other_cols:
            Names of additional target columns.

    Returns:
        A tuple containing:

        - The cleaned historical observations.
        - One row per patient containing the earliest calibrated hour.

        Both values are ``None`` when preprocessing produces no usable data.

    Raises:
        ValueError:
            If the patient identifier, time column, target column, or one of
            the additional target columns is missing.
    """

    df = _ensure_dataframe(df)

    if df.is_empty():
        print(
            " │   ├─ XX:XX (H-0) ── "
            "✕ Blocked: empty DataFrame"
        )
        return None, None

    required_columns = {
        ID_COL,
        TIME_COL,
        target_col,
        *other_cols,
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns required for data preparation: "
            f"{sorted(missing_columns)}"
        )

    # Normalize the time column and remove invalid time values.
    df = (
        df.with_columns(
            pl.col(TIME_COL).cast(
                pl.Float64,
                strict=False,
            )
        )
        .drop_nulls(subset=[TIME_COL])
    )

    if df.is_empty():
        print(
            " │   ├─ XX:XX ── "
            "✕ Blocked: no valid time values"
        )
        return None, None

    # Floor the elapsed time to obtain one integer-based hourly index.
    df = df.with_columns(
        pl.col(TIME_COL)
        .floor()
        .cast(pl.Int64)
        .alias("integer_hour")
    )

    # Recalibrate time independently for each stay:
    # 0 is the last available hour and negative values represent the past.
    df = df.with_columns(
        (
            pl.col("integer_hour")
            - pl.col("integer_hour").max().over(ID_COL)
        ).alias(CALIBRATED_TIME_COL)
    )

    df_history = df.filter(
        pl.col(CALIBRATED_TIME_COL) <= 0
    )

    if df_history.is_empty():
        print(
            " │   ├─ XX:XX ── "
            "✕ Blocked: empty historical data"
        )
        return None, None

    # Compute the earliest available calibrated hour for each stay.
    patients = (
        df_history.group_by(ID_COL)
        .agg(
            pl.col(CALIBRATED_TIME_COL)
            .min()
            .alias("min_h")
        )
        # Stable ordering guarantees deterministic alignment before sampling.
        .sort(ID_COL)
    )

    return df_history, patients


def generate_random_windows(
    patients: pl.DataFrame,
    df_agg: pl.DataFrame,
    max_hour: int,
    used_distribution: str,
    target_col: str,
    show_fig: bool,
    seed: RandomSeed = 42,
) -> pl.DataFrame | None:
    """Generate deterministic random time windows for each ICU stay.

    A local NumPy random generator is used to avoid modifying global random
    state. Three sampling strategies are supported:

    ``"uniform"``
        Uniformly sample a valid offset within each stay.

    ``"real"``
        Sample offsets from the empirical distribution of stay lengths.

    ``"flexible"``
        Prefer windows associated with a positive target value and fall back
        to a random valid window when no positive target hour is available.

    Args:
        patients:
            One row per patient containing at least ``ID_COL`` and ``min_h``.
        df_agg:
            Historical patient observations.
        max_hour:
            Number of hours preserved between the end of the selected window
            and the prediction time.
        used_distribution:
            Random-window sampling strategy.
        target_col:
            Name of the target column used by the flexible strategy.
        show_fig:
            Whether to display the sampled-offset distribution.
        seed:
            Integer random seed or existing NumPy random generator.

    Returns:
        A DataFrame containing one row per patient and selected calibrated
        hour, or ``None`` when the distribution is unsupported.

    Raises:
        ValueError:
            If required columns are missing or if ``max_hour`` is negative.
    """
    if max_hour < 0:
        raise ValueError(
            f"max_hour must be non-negative, received {max_hour}."
        )

    required_patient_columns = {ID_COL, "min_h"}
    missing_patient_columns = (
        required_patient_columns - set(patients.columns)
    )

    if missing_patient_columns:
        raise ValueError(
            "Missing patient columns required to generate windows: "
            f"{sorted(missing_patient_columns)}"
        )

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    patient_ids = patients[ID_COL].to_list()
    minimum_hours = patients["min_h"].to_list()

    if used_distribution == "uniform":
        offsets = np.asarray(
            [
                rng.integers(
                    0,
                    max(
                        1,
                        (
                            -min_h
                            - max_hour
                            - (WINDOW_SIZE - 1)
                        )
                        + 1,
                    ),
                )
                for min_h in minimum_hours
            ],
            dtype=np.int64,
        )

    elif used_distribution == "real":
        patients_with_duration = patients.with_columns(
            (-pl.col("min_h")).alias("max_h")
        )

        empirical_durations = (
            patients_with_duration["max_h"]
            .to_numpy()
        )

        sampled_offsets: list[int] = []

        for max_h in empirical_durations:
            maximum_offset = max(
                1,
                (
                    max_h
                    - max_hour
                    - (WINDOW_SIZE - 1)
                )
                + 1,
            )

            possible_offsets = empirical_durations[
                empirical_durations <= maximum_offset
            ]

            if possible_offsets.size == 0:
                sampled_offset = 0
            else:
                sampled_offset = int(
                    rng.choice(possible_offsets)
                )

            sampled_offsets.append(sampled_offset)

        offsets = np.asarray(
            sampled_offsets,
            dtype=np.int64,
        )

    elif used_distribution == "flexible":
        if target_col not in df_agg.columns:
            raise ValueError(
                f"Target column {target_col!r} is missing."
            )

        # Identify calibrated hours with a positive target for each patient.
        positive_target_hours = (
            df_agg.filter(
                (
                    pl.col(CALIBRATED_TIME_COL)
                    <= -max_hour
                )
                & (pl.col(target_col) == 1)
            )
            .group_by(ID_COL)
            .agg(
                pl.col(CALIBRATED_TIME_COL)
                .unique()
                .sort()
                .alias("positive_hours")
            )
        )

        patient_selection = (
            patients.join(
                positive_target_hours,
                on=ID_COL,
                how="left",
            )
            .sort(ID_COL)
        )

        selected_starts: list[int] = []

        for row in patient_selection.iter_rows(named=True):
            positive_hours = row["positive_hours"]
            minimum_hour = int(row["min_h"])
            selected_start: int | None = None

            if positive_hours:
                valid_positive_starts = [
                    int(hour)
                    for hour in positive_hours
                    if (
                        hour + (WINDOW_SIZE - 1)
                        <= -max_hour
                    )
                ]

                if valid_positive_starts:
                    selected_start = int(
                        rng.choice(valid_positive_starts)
                    )
                else:
                    # Shift the latest positive hour so that it appears at
                    # the end of the selected window.
                    selected_start = (
                        int(max(positive_hours))
                        - (WINDOW_SIZE - 1)
                    )

            if selected_start is None:
                latest_valid_start = (
                    -max_hour
                    - (WINDOW_SIZE - 1)
                )

                if latest_valid_start > minimum_hour:
                    selected_start = int(
                        rng.integers(
                            minimum_hour,
                            latest_valid_start + 1,
                        )
                    )
                else:
                    selected_start = minimum_hour

            selected_starts.append(selected_start)

        offsets = np.asarray(
            selected_starts,
            dtype=np.int64,
        )

    else:
        print(
            "Unsupported window distribution: "
            f"{used_distribution}"
        )
        return None

    if show_fig and offsets.size > 0:
        lower_quantile, upper_quantile = np.quantile(
            offsets,
            [0.05, 0.95],
        )

        filtered_offsets = offsets[
            (offsets >= lower_quantile)
            & (offsets <= upper_quantile)
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(
            filtered_offsets,
            bins=100,
        )
        plt.title(
            "Relative offset distribution "
            f"(sanctuary period: {max_hour} hours)\n"
            f"Distribution: {used_distribution}"
        )
        plt.xlabel("Offset")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Attach sampled values to patient identifiers explicitly to avoid
    # positional misalignment between Polars Series.
    df_offsets = pl.DataFrame(
        {
            ID_COL: patient_ids,
            "chosen_offset": offsets,
        }
    )

    if used_distribution == "flexible":
        # Flexible sampling produces absolute calibrated start hours rather
        # than offsets relative to each patient's minimum hour.
        return (
            patients.join(
                df_offsets,
                on=ID_COL,
                how="inner",
            )
            .with_columns(
                pl.int_ranges(
                    pl.col("chosen_offset"),
                    pl.col("chosen_offset")
                    + WINDOW_SIZE,
                ).alias(CALIBRATED_TIME_COL)
            )
            .explode(CALIBRATED_TIME_COL)
            .select(
                [
                    ID_COL,
                    CALIBRATED_TIME_COL,
                ]
            )
        )

    return (
        patients.join(
            df_offsets,
            on=ID_COL,
            how="inner",
        )
        .with_columns(
            (
                pl.col("min_h")
                + pl.col("chosen_offset")
            ).alias("start_h")
        )
        .with_columns(
            (
                pl.col("start_h")
                + (WINDOW_SIZE - 1)
            ).alias("end_h")
        )
        .with_columns(
            pl.int_ranges(
                pl.col("start_h"),
                pl.col("end_h") + 1,
            ).alias(CALIBRATED_TIME_COL)
        )
        .explode(CALIBRATED_TIME_COL)
        .select(
            [
                ID_COL,
                CALIBRATED_TIME_COL,
            ]
        )
    )


def generate_fixed_windows(
    patients: pl.DataFrame,
    hour_offset: int,
    max_hour: int,
) -> pl.DataFrame:
    """Generate one fixed time window for each ICU stay.

    The special value ``hour_offset=-1`` selects the latest possible window
    while respecting ``max_hour``. Other values are interpreted as offsets
    relative to each patient's earliest available hour.

    Args:
        patients:
            One row per patient containing ``ID_COL`` and ``min_h``.
        hour_offset:
            Window offset relative to the earliest available hour. A value of
            ``-1`` selects the latest valid window.
        max_hour:
            Number of hours preserved between the end of the selected window
            and the prediction time.

    Returns:
        A DataFrame containing one row per patient and selected calibrated
        hour.

    Raises:
        ValueError:
            If ``max_hour`` is negative or required columns are missing.
    """
    if max_hour < 0:
        raise ValueError(
            f"max_hour must be non-negative, received {max_hour}."
        )

    required_columns = {ID_COL, "min_h"}
    missing_columns = required_columns - set(patients.columns)

    if missing_columns:
        raise ValueError(
            "Missing patient columns required to generate windows: "
            f"{sorted(missing_columns)}"
        )

    if hour_offset == -1:
        return (
            patients.with_columns(
                pl.int_ranges(
                    -(WINDOW_SIZE - 1) - max_hour,
                    1 - max_hour,
                ).alias(CALIBRATED_TIME_COL)
            )
            .explode(CALIBRATED_TIME_COL)
            .select(
                [
                    ID_COL,
                    CALIBRATED_TIME_COL,
                ]
            )
        )

    return (
        patients.with_columns(
            pl.lit(hour_offset).alias("hour_offset")
        )
        .with_columns(
            (
                pl.col("min_h")
                + pl.col("hour_offset")
            ).alias("start_h")
        )
        .with_columns(
            (
                pl.col("start_h")
                + (WINDOW_SIZE - 1)
            ).alias("end_h")
        )
        .with_columns(
            pl.int_ranges(
                pl.col("start_h"),
                pl.col("end_h") + 1,
            ).alias(CALIBRATED_TIME_COL)
        )
        .explode(CALIBRATED_TIME_COL)
        .select(
            [
                ID_COL,
                CALIBRATED_TIME_COL,
            ]
        )
    )


def finalize_data(
    df_windows: pl.DataFrame,
    df_agg: pl.DataFrame,
    strict_mode: bool,
) -> pl.DataFrame:
    """Join selected windows with observations and apply strict filtering.

    Args:
        df_windows:
            Selected patient-hour combinations.
        df_agg:
            Historical patient observations.
        strict_mode:
            Whether to retain only windows containing all expected hourly
            observations.

    Returns:
        The finalized windowed dataset.
    """
    df_agg = df_agg.with_columns(
        pl.col(CALIBRATED_TIME_COL).cast(pl.Int64),
        pl.lit(1).alias("real_hour"),
    )

    df_full = (
        df_windows.join(
            df_agg,
            on=[
                ID_COL,
                CALIBRATED_TIME_COL,
            ],
            how="left",
        )
        .sort(
            [
                ID_COL,
                CALIBRATED_TIME_COL,
            ]
        )
    )

    if strict_mode:
        valid_ids = (
            df_full.group_by(ID_COL)
            .agg(
                pl.col("real_hour")
                .fill_null(0)
                .sum()
                .alias("observed_hour_count")
            )
            .filter(
                pl.col("observed_hour_count")
                >= WINDOW_SIZE
            )
            .select(ID_COL)
        )

        df_full = df_full.join(
            valid_ids,
            on=ID_COL,
            how="inner",
        )

    return df_full.drop("real_hour")


def prepare_data(
    df: PolarsFrame,
    hour_offset: int = 0,
    random: bool = False,
    max_hour: int = 0,
    used_distribution: str = "uniform",
    strict_mode: bool = False,
    target_col: str = "isDeceased_lt_24h",
    other_cols: list[str] | None = None,
    show_fig: bool = True,
    seed: RandomSeed = 42,
) -> pl.DataFrame:
    """Prepare fixed-length patient time windows for model training.

    The function cleans and recalibrates patient time-series data, generates
    either fixed or random 24-hour windows, and joins the selected hours with
    the original observations.

    Args:
        df:
            Patient time-series data.
        hour_offset:
            Offset used by fixed-window sampling. The special value ``-1``
            selects the latest valid window.
        random:
            Whether to generate random rather than fixed windows.
        max_hour:
            Number of hours preserved between the end of a window and the
            prediction time.
        used_distribution:
            Random sampling strategy. Supported values are ``"uniform"``,
            ``"real"``, and ``"flexible"``.
        strict_mode:
            Whether to discard windows that do not contain all 24 expected
            hourly observations.
        target_col:
            Name of the main prediction target.
        other_cols:
            Names of additional target columns that must be present.
        show_fig:
            Whether to display the random-offset distribution.
        seed:
            Integer random seed or existing NumPy random generator.

    Returns:
        The prepared DataFrame. An empty DataFrame is returned when no valid
        observations or windows can be constructed.
    """
    if other_cols is None:
        other_cols = []

    # Step 1: clean the input and recalibrate patient time.
    df_agg, patients = prepare_base_data(
        df=df,
        target_col=target_col,
        other_cols=other_cols,
    )

    if df_agg is None or patients is None:
        return pl.DataFrame()

    # Step 2: generate fixed or random time windows.
    if random:
        df_windows = generate_random_windows(
            patients=patients,
            df_agg=df_agg,
            max_hour=max_hour,
            used_distribution=used_distribution,
            target_col=target_col,
            show_fig=show_fig,
            seed=seed,
        )
    else:
        df_windows = generate_fixed_windows(
            patients=patients,
            hour_offset=hour_offset,
            max_hour=max_hour,
        )

    if df_windows is None or df_windows.is_empty():
        print(
            f" │   ├─ (H-{hour_offset}) ── "
            "✕ Blocked: no window could be generated"
        )
        return pl.DataFrame()

    # Step 3: attach observations and apply strict completeness filtering.
    return finalize_data(
        df_windows=df_windows,
        df_agg=df_agg,
        strict_mode=strict_mode,
    )


def remove_null_values(
    df: PolarsFrame,
) -> pl.DataFrame:
    """Fill missing Boolean indicators and remove incomplete static data.

    Missing values in selected Boolean clinical indicators are replaced with
    ``False``. Rows without height or admission weight are then removed.

    Args:
        df:
            Patient dataset to clean.

    Returns:
        The cleaned eager DataFrame.

    Raises:
        ValueError:
            If one of the required columns is missing.
    """
    df = _ensure_dataframe(df)

    boolean_columns = [
        "is_ventilated",
        "is_prone",
        "is_conscious",
        "is_cvvhf",
        "is_hdi",
    ]
    required_columns = {
        *boolean_columns,
        # "taille",
        # "poids_admission",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns required for null-value cleaning: "
            f"{sorted(missing_columns)}"
        )

    return (
        df.with_columns(
            [
                pl.col(column)
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
                .alias(column)
                for column in boolean_columns
            ]
        )
        # .filter(
        #     pl.col("taille").is_not_null()
        #     & pl.col("poids_admission").is_not_null()
        # )
    )