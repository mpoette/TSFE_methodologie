"""Select and apply ICU-stay windowing or resampling strategies.

This module loads feature metadata, resolves the configured temporal sampling
mode, and prepares fixed-window or resampled patient datasets for downstream
preprocessing and model training.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl

import utilitaries.extract_data_utils as extract
import utilitaries.resampling_utils as resampling
import utilitaries.timestamp_sampling_utils as tsu
import utilitaries.config_utils as config_dataclass

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def _compute_median_target_length(
    df: pl.DataFrame,
    patient_col: str = "encounterId",
    time_col: str = "delta_hour",
) -> int:
    """Compute the median number of time points per ICU stay.

    The function calculates the maximum value of ``time_col`` for each
    patient stay, then returns the rounded median of these values.

    This assumes that ``time_col`` starts at 1 and that exactly one
    observation is available per hour. Under this assumption, the maximum
    time value corresponds to the number of observations in the stay.

    Args:
        df:
            Input DataFrame containing the patient identifier and time columns.
        patient_col:
            Name of the patient or stay identifier column.
        time_col:
            Name of the elapsed-time column.

    Returns:
        The rounded median number of time points per stay.

    Raises:
        ValueError:
            If a required column is missing, the median cannot be computed,
            or the resulting target length is lower than 1.
    """
    required_columns = {patient_col, time_col}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns required to compute target length: "
            f"{sorted(missing_columns)}"
        )

    target_length = (
        df.group_by(patient_col)
        .agg(
            pl.col(time_col)
            .max()
            .alias("stay_duration")
        )
        .select(
            pl.col("stay_duration").median()
        )
        .item()
    )

    if target_length is None:
        raise ValueError(
            "Unable to compute the target length."
        )

    target_length = int(round(target_length))

    if target_length < 1:
        raise ValueError(
            f"Invalid target length: {target_length}"
        )

    return target_length


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and provide explicit error messages.

    Args:
        path:
            Path to the JSON file.

    Returns:
        The decoded JSON content.

    Raises:
        FileNotFoundError:
            If the JSON file does not exist.
        json.JSONDecodeError:
            If the file does not contain valid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================================
# PUBLIC DATASET PREPARATION PIPELINE
# ============================================================================

def prepare_dataset_from_config(
    df_merged: pl.DataFrame | pl.LazyFrame,
    config_mode: config_dataclass.ConfigFeatures,
    target_col: str,
    patient_col: str,
    targets: list[str],
    seed: int,
    dynamic_features_path: str | Path = (
        "../../Preprocessing_pipeline/preprocessing-pipelines/"
        "json/dynamic_features.json"
    ),
    calculated_features_path: str | Path = (
        "../../Preprocessing_pipeline/preprocessing-pipelines/"
        "json/calculated_features_to_original_features.json"
    ),
    show_fig: bool = True,
) -> tuple[pl.DataFrame, list[str], int | None]:
    """Prepare a merged patient dataset according to a configured mode.

    The function first removes rows containing invalid null values. Depending
    on the selected configuration, it then either extracts a time window or
    resamples each ICU stay to a common number of time points.

    Lomax-based modes generate a random prediction timestamp before applying
    window extraction or resampling.

    Supported window modes include:

    - ``"24h_alea_lomax_prio_24h_no-fill"``
    - Other window configurations handled by ``extract.prepare_data``

    Supported resampling modes include:

    - ``"resampling_x_points"``
    - ``"resampling_x_points_alea_lomax_prio_24h_no-fill"``

    Args:
        df_merged:
            Merged static and time-series patient data.
        config_mode:
            Configuration object. It must provide at least the ``mode`` and
            ``name`` attributes. Window modes may also require
            ``hour_offset``, ``random``, ``used_distribution``, and
            ``strict_mode``.
        target_col:
            Name of the prediction target column.
        patient_col:
            Name of the patient or stay identifier column.
        targets:
            Additional target-related columns passed to
            ``extract.prepare_data``.
        seed:
            Random seed used when generating Lomax-based timestamps.
        dynamic_features_path:
            Path to the JSON file containing valid dynamic feature names.
        calculated_features_path:
            Path to the JSON file mapping calculated features to their
            original features.
        show_fig:
            Whether preprocessing functions may display diagnostic figures.

    Returns:
        A tuple containing:

        - The prepared patient DataFrame.
        - The list of dynamic features used for resampling. This list is
          empty for window-based modes.
        - The number of resampling points. This value is ``None`` for
          window-based modes.

    Raises:
        FileNotFoundError:
            If one of the required JSON files does not exist.
        RuntimeError:
            If a Lomax-based mode is selected but prediction timestamps were
            not generated.
        ValueError:
            If preprocessing produces an empty DataFrame, no valid dynamic
            feature is found, the resampling output is invalid, or the
            requested mode is not supported.
    """
    if isinstance(df_merged, pl.LazyFrame):
        df_merged = df_merged.collect()
    features_list: list[str] = []
    target_length: int | None = None

    # Apply preprocessing shared by all configuration modes.
    df_clean = extract.remove_null_values(df_merged)

    if df_clean.is_empty():
        raise ValueError(
            "The DataFrame is empty after remove_null_values."
        )

    # Generate prediction timestamps only for Lomax-based configurations.
    df_timestamp: pl.DataFrame | None = None

    if "lomax" in config_mode.name.lower():
        df_timestamp = tsu.build_sampling_dataset(
            df_clean,
            sanctuary_hours=config_mode.max_hour,
            min_observation_hours=24,
            lomax_alpha=4.3085,
            lomax_lambda=1161.9368,
            seed=seed,
        )

    # Handle fixed-window configurations.
    if config_mode.mode == "windows":
        if config_mode.name == "24h_alea_lomax_prio_24h_no-fill":
            if df_timestamp is None:
                raise RuntimeError(
                    "Prediction timestamps were not generated for the "
                    "selected Lomax mode."
                )

            df_clean = (
                df_clean.join(
                    df_timestamp.select(
                        [
                            patient_col,
                            pl.col("delta_hour").alias("window_end"),
                        ]
                    ),
                    on=patient_col,
                    how="inner",
                )
                .filter(
                    (pl.col("delta_hour") <= pl.col("window_end"))
                    & (
                        pl.col("delta_hour")
                        > pl.col("window_end") - 24
                    )
                )
            )

        # Simple truncation: keep only the first 24 hours of each stay.
        elif config_mode.name == "24h_debut_rea_no-fill":
            df_clean = df_clean.filter(
                pl.col("delta_hour") < 24
            )

        else:
            df_clean = extract.prepare_data(
                df_clean,
                hour_offset=config_mode.hour_offset,
                random=config_mode.random,
                max_hour=config_mode.max_hour,
                used_distribution=config_mode.used_distribution,
                strict_mode=config_mode.strict_mode,
                target_col=target_col,
                other_cols=targets,
                show_fig=show_fig,
            )

        return df_clean, features_list, target_length

    # Handle resampling configurations.
    if config_mode.mode == "resampling":
        variables_json = _load_json(dynamic_features_path)
        calculated_features_json = _load_json(
            calculated_features_path
        )

        valid_dynamic_features = (
            set(variables_json)
            | {
                column
                for column in df_merged.columns
                if column in calculated_features_json
            }
        )

        features_list = [
            column
            for column in df_merged.columns
            if column not in {target_col, patient_col}
            and column in valid_dynamic_features
        ]

        if not features_list:
            raise ValueError(
                "No valid dynamic features were found for resampling."
            )

        columns_to_exclude = list(
            dict.fromkeys(
                features_list
                + [
                    "delta_hour",
                    "t_norm",
                    "min_h",
                    "max_h",
                    "duree_reelle_sejour",
                    "observed_duration",
                    "real_time_hours",
                    "window_end",
                ]
            )
        )

        # Resample the complete available ICU stay.
        if config_mode.name == "resampling_x_points":
            target_length = _compute_median_target_length(
                df_clean,
                patient_col=patient_col,
                time_col="delta_hour",
            )

            df_clean = resampling.resample_icu_stays(
                df=df_clean,
                target_length=target_length,
                features=features_list,
                variables_json=variables_json,
                help_json=calculated_features_json,
            )

            if "duree_reelle_sejour" not in df_clean.columns:
                raise ValueError(
                    "resample_icu_stays did not produce the "
                    "'duree_reelle_sejour' column."
                )

            # Map resampled indices back to the original stay timeline.
            #
            # This assumes that delta_hour ranges from 0 to target_length - 1
            # after resampling. The first resampled point corresponds to time 0,
            # while the last point corresponds to the full observed stay duration.
            if target_length == 1:
                df_clean = df_clean.with_columns(
                    pl.lit(0.0).alias("real_time_hours")
                )
            else:
                df_clean = df_clean.with_columns(
                    (
                        pl.col("duree_reelle_sejour")
                        * pl.col("delta_hour")
                        / (target_length - 1)
                    ).alias("real_time_hours")
                )

            # Remove the complete stay duration to prevent data leakage.
            df_clean = df_clean.drop("duree_reelle_sejour")

        # Sample an observation cutoff with Lomax, then resample the
        # observed part of the stay.
        elif (
            config_mode.name
            == "resampling_x_points_alea_lomax_prio_24h_no-fill"
        ):
            if df_timestamp is None:
                raise RuntimeError(
                    "Prediction timestamps were not generated for the "
                    "selected Lomax mode."
                )

            df_clean = (
                df_clean.join(
                    df_timestamp.select(
                        [
                            patient_col,
                            pl.col("delta_hour").alias("window_end"),
                        ]
                    ),
                    on=patient_col,
                    how="inner",
                )
                .filter(
                    pl.col("delta_hour") <= pl.col("window_end")
                )
            )

            if df_clean.is_empty():
                raise ValueError(
                    "Lomax filtering removed all observations."
                )

            target_length = _compute_median_target_length(
                df_clean,
                patient_col=patient_col,
                time_col="delta_hour",
            )

            df_clean = resampling.resample_icu_stays(
                df=df_clean,
                target_length=target_length,
                features=features_list,
                variables_json=variables_json,
                help_json=calculated_features_json,
            )

            if "duree_reelle_sejour" not in df_clean.columns:
                raise ValueError(
                    "resample_icu_stays did not produce the "
                    "'duree_reelle_sejour' column."
                )

            # In this mode, the duration represents the data available at
            # prediction time rather than the patient's complete ICU stay.
            df_clean = df_clean.rename(
                {
                    "duree_reelle_sejour": "observed_duration",
                }
            )

        else:
            raise ValueError(
                "Unsupported resampling configuration: "
                f"{config_mode.name}"
            )

        # Recover one row of static information per patient.
        static_columns = [
            column
            for column in df_merged.columns
            if column not in columns_to_exclude
        ]

        if patient_col not in static_columns:
            static_columns.append(patient_col)

        df_static_patient = (
            df_merged.select(static_columns)
            .unique(
                subset=[patient_col],
                keep="first",
            )
        )

        # Attach static patient information to the resampled observations.
        df_clean = df_clean.join(
            df_static_patient,
            on=patient_col,
            how="inner",
        )

        return df_clean, features_list, target_length

    raise ValueError(
        f"Unsupported preprocessing mode: {config_mode.mode}"
    )
