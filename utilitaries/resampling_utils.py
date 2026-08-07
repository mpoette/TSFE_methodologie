"""Resampling utilities for ICU patient stay time-series.

Provides functions to resample ICU stays to a fixed number of time points
using interpolation-based strategies, ensuring consistent sequence lengths
for time-series model training.
"""

import polars as pl
import numpy as np


def resample_icu_stays(
    df: pl.DataFrame,
    target_length: int,
    features: list[str],
    variables_json,
    help_json=None,
) -> pl.DataFrame:
    """Resample variable-length ICU stays to a fixed number of time points.

    The function normalizes the time axis of each ICU stay between 0 and 1,
    aggregates observations occurring at the same normalized time point, and
    creates a regular target grid containing ``target_length`` points.

    Numerical features are linearly interpolated, while categorical and
    cumulative features are propagated using forward and backward filling.

    The implementation builds a shared target grid for all patients and uses
    vertical concatenation to limit memory usage.

    Args:
        df:
            Input DataFrame containing one or more observations per ICU stay.
            It must include ``"encounterId"``, ``"delta_hour"``, and all
            columns listed in ``features``.
        target_length:
            Number of time points to generate for each ICU stay.
        features:
            Feature columns to resample.
        variables_json:
            Feature metadata mapping. Each feature must provide a
            ``"type_feat"`` value.
        help_json:
            Optional mapping from calculated features to their original
            feature names. It is used when a feature is not found directly
            in ``variables_json``.

    Returns:
        A DataFrame containing exactly ``target_length`` rows per ICU stay,
        with the following columns:

        - ``"encounterId"``: ICU stay identifier.
        - ``"delta_hour"``: resampled time-point index.
        - ``"duree_reelle_sejour"``: original maximum elapsed time.
        - The resampled feature columns.

    Raises:
        KeyError:
            If a feature cannot be found in either ``variables_json`` or
            ``help_json``.
        ValueError:
            If a feature type is not supported.
    """
    # Store the minimum and maximum elapsed time for each ICU stay.
    df = df.with_columns(
        min_h=pl.col("delta_hour").min().over("encounterId"),
        max_h=pl.col("delta_hour").max().over("encounterId"),
    ).with_columns(
        duree_reelle_sejour=pl.col("max_h")
    )

    # Normalize each ICU stay timeline between 0 and 1.
    df = df.with_columns(
        t_norm=(
            pl.when(
                pl.col("max_h") == pl.col("min_h")
            )
            .then(
                pl.lit(0.0)
            )
            .otherwise(
                (
                    pl.col("delta_hour")
                    - pl.col("min_h")
                )
                / (
                    pl.col("max_h")
                    - pl.col("min_h")
                )
            )
        )
    )

    # Separate numerical and non-numerical features and build their
    # aggregation expressions.
    num_features = []
    cat_features = []
    expressions_agreg = []

    for f in features:
        try:
            type_feat = variables_json[f]["type_feat"]

        except KeyError:
            try:
                type_feat = (
                    variables_json[
                        help_json[f][0]
                    ]["type_feat"]
                )

            except:
                raise KeyError(
                    f"Feature {f!r} could not be found in the metadata."
                )

        if type_feat == "num_features":
            expressions_agreg.append(
                pl.col(f).median().alias(f)
            )
            num_features.append(f)

        elif type_feat == "sum_features":
            expressions_agreg.append(
                pl.col(f).sum().alias(f)
            )
            cat_features.append(f)

        elif type_feat == "str_features":
            expressions_agreg.append(
                pl.col(f).mode().first().alias(f)
            )
            cat_features.append(f)

        else:
            raise ValueError(
                f"Feature type {type_feat!r} is not currently supported."
            )

    # Aggregate observations sharing the same normalized time point.
    # Additional columns are created in advance so that the original data
    # and target grid have identical schemas during vertical concatenation.
    df_clinique = (
        df.group_by(
            [
                "encounterId",
                "t_norm",
                "duree_reelle_sejour",
            ]
        )
        .agg(
            expressions_agreg
        )
        .with_columns(
            pl.lit(True).alias("est_origine"),
            pl.lit(
                None,
                dtype=pl.Int64,
            ).alias("heure_virtuelle"),
        )
    )

    # Create one regular normalized time grid shared by all ICU stays.
    patients_uniques = (
        df_clinique
        .select(
            [
                "encounterId",
                "duree_reelle_sejour",
            ]
        )
        .unique()
        .sort("encounterId")
    )

    grille_t = np.linspace(
        0,
        1,
        target_length,
    )

    df_grille = pl.DataFrame(
        {
            "t_norm": grille_t,
            "heure_virtuelle": np.arange(
                target_length,
                dtype=np.int64,
            ),
        }
    )

    # Cross-join every ICU stay with the target time grid.
    # Feature columns are initialized with null values using the schema of
    # the aggregated clinical data.
    grille_cible = (
        patients_uniques
        .join(
            df_grille,
            how="cross",
        )
        .with_columns(
            pl.lit(False).alias("est_origine"),
            *[
                pl.lit(
                    None,
                    dtype=df_clinique.schema[f],
                ).alias(f)
                for f in features
            ],
        )
    )

    # Concatenate original observations and virtual target-grid points.
    # Columns are explicitly ordered to allow efficient vertical
    # concatenation.
    colonnes_ordre = [
        "encounterId",
        "t_norm",
        "duree_reelle_sejour",
        "est_origine",
        "heure_virtuelle",
    ] + features

    df_merged = (
        pl.concat(
            [
                df_clinique.select(
                    colonnes_ordre
                ),
                grille_cible.select(
                    colonnes_ordre
                ),
            ],
            how="vertical",
        )
        .sort(
            [
                "encounterId",
                "t_norm",
                "est_origine",
            ],
            descending=[
                False,
                False,
                True,
            ],
        )
    )

    # Build interpolation expressions according to feature type.
    expressions_interp = []

    if num_features:
        expressions_interp.extend(
            [
                pl.col(f)
                .interpolate()
                .forward_fill()
                .backward_fill()
                for f in num_features
            ]
        )

    if cat_features:
        expressions_interp.extend(
            [
                pl.col(f)
                .forward_fill()
                .backward_fill()
                for f in cat_features
            ]
        )

    # Interpolate each ICU stay independently, keep only the virtual target
    # grid, and restore the expected output column names.
    df_resampled = (
        df_merged
        .group_by(
            "encounterId",
            maintain_order=True,
        )
        .agg(
            [
                pl.col("heure_virtuelle"),
                pl.col("duree_reelle_sejour"),
                pl.col("est_origine"),
                *expressions_interp,
            ]
        )
        .explode(
            pl.all().exclude("encounterId")
        )
        .filter(
            ~pl.col("est_origine")
        )
        .select(
            [
                pl.col("encounterId"),
                pl.col("heure_virtuelle").alias(
                    "delta_hour"
                ),
                pl.col("duree_reelle_sejour"),
            ]
            + features
        )
    )

    return df_resampled