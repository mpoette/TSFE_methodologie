"""Timestamp sampling pipeline for ICU survival prediction at H24.

One timestamp is selected per patient.

Strategy:
    1. Pre-filter invalid rows based on the sanctuary period and minimum
       observation duration.
    2. Census: enumerate the mortality profiles reachable by each patient.
    3. Assignment:
        a. Assign mono-profile patients immediately.
        b. Assign multi-profile patients to balance classes while explicitly
           protecting profile ``(1, 1)``, the positive class for H24
           mortality prediction.
    4. Draw one timestamp within the assigned profile zone using Lomax
       weights, favoring earlier timestamps in the zone.

Expected input columns:
    encounterId:
        Patient identifier.
    delta_hour:
        Hours since admission, with one row per patient-hour.
    isDeceased_lt_24h:
        Whether death occurs within 24 hours after the current timestamp.
    isDeceased_lt_28d:
        Whether death occurs before day 28.

Profile taxonomy:
    ``(1, 1)``:
        Death within the next 24 hours.
    ``(0, 1)``:
        No death within 24 hours, but death before day 28.
    ``(0, 0)``:
        Survival beyond day 28.
    ``(1, 0)``:
        Impossible by construction.
"""

from collections import defaultdict

import numpy as np
import polars as pl
from scipy.stats import lomax


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

SANCTUARY_HOURS = 6
MIN_OBSERVATION_HOURS = 24
LOMAX_ALPHA = 4.3085
LOMAX_LAMBDA = 1161.9368
RANDOM_SEED = 42

# Protect the H24-positive class during profile assignment.
PRIORITY_PROFILE = (1, 1)


# ---------------------------------------------------------------------
# Step 0: label preparation
# ---------------------------------------------------------------------

# ============================================================================
# LABEL PREPARATION
# ============================================================================

def prepare_labels(df, mode):
    """Create mortality labels according to the selected temporal definition.

    Args:
        df:
            Input DataFrame containing ``deces_datediff_days`` and
            ``delta_hour``.
        mode:
            Labeling mode. Supported values are ``"relative"``,
            ``"absolute"``, and ``"mixed"``.

    Returns:
        The input DataFrame with mortality target columns added.
    """
    if mode == "relative":
        return df.with_columns(
            [
                # Death within 24 hours after the current timestamp.
                (
                    pl.col("deces_datediff_days") * 24
                    <= pl.col("delta_hour") + 24
                )
                .fill_null(False)
                .alias("isDeceased_lt_24h"),

                # Death within 7 days after the current timestamp.
                (
                    pl.col("deces_datediff_days") * 24
                    <= pl.col("delta_hour") + 168
                )
                .fill_null(False)
                .alias("isDeceased_lt_7d"),

                # Death within 28 days after the current timestamp.
                (
                    pl.col("deces_datediff_days") * 24
                    <= pl.col("delta_hour") + 672
                )
                .fill_null(False)
                .alias("isDeceased_lt_28d"),

                # Death within 3 months after the current timestamp.
                (
                    pl.col("deces_datediff_days") * 24
                    <= pl.col("delta_hour") + 2160
                )
                .fill_null(False)
                .alias("isDeceased_lt_3m"),
            ]
        )

    elif mode == "absolute":
        return df.with_columns(
            [
                (
                    pl.col("deces_datediff_days") * 24
                    <= 24
                )
                .fill_null(False)
                .alias("isDeceased_lt_24h"),

                (
                    pl.col("deces_datediff_days")
                    <= 7
                )
                .fill_null(False)
                .alias("isDeceased_lt_7d"),

                (
                    pl.col("deces_datediff_days")
                    <= 28
                )
                .fill_null(False)
                .alias("isDeceased_lt_28d"),

                (
                    pl.col("deces_datediff_days")
                    <= 90
                )
                .fill_null(False)
                .alias("isDeceased_lt_3m"),
            ]
        )

    elif mode == "mixed":
        return df.with_columns(
            [
                # Keep H24 relative to the current timestamp.
                (
                    (
                        pl.col("deces_datediff_days") * 24
                        > pl.col("delta_hour")
                    )
                    & (
                        pl.col("deces_datediff_days") * 24
                        <= pl.col("delta_hour") + 24
                    )
                )
                .fill_null(False)
                .alias("isDeceased_lt_24h"),

                # Use absolute mortality horizons from admission for the
                # remaining targets.
                (
                    pl.col("deces_datediff_days")
                    <= 7
                )
                .fill_null(False)
                .alias("isDeceased_lt_7d"),

                (
                    pl.col("deces_datediff_days")
                    <= 28
                )
                .fill_null(False)
                .alias("isDeceased_lt_28d"),

                (
                    pl.col("deces_datediff_days")
                    <= 90
                )
                .fill_null(False)
                .alias("isDeceased_lt_3m"),
            ]
        )


# ---------------------------------------------------------------------
# Step 1: pre-filtering
# ---------------------------------------------------------------------

# ============================================================================
# SAMPLING ELIGIBILITY
# ============================================================================

def prefilter(
    df: pl.DataFrame,
    sanctuary_hours,
    min_observation_hours,
) -> pl.DataFrame:
    """Remove rows outside the valid timestamp-sampling interval.

    Rows are excluded when they occur before the minimum observation duration
    or within the sanctuary period preceding the end of the ICU stay.

    Patients with no remaining valid row are excluded automatically.

    Args:
        df:
            Input patient-hour DataFrame.
        sanctuary_hours:
            Number of hours excluded before the end of the ICU stay.
        min_observation_hours:
            Minimum elapsed time required before a timestamp can be sampled.

    Returns:
        The filtered DataFrame.
    """
    length_of_stay = (
        df.group_by("encounterId")
        .agg(
            pl.col("delta_hour")
            .max()
            .alias("los_hours")
        )
    )

    df = (
        df.join(
            length_of_stay,
            on="encounterId",
            how="left",
        )
        .filter(
            (
                pl.col("delta_hour")
                >= min_observation_hours
            )
            & (
                pl.col("delta_hour")
                <= pl.col("los_hours") - sanctuary_hours
            )
        )
        .drop("los_hours")
    )

    return df


# ---------------------------------------------------------------------
# Step 2: profile validation
# ---------------------------------------------------------------------

# ============================================================================
# MORTALITY-PROFILE VALIDATION AND CENSUS
# ============================================================================

def validate_profiles(df: pl.DataFrame) -> pl.DataFrame:
    """Cast mortality labels and validate profile consistency.

    The profile ``(1, 0)`` is impossible because death within 24 hours implies
    death within 28 days.

    Args:
        df:
            Input DataFrame containing H24 and J28 mortality labels.

    Returns:
        The DataFrame with mortality labels cast to ``Int8``.

    Raises:
        ValueError:
            If one or more rows contain the impossible profile ``(1, 0)``.
    """
    df = df.with_columns(
        [
            pl.col("isDeceased_lt_24h").cast(pl.Int8),
            pl.col("isDeceased_lt_28d").cast(pl.Int8),
        ]
    )

    impossible = df.filter(
        (
            pl.col("isDeceased_lt_24h") == 1
        )
        & (
            pl.col("isDeceased_lt_28d") == 0
        )
    )

    if len(impossible) > 0:
        raise ValueError(
            f"{len(impossible)} rows contain the impossible profile "
            "(isDeceased_lt_24h=1, isDeceased_lt_28d=0)."
        )

    return df


# ---------------------------------------------------------------------
# Step 3: profile census
# ---------------------------------------------------------------------

def census(df: pl.DataFrame) -> dict:
    """Enumerate the mortality profiles reachable by each patient.

    Args:
        df:
            Filtered patient-hour DataFrame.

    Returns:
        A dictionary mapping each patient identifier to a frozenset of
        reachable ``(H24, J28)`` profiles.
    """
    # Create one structure per patient containing all unique reachable
    # mortality profiles.
    df_agg = (
        df.group_by("encounterId")
        .agg(
            pl.struct(
                [
                    "isDeceased_lt_24h",
                    "isDeceased_lt_28d",
                ]
            )
            .unique()
            .alias("profiles")
        )
    )

    # Convert the aggregated profile structures into frozensets.
    return {
        row["encounterId"]: frozenset(
            (
                int(profile["isDeceased_lt_24h"]),
                int(profile["isDeceased_lt_28d"]),
            )
            for profile in row["profiles"]
        )
        for row in df_agg.iter_rows(named=True)
    }


def print_census_summary(patient_profiles: dict) -> None:
    """Print reachable-profile and patient-flexibility statistics.

    Args:
        patient_profiles:
            Mapping between patient identifiers and reachable profiles.
    """
    profile_reach = defaultdict(int)
    flexibility = defaultdict(int)

    for profiles in patient_profiles.values():
        for profile in profiles:
            profile_reach[profile] += 1

        flexibility[len(profiles)] += 1

    print(
        "\nCensus — patients who can reach each profile:"
    )

    for profile in sorted(profile_reach):
        label = {
            (1, 1): (
                "death within 24h          (1,1)"
            ),
            (0, 1): (
                "no 24h death, dead by J28 (0,1)"
            ),
            (0, 0): (
                "survivor J28               (0,0)"
            ),
        }.get(
            profile,
            str(profile),
        )

        print(
            f"  {label} : "
            f"{profile_reach[profile]} patients"
        )

    print(
        "\nPatient flexibility "
        "(number of reachable profiles):"
    )

    for number_profiles, count in sorted(
        flexibility.items()
    ):
        print(
            f"  {number_profiles} profile(s) reachable : "
            f"{count} patients"
        )


# ---------------------------------------------------------------------
# Step 4: profile assignment
# ---------------------------------------------------------------------

# ============================================================================
# PROFILE ASSIGNMENT
# ============================================================================

def assign_profiles(
    patient_profiles: dict,
    rng: np.random.Generator,
) -> dict:
    """Assign exactly one mortality profile to each patient.

    Assignment prioritizes class balance while protecting the H24-positive
    profile ``(1, 1)``.

    Assignment order:

    1. Mono-profile patients are assigned to their only reachable profile.
    2. Multi-profile patients capable of reaching ``(1, 1)`` are assigned to
       that profile until it reaches the level of the other classes.
    3. Remaining patients are assigned to the least represented profile among
       their reachable profiles.

    Args:
        patient_profiles:
            Mapping between patient identifiers and reachable profiles.
        rng:
            NumPy random generator used to avoid ordering bias.

    Returns:
        A dictionary mapping each patient identifier to one assigned
        ``(H24, J28)`` profile.
    """
    assignment = {}
    counts = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 1): 0,
    }

    mono = {
        patient_id: list(profiles)[0]
        for patient_id, profiles
        in patient_profiles.items()
        if len(profiles) == 1
    }

    multi = {
        patient_id: profiles
        for patient_id, profiles
        in patient_profiles.items()
        if len(profiles) > 1
    }

    # Step 1: assign patients with only one reachable profile.
    for patient_id, profile in mono.items():
        assignment[patient_id] = profile
        counts[profile] += 1

    print(
        f"\nAfter mono-profile assignment: {counts}"
    )

    # Step 2: prioritize patients capable of reaching the H24-positive class.
    can_reach_priority = {
        patient_id: profiles
        for patient_id, profiles in multi.items()
        if PRIORITY_PROFILE in profiles
    }

    cannot_reach_priority = {
        patient_id: profiles
        for patient_id, profiles in multi.items()
        if PRIORITY_PROFILE not in profiles
    }

    # Shuffle patient identifiers to avoid assignment-order bias.
    priority_patient_ids = list(
        can_reach_priority.keys()
    )

    rng.shuffle(priority_patient_ids)

    other_counts = {
        profile: count
        for profile, count in counts.items()
        if profile != PRIORITY_PROFILE
    }

    target_priority_count = (
        max(other_counts.values())
        if other_counts
        else 0
    )

    remaining_for_general = []

    for patient_id in priority_patient_ids:
        profiles = can_reach_priority[
            patient_id
        ]

        if (
            counts[PRIORITY_PROFILE]
            < target_priority_count
        ):
            # Assign to the priority profile until its target count is reached.
            assignment[patient_id] = (
                PRIORITY_PROFILE
            )

            counts[PRIORITY_PROFILE] += 1

        else:
            # Defer remaining patients to the general balancing step.
            remaining_for_general.append(
                (
                    patient_id,
                    profiles,
                )
            )

    print(
        "After (1,1) priority fill: "
        f"{counts}"
    )

    # Step 3: assign remaining patients to their least represented reachable
    # profile.
    general_patients = (
        remaining_for_general
        + [
            (
                patient_id,
                profiles,
            )
            for patient_id, profiles
            in cannot_reach_priority.items()
        ]
    )

    rng.shuffle(general_patients)

    for patient_id, profiles in general_patients:
        best_profile = min(
            profiles,
            key=lambda profile: counts[profile],
        )

        assignment[patient_id] = best_profile
        counts[best_profile] += 1

    print(
        f"After general balancing: {counts}"
    )

    return assignment


# ---------------------------------------------------------------------
# Step 5: Lomax-weighted timestamp sampling
# ---------------------------------------------------------------------

# ============================================================================
# TIMESTAMP DRAWING
# ============================================================================

def draw_timestamps(
    df: pl.DataFrame,
    assignment: dict,
    lomax_alpha: float,
    lomax_lambda: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Draw one Lomax-weighted timestamp per patient.

    Each patient is restricted to timestamps matching the mortality profile
    assigned during the balancing step. Lomax weights are then computed from
    the distance to the end of the assigned profile zone.

    Sampling is performed with inverse transform sampling without iterating
    over DataFrame rows.

    Args:
        df:
            Filtered patient-hour DataFrame.
        assignment:
            Mapping between patient identifiers and assigned profiles.
        lomax_alpha:
            Lomax shape parameter.
        lomax_lambda:
            Lomax scale parameter.
        rng:
            NumPy random generator.

    Returns:
        A DataFrame containing at most one sampled timestamp per patient.
    """
    if not assignment:
        return pl.DataFrame(
            schema=df.schema
        )

    # Convert the patient assignment dictionary into a Polars DataFrame for
    # an efficient join.
    assignment_df = pl.DataFrame(
        [
            {
                "encounterId": patient_id,
                "assigned_h24": h24,
                "assigned_j28": j28,
            }
            for patient_id, (h24, j28)
            in assignment.items()
        ],
        schema={
            "encounterId": df.schema[
                "encounterId"
            ],
            "assigned_h24": pl.Int8,
            "assigned_j28": pl.Int8,
        },
    )

    # Keep only timestamps matching the assigned mortality profile.
    df_assigned = (
        df.join(
            assignment_df,
            on="encounterId",
            how="inner",
        )
        .filter(
            (
                pl.col("isDeceased_lt_24h")
                == pl.col("assigned_h24")
            )
            & (
                pl.col("isDeceased_lt_28d")
                == pl.col("assigned_j28")
            )
        )
    )

    if df_assigned.is_empty():
        return pl.DataFrame(
            schema=df.schema
        )

    # Sort chronologically before cumulative computations.
    df_assigned = df_assigned.sort(
        [
            "encounterId",
            "delta_hour",
        ]
    )

    # Compute the remaining time until the end of the assigned profile zone.
    df_assigned = (
        df_assigned.with_columns(
            [
                pl.col("delta_hour")
                .max()
                .over("encounterId")
                .alias("h_max")
            ]
        )
        .with_columns(
            (
                pl.col("h_max")
                - pl.col("delta_hour")
            ).alias("distance_fin_zone")
        )
    )

    # Compute Lomax weights in a single NumPy pass.
    distances_np = (
        df_assigned[
            "distance_fin_zone"
        ]
        .to_numpy()
    )

    raw_weights = lomax.pdf(
        distances_np,
        c=lomax_alpha,
        scale=lomax_lambda,
    )

    # Normalize weights independently for each patient.
    df_assigned = df_assigned.with_columns(
        pl.Series(
            "raw_weight",
            raw_weights,
        )
    )

    df_assigned = (
        df_assigned.with_columns(
            pl.col("raw_weight")
            .sum()
            .over("encounterId")
            .alias("total_weight")
        )
        .with_columns(
            pl.when(
                pl.col("total_weight") > 0
            )
            .then(
                pl.col("raw_weight")
                / pl.col("total_weight")
            )
            .otherwise(
                1.0
                / pl.col("raw_weight")
                .count()
                .over("encounterId")
            )
            .alias("weight")
        )
    )

    # Compute cumulative weights for inverse transform sampling.
    df_assigned = df_assigned.with_columns(
        pl.col("weight")
        .cum_sum()
        .over("encounterId")
        .alias("cum_weight")
    )

    # Draw one uniform random value for each patient.
    unique_patients = (
        df_assigned
        .select("encounterId")
        .unique()
    )

    uniform_values = rng.random(
        len(unique_patients)
    )

    patient_uniform_df = (
        unique_patients.with_columns(
            pl.Series(
                "u",
                uniform_values,
            )
        )
    )

    # Join patient-specific random values and identify candidate timestamps.
    df_assigned = (
        df_assigned.join(
            patient_uniform_df,
            on="encounterId",
            how="left",
        )
        .with_columns(
            pl.col("delta_hour")
            .max()
            .over("encounterId")
            .alias("max_hour")
        )
    )

    # Keep timestamps whose cumulative probability exceeds the sampled
    # threshold. The final timestamp is retained as a safety fallback.
    df_candidates = df_assigned.filter(
        (
            pl.col("cum_weight")
            >= pl.col("u")
        )
        | (
            pl.col("delta_hour")
            == pl.col("max_hour")
        )
    )

    # Retain the first timestamp crossing the cumulative threshold.
    df_candidates = (
        df_candidates.with_columns(
            pl.col("delta_hour")
            .min()
            .over("encounterId")
            .alias("min_candidate_hour")
        )
    )

    result = df_candidates.filter(
        pl.col("delta_hour")
        == pl.col("min_candidate_hour")
    )

    # Remove intermediate computation columns and restore the original schema.
    cols_to_drop = [
        "assigned_h24",
        "assigned_j28",
        "h_max",
        "distance_fin_zone",
        "raw_weight",
        "total_weight",
        "weight",
        "cum_weight",
        "u",
        "max_hour",
        "min_candidate_hour",
    ]

    return (
        result.drop(cols_to_drop)
        .sort(
            [
                "encounterId",
                "delta_hour",
            ]
        )
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

# ============================================================================
# PUBLIC SAMPLING PIPELINE
# ============================================================================

def build_sampling_dataset(
    df: pl.DataFrame,
    sanctuary_hours: int = SANCTUARY_HOURS,
    min_observation_hours: int = MIN_OBSERVATION_HOURS,
    lomax_alpha: float = LOMAX_ALPHA,
    lomax_lambda: float = LOMAX_LAMBDA,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Build a one-timestamp-per-patient sampling dataset.

    The function applies pre-filtering, validates mortality profiles,
    enumerates reachable profiles, assigns one profile to each patient, and
    samples one timestamp using Lomax-weighted probabilities.

    Args:
        df:
            Input patient-hour DataFrame.
        sanctuary_hours:
            Number of hours excluded before the end of each ICU stay.
        min_observation_hours:
            Minimum elapsed time required before sampling.
        lomax_alpha:
            Lomax shape parameter.
        lomax_lambda:
            Lomax scale parameter.
        seed:
            Random seed controlling profile assignment and timestamp sampling.

    Returns:
        A DataFrame containing one sampled timestamp per retained patient.
    """
    rng = np.random.default_rng(seed)

    n_patients_in = (
        df["encounterId"]
        .n_unique()
    )

    print(
        f"Input: {n_patients_in} patients, "
        f"{len(df)} rows"
    )

    # Step 1: pre-filter invalid timestamps.
    df = prefilter(
        df,
        sanctuary_hours,
        min_observation_hours,
    )

    n_patients_kept = (
        df["encounterId"]
        .n_unique()
    )

    print(
        "After pre-filtering: "
        f"{n_patients_kept} patients "
        f"({n_patients_in - n_patients_kept} excluded)"
    )

    # Step 2: validate mortality profiles.
    df = validate_profiles(df)

    # Step 3: enumerate reachable patient profiles.
    patient_profiles = census(df)

    print_census_summary(
        patient_profiles
    )

    # Step 4: assign one profile per patient.
    assignment = assign_profiles(
        patient_profiles,
        rng,
    )

    # Step 5: draw one timestamp per patient.
    result = draw_timestamps(
        df,
        assignment,
        lomax_alpha,
        lomax_lambda,
        rng,
    )

    print(
        f"\nFinal dataset: {len(result)} patients, "
        "one timestamp each"
    )

    return result


# ---------------------------------------------------------------------
# Usage example with synthetic data
# ---------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_patients = 300
    rows = []

    for patient_id in range(n_patients):
        length_of_stay = int(
            rng.integers(
                12,
                400,
            )
        )

        dead_by_day_28 = (
            rng.random() < 0.35
        )

        death_hour = (
            int(
                rng.integers(
                    6,
                    min(
                        length_of_stay,
                        672,
                    ),
                )
            )
            if dead_by_day_28
            else None
        )

        for hour in range(
            length_of_stay + 1
        ):
            is_deceased_lt_24h = (
                dead_by_day_28
                and death_hour is not None
                and (
                    hour
                    < death_hour
                    <= hour + 24
                )
            )

            rows.append(
                {
                    "encounterId": patient_id,
                    "delta_hour": hour,
                    "deces_datediff_days": (
                        death_hour / 24
                        if death_hour is not None
                        else None
                    ),
                }
            )

    df_raw = pl.DataFrame(rows)

    df_raw = prepare_labels(
        df_raw,
        "mixed",
    )

    result = build_sampling_dataset(
        df_raw
    )

    print(
        result.head(10)
    )