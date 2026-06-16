"""
Timestamp sampling pipeline for ICU survival prediction at H24.
One timestamp per patient.

Strategy:
  1. Pre-filter invalid rows (sanctuary + insufficient lookback).
  2. Census: for each patient, enumerate which profiles are reachable.
  3. Assignment:
       a. Mono-profile patients are assigned immediately (no choice).
       b. Multi-profile patients are assigned to best balance classes,
          with explicit priority on protecting profile (1,1) — the
          positive class for H24 mortality prediction.
  4. Within the assigned zone, draw one timestamp using Lomax weights
     (favoring earlier timestamps in the zone).

Input dataset columns:
    - encounterId : patient identifier
    - delta_hour  : hours since admission (1 row per patient-hour)
    - dc_h24      : bool, death within 24h after this timestamp
    - dc_j28      : bool, death before day 28 (absolute, patient-level)

Profile taxonomy:
    (1, 1) -> death within next 24h  [positive class — protect first]
    (0, 1) -> no 24h death, dies before J28
    (0, 0) -> survives J28
    (1, 0) -> impossible by construction
"""

import numpy as np
import polars as pl
from scipy.stats import lomax
from collections import defaultdict


# ──────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────

SANCTUARY_HOURS = 6
MIN_OBSERVATION_HOURS = 24
LOMAX_ALPHA = 1.5
LOMAX_LAMBDA = 1.0
RANDOM_SEED = 42

PRIORITY_PROFILE = (1, 1)   # protect this class first


# ──────────────────────────────────────────────
# STEP 1 — PRE-FILTERING
# ──────────────────────────────────────────────

def prefilter(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows outside the valid sampling window once and for all:
      - before MIN_OBSERVATION_HOURS  (no lookback available)
      - within SANCTUARY_HOURS of end of stay  (prediction too obvious)
    Patients with no remaining valid rows are excluded entirely.
    """
    los = (
        df.group_by("encounterId")
        .agg(pl.col("delta_hour").max().alias("los_hours"))
    )
    df = (
        df.join(los, on="encounterId", how="left")
        .filter(
            (pl.col("delta_hour") >= MIN_OBSERVATION_HOURS) &
            (pl.col("delta_hour") <= pl.col("los_hours") - SANCTUARY_HOURS)
        )
        .drop("los_hours")
    )
    return df


# ──────────────────────────────────────────────
# STEP 2 — PROFILE VALIDATION
# ──────────────────────────────────────────────

def validate_profiles(df: pl.DataFrame) -> pl.DataFrame:
    """Cast booleans and check that (1,0) never occurs."""
    df = df.with_columns([
        pl.col("dc_h24").cast(pl.Int8),
        pl.col("dc_j28").cast(pl.Int8),
    ])
    impossible = df.filter(
        (pl.col("dc_h24") == 1) & (pl.col("dc_j28") == 0)
    )
    if len(impossible) > 0:
        raise ValueError(
            f"{len(impossible)} rows with impossible profile (dc_h24=1, dc_j28=0)."
        )
    return df


# ──────────────────────────────────────────────
# STEP 3 — CENSUS
# ──────────────────────────────────────────────

def census(df: pl.DataFrame) -> dict:
    """
    For each patient, enumerate the set of reachable profiles
    within their valid sampling window.

    Returns a dict:
        patient_profiles[encounter_id] = frozenset of reachable (dc_h24, dc_j28) tuples
    """
    patient_profiles = {}
    for encounter_id, group in df.group_by("encounterId"):
        profiles = frozenset(
            (int(row["dc_h24"]), int(row["dc_j28"]))
            for row in group.select(["dc_h24", "dc_j28"]).unique().iter_rows(named=True)
        )
        patient_profiles[encounter_id] = profiles
    return patient_profiles


def print_census_summary(patient_profiles: dict) -> None:
    """Print how many patients can reach each profile, and flexibility stats."""
    profile_reach = defaultdict(int)
    flexibility = defaultdict(int)

    for profiles in patient_profiles.values():
        for p in profiles:
            profile_reach[p] += 1
        flexibility[len(profiles)] += 1

    print("\nCensus — patients who can reach each profile:")
    for profile in sorted(profile_reach):
        label = {
            (1, 1): "death within 24h          (1,1)",
            (0, 1): "no 24h death, dead by J28 (0,1)",
            (0, 0): "survivor J28               (0,0)",
        }.get(profile, str(profile))
        print(f"  {label} : {profile_reach[profile]} patients")

    print("\nPatient flexibility (number of reachable profiles):")
    for n, count in sorted(flexibility.items()):
        print(f"  {n} profile(s) reachable : {count} patients")


# ──────────────────────────────────────────────
# STEP 4 — ASSIGNMENT
# ──────────────────────────────────────────────

def assign_profiles(patient_profiles: dict, rng: np.random.Generator) -> dict:
    """
    Assign each patient to exactly one profile, prioritizing balance
    while protecting the (1,1) class first.

    Assignment order:
      1. Mono-profile patients: forced assignment, no choice.
      2. Multi-profile patients who can reach (1,1): assign to (1,1)
         until its count reaches the level of the other classes.
      3. Remaining multi-profile patients: assign to the most
         under-represented profile among their reachable set.

    Returns:
        assignment[encounter_id] = (dc_h24, dc_j28) tuple
    """
    assignment = {}
    counts = {(0, 0): 0, (0, 1): 0, (1, 1): 0}

    mono = {pid: list(profiles)[0]
            for pid, profiles in patient_profiles.items()
            if len(profiles) == 1}
    multi = {pid: profiles
             for pid, profiles in patient_profiles.items()
             if len(profiles) > 1}

    # step 1: assign mono-profile patients
    for pid, profile in mono.items():
        assignment[pid] = profile
        counts[profile] += 1

    print(f"\nAfter mono-profile assignment: {counts}")

    # step 2: priority fill for (1,1)
    # compute target = max count among non-(1,1) profiles after mono assignment
    # we want (1,1) to reach at least that level before doing general balancing
    can_reach_priority = {
        pid: profiles
        for pid, profiles in multi.items()
        if PRIORITY_PROFILE in profiles
    }
    cannot_reach_priority = {
        pid: profiles
        for pid, profiles in multi.items()
        if PRIORITY_PROFILE not in profiles
    }

    # shuffle to avoid order bias
    priority_pids = list(can_reach_priority.keys())
    rng.shuffle(priority_pids)

    other_counts = {p: c for p, c in counts.items() if p != PRIORITY_PROFILE}
    target_11 = max(other_counts.values()) if other_counts else 0

    remaining_for_general = []

    for pid in priority_pids:
        profiles = can_reach_priority[pid]
        if counts[PRIORITY_PROFILE] < target_11:
            # still under target: assign to (1,1)
            assignment[pid] = PRIORITY_PROFILE
            counts[PRIORITY_PROFILE] += 1
        else:
            # (1,1) is sufficiently represented: defer to general balancing
            remaining_for_general.append((pid, profiles))

    print(f"After (1,1) priority fill: {counts}")

    # step 3: general balancing for remaining multi-profile patients
    general_pids = remaining_for_general + [
        (pid, profiles) for pid, profiles in cannot_reach_priority.items()
    ]
    rng.shuffle(general_pids)

    for pid, profiles in general_pids:
        # assign to the least represented reachable profile
        best_profile = min(profiles, key=lambda p: counts[p])
        assignment[pid] = best_profile
        counts[best_profile] += 1

    print(f"After general balancing: {counts}")

    return assignment


# ──────────────────────────────────────────────
# STEP 5 — LOMAX-WEIGHTED TIMESTAMP DRAW
# ──────────────────────────────────────────────

def lomax_weights(positions: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """Lomax weights over normalized [0,1] positions. Falls back to uniform."""
    raw = lomax.pdf(positions, c=alpha, scale=lam)
    total = raw.sum()
    if total == 0:
        return np.ones(len(positions)) / len(positions)
    return raw / total


def draw_timestamps(
    df: pl.DataFrame,
    assignment: dict,
    lomax_alpha: float,
    lomax_lambda: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """
    For each patient, draw one timestamp from the rows matching
    their assigned profile, using Lomax weights.
    """
    sampled_rows = []

    for encounter_id, group in df.group_by("encounterId"):
        if encounter_id not in assignment:
            continue

        h24, j28 = assignment[encounter_id]
        zone = group.filter(
            (pl.col("dc_h24") == h24) & (pl.col("dc_j28") == j28)
        ).sort("delta_hour")

        if len(zone) == 0:
            continue

        hours = zone["delta_hour"].to_numpy().astype(float)
        h_min, h_max = hours.min(), hours.max()
        positions = (hours - h_min) / (h_max - h_min) if h_max > h_min else np.zeros(len(hours))

        weights = lomax_weights(positions, alpha=lomax_alpha, lam=lomax_lambda)
        chosen_idx = rng.choice(len(zone), p=weights)
        sampled_rows.append(zone[chosen_idx])

    return pl.concat(sampled_rows).sort(["encounterId", "delta_hour"])


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def build_sampling_dataset(
    df: pl.DataFrame,
    sanctuary_hours: int = SANCTUARY_HOURS,
    min_obs_hours: int = MIN_OBSERVATION_HOURS,
    lomax_alpha: float = LOMAX_ALPHA,
    lomax_lambda: float = LOMAX_LAMBDA,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:

    rng = np.random.default_rng(seed)

    n_patients_in = df["encounterId"].n_unique()
    print(f"Input: {n_patients_in} patients, {len(df)} rows")

    # step 1: pre-filter
    df = prefilter(df)
    n_patients_kept = df["encounterId"].n_unique()
    print(f"After pre-filtering: {n_patients_kept} patients "
          f"({n_patients_in - n_patients_kept} excluded, LOS too short)")

    # step 2: validate
    df = validate_profiles(df)

    # step 3: census
    patient_profiles = census(df)
    print_census_summary(patient_profiles)

    # step 4: assignment (mono → priority (1,1) → general)
    assignment = assign_profiles(patient_profiles, rng)

    # step 5: draw one timestamp per patient
    result = draw_timestamps(df, assignment, lomax_alpha, lomax_lambda, rng)

    print(f"\nFinal dataset: {len(result)} patients, one timestamp each")
    return result


# ──────────────────────────────────────────────
# USAGE EXAMPLE (synthetic data)
# ──────────────────────────────────────────────

if __name__ == "__main__":

    rng = np.random.default_rng(0)
    n_patients = 300
    rows = []

    for pid in range(n_patients):
        los = int(rng.integers(12, 400))
        dead_j28 = rng.random() < 0.35
        death_hour = int(rng.integers(6, min(los, 672))) if dead_j28 else None

        for h in range(los + 1):
            dc_h24 = (
                dead_j28
                and death_hour is not None
                and (h < death_hour <= h + 24)
            )
            rows.append({
                "encounterId": pid,
                "delta_hour": h,
                "dc_h24": dc_h24,
                "dc_j28": dead_j28,
            })

    df_raw = pl.DataFrame(rows).with_columns([
        pl.col("dc_h24").cast(pl.Boolean),
        pl.col("dc_j28").cast(pl.Boolean),
    ])

    result = build_sampling_dataset(df_raw)
    print(result.head(10))
