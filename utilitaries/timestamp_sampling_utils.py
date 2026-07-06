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
    - isDeceased_lt_24h      : bool, death within 24h after this timestamp
    - isDeceased_lt_28d      : bool, death before day 28 (absolute, patient-level)

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
LOMAX_ALPHA = 4.3085
LOMAX_LAMBDA = 1161.9368
RANDOM_SEED = 42

PRIORITY_PROFILE = (1, 1)   # protect this class first

# ──────────────────────────────────────────────
# STEP 0 — LABELIZATION (relative or absolute)
# ──────────────────────────────────────────────

def prepare_labels(df, mode):
    if mode == "relative":
        return df.with_columns([
            # H24 (delta + 24h)
            ((pl.col("deces_datediff_days") * 24 > pl.col("delta_hour")) & 
             (pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24))
            .fill_null(False).alias("isDeceased_lt_24h"),
            
            # J7 (delta + 168h)
            ((pl.col("deces_datediff_days") * 24 > pl.col("delta_hour")) & 
             (pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 168))
            .fill_null(False).alias("isDeceased_lt_7d"),
            
            # J28 (delta + 672h)
            ((pl.col("deces_datediff_days") * 24 > pl.col("delta_hour")) & 
             (pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672))
            .fill_null(False).alias("isDeceased_lt_28d"),
            
            # 3 Mois / J90 (delta + 2160h)
            ((pl.col("deces_datediff_days") * 24 > pl.col("delta_hour")) & 
             (pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 2160))
            .fill_null(False).alias("isDeceased_lt_3m")
        ])
        
    elif mode == "absolute":
        return df.with_columns([
            (pl.col("deces_datediff_days") * 24 <= 24).fill_null(False).alias("isDeceased_lt_24h"),
            (pl.col("deces_datediff_days") <= 7).fill_null(False).alias("isDeceased_lt_7d"),
            (pl.col("deces_datediff_days") <= 28).fill_null(False).alias("isDeceased_lt_28d"),
            (pl.col("deces_datediff_days") <= 90).fill_null(False).alias("isDeceased_lt_3m")
        ])
        
    elif mode == "mixed": 
        return df.with_columns([
            # H24 reste en relatif
            ((pl.col("deces_datediff_days") * 24 > pl.col("delta_hour")) & 
             (pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24))
            .fill_null(False).alias("isDeceased_lt_24h"),
            
            # Le reste passe en absolu à partir de J0
            (pl.col("deces_datediff_days") <= 7).fill_null(False).alias("isDeceased_lt_7d"),
            (pl.col("deces_datediff_days") <= 28).fill_null(False).alias("isDeceased_lt_28d"),
            (pl.col("deces_datediff_days") <= 90).fill_null(False).alias("isDeceased_lt_3m")
        ])

# ──────────────────────────────────────────────
# STEP 1 — PRE-FILTERING
# ──────────────────────────────────────────────

def prefilter(df: pl.DataFrame, sanctuary_hours, min_observation_hours) -> pl.DataFrame:
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
            (pl.col("delta_hour") >= min_observation_hours) &
            (pl.col("delta_hour") <= pl.col("los_hours") - sanctuary_hours)
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
        pl.col("isDeceased_lt_24h").cast(pl.Int8),
        pl.col("isDeceased_lt_28d").cast(pl.Int8),
    ])
    impossible = df.filter(
        (pl.col("isDeceased_lt_24h") == 1) & (pl.col("isDeceased_lt_28d") == 0)
    )
    if len(impossible) > 0:
        raise ValueError(
            f"{len(impossible)} rows with impossible profile (isDeceased_lt_24h=1, isDeceased_lt_28d=0)."
        )
    return df


# ──────────────────────────────────────────────
# STEP 3 — CENSUS
# ──────────────────────────────────────────────
def census(df: pl.DataFrame) -> dict:
    """
    For each patient, enumerate the set of reachable profiles
    within their valid sampling window.
    """
    # 1. On crée une structure (paire) et on extrait les valeurs uniques par patient
    df_agg = (
        df.group_by("encounterId")
        .agg(
            pl.struct(["isDeceased_lt_24h", "isDeceased_lt_28d"])
            .unique()
            .alias("profiles")
        )
    )
    
    # 2. On reconstruit le dictionnaire de frozensets
    # Chaque élément de 'profiles' est un dictionnaire Python {'isDeceased_lt_24h': v1, 'isDeceased_lt_28d': v2}
    return {
        row["encounterId"]: frozenset(
            (int(p["isDeceased_lt_24h"]), int(p["isDeceased_lt_28d"]))
            for p in row["profiles"]
        )
        for row in df_agg.iter_rows(named=True)
    }


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
        assignment[encounter_id] = (isDeceased_lt_24h, isDeceased_lt_28d) tuple
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

def draw_timestamps(
    df: pl.DataFrame,
    assignment: dict,
    lomax_alpha: float,
    lomax_lambda: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Sélectionne un timestamp par patient via Lomax, sans aucune boucle sur le DF."""
    if not assignment:
        return pl.DataFrame(schema=df.schema)

    # 1. Conversion du dictionnaire d'assignation en DataFrame Polars pour jointure
    assignment_df = pl.DataFrame(
        [{"encounterId": pid, "assigned_h24": h24, "assigned_j28": j28} for pid, (h24, j28) in assignment.items()],
        schema={"encounterId": df.schema["encounterId"], "assigned_h24": pl.Int8, "assigned_j28": pl.Int8}
    )

    # 2. Filtrage global des lignes qui matchent le profil affecté
    df_assigned = df.join(assignment_df, on="encounterId", how="inner").filter(
        (pl.col("isDeceased_lt_24h") == pl.col("assigned_h24")) &
        (pl.col("isDeceased_lt_28d") == pl.col("assigned_j28"))
    )

    if df_assigned.is_empty():
        return pl.DataFrame(schema=df.schema)

    # 3. Tri chronologique indispensable pour le calcul des cumsums
    df_assigned = df_assigned.sort(["encounterId", "delta_hour"])

    # 4. Au lieu d'une position [0, 1], on calcule le nombre d'heures réelles 
    # qui séparent la ligne actuelle de la fin de la fenêtre de ce profil.
    df_assigned = df_assigned.with_columns([
        pl.col("delta_hour").max().over("encounterId").alias("h_max")
    ]).with_columns(
        (pl.col("h_max") - pl.col("delta_hour")).alias("distance_fin_zone")
    )

    # 5. Calcul des poids Lomax en une seule passe NumPy
    distances_np = df_assigned["distance_fin_zone"].to_numpy()
    raw_weights = lomax.pdf(distances_np, c=lomax_alpha, scale=lomax_lambda)

    # 6. Normalisation des poids à l'échelle de chaque patient
    df_assigned = df_assigned.with_columns(pl.Series("raw_weight", raw_weights))
    df_assigned = df_assigned.with_columns(
        pl.col("raw_weight").sum().over("encounterId").alias("total_weight")
    ).with_columns(
        pl.when(pl.col("total_weight") > 0)
        .then(pl.col("raw_weight") / pl.col("total_weight"))
        .otherwise(1.0 / pl.col("raw_weight").count().over("encounterId"))
        .alias("weight")
    )

    # 7. Cumsum des poids par patient pour préparer la transformation inverse
    df_assigned = df_assigned.with_columns(
        pl.col("weight").cum_sum().over("encounterId").alias("cum_weight")
    )

    # 8. Génération d'un nombre aléatoire uniforme U(0,1) UNIQUE par patient
    unique_patients = df_assigned.select("encounterId").unique()
    u_values = rng.random(len(unique_patients))
    patient_u_df = unique_patients.with_columns(pl.Series("u", u_values))

    # 9. Jointure de la valeur U et sélection de la première ligne qui dépasse ce seuil
    df_assigned = df_assigned.join(patient_u_df, on="encounterId", how="left")
    df_assigned = df_assigned.with_columns(
        pl.col("delta_hour").max().over("encounterId").alias("max_hour")
    )

    # Le filtre garde toutes les lignes au-dessus du seuil aléatoire (+ sécurité borne supérieure)
    df_candidates = df_assigned.filter(
        (pl.col("cum_weight") >= pl.col("u")) | (pl.col("delta_hour") == pl.col("max_hour"))
    )

    # On ne retient que l'heure minimale (la toute première à avoir franchi le cap du cumsum)
    df_candidates = df_candidates.with_columns(
        pl.col("delta_hour").min().over("encounterId").alias("min_candidate_hour")
    )
    result = df_candidates.filter(pl.col("delta_hour") == pl.col("min_candidate_hour"))

    # Nettoyage final des colonnes de calcul pour coller au schéma d'origine
    cols_to_drop = [
        "assigned_h24", "assigned_j28", "h_max", "distance_fin_zone", 
        "raw_weight", "total_weight", "weight", "cum_weight", "u", 
        "max_hour", "min_candidate_hour"
    ]
    return result.drop(cols_to_drop).sort(["encounterId", "delta_hour"])


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def build_sampling_dataset(
    df: pl.DataFrame,
    sanctuary_hours: int = SANCTUARY_HOURS,
    min_observation_hours: int = MIN_OBSERVATION_HOURS,
    lomax_alpha: float = LOMAX_ALPHA,
    lomax_lambda: float = LOMAX_LAMBDA,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:

    rng = np.random.default_rng(seed)

    n_patients_in = df["encounterId"].n_unique()
    print(f"Input: {n_patients_in} patients, {len(df)} rows")

    # step 1: pre-filter
    df = prefilter(df, sanctuary_hours, min_observation_hours)
    n_patients_kept = df["encounterId"].n_unique()
    print(f"After pre-filtering: {n_patients_kept} patients ({n_patients_in - n_patients_kept} excluded)")

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
            isDeceased_lt_24h = (
                dead_j28
                and death_hour is not None
                and (h < death_hour <= h + 24)
            )
            rows.append({
                "encounterId": pid,
                "delta_hour": h,
                "deces_datediff_days": (death_hour / 24) if death_hour is not None else None
            })

    df_raw = pl.DataFrame(rows)
    df_raw = prepare_labels(df_raw, "mixed")
    result = build_sampling_dataset(df_raw)
    print(result.head(10))
