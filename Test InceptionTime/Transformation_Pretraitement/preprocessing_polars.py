import polars as pl
import numpy as np
import config as cfg
import utils_polars as utils
 
import pandas as pd
from datetime import timedelta

def prepare_data(df, hour_offset=0):
    # Extraction des features importantes
    thesaurus = utils.load_thesaurus("thesaurus.json")
 
    if df.is_empty():
        print(" │   ├─ XX:XX (H-0) ── ✕ Blocage : DataFrame vide")
        return pl.DataFrame()
 
    if cfg.TIME_COL not in df.columns:
        raise ValueError(f"Colonne temporelle absente : {cfg.TIME_COL}")
 
    if cfg.ID_COL not in df.columns:
        raise ValueError(f"Colonne identifiant absente : {cfg.ID_COL}")
 
    # Affichage seulement
    if cfg.COL_DATE_MESURE in df.columns:
        real_max_time = df.select(
            pl.col(cfg.COL_DATE_MESURE).str.strptime(pl.Datetime, strict=False, exact=False).max()
        ).item()

        if real_max_time is not None:
            simulated_time = real_max_time - timedelta(hours=hour_offset)
            time_str = simulated_time.strftime("%H:%M")
        else:
            time_str = "XX:XX"
    else:
        time_str = "XX:XX"
 
    # par précaution, on retransforme delta_hour en float si c'est pas déjà le cas
    
    df = df.with_columns(pl.col(cfg.TIME_COL).cast(pl.Float64, strict = False))
    df = df.drop_nulls(subset=cfg.TIME_COL)
 
    if df.is_empty():
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Temps invalides")
        return pl.DataFrame()
 
    # 1) Temps entier : on arrondit en coupant les composantes à virgules
    df = df.with_columns(pl.col(cfg.TIME_COL).floor().alias("heure_entiere"))
 
    # 2) Recalage par rapport à la sortie : 0 = dernière heure, négatif = passé
    df = df.with_columns(
    (
        pl.col("heure_entiere")
        - pl.col("heure_entiere").max().over(cfg.ID_COL)
    ).alias("heure_calibree")
    )
 
    # 3) On garde uniquement l'historique jusqu'à l'heure cible
    # Pourquoi garder plus sachant que pour l'instant la fenêtre n'est pas glissante ?
    df_history = df.filter(pl.col("heure_calibree") <= 0)
 
    if df_history.is_empty():
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Historique vide")
        return pl.DataFrame()
    # 4) Agrégation
    df_agg = utils.apply_generic_aggregation(
        df_history,
        thesaurus,
        cfg.ID_COL,
        "heure_calibree"
    )
    df_windows = (
        df_agg
        .group_by(cfg.ID_COL)
        .agg(
            pl.col("heure_calibree").min().alias("start_h")
        )
        .with_columns(
            (pl.col("start_h") + (cfg.WINDOW_SIZE - 1)).alias("end_h")
        )
        .with_columns(
            pl.int_ranges(
                pl.col("start_h"),
                pl.col("end_h") + 1,
            ).alias("heure_calibree")
        )
        .explode("heure_calibree")
        .select([cfg.ID_COL, "heure_calibree"])
    )
    
    # Join avec les données agrégées pour faire apparaître les heures manquantes
    df_agg = df_agg.with_columns(pl.col("heure_calibree").cast(pl.Int64))
    df_full = (
        df_windows
        .join(
            df_agg,
            on=[cfg.ID_COL, "heure_calibree"],
            how="left"
        )
        .sort([cfg.ID_COL, "heure_calibree"])
    )
    
    if df_full.is_empty():
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Aucune fenêtre construite")
        return pl.DataFrame()

    # 6) Imputation
    df_full = utils.apply_generic_imputation(df_full, thesaurus, cfg.ID_COL)

    # 7) Gestion de la dialyse
    missing_dial_cols = []
    for _col, _def in [("dialyse_hdi", 0), ("dialyse_cvvhf", 0), ("abs_dialyse", 1)]:
        if _col not in df_full.columns:
            missing_dial_cols.append(
                pl.lit(_def).cast(pl.Float64).alias(_col)
            )
    
    if missing_dial_cols:
        df_full = df_full.with_columns(missing_dial_cols)
    
    df_full = df_full.with_columns(
        (
            (
                (pl.col("dialyse_hdi") == 0) &
                (pl.col("dialyse_cvvhf") == 0)
            )
            .cast(pl.Int64)
            .alias("abs_dialyse")
        )
    )
    
    return df_full