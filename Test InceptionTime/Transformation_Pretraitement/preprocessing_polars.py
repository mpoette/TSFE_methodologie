import polars as pl
import numpy as np
import config as cfg
import utils_polars as utils
import random as rd
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
def prepare_data(df, hour_offset=0, random = False, max_hour = 0, used_distribution = "uniform", strict_mode = False):
    # Extraction des features importantes
    thesaurus = utils.load_thesaurus("thesaurus.json")
 
    if df.is_empty():
        print(" │   ├─ XX:XX (H-0) ── ✕ Blocage : DataFrame vide")
        return pl.DataFrame()
 
    if cfg.TIME_COL not in df.columns:
        raise ValueError(f"Colonne temporelle absente : {cfg.TIME_COL}")
 
    if cfg.ID_COL not in df.columns:
        raise ValueError(f"Colonne identifiant absente : {cfg.ID_COL}")

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
    # ligne peut-être inutile mais dans le doute je la laisse
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
    # 5) Si c'est aléatoire, on calcule l'offset pour chaque patient
    patients = (
        df_agg
        .group_by(cfg.ID_COL)
        .agg(
            pl.col("heure_calibree").min().alias("min_h")
            )
    )
    if random :
        if used_distribution == "uniform" : 
            offsets = np.array([
                np.random.randint(0, max(1,(-min_h - max_hour - (cfg.WINDOW_SIZE - 1)) + 1))
                for min_h in patients["min_h"].to_list()
            ])
        elif used_distribution == "real" :
            patients = patients.with_columns(
                (-pl.col("min_h")).alias("max_h")
            )
            real_distribution = patients["max_h"].to_numpy()
            offsets = []
            count = 0
            for max_h in real_distribution:

                max_offset = max(1, (max_h - max_hour - (cfg.WINDOW_SIZE - 1)) + 1)

                possible_offsets = real_distribution[real_distribution <= max_offset]

                if len(possible_offsets) == 0:
                    print("oups")
                    offset = 0  # fallback safe
                    count +=1
                else:
                    offset = np.random.choice(possible_offsets)

                offsets.append(offset)
        else:
            print(f"Distribution {used_distribution} non prise en charge ")
            return pl.DataFrame()
        x = np.array(offsets)
        q99 = np.quantile(x, 0.99)

        x_filtered = x[x <= q99]

        plt.hist(x_filtered, bins=200)
        plt.title(f"Distribution de l'offset pour une loi '{used_distribution}'")
        plt.show()

        patients = patients.with_columns(
            pl.Series("hour_offset", offsets)
        )
    else:
        patients = patients.with_columns(
            pl.lit(hour_offset).alias("hour_offset")
        )
    if hour_offset == -1:
        df_windows = (
            patients
            .with_columns(
                pl.int_ranges(
                    -(cfg.WINDOW_SIZE - 1) - max_hour,
                    1 - max_hour
                ).alias("heure_calibree")
            )
            .explode("heure_calibree")
            .select([cfg.ID_COL, "heure_calibree"])
        )
    else:
        df_windows = (
            patients
            .with_columns(
                (pl.col("min_h") + pl.col("hour_offset")).alias("start_h")
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
    df_agg = df_agg.with_columns([pl.col("heure_calibree").cast(pl.Int64),
                                  pl.lit(1).alias("real_hour"),
                                  ])
    df_full = (
        df_windows
        .join(
            df_agg,
            on=[cfg.ID_COL, "heure_calibree"],
            how="left"
        )
        .sort([cfg.ID_COL, "heure_calibree"])
    )
    
    if strict_mode:
        valid_ids = (df_full.group_by(cfg.ID_COL).agg(
            pl.col("real_hour").fill_null(0).sum().alias("nb_hour_present")
        )
        .filter(pl.col("nb_hour_present") >= cfg.WINDOW_SIZE).select(cfg.ID_COL)
        )
        df_full = df_full.join(valid_ids, on = cfg.ID_COL, how = "inner")
    # df_full = df_full.drop('real_hour')

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


    
    

    
