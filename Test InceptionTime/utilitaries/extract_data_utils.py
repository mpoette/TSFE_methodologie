import polars as pl
import os
import numpy as np
import random as rd
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import json


# ============================================================
# NOMS DES COLONNES TECHNIQUES (fixes, indépendants du thésaurus)
# ============================================================
COL_DATE_MESURE    = 'utcChartTime'
COL_DATE_ADMISSION = 'utcInTime'
ID_COL      = 'encounterId'
TIME_COL    = 'delta_hour'
WINDOW_SIZE = 24
THESAURUS_PATH = "utilitaries/thesaurus.json"
def load_thesaurus(filepath: str = THESAURUS_PATH) -> dict:
    """Charge la configuration des variables cliniques."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_generic_aggregation(
    df: pl.DataFrame,
    thesaurus: dict,
    id_col: str,
    time_col: str
) -> pl.DataFrame:
    """Agrège les données selon les méthodes définies (median, sum, max)."""
 
    agg_map = {}
 
    for feat_id, config in thesaurus["features"].items():
        if config.get("type") in ("static", "source_only", "derived"):
            continue
 
        cols = config["column"] if "column" in config else config.get("columns", [])
        method = config.get("agg_method", "median")
 
        if isinstance(cols, list):
            for c in cols:
                agg_map[c] = method
        else:
            agg_map[cols] = method
 
    # Ajouter les colonnes manquantes
    missing_cols = [col for col in agg_map if col not in df.columns]
    if missing_cols:
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(col) # constante qui vaut None pour toutes les lignes (lit pour littéral)
            for col in missing_cols
        ])
 
    # Construire les expressions d'agrégation
    agg_exprs = []
    for col, method in agg_map.items():
        if method == "median":
            agg_exprs.append(pl.col(col).median().alias(col))
        elif method == "sum":
            agg_exprs.append(pl.col(col).sum().alias(col))
        elif method == "max":
            agg_exprs.append(pl.col(col).max().alias(col))
        elif method == "first":
            agg_exprs.append(pl.col(col).fill_nan(None).drop_nulls().first().alias(col))
        else:
            raise ValueError(f"Méthode d'agrégation non supportée pour {col}: {method}")
 
    return df.group_by([id_col, time_col]).agg(agg_exprs)
 
def apply_generic_imputation(
    df: pl.DataFrame,
    thesaurus: dict,
    id_col: str,
) -> pl.DataFrame:
    """Impute les colonnes selon la stratégie définie dans le thesaurus."""
 
    exprs = []
 
    for feat_id, config in thesaurus["features"].items():
        if config.get("type") in ("static", "source_only", "derived"):
            continue
 
        cols = config["column"] if "column" in config else config.get("columns", [])
        method = config.get("imputation_method")
        default = config.get("default_value", 0)
 
        if not isinstance(cols, list):
            cols = [cols]
 
        for col in cols:
            # Si la colonne n'existe pas, on la crée remplie avec la valeur par défaut
            if col not in df.columns:
                exprs.append(
                    pl.lit(default).cast(pl.Float64).alias(col)
                )
                continue
 
            expr = pl.col(col)
 
            if method == "interpolate_ffill_bfill":
                expr = (
                    expr
                    .interpolate()
                    .over(id_col)
                    .forward_fill()
                    .over(id_col)
                    .backward_fill()
                    .over(id_col)
                )
            # pas d'équivalent polars. Je dois donc temporairement passer en pandas..
            elif method == "interpolate_limit":
                pdf = df.select([id_col, col]).to_pandas()
            
                pdf[col] = (
                    pdf.groupby(id_col)[col]
                    .transform(lambda x: x.interpolate(method="linear", limit=6).ffill().bfill())
                    .fillna(default)
                )
            
                df = df.with_columns(
                    pl.Series(col, pdf[col])
                )
            
                continue  # important pour éviter de faire ce qui se trouve après les branchements conditionnels
 
            elif method == "ffill":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .fill_null(default)
                )
 
            elif method == "ffill_zero":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .fill_null(0)
                )
 
            elif method == "ffill_bfill":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .backward_fill()
                    .over(id_col)
                )
 
            elif method == "fillna_zero":
                expr = expr.fill_null(0)
 
            expr = expr.fill_null(default)
 
            exprs.append(expr.alias(col))
 
    return df.with_columns(exprs)


def extract_data_survie(file_path):
    if not os.path.exists(file_path):
        print("le Fichier n'existe pas")
        return None
 
    if file_path.endswith('.parquet'):
        df = pl.read_parquet(file_path)
    else:
        df = pl.read_csv(file_path)
    df = df.with_columns([
        pl.col("delta_hour").cast(pl.Float64, strict=False),
        pl.col("pam").cast(pl.Float64, strict=False),
        # Je corrige ici le mauvais stockage de tracheo et ecmo_type en str et non en bool
        # Pour l'instant la base est propre donc on a que des "true" et "false" mais ici
        # cela fonctionnerait même si on avait "FaLsE" ou "True"
        pl.col("tracheo").str.to_lowercase().str.strip_chars().eq("true").alias("tracheo"),
        pl.col("ecmo_type").str.to_lowercase().str.strip_chars().eq("true").alias("ecmo_type"),
    ])
 
    print(f"{df.height} lignes chargées.")
    return df


def prepare_data(df, hour_offset=0, random = False, max_hour = 0, used_distribution = "uniform", strict_mode = False,
 target_col = "isDeceased_lt_24h"):
    # Extraction des features importantes
    thesaurus = load_thesaurus(THESAURUS_PATH)
 
    if df.is_empty():
        print(" │   ├─ XX:XX (H-0) ── ✕ Blocage : DataFrame vide")
        return pl.DataFrame()
 
    if TIME_COL not in df.columns:
        raise ValueError(f"Colonne temporelle absente : {TIME_COL}")
 
    if ID_COL not in df.columns:
        raise ValueError(f"Colonne identifiant absente : {ID_COL}")

    time_str = "XX:XX"
 
    # par précaution, on retransforme delta_hour en float si c'est pas déjà le cas
    
    df = df.with_columns(pl.col(TIME_COL).cast(pl.Float64, strict = False))
    df = df.drop_nulls(subset=TIME_COL)
 
    if df.is_empty():
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Temps invalides")
        return pl.DataFrame()
 
    # 1) Temps entier : on arrondit en coupant les composantes à virgules
    df = df.with_columns(pl.col(TIME_COL).floor().alias("heure_entiere"))
 
    # 2) Recalage par rapport à la sortie : 0 = dernière heure, négatif = passé
    df = df.with_columns(
    (
        pl.col("heure_entiere")
        - pl.col("heure_entiere").max().over(ID_COL)
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
    df_agg = apply_generic_aggregation(
        df_history,
        thesaurus,
        ID_COL,
        "heure_calibree"
    )
    if used_distribution == "flexible" :
        # TODO : petite bidouille pour garder target_col mais faudra que je le fasse plus propre que ça
        target_by_hour = df_history.select([ID_COL, "heure_calibree", target_col]).unique()
        df_agg = df_agg.join(target_by_hour, on=[ID_COL, "heure_calibree"], how="left")
    # 5) Si c'est aléatoire, on calcule l'offset pour chaque patient
    patients = (
        df_agg
        .group_by(ID_COL)
        .agg(
            pl.col("heure_calibree").min().alias("min_h")
            )
    )
    if random :
        if used_distribution == "uniform" : 
            offsets = np.array([
                np.random.randint(0, max(1,(-min_h - max_hour - (WINDOW_SIZE - 1)) + 1))
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

                max_offset = max(1, (max_h - max_hour - (WINDOW_SIZE - 1)) + 1)

                possible_offsets = real_distribution[real_distribution <= max_offset]

                if len(possible_offsets) == 0:
                    print("oups")
                    offset = 0  # fallback safe
                    count +=1
                else:
                    offset = np.random.choice(possible_offsets)

                offsets.append(offset)
        elif used_distribution == "flexible" :
            # 1) On identifie pour chaque patient les heures où target_col == 1
            # On filtre déjà pour respecter la sanctuarisation (max_hour)
            # On récupère la liste des heures avec cible = 1 par patient
            target_info = (
                df_agg
                .filter((pl.col("heure_calibree") <= -max_hour) & (pl.col(target_col) == 1))
                .group_by(ID_COL)
                .agg(pl.col("heure_calibree").alias("heures_positives"))
            )

            # 2) On rejoint cette info avec nos patients pour décider du start_h
            patients_selection = patients.join(target_info, on=ID_COL, how="left")
            
            offsets = []
            for row in patients_selection.iter_rows(named=True):
                h_pos = row["heures_positives"]
                min_h = row["min_h"]
                
                # Détermination du début de fenêtre (start_h)
                chosen_start = None
                
                # Règle : Si on a des moments où target_col == 1
                if h_pos is not None and len(h_pos) > 0:
                    # On ne garde que les départs qui permettent de tenir la WINDOW_SIZE sans dépasser -max_hour
                    # Si on veut que la cible soit DANS la fenêtre, le start_h doit être 
                    # entre (h - WINDOW_SIZE + 1) et h :
                    possibilites = [h for h in h_pos if h + (WINDOW_SIZE - 1) <= -max_hour]
                    
                    if possibilites:
                        chosen_start = rd.choice(possibilites)
                    else:
                        # Si on ne peut pas avoir 24h d'affilé après le '1', on prend le max possible
                        chosen_start = max(h_pos) - (WINDOW_SIZE - 1)
                
                # Si pas de cible == 1 ou si aucune fenêtre valide trouvée au dessus
                if chosen_start is None:
                    max_possible_start = -max_hour - (WINDOW_SIZE - 1)
                    if max_possible_start > min_h:
                        chosen_start = rd.randint(int(min_h), int(max_possible_start))
                    else:
                        chosen_start = min_h

                offsets.append(chosen_start)
            offsets = [int(x) for x in offsets]
            # 3) On génère la structure finale pour la distribution 'flexible'
            df_windows = (
                patients
                .with_columns(pl.Series("start_h", offsets))
                .with_columns(
                    pl.int_ranges(
                        pl.col("start_h"),
                        pl.col("start_h") + WINDOW_SIZE
                    ).alias("heure_calibree")
                )
                .explode("heure_calibree")
                .select([ID_COL, "heure_calibree"])
            )
        else:
            print(f"Distribution {used_distribution} non prise en charge ")
            return pl.DataFrame()
        x = np.array(offsets)
        q05, q95 = np.quantile(x, [0.05, 0.95])
        x_filtered = x[(x >= q05) & (x <= q95)]
        plt.figure(figsize=(10, 6))
        plt.hist(x_filtered, bins=100, color='skyblue', edgecolor='black')
        plt.title(f"Distribution des offsets relatifs (Début admission + X heures)\nLoi: {used_distribution}")
        plt.xlabel("Heures après le début de l'admission")
        plt.ylabel("Nombre de patients")
        plt.grid(alpha=0.3)
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
                    -(WINDOW_SIZE - 1) - max_hour,
                    1 - max_hour
                ).alias("heure_calibree")
            )
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
        )

    else:
        df_windows = (
            patients
            .with_columns(
                (pl.col("min_h") + pl.col("hour_offset")).alias("start_h")
            )
            .with_columns(
                (pl.col("start_h") + (WINDOW_SIZE - 1)).alias("end_h")
            )
            .with_columns(
                pl.int_ranges(
                    pl.col("start_h"),
                    pl.col("end_h") + 1,
                ).alias("heure_calibree")
            )
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
            )
    # Join avec les données agrégées pour faire apparaître les heures manquantes
    df_agg = df_agg.with_columns([pl.col("heure_calibree").cast(pl.Int64),
                                  pl.lit(1).alias("real_hour"),
                                  ])
    df_full = (
        df_windows
        .join(
            df_agg,
            on=[ID_COL, "heure_calibree"],
            how="left"
        )
        .sort([ID_COL, "heure_calibree"])
    )
    # là actuellement, un patient n'est gardé que s'il a au moins une mesure par heure, 
    # n'importe laquelle, d'où le faible nombre de données gardées
    # TODO : Il faudrait donc baisser le seuil.
    # Par exemple, dire qu'on garde que si le patient a 50% 
    # des heures avec des mesures
    seuil = 1
    if strict_mode:
        valid_ids = (df_full.group_by(ID_COL).agg(
            pl.col("real_hour").fill_null(0).sum().alias("nb_hour_present")
        )
        .filter(pl.col("nb_hour_present") >= WINDOW_SIZE * seuil).select(ID_COL)
        )
        df_full = df_full.join(valid_ids, on = ID_COL, how = "inner")
    df_full = df_full.drop('real_hour')

    if df_full.is_empty():
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Aucune fenêtre construite")
        return pl.DataFrame()

    # 6) Imputation
    df_full = apply_generic_imputation(df_full, thesaurus, ID_COL)

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


    
    

    
