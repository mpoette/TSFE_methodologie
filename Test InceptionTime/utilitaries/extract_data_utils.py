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


def _load_thesaurus(filepath: str = THESAURUS_PATH) -> dict:
    """Charge la configuration des variables cliniques."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def _apply_generic_aggregation(
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
 
def _apply_generic_imputation(
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


# ==========================================
# 1. FONCTIONS AUXILIAIRES A PREPARE_DATA
# ==========================================

def _prepare_base_data(df, target_col, other_cols, used_distribution):
    """Nettoie, recale le temps et agrège les données historiques."""
    thesaurus = _load_thesaurus(THESAURUS_PATH)
    time_str = "XX:XX"
    
    if df.is_empty():
        print(" │   ├─ XX:XX (H-0) ── ✕ Blocage : DataFrame vide")
        return None, None
        
    if TIME_COL not in df.columns or ID_COL not in df.columns:
        raise ValueError(f"Colonnes temporelle ou ID absentes.")

    # Cast et nettoyage (par précaution)
    df = df.with_columns(pl.col(TIME_COL).cast(pl.Float64, strict=False)).drop_nulls(subset=TIME_COL)
    if df.is_empty():
        print(f" │   ├─ {time_str} ── ✕ Blocage : Temps invalides")
        return None, None

    # Recalage temporel par rapport à la sortie : 0 = dernière heure, négatif = passé
    # On arrondit en coupant les composantes à virgules
    df = df.with_columns(pl.col(TIME_COL).floor().alias("heure_entiere"))
    df = df.with_columns(
        (pl.col("heure_entiere") - pl.col("heure_entiere").max().over(ID_COL)).alias("heure_calibree")
    )
    
    df_history = df.filter(pl.col("heure_calibree") <= 0)
    if df_history.is_empty():
        print(f" │   ├─ {time_str} ── ✕ Blocage : Historique vide")
        return None, None

    # Agrégation
    df_agg = _apply_generic_aggregation(df_history, thesaurus, ID_COL, "heure_calibree")
    
    if used_distribution == "flexible":
        target_by_hour = df_history.select([ID_COL, "heure_calibree", target_col] + other_cols).unique()
        df_agg = df_agg.join(target_by_hour, on=[ID_COL, "heure_calibree"], how="left")

    # Calcul des patients (min_h)
    patients = df_agg.group_by(ID_COL).agg(pl.col("heure_calibree").min().alias("min_h"))
    
    return df_agg, patients


def _generate_random_windows(patients, df_agg, max_hour, used_distribution, target_col, show_fig):
    """Construit les fenêtres temporelles selon une distribution aléatoire."""
    if used_distribution == "uniform": 
        offsets = np.array([
            np.random.randint(0, max(1, (-min_h - max_hour - (WINDOW_SIZE - 1)) + 1))
            for min_h in patients["min_h"].to_list()
        ])
    elif used_distribution == "real":
        patients = patients.with_columns((-pl.col("min_h")).alias("max_h"))
        real_distribution = patients["max_h"].to_numpy()
        offsets = []
        for max_h in real_distribution:
            max_offset = max(1, (max_h - max_hour - (WINDOW_SIZE - 1)) + 1)
            possible_offsets = real_distribution[real_distribution <= max_offset]
            if len(possible_offsets) == 0:
                offset = 0  # fallback safe
            else:
                offset = np.random.choice(possible_offsets)
            offsets.append(offset)
            
    elif used_distribution == "flexible":
        target_info = (
            df_agg.filter((pl.col("heure_calibree") <= -max_hour) & (pl.col(target_col) == 1))
            .group_by(ID_COL).agg(pl.col("heure_calibree").alias("heures_positives"))
        )
        patients_selection = patients.join(target_info, on=ID_COL, how="left")
        
        offsets = []
        for row in patients_selection.iter_rows(named=True):
            h_pos = row["heures_positives"]
            min_h = row["min_h"]
            chosen_start = None
            
            if h_pos is not None and len(h_pos) > 0:
                possibilites = [h for h in h_pos if h + (WINDOW_SIZE - 1) <= -max_hour]
                if possibilites:
                    chosen_start = rd.choice(possibilites)
                else:
                    chosen_start = max(h_pos) - (WINDOW_SIZE - 1)
            
            if chosen_start is None:
                max_possible_start = -max_hour - (WINDOW_SIZE - 1)
                if max_possible_start > min_h:
                    chosen_start = rd.randint(int(min_h), int(max_possible_start))
                else:
                    chosen_start = min_h
            offsets.append(chosen_start)
            
        offsets = [int(x) for x in offsets]
    else:
        print(f"Distribution {used_distribution} non prise en charge")
        return None

    if show_fig:
        x = np.array(offsets)
        q05, q95 = np.quantile(x, [0.05, 0.95])
        x_filtered = x[(x >= q05) & (x <= q95)]
        plt.figure(figsize=(10, 6))
        plt.hist(x_filtered, bins=100, color='skyblue', edgecolor='black')
        plt.title(f"Distribution des offsets relatifs (sanctuarisation : {max_hour})\nLoi: {used_distribution}")
        plt.show()
    
    if used_distribution == "flexible" :
        # Retour anticipé spécifique à "flexible" car il utilise des offsets absolus
        return (
            patients.with_columns(pl.Series("start_h", offsets))
            .with_columns(pl.int_ranges(pl.col("start_h"), pl.col("start_h") + WINDOW_SIZE).alias("heure_calibree"))
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
        )

    # Application des offsets (pour Uniform et Real)
    patients = patients.with_columns(pl.Series("hour_offset", offsets))
    return (
        patients.with_columns((pl.col("min_h") + pl.col("hour_offset")).alias("start_h"))
        .with_columns((pl.col("start_h") + (WINDOW_SIZE - 1)).alias("end_h"))
        .with_columns(pl.int_ranges(pl.col("start_h"), pl.col("end_h") + 1).alias("heure_calibree"))
        .explode("heure_calibree")
        .select([ID_COL, "heure_calibree"])
    )


def _generate_fixed_windows(patients, hour_offset, max_hour):
    """Construit les fenêtres de manière fixe, sans aléatoire."""
    patients = patients.with_columns(pl.lit(hour_offset).alias("hour_offset"))
    
    if hour_offset == -1:
        return (
            patients.with_columns(pl.int_ranges(-(WINDOW_SIZE - 1) - max_hour, 1 - max_hour).alias("heure_calibree"))
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
        )
    else:
        return (
            patients.with_columns((pl.col("min_h") + pl.col("hour_offset")).alias("start_h"))
            .with_columns((pl.col("start_h") + (WINDOW_SIZE - 1)).alias("end_h"))
            .with_columns(pl.int_ranges(pl.col("start_h"), pl.col("end_h") + 1).alias("heure_calibree"))
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
        )


def _finalize_data(df_windows, df_agg, strict_mode):
    """Effectue la jointure, le filtrage strict, l'imputation et les features finales."""
    thesaurus = _load_thesaurus(THESAURUS_PATH)
    
    # Jointure
    df_agg = df_agg.with_columns([pl.col("heure_calibree").cast(pl.Int64), pl.lit(1).alias("real_hour")])
    df_full = df_windows.join(df_agg, on=[ID_COL, "heure_calibree"], how="left").sort([ID_COL, "heure_calibree"])
    
    # Filtrage strict
    if strict_mode:
        seuil = 1
        valid_ids = (
            df_full.group_by(ID_COL)
            .agg(pl.col("real_hour").fill_null(0).sum().alias("nb_hour_present"))
            .filter(pl.col("nb_hour_present") >= WINDOW_SIZE * seuil)
            .select(ID_COL)
        )
        df_full = df_full.join(valid_ids, on=ID_COL, how="inner")
        
    df_full = df_full.drop('real_hour')
    if df_full.is_empty():
        return df_full

    # Imputation
    df_full = _apply_generic_imputation(df_full, thesaurus, ID_COL)

    # Variables Dialyse
    missing_dial_cols = []
    for _col, _def in [("dialyse_hdi", 0), ("dialyse_cvvhf", 0), ("abs_dialyse", 1)]:
        if _col not in df_full.columns:
            missing_dial_cols.append(pl.lit(_def).cast(pl.Float64).alias(_col))
            
    if missing_dial_cols:
        df_full = df_full.with_columns(missing_dial_cols)
        
    df_full = df_full.with_columns(
        ((pl.col("dialyse_hdi") == 0) & (pl.col("dialyse_cvvhf") == 0)).cast(pl.Int64).alias("abs_dialyse")
    )
    
    return df_full


# ==========================================
# 2. FONCTION PRINCIPALE (L'Orchestrateur)
# ==========================================

def prepare_data(df, hour_offset=0, random=False, max_hour=0, used_distribution="uniform", 
                 strict_mode=False, target_col="isDeceased_lt_24h", other_cols=None, show_fig = True):
    
    if other_cols is None: other_cols = []
    
    # Etape 1 : Nettoyage et Agrégation
    df_agg, patients = _prepare_base_data(df, target_col, other_cols, used_distribution)
    if df_agg is None:
        return pl.DataFrame()

    # Etape 2 : Construction des fenêtres (Routing Random vs Fixed)
    if random:
        df_windows = _generate_random_windows(patients, df_agg, max_hour, used_distribution, target_col, show_fig)
    else:
        df_windows = _generate_fixed_windows(patients, hour_offset, max_hour)

    if df_windows is None or df_windows.is_empty():
        print(f" │   ├─ (H-{hour_offset}) ── ✕ Blocage : Aucune fenêtre construite")
        return pl.DataFrame()

    # Etape 3 : Finalisation (Jointure, Imputation, Features)
    df_full = _finalize_data(df_windows, df_agg, strict_mode)

    return df_full


    
    

    
