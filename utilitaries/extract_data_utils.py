import polars as pl
import os
import numpy as np
# Suppression de random as rd pour tout centraliser sur le Generator NumPy
import matplotlib.pyplot as plt

# ============================================================
# NOMS DES COLONNES TECHNIQUES (fixes, indépendants du thésaurus)
# ============================================================
COL_DATE_MESURE    = 'utcChartTime'
COL_DATE_ADMISSION = 'utcInTime'
ID_COL      = 'encounterId'
TIME_COL    = 'delta_hour'
TIME_COL2   = 'heure_calibree'
WINDOW_SIZE = 24
THESAURUS_PATH = "utilitaries/thesaurus.json"

def extract_data_survie(file_path):
    if not os.path.exists(file_path):
        print("le Fichier n'existe pas")
        return None
    df = pl.scan_parquet(file_path)
    df = df.with_columns([
        pl.col("delta_hour").cast(pl.Float64, strict=False),
        pl.col("pam").cast(pl.Float64, strict=False),
        # Je corrige ici le mauvais stockage de ecmo_type en str et non en bool
        pl.col("ecmo_all").str.to_lowercase().str.strip_chars().eq("true").alias("ecmo_all"),
    ])
    return df


# ==========================================
# 1. FONCTIONS AUXILIAIRES A PREPARE_DATA
# ==========================================

def prepare_base_data(df, target_col, other_cols, used_distribution):
    """Nettoie, recale le temps et agrège les données historiques."""
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

    df_agg = df_history
    
    if used_distribution == "flexible":
        target_by_hour = df_history.select([ID_COL, "heure_calibree", target_col] + other_cols).unique()
        df_agg = df_agg.join(target_by_hour, on=[ID_COL, "heure_calibree"], how="left")

    # Calcul des patients (min_h)
    patients = (
        df_agg.group_by(ID_COL)
        .agg(pl.col("heure_calibree").min().alias("min_h"))
        # Ajout du sort pour garantir que l'ordre des patients est identique avant l'échantillonnage
        .sort(ID_COL)
    )
    
    return df_agg, patients


def generate_random_windows(patients, df_agg, max_hour, used_distribution, target_col, show_fig, seed=42):
    """Construit les fenêtres temporelles selon une distribution aléatoire déterministe via Generator."""
    
    # On garantit la création d'un générateur d'état local et étanche
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    
    # On extrait l'ID_COL parallèlement pour être sûr de reconstruire proprement l'alignement
    list_ids = patients[ID_COL].to_list()
    if used_distribution == "uniform": 
        offsets = np.array([
            rng.integers(0, max(1, (-min_h - max_hour - (WINDOW_SIZE - 1)) + 1))
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
                offset = 0 # fallback safe
            else:
                offset = rng.choice(possible_offsets)
            offsets.append(offset)
            
    elif used_distribution == "flexible":
        target_info = (
            df_agg.filter((pl.col("heure_calibree") <= -max_hour) & (pl.col(target_col) == 1))
            .group_by(ID_COL).agg(pl.col("heure_calibree").alias("heures_positives"))
        )
        patients_selection = patients.join(target_info, on=ID_COL, how="left").sort(ID_COL)
        
        offsets = []
        for row in patients_selection.iter_rows(named=True):
            h_pos = row["heures_positives"]
            min_h = row["min_h"]
            chosen_start = None
            
            if h_pos is not None and len(h_pos) > 0:
                possibilites = [h for h in h_pos if h + (WINDOW_SIZE - 1) <= -max_hour]
                if possibilites:
                    chosen_start = rng.choice(possibilites)
                else:
                    chosen_start = max(h_pos) - (WINDOW_SIZE - 1)
            
            if chosen_start is None:
                max_possible_start = -max_hour - (WINDOW_SIZE - 1)
                if max_possible_start > min_h:
                    chosen_start = rng.integers(int(min_h), int(max_possible_start) + 1) # +1 car exclusif dans rng
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

    # Reconstruction via un df indexé sur ID_COL pour éviter les désalignements de Series
    df_offsets = pl.DataFrame({
        ID_COL: list_ids,
        "chosen_offset": offsets
    })
    if used_distribution == "flexible" :
        # Retour anticipé spécifique à "flexible" car il utilise des offsets absolus
        return (
            patients.join(df_offsets, on=ID_COL, how="inner")
            .with_columns(pl.int_ranges(pl.col("chosen_offset"), pl.col("chosen_offset") + WINDOW_SIZE).alias("heure_calibree"))
            .explode("heure_calibree")
            .select([ID_COL, "heure_calibree"])
        )

    return (
        patients.join(df_offsets, on=ID_COL, how="inner")
        .with_columns((pl.col("min_h") + pl.col("chosen_offset")).alias("start_h"))
        .with_columns((pl.col("start_h") + (WINDOW_SIZE - 1)).alias("end_h"))
        .with_columns(pl.int_ranges(pl.col("start_h"), pl.col("end_h") + 1).alias("heure_calibree"))
        .explode("heure_calibree")
        .select([ID_COL, "heure_calibree"])
    )


def generate_fixed_windows(patients, hour_offset, max_hour):
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


def finalize_data(df_windows, df_agg, strict_mode):
    """Effectue la jointure, le filtrage strict, l'imputation et les features finales."""
    df_agg = df_agg.with_columns([pl.col("heure_calibree").cast(pl.Int64), pl.lit(1).alias("real_hour")])
    df_full = df_windows.join(df_agg, on=[ID_COL, "heure_calibree"], how="left").sort([ID_COL, "heure_calibree"])
    
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
    return df_full


# ==========================================
# 2. FONCTION PRINCIPALE (L'Orchestrateur)
# ==========================================

def prepare_data(df, hour_offset=0, random=False, max_hour=0, used_distribution="uniform", 
                 strict_mode=False, target_col="isDeceased_lt_24h", other_cols=None, show_fig=True, seed=42):
    
    if other_cols is None: other_cols = []
    
    # Etape 1 : Nettoyage et Agrégation 
    df_agg, patients = prepare_base_data(df, target_col, other_cols, used_distribution)
    if df_agg is None:
        return pl.DataFrame()

    # Etape 2 : Construction des fenêtres
    if random:
        # Passage explicite de la seed (ou du Generator) à la fonction
        df_windows = generate_random_windows(patients, df_agg, max_hour, used_distribution, target_col, show_fig, seed=seed)
    else:
        df_windows = generate_fixed_windows(patients, hour_offset, max_hour)

    if df_windows is None or df_windows.is_empty():
        print(f" │   ├─ (H-{hour_offset}) ── ✕ Blocage : Aucune fenêtre construite")
        return pl.DataFrame()

    # Etape 3 : Finalisation
    df_full = finalize_data(df_windows, df_agg, strict_mode)

    return df_full