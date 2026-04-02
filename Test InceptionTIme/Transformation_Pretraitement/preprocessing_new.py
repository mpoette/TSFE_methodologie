import pandas as pd
import numpy as np
import config as cfg
import utils
 
 
def prepare_data(df, hour_offset=0):
    thesaurus = utils.load_thesaurus("thesaurus.json")
 
    if df.empty:
        print(" │   ├─ XX:XX (H-0) ── ✕ Blocage : DataFrame vide")
        return pd.DataFrame()
 
    if cfg.TIME_COL not in df.columns:
        raise ValueError(f"Colonne temporelle absente : {cfg.TIME_COL}")
 
    if cfg.ID_COL not in df.columns:
        raise ValueError(f"Colonne identifiant absente : {cfg.ID_COL}")
 
    # Affichage seulement
    if cfg.COL_DATE_MESURE in df.columns:
        real_max_time = pd.to_datetime(df[cfg.COL_DATE_MESURE], errors="coerce").max()
        if pd.notna(real_max_time):
            simulated_time = real_max_time - pd.Timedelta(hours=hour_offset)
            time_str = simulated_time.strftime("%H:%M")
        else:
            time_str = "XX:XX"
    else:
        time_str = "XX:XX"
 
    df = df.copy()
    # par précaution, on retransforme delta_hour en float/int si c'est pas déjà le cas
    df[cfg.TIME_COL] = pd.to_numeric(df[cfg.TIME_COL], errors="coerce")
    df = df.dropna(subset=[cfg.TIME_COL])
 
    if df.empty:
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Temps invalides")
        return pd.DataFrame()
 
    # 1) Temps entier : on arrondit en coupant les composantes à virgules
    df["heure_entiere"] = np.floor(df[cfg.TIME_COL]).astype(int)
 
    # 2) Recalage par rapport à la sortie : 0 = dernière heure, négatif = passé
    max_time = df.groupby(cfg.ID_COL)["heure_entiere"].transform("max") - hour_offset
    df["heure_calibree"] = df["heure_entiere"] - max_time
 
    # 3) On garde uniquement l'historique jusqu'à l'heure cible
    # Pourquoi garder plus sachant que pour l'instant la fenêtre n'est pas glissante ?
    df_history = df[df["heure_calibree"] <= 0].copy()
 
    if df_history.empty:
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Historique vide")
        return pd.DataFrame()
 
    # 4) Agrégation
    df_agg = utils.apply_generic_aggregation(
        df_history,
        thesaurus,
        cfg.ID_COL,
        "heure_calibree"
    )
 
    # 5) Pour chaque patient, prendre les 24 premières heures du séjour
    #    en gardant le repère négatif basé sur la sortie.
    pieces = []
 
    for pid in df_agg.index.get_level_values(cfg.ID_COL).unique():
        df_pid = df_agg.xs(pid, level=cfg.ID_COL).sort_index()
 
        if not df_pid.empty:
 
            min_h = int(df_pid.index.min())
            start_h = min_h
            end_h = min_h + cfg.WINDOW_SIZE - 1
    
            idx_pid = pd.Index(range(start_h, end_h + 1), name="heure_calibree")
            df_pid = df_pid.reindex(idx_pid)
    
            # Remettre l'identifiant patient dans l'index MultiIndex
            df_pid[cfg.ID_COL] = pid
            df_pid = df_pid.set_index(cfg.ID_COL, append=True)
            df_pid = df_pid.reorder_levels([cfg.ID_COL, "heure_calibree"])
    
            pieces.append(df_pid)
 
    if not pieces:
        print(f" │   ├─ {time_str} (H-{hour_offset}) ── ✕ Blocage : Aucune fenêtre construite")
        return pd.DataFrame()
 
    df_full = pd.concat(pieces).sort_index()
 
    # 6) Imputation
    df_full = utils.apply_generic_imputation(df_full, thesaurus)
 
    # 7) Gestion de la dialyse
    for _col, _def in [("dialyse_hdi", 0), ("dialyse_cvvhf", 0), ("abs_dialyse", 1)]:
        if _col not in df_full.columns:
            df_full[_col] = _def
 
    mask_absent = (
        (df_full["dialyse_hdi"] == 0) &
        (df_full["dialyse_cvvhf"] == 0)
    )
    df_full["abs_dialyse"] = mask_absent.astype(int)
 
    print(f" │   ├─ {time_str} (H-{hour_offset}) ")
 
    return df_full.reset_index()