import utilitaries.extract_data_utils as edu 
import polars as pl
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import utilitaries.extract_data_utils as extract
import utilitaries.features_extraction_utils as extract_feat
def scaling(df_train: pl.DataFrame, df_test: pl.DataFrame):
    """
    Scales numerical columns using StandardScaler, excluding booleans 
    and columns containing only 0 and 1.
    """

    # 1. Identifier les colonnes numériques (float/int)
    # On exclut d'office la colonne patient et les booléens natifs
    all_num_cols = df_train.select(
        pl.col(pl.NUMERIC_DTYPES).exclude(edu.ID_COL)
    ).columns

    # 2. Filtrer pour exclure les colonnes qui ne contiennent QUE (0, 1, ou Null)
    num_cols_to_scale = []
    for col in all_num_cols:
        # On vérifie si les valeurs uniques sont incluses dans {0, 1, None}
        unique_vals = df_train.select(pl.col(col).unique()).to_series().to_list()
        is_binary = all(v in [0, 1, None] for v in unique_vals)
        
        if not is_binary:
            num_cols_to_scale.append(col)

    if not num_cols_to_scale:
        return df_train, df_test

    # 3. Scaling via Scikit-Learn
    # On ne transforme en Pandas QUE les colonnes nécessaires au dernier moment
    scaler = StandardScaler()
    
    # Fit & Transform sur le train
    train_scaled_values = scaler.fit_transform(
        df_train.select(num_cols_to_scale).to_pandas()
    )
    
    # Transform sur le test
    test_scaled_values = scaler.transform(
        df_test.select(num_cols_to_scale).to_pandas()
    )

    # 4. Reconstruction des DataFrames Polars (évite toute contamination)
    # On remplace les anciennes colonnes par les nouvelles scalées
    df_train_final = df_train.with_columns([
        pl.Series(name, train_scaled_values[:, i]) 
        for i, name in enumerate(num_cols_to_scale)
    ])
    
    df_test_final = df_test.with_columns([
        pl.Series(name, test_scaled_values[:, i]) 
        for i, name in enumerate(num_cols_to_scale)
    ])

    return df_train_final, df_test_final

def build_sequences(df, patient_col, target_col, expected_length, keep_features):
    """
    Fonction qui permet d'extraire des dataframes train et test, les features appropriées et les split entre X et y
    """

    # On garantit un ordre des groupes strict
    df = df.sort(patient_col)

    # Pareil on fige l'ordre alphabétique des features

    keep_features = sorted(list(keep_features))
    X_list = []
    y_list = []
 
    for subdf in df.partition_by(patient_col, maintain_order=True):
        if subdf.height != expected_length:
            raise ValueError(f"Le groupe {subdf[patient_col][0]} n'a pas {expected_length} lignes.")
 
        X_list.append(subdf.select(keep_features).to_numpy())
        y_list.append(subdf[target_col][0])  # un seul label par séquence
 
    X_3d = np.stack(X_list).astype(np.float32)
    y_1d = np.array(y_list).astype(np.int64)
 
    return X_3d, y_1d



def downsample_train_patients(
    train_df: pl.DataFrame,
    patient_col: str,
    col_24h: str = "isDeceased_lt_24h",
    col_28d: str = "isDeceased_lt_28d",
    seed: int = 42,
) -> pl.DataFrame:
    # 1) Une ligne par patient avec les labels patient-level
    # On suppose que les colonnes de décès sont constantes sur les 24 lignes du patient
    # ce qui est le cas plus de 95% du temps d'après testDataSplit
    patient_labels = (
        train_df
        .group_by(patient_col)
        .agg([
            pl.last(col_24h).alias(col_24h),
            pl.last(col_28d).alias(col_28d),
        ])
        .sort(patient_col) # tri pour garantir l'ordre
        .with_columns([
            pl.when(pl.col(col_24h) == 1)
              .then(pl.lit("lt_24h"))
              .when((pl.col(col_24h) == 0) & (pl.col(col_28d) == 1))
              .then(pl.lit("lt_28d_only"))
              .otherwise(pl.lit("healthy"))
              .alias("group")
        ])
    )

    # 2) Séparer les IDs patients par groupe
    ids_24h = patient_labels.filter(pl.col("group") == "lt_24h").select(patient_col)
    ids_28d = patient_labels.filter(pl.col("group") == "lt_28d_only").select(patient_col)
    ids_healthy = patient_labels.filter(pl.col("group") == "healthy").select(patient_col)

    # On prend leur taille respective
    n_24h = ids_24h.height
    n_28d = ids_28d.height
    n_healthy = ids_healthy.height

    if n_24h == 0:
        raise ValueError("Aucun patient dans le groupe lt_24h.")

    # On veut : n_28d_sample + n_healthy_sample = n_24h
    if n_28d + n_healthy < n_24h:
        raise ValueError(
            f"Pas assez de patients dans les groupes lt_28d_only + healthy "
            f"pour matcher lt_24h : {n_28d} + {n_healthy} < {n_24h}"
        )

    n_28d_sample = min(n_28d, n_24h // 2)
    n_healthy_sample = n_24h - n_28d_sample

    # Si 28d insuffisant (ce qui sera le cas), on complète avec healthy
    if n_28d_sample > n_28d:
        manque = n_28d_sample - n_28d
        n_28d_sample = n_28d
        n_healthy_sample = n_healthy_sample + manque

    # 3) Sample des IDs patients
    sampled_28d = ids_28d.sample(n=n_28d_sample, with_replacement=False, shuffle=True, seed=seed)
    sampled_healthy = ids_healthy.sample(n=n_healthy_sample, with_replacement=False, shuffle=True, seed=seed)

    # On prend donc : toutes les lignes de isDeceased_lt_24h,et les samples des 2 autres
    selected_ids = pl.concat([ids_24h, sampled_28d, sampled_healthy])

    # 4) Rejoindre avec le train_df complet (24 lignes par patient conservées)
    train_down = (
        train_df
        .join(selected_ids, on=patient_col, how="inner")
        .sort(patient_col)
    )
    return train_down



def equilibrer_dataset_tabulaire(df: pl.DataFrame, id_col: str, target_col: str, method: str, seed: int = 42) -> pl.DataFrame:
    """
    Équilibre un DataFrame Polars. 
    """
    
    # On compte les effectifs avant sampling
    n_sains_avant = df.filter(pl.col(target_col) == 0).height
    n_malades_avant = df.filter(pl.col(target_col) == 1).height

    if method == "":
        return df, None  # On ne fait rien ici (soit pas d'équilibrage, soit géré par méthode custom)
    elif method == "downsampling_homemade":
        df_balanced = downsample_train_patients(df, patient_col = id_col, seed = seed)
        stats = {
            "n_sains_avant": n_sains_avant,
            "n_malades_avant": n_malades_avant,
            "n_sains_apres": df_balanced.filter(pl.col(target_col) == 0).height,
            "n_malades_apres": df_balanced.filter(pl.col(target_col) == 1).height
        }
        return df_balanced, stats

    # Séparation en Pandas pour imblearn
    X_p = df.select(pl.all().exclude(target_col)).to_pandas()
    y_p = df.select(target_col).to_pandas().values.ravel()
    
    if method == "downsampling_50-50":
        from imblearn.under_sampling import RandomUnderSampler
        rs = RandomUnderSampler(random_state=seed)
    elif method == "upsampling_50-50":
        from imblearn.over_sampling import RandomOverSampler
        rs = RandomOverSampler(random_state=seed)
    else:
        raise ValueError(f"Méthode d'équilibrage '{method}' non reconnue.")
        
    X_res, y_res = rs.fit_resample(X_p, y_p)
    
    # Reconstruction immédiate en Polars
    df_balanced = pl.from_pandas(X_res).with_columns(pl.Series(target_col, y_res))

    # Effectifs après sampling
    stats = {
        "n_sains_avant": n_sains_avant,
        "n_malades_avant": n_malades_avant,
        "n_sains_apres": np.sum(y_res == 0),
        "n_malades_apres": np.sum(y_res == 1)
    }
    return df_balanced, stats

def process_tsfel_fold(fold_idx, train_idx, test_idx, X, y, groups, seed, **kwargs):
    """Pipeline de traitement pour l'extraction TSFEL sur un fold."""
    
    # --- Extraction de la configuration ---
    patient_col = kwargs['patient_col']
    target_col = kwargs['target_col']
    train_init_tsfel = kwargs['train_init']
    boruta_filter = kwargs['boruta_filter']
    exp = kwargs['exp']
    balance_method = kwargs['balance_method']
    
    # Récupération des IDs patients correspondants au split de ce fold
    train_patients = X[train_idx].select(patient_col).unique()
    test_patients = X[test_idx].select(patient_col).unique()
    
    # Filtrage du gros DataFrame train pré-calculé
    train_fold_tsfel = train_init_tsfel.join(train_patients, on=patient_col, how="inner").sort(patient_col)
    test_fold_tsfel = train_init_tsfel.join(test_patients, on=patient_col, how="inner").sort(patient_col)
    
    # Filtrage corrélation/variance
    train_clean, test_clean, keepVariableList_1 = extract_feat.filtrage_corr_var(
        train_fold_tsfel, test_fold_tsfel, patient_col, target_col
    )
    
    if boruta_filter:
        filename_train_boruta = exp.get_tsfel_boruta("train", fold_idx)
        filename_test_boruta = exp.get_tsfel_boruta("test", fold_idx)

        if os.path.exists(filename_train_boruta) and os.path.exists(filename_test_boruta):
            print(f"Lecture des fichiers Boruta existants pour le fold {fold_idx} (Graine {seed}).")
            train_clean = pl.read_parquet(filename_train_boruta)
            test_clean = pl.read_parquet(filename_test_boruta)
        else:
            train_clean, test_clean, keepVariableList_2 = extract_feat.filtrage_boruta(
                train_clean, test_clean, patient_col, target_col, max_iter=100, seed=seed
            )
            train_clean.write_parquet(filename_train_boruta)
            test_clean.write_parquet(filename_test_boruta)
            print(f"Save de Boruta pour le fold {fold_idx} (Graine {seed}).")

            parent_folder2 = filename_train_boruta.parent
            np.save(parent_folder2 / f"keepVariableList_1_fold_{fold_idx}.npy", keepVariableList_1)
            np.save(parent_folder2 / f"keepVariableList_2_fold_{fold_idx}.npy", keepVariableList_2)

    # Tri et équilibrage
    train_clean = train_clean.sort(patient_col)
    test_clean = test_clean.sort(patient_col)
    train_clean, stats = equilibrer_dataset_tabulaire(
        train_clean, patient_col, target_col, method=balance_method, seed=seed
    )
    
    # Si on fait de l'upsampling, on duplique des patients dans tous les cas ^^'
    if balance_method !="upsampling_50-50":
        # Tests d'intégrité
        assert train_clean.height == train_clean[patient_col].n_unique(), f"Erreur d'alignement Train TSFEL Fold {fold_idx}"
        assert test_clean.height == test_clean[patient_col].n_unique(), f"Erreur d'alignement Test TSFEL Fold {fold_idx}"

    y_train_fold = train_clean[target_col].to_numpy()
    y_test_fold = test_clean[target_col].to_numpy()
    groups_fold = train_clean[patient_col].to_numpy()
    
    train_clean = train_clean.select(pl.exclude(patient_col, target_col))
    test_clean = test_clean.select(pl.exclude(patient_col, target_col))

    # Scaling final
    X_train_fold, X_test_fold = scaling(train_clean, test_clean)

    return X_train_fold, X_test_fold, y_train_fold, y_test_fold, groups_fold, stats

def process_time_fold(fold_idx, train_idx, test_idx, seed, **kwargs):
    """Pipeline de traitement pour l'extraction temporelle (3D) sur un fold."""
    
    # --- Extraction de la configuration ---
    patient_col = kwargs['patient_col']
    time_col = kwargs['time_col']
    target_col = kwargs['target_col']
    train_init_df = kwargs['train_init']
    final_features = kwargs['final_features']
    balance_method = kwargs['balance_method']
    expected_length = kwargs['expected_length']
    exp = kwargs['exp']

    train_df = train_init_df[train_idx].sort([patient_col, time_col])
    test_df = train_init_df[test_idx].sort([patient_col, time_col])

    # Équilibrage fait maison (Polars)
    if balance_method in ["downsampling_homemade", ""]:
        train_df, _ = equilibrer_dataset_tabulaire(
            train_df, patient_col, target_col, method=balance_method, seed=seed
        )

    # Scaling & Passage en 3D
    train_df, test_df = scaling(train_df, test_df)
    X_train_fold, y_train_fold = build_sequences(train_df, patient_col, target_col, expected_length, final_features)
    X_test_fold, y_test_fold = build_sequences(test_df, patient_col, target_col, expected_length, final_features)

    patients_time_fold = train_df[patient_col].unique().sort().to_numpy()

    # Équilibrage Imblearn (sur tableau 3D aplati en 2D)
    if balance_method not in ["downsampling_homemade", ""]:
        n_sains_avant = int(np.sum(y_train_fold == 0))
        n_malades_avant = int(np.sum(y_train_fold == 1))

        n_samples, n_timesteps, n_feats = X_train_fold.shape
        X_train_fold_2d = X_train_fold.reshape(n_samples, n_timesteps * n_feats)

        if balance_method == "downsampling_50-50":
            from imblearn.under_sampling import RandomUnderSampler
            rs = RandomUnderSampler(random_state=seed)
        elif balance_method == "upsampling_50-50":
            from imblearn.over_sampling import RandomOverSampler
            rs = RandomOverSampler(random_state=seed)
        else:
            raise ValueError(f"L'équilibrage '{balance_method}' n'est pas implémenté")

        indices_arr = np.arange(n_samples).reshape(-1, 1)
        indices_resampled, y_train_fold = rs.fit_resample(indices_arr, y_train_fold)
        indices_resampled = indices_resampled.flatten()
        
        X_train_fold = X_train_fold_2d[indices_resampled].reshape(-1, n_timesteps, n_feats)
        groups_fold = patients_time_fold[indices_resampled]

        stats = {
            "n_sains_avant": n_sains_avant,
            "n_malades_avant": n_malades_avant,
            "n_sains_apres": int(np.sum(y_train_fold == 0)),
            "n_malades_apres": int(np.sum(y_train_fold == 1))
        }
    else:
        stats = None
        groups_fold = patients_time_fold

    # Gestion des NaN
    total_nan = np.isnan(X_train_fold).sum()
    print(f"Nombre total de valeurs NaN : {total_nan}")

    X_train_fold = np.nan_to_num(X_train_fold, nan=0.0)
    X_test_fold = np.nan_to_num(X_test_fold, nan=0.0)
    
    np.save(exp.get_time_path(mode="train", fold_idx=fold_idx), X_train_fold)
    np.save(exp.get_time_path(mode="test", fold_idx=fold_idx), X_test_fold)
    
    return X_train_fold, X_test_fold, y_train_fold, y_test_fold, groups_fold, stats
