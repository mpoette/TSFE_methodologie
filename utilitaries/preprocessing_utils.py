import utilitaries.extract_data_utils as edu 
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    # 3) Sample des IDs patients (intégré à polars ^^)
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
    ATTENTION : Les méthodes imblearn "50-50" ne sont viables QUE sur du pur tabulaire (1 ligne = 1 patient).
    Si le DataFrame contient des fenêtres temporelles (ex: 24h par patient), utilisez exclusivement "downsampling_homemade".
    """
    if method == "":
        return df  # On ne fait rien ici (soit pas d'équilibrage, soit géré par ta méthode custom)
    elif method == "downsampling_homemade":
        return downsample_train_patients(df, patient_col = id_col, seed = seed)

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
    df_balanced = pl.from_pandas(X_res)
    return df_balanced.with_columns(pl.Series(target_col, y_res))