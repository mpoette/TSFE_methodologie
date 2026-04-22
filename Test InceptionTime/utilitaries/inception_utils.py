import numpy as np
import polars as pl
import optuna
from sklearn.preprocessing import StandardScaler
 
from utilitaries.models.inceptionTimeModified import (
    train_inception_time,
)

# nommage des variables et fixation des paramètres
patient_col="encounterId"
time_col="heure_calibree"
expected_length=24



# PREPROCESSING

def scaling(df_train, df_test):
    """
    Fonction qui permet de scaler en utilisant StandardScaler, uniquement les entiers et pas les booléens
    """
    # StandardScaler ne supporte pas directement Polars mais uniquement pandas donc transfert obligatoire...
    df_train_pd = df_train.to_pandas()
    df_test_pd = df_test.to_pandas()
    
    # colonnes numériques sauf bool
    num_cols = df_train_pd.select_dtypes(include=["number"]).columns
    bool_cols = df_train_pd.select_dtypes(include=["bool"]).columns
    
    num_cols = [c for c in num_cols if c not in bool_cols and c not in [patient_col]]
    
    scaler = StandardScaler()
    
    df_train_pd[num_cols] = scaler.fit_transform(df_train_pd[num_cols])
    df_test_pd[num_cols] = scaler.transform(df_test_pd[num_cols])
    return df_train_pd, df_test_pd


def scaling(df_train: pl.DataFrame, df_test: pl.DataFrame):
    """
    Scales numerical columns using StandardScaler, excluding booleans 
    and columns containing only 0 and 1.
    """

    # 1. Identifier les colonnes numériques (float/int)
    # On exclut d'office la colonne patient et les booléens natifs
    all_num_cols = df_train.select(
        pl.col(pl.NUMERIC_DTYPES).exclude(patient_col)
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

    # Petit contrôle
    check = (
        train_down
        .group_by(patient_col)
        .agg([
            pl.first(col_24h).alias(col_24h),
            pl.first(col_28d).alias(col_28d),
        ])
        .with_columns([
            pl.when(pl.col(col_24h) == 1)
              .then(pl.lit("lt_24h"))
              .when((pl.col(col_24h) == 0) & (pl.col(col_28d) == 1))
              .then(pl.lit("lt_28d_only"))
              .otherwise(pl.lit("healthy"))
              .alias("group")
        ])
        .group_by("group")
        .len()
        .sort("group")
    )

    print(check)

    return train_down

######

# Recherche des meilleurs hyperparamètres

def extract_best_val_loss(history):
    """
    Fonction qui extrait la meilleure valeur de loss sur validation (val_loss)
    """
    if history is None:
        raise ValueError("history est None.")
    if "val_loss" not in history:
        raise ValueError("La clé 'val_loss' est absente de history.")
 
    values = [v for v in history["val_loss"] if v is not None and np.isfinite(v)]
    if not values:
        raise ValueError("Aucune val_loss valide trouvée.")
 
    return float(min(values))


# Au début, totalement au hasard car on ne sait rien
def make_objective_stage1(
    X_train_3d,
    y_train_seq,
    metric_name="val_loss",
    fixed_params=None,
):
    fixed_params = fixed_params or {}
 
    def objective(trial):
        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,
 
            # Hyperparams à explorer largement
            "num_blocks": trial.suggest_int("num_blocks", 3, 8),
            "out_channels": trial.suggest_categorical("out_channels", [16, 32, 64, 128]),
            "bottleneck_channels": trial.suggest_categorical("bottleneck_channels", [8, 16, 32, 64]),
            "kernel_sizes": trial.suggest_categorical("kernel_sizes", [15, 21, 31, 41, 51, 61]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0]),
            "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        }
 
        try:
            model, T, history, splits = train_inception_time(
                X_train_3d,
                y_train_seq,
                **params,
            )
 
            score = extract_best_val_loss(history)
 
            if not np.isfinite(score):
                raise FloatingPointError("Score non fini.")
 
            return score
 
        except FloatingPointError:
            raise
        except Exception as e:
            # si un essai plante, on le marque comme mauvais essai et on le coupe
            raise optuna.TrialPruned(f"Trial échoué: {e}")
 
    return objective


def run_stage1_search(
    X_train_3d,
    y_train_seq,
    study_name,
    n_trials=40,
    storage=None,
    metric_name="val_loss",
    fixed_params=None,
):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5)
 
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
 
    objective = make_objective_stage1(
        X_train_3d=X_train_3d,
        y_train_seq=y_train_seq,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )
 
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
 
    print("=== STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)
 
    return study

# Maintenant, on peut faire une heuristique : prendre les voisins autour des 
# meilleurs hyperparamètres pour améliorer légèrement le score

def _neighbors_from_choices(best_value, choices):
    """
    Pour un hyperparam catégoriel ordinal, on prend le voisin du dessous,
    lui-même, et le voisin du dessus.
    """
    choices = list(choices)
    idx = choices.index(best_value)
    low_idx = max(0, idx - 1)
    high_idx = min(len(choices) - 1, idx + 1)
    return choices[low_idx:high_idx + 1]


def make_objective_stage2(
    X_train_3d,
    y_train_seq,
    best_stage1_params,
    metric_name="val_loss",
    fixed_params=None,
):
    fixed_params = fixed_params or {}
 
    # espaces globaux de référence
    out_choices = [16, 32, 64, 128]
    bottleneck_choices = [8, 16, 32, 64]
    kernel_choices = [15, 21, 31, 41, 51, 61]
    batch_choices = [16, 32, 64, 128]
    clip_choices = [0.5, 1.0, 2.0]
 
    # bornes fines autour du meilleur stage1
    best_lr = best_stage1_params["lr"]
    lr_low = max(1e-5, best_lr / 3.0)
    lr_high = min(1e-2, best_lr * 3.0)
 
    best_wd = best_stage1_params["weight_decay"]
    wd_low = max(1e-7, best_wd / 10.0)
    wd_high = min(1e-1, best_wd * 10.0)
 
    num_blocks_best = best_stage1_params["num_blocks"]
    num_blocks_low = max(2, num_blocks_best - 1)
    num_blocks_high = min(10, num_blocks_best + 1)
 
    out_candidates = _neighbors_from_choices(best_stage1_params["out_channels"], out_choices)
    bottleneck_candidates = _neighbors_from_choices(best_stage1_params["bottleneck_channels"], bottleneck_choices)
    kernel_candidates = _neighbors_from_choices(best_stage1_params["kernel_sizes"], kernel_choices)
    batch_candidates = _neighbors_from_choices(best_stage1_params["batch_size"], batch_choices)
    clip_candidates = _neighbors_from_choices(best_stage1_params["clip_grad"], clip_choices)
 
    scheduler_best = best_stage1_params["use_scheduler"]
 
    def objective(trial):
        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,
 
            # recherche fine
            "num_blocks": trial.suggest_int("num_blocks", num_blocks_low, num_blocks_high),
            "out_channels": trial.suggest_categorical("out_channels", out_candidates),
            "bottleneck_channels": trial.suggest_categorical("bottleneck_channels", bottleneck_candidates),
            "kernel_sizes": trial.suggest_categorical("kernel_sizes", kernel_candidates),
            "batch_size": trial.suggest_categorical("batch_size", batch_candidates),
            "lr": trial.suggest_float("lr", lr_low, lr_high, log=True),
            "weight_decay": trial.suggest_float("weight_decay", wd_low, wd_high, log=True),
            "clip_grad": trial.suggest_categorical("clip_grad", clip_candidates),
            "use_scheduler": trial.suggest_categorical("use_scheduler", [scheduler_best]),
        }
 
        try:
            model, T, history, splits = train_inception_time(
                X_train_3d,
                y_train_seq,
                **params,
            )
 
            score = extract_best_val_loss(history)
 
            if not np.isfinite(score):
                raise FloatingPointError("Score non fini.")
 
            return score
 
        except FloatingPointError:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")
 
    return objective
 
 
def run_stage2_search(
    X_train_3d,
    y_train_seq,
    study_stage1,
    n_trials=25,
    study_name="inception_stage2",
    storage=None,
    metric_name="val_loss",
    fixed_params=None,
):
    best_stage1_params = study_stage1.best_params
 
    sampler = optuna.samplers.TPESampler(seed=43)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
 
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
 
    objective = make_objective_stage2(
        X_train_3d=X_train_3d,
        y_train_seq=y_train_seq,
        best_stage1_params=best_stage1_params,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )
 
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
 
    print("=== STAGE 2 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)
 
    return study


#############