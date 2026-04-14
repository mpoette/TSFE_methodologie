import numpy as np
import polars as pl
import optuna
from sklearn.preprocessing import StandardScaler
 
from inceptionTimeModified import (
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