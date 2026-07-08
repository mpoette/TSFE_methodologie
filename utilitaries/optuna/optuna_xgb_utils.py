import numpy as np
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def _to_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)

def make_objective_xgb_stage1_anti_overfit(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        # Configuration des hyperparamètres blindée contre le surapprentissage
        params = {
            # 1. On plafonne les arbres mais on ralentit drastiquement le pas
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            
            # CRUCIAL : On force un apprentissage lent (max 0.05 au lieu de 0.3)
            # Ça évite que le learning_rate cannibalise toute l'étude Optuna
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            
            # 2. On restreint sévèrement la structure des arbres
            # Des "weak learners" purs (profondeur 2 à 5 max)
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            
            # On force le modèle à avoir une assise solide par feuille (anti-longue traîne)
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 80),
            
            # Gain minimal requis pour couper un nœud (pénalité sur la complexité)
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            
            # 3. Sous-échantillonnage drastique pour perturber la mémorisation
            # Chaque arbre ne voit qu'une fraction des lignes et des colonnes TSFEL
            "subsample": trial.suggest_float("subsample", 0.4, 0.7, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7, step=0.1),
            
            # 4. Régularisation L1 (Lasso) et L2 (Ridge) sur les poids des feuilles
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 20.0, log=True),
            
            # Paramètres fixes indispensables
            "random_state": fixed_params.get("random_state", 42),
            "n_jobs": fixed_params.get("n_jobs", -1),
            "eval_metric": "logloss",
        }

        clf = XGBClassifier(**params)

        cv = StratifiedKFold(
            n_splits=fixed_params.get("n_splits", 5),
            shuffle=True,
            random_state=fixed_params.get("random_state", 42),
        )

        try:
            scores = cross_val_score(
                clf,
                X_train,
                y_train,
                cv=cv,
                scoring=metric_name,
                n_jobs=fixed_params.get("cv_n_jobs", 1),
            )

            score = np.mean(scores)

            if not np.isfinite(score):
                raise FloatingPointError("Score non fini.")

            return score

        except FloatingPointError:
            raise optuna.TrialPruned("FloatingPointError détecté.")
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective

def make_objective_xgb_stage1(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        # Configuration des hyperparamètres spécifiques à XGBoost
        params = {
            # Nombre d'arbres
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            
            # On baisse le plafond de profondeur (8 au lieu de 10)
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            
            # Empêche de diviser le nœud pour des broutilles
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            
            # Gain minimal requis pour faire un split
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
            
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            
            "random_state": fixed_params.get("random_state", 42),
            "n_jobs": fixed_params.get("n_jobs", -1),
            "eval_metric": "logloss",
        }

        clf = XGBClassifier(**params)

        cv = StratifiedKFold(
            n_splits=fixed_params.get("n_splits", 5),
            shuffle=True,
            random_state=fixed_params.get("random_state", 42),
        )

        try:
            scores = cross_val_score(
                clf,
                X_train,
                y_train,
                cv=cv,
                scoring=metric_name,
                n_jobs=fixed_params.get("cv_n_jobs", 1),
            )

            score = np.mean(scores)

            if not np.isfinite(score):
                raise FloatingPointError("Score non fini.")

            return score

        except FloatingPointError:
            raise optuna.TrialPruned("FloatingPointError détecté.")
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective


def run_xgb_stage1_search(
    X_train,
    y_train,
    study_name="xgb_stage1",
    n_trials=50, # Un peu plus de trials car l'espace XGBoost est plus grand
    storage=OPTUNA_STORAGE,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective_xgb_stage1_anti_overfit(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== XGBOOST STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)

    return study