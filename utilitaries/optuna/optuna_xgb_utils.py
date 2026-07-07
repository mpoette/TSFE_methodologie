import numpy as np
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def _to_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


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
            
            # Profondeur : JAMAIS de None ou 0 en boosting. On cherche généralement bas (3 à 10)
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            
            # Taux d'apprentissage (learning rate / eta) - Crucial en log=True
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            
            # Sous-échantillonnage pour éviter l'overfitting (équivalent du bootstrap)
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
            
            # Régularisation (L1 et L2)
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            
            # Gestion du déséquilibre des classes (remplace class_weight)
            # scale_pos_weight est utile en binaire. Si tu es en multiclasse, on gère autrement.
            "random_state": fixed_params.get("random_state", 42),
            "n_jobs": fixed_params.get("n_jobs", -1),
            "eval_metric": "logloss", # Évite les warnings XGBoost
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

    objective = make_objective_xgb_stage1(
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