import numpy as np
import optuna

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def _to_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


def make_objective_svc_stage1(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        # 1. Choix du noyau
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        
        params = {
            # Paramètre de régularisation : crucial en échelle logarithmique
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "kernel": kernel,
            "probability": fixed_params.get("probability", True), # Souvent requis pour certaines métriques (ex: AUC)
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": fixed_params.get("random_state", 42),
        }

        # 2. Conditionner gamma uniquement pour les noyaux qui l'utilisent (RBF et Poly)
        if kernel in ["rbf", "poly"]:
            # On propose soit la string 'scale'/'auto', soit une valeur numérique continue en log
            gamma_type = trial.suggest_categorical("gamma_type", ["scale", "auto", "value"])
            if gamma_type == "value":
                params["gamma"] = trial.suggest_float("gamma_value", 1e-4, 1e1, log=True)
            else:
                params["gamma"] = gamma_type
        
        # 3. Conditionner le degré uniquement pour le noyau polynomial
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)

        clf = SVC(**params)

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


def run_svc_stage1_search(
    X_train,
    y_train,
    study_name="svc_stage1",
    n_trials=35, # Un peu moins de trials car le SVC peut être très long à s'entraîner
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

    objective = make_objective_svc_stage1(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== SVC STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)

    return study