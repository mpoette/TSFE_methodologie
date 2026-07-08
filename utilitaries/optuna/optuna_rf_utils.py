import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def _to_numpy(X):
    """
    Accepte Polars / Pandas / numpy.
    """
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


def make_objective_rf_stage1(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        # 1. Remplacement du categorical par un int ordonné (0 correspond à None)
        real_max_depth = trial.suggest_int("max_depth", 3, 12)

        # 2. Enregistrement de la vraie valeur pour les analyses
        trial.set_user_attr("actual_max_depth", real_max_depth)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": real_max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                ["balanced", "balanced_subsample", None],
            ),
            "random_state": fixed_params.get("random_state", 42),
            "n_jobs": fixed_params.get("n_jobs", -1),
        }

        clf = RandomForestClassifier(**params)

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


def run_rf_stage1_search(
    X_train,
    y_train,
    study_name="rf_stage1",
    n_trials=40,
    storage=OPTUNA_STORAGE,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    # useless sauf pour erreurs
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

    objective = make_objective_rf_stage1(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== RANDOM FOREST STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params (internal):", study.best_params)
    print("Best real values      :", study.best_trial.user_attrs)

    return study