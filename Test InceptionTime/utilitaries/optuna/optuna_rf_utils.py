import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer


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
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 16, 24, 32]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
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
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective


def run_rf_stage1_search(
    X_train,
    y_train,
    study_name="rf_stage1",
    n_trials=40,
    storage=None,
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

    objective = make_objective_rf_stage1(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== RANDOM FOREST STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)

    return study





def _int_window(best, low_abs, high_abs, delta):
    return max(low_abs, best - delta), min(high_abs, best + delta)


def make_objective_rf_stage2(
    X_train,
    y_train,
    best_stage1_params,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    best_n_estimators = best_stage1_params["n_estimators"]
    n_est_low, n_est_high = _int_window(best_n_estimators, 50, 1500, 200)

    best_min_split = best_stage1_params["min_samples_split"]
    split_low, split_high = _int_window(best_min_split, 2, 30, 5)

    best_min_leaf = best_stage1_params["min_samples_leaf"]
    leaf_low, leaf_high = _int_window(best_min_leaf, 1, 30, 5)

    max_depth_best = best_stage1_params["max_depth"]

    if max_depth_best is None:
        depth_candidates = [None, 16, 24, 32, 48]
    else:
        depth_candidates = sorted(set([
            max(2, max_depth_best - 4),
            max_depth_best,
            max_depth_best + 4,
            max_depth_best + 8,
        ]))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                n_est_low,
                n_est_high,
                step=50,
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                depth_candidates,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                split_low,
                split_high,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                leaf_low,
                leaf_high,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                [best_stage1_params["max_features"]],
            ),
            "bootstrap": trial.suggest_categorical(
                "bootstrap",
                [best_stage1_params["bootstrap"]],
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [best_stage1_params["class_weight"]],
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
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective


def run_rf_stage2_search(
    X_train,
    y_train,
    study_stage1,
    n_trials=25,
    study_name="rf_stage2",
    storage=None,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    best_stage1_params = study_stage1.best_params

    sampler = optuna.samplers.TPESampler(seed=43)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective_rf_stage2(
        X_train=X_train,
        y_train=y_train,
        best_stage1_params=best_stage1_params,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== RANDOM FOREST STAGE 2 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)

    return study