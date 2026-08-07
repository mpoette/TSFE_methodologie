"""Optuna hyperparameter search utilities for SVC models.

This module provides an Optuna objective function and a high-level search
routine for tuning Support Vector Classifier hyperparameters using
cross-validation.
"""

import warnings

import numpy as np
import optuna

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

# Suppress SVC convergence warnings during Optuna search (many trials use
# suboptimal C values that fail to converge within max_iter).
warnings.filterwarnings(
    action="ignore",
    category=ConvergenceWarning,
    module="sklearn.svm._base",
)


def _to_numpy(X):
    """Convert an array-like or DataFrame to a NumPy array.

    Args:
        X:
            Input data, either a NumPy array, a Polars/Pandas DataFrame,
            or any array-like object.

    Returns:
        A two-dimensional NumPy array.
    """
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


def make_objective_svc_stage1(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Create an Optuna objective function for SVC hyperparameter tuning.

    The objective evaluates a candidate set of SVC hyperparameters using
    stratified k-fold cross-validation and returns the mean score.

    Args:
        X_train:
            Training feature matrix.
        y_train:
            Training labels.
        metric_name:
            Scikit-learn scoring metric used for evaluation.
            Defaults to ``"balanced_accuracy"``.
        fixed_params:
            Dictionary of fixed hyperparameters passed to the SVC
            constructor and cross-validation. Defaults to ``{}``.

    Returns:
        A callable accepting an Optuna ``Trial`` and returning the mean
        cross-validation score.
    """
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        """Evaluate one set of SVC hyperparameters.

        Args:
            trial:
                Current Optuna trial.

        Returns:
            The mean cross-validation score.

        Raises:
            optuna.TrialPruned:
                If the score is non-finite or an error occurs during
                cross-validation.
        """
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])

        params = {
            "C": trial.suggest_float("C", 0.1, 100, log=True),
            "kernel": kernel,
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": fixed_params.get("random_state", 42),
            "cache_size": fixed_params.get("cache_size", 4000),
            "max_iter": fixed_params.get("max_iter", 50000),
        }

        if kernel in ["rbf", "poly"]:
            params["gamma"] = "scale"

        if kernel == "poly":
            params["degree"] = 2

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
                n_jobs=fixed_params.get("cv_n_jobs", 5),
                pre_dispatch=fixed_params.get("pre_dispatch", "n_jobs"),
            )

            score = np.mean(scores)

            if not np.isfinite(score):
                raise FloatingPointError("Non-finite score encountered.")

            return score

        except FloatingPointError:
            raise optuna.TrialPruned("FloatingPointError detected.")
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {e}")

    return objective


def run_svc_stage1_search(
    X_train,
    y_train,
    study_name="svc_stage1",
    n_trials=20,
    storage=OPTUNA_STORAGE,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Run an Optuna hyperparameter search for an SVC model.

    The search uses a TPE sampler with sequential execution to ensure
    proper Bayesian updates across trials.

    Args:
        X_train:
            Training feature matrix.
        y_train:
            Training labels.
        study_name:
            Name of the Optuna study. Defaults to ``"svc_stage1"``.
        n_trials:
            Number of optimization trials. Defaults to ``20``.
        storage:
            Optuna storage URL. Defaults to an RDB backend.
        metric_name:
            Scoring metric passed to the objective function.
        fixed_params:
            Dictionary of fixed parameters forwarded to both the SVC
            constructor and the cross-validation routine.

    Returns:
        The completed Optuna ``Study`` containing the best hyperparameters
        and trial history.
    """
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective_svc_stage1(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    # Sequential execution ensures proper TPE sampler updates across trials.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    print("=== SVC STAGE 1 ===")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study