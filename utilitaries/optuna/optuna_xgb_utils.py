import numpy as np
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def _to_numpy(X):
    """Convert an array-like object to a NumPy array.

    Args:
        X:
            Input data exposing an optional ``to_numpy`` method.

    Returns:
        A NumPy representation of the input data.
    """
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)

def make_objective_xgb_stage1_anti_overfit(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Create a strongly regularized XGBoost Optuna objective.

    Args:
        X_train:
            Training feature matrix.
        y_train:
            Training target labels.
        metric_name:
            Scikit-learn scoring metric used during cross-validation.
        fixed_params:
            Optional dictionary containing fixed XGBoost and
            cross-validation parameters.

    Returns:
        A callable Optuna objective returning the mean cross-validation score.
    """
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        # Define a hyperparameter space designed to reduce overfitting.
        """Evaluate one XGBoost hyperparameter trial."""
        params = {
            # 1. Limit the number of trees and use a very small learning rate.
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            
            # Force slow learning (maximum 0.05 instead of 0.3).
            # This prevents the learning rate from dominating the Optuna study.
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            
            # 2. Strongly constrain the tree structure.
            # Use shallow weak learners with depths between 2 and 5.
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            
            # Require sufficient support for each leaf.
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 80),
            
            # Require a minimum gain before splitting a node.
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            
            # 3. Use aggressive subsampling to reduce memorization.
            # Each tree sees only a fraction of rows and TSFEL features.
            "subsample": trial.suggest_float("subsample", 0.4, 0.7, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7, step=0.1),
            
            # 4. Apply L1 and L2 regularization to leaf weights.
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 20.0, log=True),
            
            # Required fixed parameters.
            "random_state": fixed_params.get("random_state", 42),
            "n_jobs": fixed_params.get("n_jobs", 1),
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
                n_jobs=fixed_params.get("cv_n_jobs", 5),
                pre_dispatch=fixed_params.get("pre_dispatch", "n_jobs"),
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
    n_trials=50, # More trials because the XGBoost search space is larger.
    storage=OPTUNA_STORAGE,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Run the first-stage Optuna search for an XGBoost classifier.

    Args:
        X_train:
            Training feature matrix.
        y_train:
            Training target labels.
        study_name:
            Name of the Optuna study.
        n_trials:
            Number of optimization trials.
        storage:
            Optuna storage URL.
        metric_name:
            Scikit-learn scoring metric used during cross-validation.
        fixed_params:
            Optional dictionary containing fixed XGBoost and
            cross-validation parameters.

    Returns:
        The optimized Optuna study.
    """
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
