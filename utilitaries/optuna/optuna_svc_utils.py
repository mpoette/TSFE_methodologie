import numpy as np
import optuna

from sklearn.svm import SVC
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


def make_objective_svc_stage1(
    X_train,
    y_train,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Create the Optuna objective used for the first SVC search stage.

    The objective evaluates each hyperparameter configuration with stratified
    cross-validation and returns the mean validation score.

    Args:
        X_train:
            Training feature matrix.
        y_train:
            Training target labels.
        metric_name:
            Scikit-learn scoring metric used during cross-validation.
        fixed_params:
            Optional dictionary containing fixed SVC and cross-validation
            parameters.

    Returns:
        A callable Optuna objective accepting a trial and returning its mean
        cross-validation score.
    """
    fixed_params = fixed_params or {}

    X_train = _to_numpy(X_train)
    y_train = np.asarray(y_train)

    def objective(trial):
        """Evaluate one SVC hyperparameter trial."""
        # 1. Select the kernel (kept poly for completeness, but constrained for speed).
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])

        params = {
            "C": trial.suggest_float("C", 0.1, 100, log=True),
            "kernel": kernel,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": fixed_params.get("random_state", 42),
        }

        # 2. Gamma fixed to 'scale' for baseline speed (RBF and poly kernels only).
        if kernel in ["rbf", "poly"]:
            params["gamma"] = "scale"

        # 3. Degree fixed to 2 for polynomial kernel (baseline constraint).
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
    n_trials=20, # Reduced for baseline speed.
    storage=OPTUNA_STORAGE,
    metric_name="balanced_accuracy",
    fixed_params=None,
):
    """Run the first-stage Optuna hyperparameter search for an SVC model.

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
            Optional dictionary containing fixed SVC and cross-validation
            parameters.

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
