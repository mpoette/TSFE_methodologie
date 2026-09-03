"""Optuna hyperparameter optimization utilities for LSTM models.

Provides objective functions and study runners for the first stage of
Optuna-based hyperparameter search targeting LSTM (Long Short-Term Memory)
architectures. Includes categorical search spaces for fully-connected units
and handles gradient clipping suggestions.
"""

import numpy as np
import optuna

import utilitaries.optuna.optuna_utils as optuna_utils
from utilitaries.models.lstmTimeModified import train_lstm_model
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

FC_UNITS_MAP = {
    "none": None,
    "64": (64,),
    "128": (128,),
    "256": (256,),
    "128_64": (128, 64),
    "256_128": (256, 128),
}

FC_UNITS_KEYS = ["none", "64", "128", "256", "128_64", "256_128"]


# ============================================================================
# LSTM OPTUNA SEARCH
# ============================================================================

def make_objective_lstm_stage1(
    X_train,
    y_train,
    metric_name="val_auc",
    fixed_params=None,
):
    """Create the first-stage Optuna objective for an LSTM model.

    Args:
        X_train:
            Training feature array.
        y_train:
            Training target labels.
        metric_name:
            Validation metric optimized by Optuna.
        fixed_params:
            Optional dictionary containing fixed training parameters.

    Returns:
        A callable Optuna objective returning the selected validation metric.
    """
    fixed_params = fixed_params or {}

    def objective(trial):
        """Evaluate one LSTM hyperparameter trial."""
        fc_units_key = trial.suggest_categorical("fc_units", FC_UNITS_KEYS)

        # 1. Suggest exponents used to generate powers of two.
        hidden_size_exp = trial.suggest_int("hidden_size_exp", 5, 8)  
        batch_size_exp = trial.suggest_int("batch_size_exp", 4, 7) 
        
        # 2. Convert the exponents to their actual numeric values.
        real_hidden_size = 2 ** hidden_size_exp
        real_batch_size = 2 ** batch_size_exp
        
        # Represent disabled gradient clipping with 0.0 in the search space.
        clip_grad_val = trial.suggest_float("clip_grad", 0.0, 2.0, step=0.5)
        real_clip_grad = None if clip_grad_val == 0.0 else clip_grad_val

        # 3. Store actual values as user attributes for final analysis.
        trial.set_user_attr("actual_hidden_size", real_hidden_size)
        trial.set_user_attr("actual_batch_size", real_batch_size)
        trial.set_user_attr("actual_clip_grad", real_clip_grad)

        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,

            "hidden_size": real_hidden_size,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "fc_units": FC_UNITS_MAP[fc_units_key],
            "layernorm": trial.suggest_categorical("layernorm", [True, False]),
            "batch_size": real_batch_size,
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "clip_grad": real_clip_grad,
            "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        }

        # PyTorch LSTMs ignore dropout when only one recurrent layer is used.
        if params["num_layers"] == 1:
            params["dropout"] = 0.0

        try:
            model, T, history, splits = train_lstm_model(
                X_train,
                y_train,
                **params,
            )

            score = optuna_utils.extract_best_val_loss(history, metric_name)

            if not np.isfinite(score):
                raise FloatingPointError("Non-finite score.")

            return score

        except FloatingPointError:
            raise optuna.TrialPruned("FloatingPointError: Infinite loss.")
        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {e}")

    return objective


def run_lstm_stage1_search(
    X_train,
    y_train,
    study_name,
    n_trials=40,
    storage=OPTUNA_STORAGE,
    metric_name="val_auc",
    fixed_params=None,
):
    """Run the first-stage Optuna search for an LSTM model.

    Args:
        X_train:
            Training feature array.
        y_train:
            Training target labels.
        study_name:
            Name of the Optuna study.
        n_trials:
            Number of optimization trials.
        storage:
            Optuna storage URL.
        metric_name:
            Validation metric optimized by Optuna.
        fixed_params:
            Optional dictionary containing fixed training parameters.

    Returns:
        The optimized Optuna study.
    """
    direction = "maximize" if metric_name == "val_auc" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective_lstm_stage1(
        X_train=X_train,
        y_train=y_train,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== LSTM STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params (exponents):", study.best_params)
    print("Best real values       :", study.best_trial.user_attrs)

    return study
