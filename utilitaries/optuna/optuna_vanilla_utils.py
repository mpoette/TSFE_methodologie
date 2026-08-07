"""Optuna hyperparameter optimization utilities for Vanilla Transformer models.

Provides objective functions and study runners for the first stage of
Optuna-based hyperparameter search targeting Vanilla Transformer architectures
with attention-based sequence modelling and configurable encoder dimensions.
"""

import numpy as np
import optuna

from utilitaries.models.vanillaTransformerModified import (
    train_vanilla_transformer,
)
import utilitaries.optuna.optuna_utils as optuna_utils
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE


def make_objective_stage1(
    X_train_3d,
    y_train_seq,
    fixed_params=None,
):
    """Create the first-stage Optuna objective for Vanilla Transformer (TST).

    For short time series (sequence length < 50), the search space is
    automatically constrained to prevent overly large embedding dimensions
    and model depth that lead to overfitting.

    Args:
        X_train_3d:
            Three-dimensional training feature array.
        y_train_seq:
            Training target sequence.
        fixed_params:
            Optional dictionary containing fixed training parameters.

    Returns:
        A callable Optuna objective returning the best validation loss.
    """
    fixed_params = fixed_params or {}

    # Detect short sequences and adapt search space accordingly.
    seq_length = X_train_3d.shape[1]
    is_short_sequence = seq_length < 50

    def objective(trial):
        """Evaluate one Vanilla Transformer hyperparameter trial."""
        # For short sequences, use smaller architectures to avoid overfitting.
        if is_short_sequence:
            d_model_exp = trial.suggest_int("d_model_exp", 4, 6)  # 2^4 = 16 to 2^6 = 64
            dim_feedforward_exp = trial.suggest_int("dim_feedforward_exp", 5, 7)  # 32 to 128
            batch_size_exp = trial.suggest_int("batch_size_exp", 4, 7)  # 16 to 128
            num_layers = trial.suggest_int("num_layers", 1, 3)
            # Constrain heads for smaller d_model
            nhead_options = [2, 4]
        else:
            d_model_exp = trial.suggest_int("d_model_exp", 5, 7)  # 32 to 128
            dim_feedforward_exp = trial.suggest_int("dim_feedforward_exp", 6, 8)  # 64 to 256
            batch_size_exp = trial.suggest_int("batch_size_exp", 4, 7)
            num_layers = trial.suggest_int("num_layers", 2, 6)
            nhead_options = [2, 4, 8]

        # Convert exponents to powers of two.
        real_d_model = 2 ** d_model_exp
        real_dim_feedforward = 2 ** dim_feedforward_exp
        real_batch_size = 2 ** batch_size_exp

        # Filter valid nhead values (nhead must divide d_model)
        valid_nheads = [h for h in nhead_options if real_d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", valid_nheads)

        # Store actual values in Optuna for final analysis.
        trial.set_user_attr("actual_d_model", real_d_model)
        trial.set_user_attr("actual_dim_feedforward", real_dim_feedforward)
        trial.set_user_attr("actual_batch_size", real_batch_size)
        trial.set_user_attr("sequence_length", seq_length)
        trial.set_user_attr("is_short_sequence", is_short_sequence)

        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,

            # Pass model hyperparameters explicitly.
            "d_model": real_d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": real_dim_feedforward,
            "dropout": trial.suggest_float("dropout", 0.05, 0.3, step=0.05),
            "batch_size": real_batch_size,
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "clip_grad": trial.suggest_float("clip_grad", 0.5, 2.0, step=0.5),
            "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        }

        try:
            _, _, history, _ = train_vanilla_transformer(
                X_train_3d,
                y_train_seq,
                **params,
            )

            score = optuna_utils.extract_best_val_loss(history, "val_loss")

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


def run_stage1_search(
    X_train_3d,
    y_train_seq,
    study_name,
    n_trials=40,
    storage=OPTUNA_STORAGE,
    fixed_params=None,
):
    """Run the first-stage Optuna search for a Vanilla Transformer model.

    Args:
        X_train_3d:
            Three-dimensional training feature array.
        y_train_seq:
            Training target sequence.
        study_name:
            Name of the Optuna study.
        n_trials:
            Number of optimization trials.
        storage:
            Optuna storage URL.
        fixed_params:
            Optional dictionary containing fixed training parameters.

    Returns:
        The optimized Optuna study.
    """
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
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== STAGE 1 ===")
    print("Best value :", study.best_value)
    print("Best params (exponents):", study.best_params)
    print("Best real values       :", study.best_trial.user_attrs)

    return study