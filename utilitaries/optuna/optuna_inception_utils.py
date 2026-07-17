import numpy as np
import optuna
 
from utilitaries.models.inceptionTimeModified import (
    train_inception_time,
)
import utilitaries.optuna.optuna_utils as optuna_utils
from utilitaries.optuna.optuna_utils import OPTUNA_STORAGE

def make_objective_stage1(
    X_train_3d,
    y_train_seq,
    fixed_params=None,
):
    """Create the first-stage Optuna objective for InceptionTime.

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
 
    def objective(trial):
        # 1. Suggest exponents with distinct names to avoid conflicts.
        """Evaluate one InceptionTime hyperparameter trial."""
        out_channels_exp = trial.suggest_int("out_channels_exp", 4, 7)
        bottleneck_exp = trial.suggest_int("bottleneck_channels_exp", 3, 6)
        batch_size_exp = trial.suggest_int("batch_size_exp", 4, 7)
        
        # 2. Convert the exponents to powers of two.
        real_out_channels = 2 ** out_channels_exp
        real_bottleneck = 2 ** bottleneck_exp
        real_batch_size = 2 ** batch_size_exp
        
        # 3. Store the actual values in Optuna for final analysis.
        trial.set_user_attr("actual_out_channels", real_out_channels)
        trial.set_user_attr("actual_bottleneck_channels", real_bottleneck)
        trial.set_user_attr("actual_batch_size", real_batch_size)

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
            "num_blocks": trial.suggest_int("num_blocks", 3, 8),
            "out_channels": real_out_channels,
            "bottleneck_channels": real_bottleneck,    
            "kernel_sizes": trial.suggest_int("kernel_sizes", 15, 61, step=2),
            "batch_size": real_batch_size,
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "clip_grad": trial.suggest_float("clip_grad", 0.5, 2.0, step=0.5),
            "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        }
 
        try:
            _, _, history, _ = train_inception_time(
                X_train_3d,
                y_train_seq,
                **params,
            )
 
            score = optuna_utils.extract_best_val_loss(history, "val_loss")
 
            if not np.isfinite(score):
                raise FloatingPointError("Score non fini.")
 
            return score
 
        except FloatingPointError:
            raise optuna.TrialPruned("FloatingPointError: Perte infinie.")
        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")
 
    return objective


def run_stage1_search(
    X_train_3d,
    y_train_seq,
    study_name,
    n_trials=40,
    storage=OPTUNA_STORAGE,
    fixed_params=None,
):
    """Run the first-stage Optuna search for an InceptionTime model.

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
