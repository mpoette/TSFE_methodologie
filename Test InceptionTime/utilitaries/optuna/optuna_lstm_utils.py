import numpy as np
import optuna

import utilitaries.optuna.optuna_utils as optuna_utils
from utilitaries.models.lstmTimeModified import train_lstm_model


FC_UNITS_MAP = {
    "none": None,
    "64": (64,),
    "128": (128,),
    "256": (256,),
    "128_64": (128, 64),
    "256_128": (256, 128),
}

FC_UNITS_KEYS = ["none", "64", "128", "256", "128_64", "256_128"]


def make_objective_lstm_stage1(
    X_train,
    y_train,
    metric_name="val_loss",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    def objective(trial):
        fc_units_key = trial.suggest_categorical(
            "fc_units",
            FC_UNITS_KEYS,
        )

        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,

            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "bidirectional": trial.suggest_categorical(
                "bidirectional", [True, False]
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "fc_units": FC_UNITS_MAP[fc_units_key],
            "layernorm": trial.suggest_categorical(
                "layernorm", [True, False]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [16, 32, 64, 128]
            ),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-2, log=True
            ),
            "clip_grad": trial.suggest_categorical(
                "clip_grad", [0.5, 1.0, 2.0, None]
            ),
            "use_scheduler": trial.suggest_categorical(
                "use_scheduler", [True, False]
            ),
        }

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
                raise FloatingPointError("Score non fini.")

            return score

        except FloatingPointError:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective


def run_lstm_stage1_search(
    X_train,
    y_train,
    study_name,
    n_trials=40,
    storage=None,
    metric_name="val_loss",
    fixed_params=None,
):
    if metric_name == "val_auc":
        direction = "maximize"
    else:
        direction = "minimize"
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
    print("Best params:", study.best_params)

    return study


def make_objective_lstm_stage2(
    X_train,
    y_train,
    best_stage1_params,
    metric_name="val_loss",
    fixed_params=None,
):
    fixed_params = fixed_params or {}

    hidden_choices = [32, 64, 128, 256]
    batch_choices = [16, 32, 64, 128]
    clip_choices = [0.5, 1.0, 2.0, None]

    hidden_candidates = optuna_utils.neighbors_from_choices(
        best_stage1_params["hidden_size"],
        hidden_choices,
    )

    batch_candidates = optuna_utils.neighbors_from_choices(
        best_stage1_params["batch_size"],
        batch_choices,
    )

    clip_candidates = optuna_utils.neighbors_from_choices(
        best_stage1_params["clip_grad"],
        clip_choices,
    )

    fc_best_key = best_stage1_params["fc_units"]

    fc_candidates = optuna_utils.neighbors_from_choices(
        fc_best_key,
        FC_UNITS_KEYS,
    )

    best_lr = best_stage1_params["lr"]
    lr_low = max(1e-5, best_lr / 3.0)
    lr_high = min(1e-2, best_lr * 3.0)

    best_wd = best_stage1_params["weight_decay"]
    wd_low = max(1e-7, best_wd / 10.0)
    wd_high = min(1e-1, best_wd * 10.0)

    best_dropout = best_stage1_params["dropout"]
    dropout_low = max(0.0, best_dropout - 0.15)
    dropout_high = min(0.6, best_dropout + 0.15)

    best_num_layers = best_stage1_params["num_layers"]
    num_layers_low = max(1, best_num_layers - 1)
    num_layers_high = min(5, best_num_layers + 1)

    bidirectional_best = best_stage1_params["bidirectional"]
    layernorm_best = best_stage1_params["layernorm"]
    scheduler_best = best_stage1_params["use_scheduler"]

    def objective(trial):
        fc_units_key = trial.suggest_categorical(
            "fc_units",
            fc_candidates,
        )

        params = {
            "val_ratio": fixed_params.get("val_ratio", 0.2),
            "epochs": fixed_params.get("epochs", 100),
            "patience": fixed_params.get("patience", 10),
            "min_delta": fixed_params.get("min_delta", 0.0),
            "calibrate": fixed_params.get("calibrate", False),
            "save_best_path": None,
            "device": fixed_params.get("device", "cuda"),
            "progress": False,

            "hidden_size": trial.suggest_categorical(
                "hidden_size", hidden_candidates
            ),
            "num_layers": trial.suggest_int(
                "num_layers", num_layers_low, num_layers_high
            ),
            "bidirectional": trial.suggest_categorical(
                "bidirectional", [bidirectional_best]
            ),
            "dropout": trial.suggest_float(
                "dropout", dropout_low, dropout_high
            ),
            "fc_units": FC_UNITS_MAP[fc_units_key],
            "layernorm": trial.suggest_categorical(
                "layernorm", [layernorm_best]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_candidates
            ),
            "lr": trial.suggest_float(
                "lr", lr_low, lr_high, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", wd_low, wd_high, log=True
            ),
            "clip_grad": trial.suggest_categorical(
                "clip_grad", clip_candidates
            ),
            "use_scheduler": trial.suggest_categorical(
                "use_scheduler", [scheduler_best]
            ),
        }

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
                raise FloatingPointError("Score non fini.")

            return score

        except FloatingPointError:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial échoué: {e}")

    return objective


def run_lstm_stage2_search(
    X_train,
    y_train,
    study_stage1,
    n_trials=25,
    study_name="lstm_stage2",
    storage=None,
    metric_name="val_loss",
    fixed_params=None,
):
    if metric_name == "val_auc":
        direction = "maximize"
    else:
        direction = "minimize"
    best_stage1_params = study_stage1.best_params

    sampler = optuna.samplers.TPESampler(seed=43)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective_lstm_stage2(
        X_train=X_train,
        y_train=y_train,
        best_stage1_params=best_stage1_params,
        metric_name=metric_name,
        fixed_params=fixed_params,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("=== LSTM STAGE 2 ===")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)

    return study