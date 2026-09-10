"""ICU adapter: stability analysis for pre-fitted Logistic Regression models.

Performs patient-level clustered bootstrap evaluations conditional on
standardization and feature extraction pipelines. Exports publication-ready
metrics, summaries, and figures in PDF, PNG, and SVG formats.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from utilitaries.figures.feature_names import short_feature_name
from .stability_caterpillar import (
    bootstrap_coefficients,
    bootstrap_predictions,
    caterpillar_pair,
    coefficient_summary,
    compare_fixed_models,
    plot_performance,
)


def root_lr(model: Any) -> LogisticRegression:
    """Unwrap fitted estimator chains to retrieve the underlying LogisticRegression.

    Args:
        model: Fitted estimator or calibration wrapper.

    Returns:
        The core fitted LogisticRegression estimator.

    Raises:
        ValueError: If circular references exist, or classes are not binary [0, 1].
        TypeError: If an unsupported wrapper class is supplied.
    """
    seen = set()
    while not isinstance(model, LogisticRegression):
        if id(model) in seen:
            raise ValueError("Cyclic wrapper reference detected.")
        seen.add(id(model))
        name = type(model).__name__
        if name == "CalibratedClassifierCV":
            fitted = getattr(model, "calibrated_classifiers_", [])
            if len(fitted) != 1:
                raise ValueError("Expected a single frozen estimator, not a calibration ensemble.")
            model = fitted[0].estimator
        elif name in ("FrozenEstimator", "TemperatureScaledEstimator"):
            model = model.estimator
        elif name == "PriorCorrectionWrapper":
            model = model.base_estimator
        else:
            raise TypeError(f"Unsupported estimator wrapper: {name}")

    if not np.array_equal(model.classes_, [0, 1]) or model.coef_.shape[0] != 1:
        raise ValueError("Fitted binary LogisticRegression with classes [0, 1] required.")
    return model


def as_frame(X: Any, names: Sequence[str]) -> pd.DataFrame:
    """Convert input array or DataFrame ensuring strict column alignment.

    Args:
        X: Design matrix or DataFrame.
        names: Expected feature column ordering.

    Returns:
        A validated pandas DataFrame.

    Raises:
        ValueError: If column order, shape, or finiteness checks fail.
    """
    if hasattr(X, "columns") and list(X.columns) != list(names):
        raise ValueError("Feature column order mismatch.")
    values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError("Invalid or non-finite design matrix.")
    return pd.DataFrame(values, columns=names)


def exact_training_subset(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    calibration_enabled: bool,
    balance_method: str,
    seed: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reproduce the exact training partition used during the calibration stage.

    Args:
        X: Feature DataFrame.
        y: Target label array.
        groups: Group identifiers (e.g., patient stays).
        calibration_enabled: Whether internal calibration splitting was active.
        balance_method: Data balancing configuration flag.
        seed: Random partition seed.

    Returns:
        A tuple of (X_subset, y_subset, groups_subset).

    Raises:
        ValueError: If array lengths are inconsistent or contain NaNs.
    """
    y, groups = np.asarray(y).reshape(-1), np.asarray(groups).reshape(-1)
    if len(X) != len(y) or len(y) != len(groups) or pd.isna(groups).any():
        raise ValueError("Mismatched dimensions or missing groups in training subset.")

    idx = np.arange(len(y))
    if calibration_enabled and balance_method == "":
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        idx, _ = next(cv.split(X, y, groups=groups))

    return X.iloc[idx].reset_index(drop=True), y[idx], groups[idx]


def _fingerprint(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    scale: np.ndarray,
    model: LogisticRegression,
    n_bootstrap: int,
    seed: int
) -> str:
    """Compute a deterministic hash fingerprint to validate bootstrap caches."""
    digest = hashlib.sha256()
    for value in [X.to_numpy(dtype=float), y, scale, model.coef_, model.intercept_]:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(list(map(str, groups))).encode())
    digest.update(json.dumps(list(X.columns)).encode())
    digest.update(repr(sorted(model.get_params().items())).encode())
    digest.update(f"icu-v2.2/{sklearn.__version__}/{n_bootstrap}/{seed}".encode())
    return digest.hexdigest()[:24]


def run_icu_stability(
    *,
    exp: Any,
    models: Sequence[Any],
    feature_names_per_model: Sequence[Sequence[str]],
    X_train_per_model: Sequence[Any],
    y_train_per_model: Sequence[Any],
    groups_per_model: Sequence[Any],
    calibration_enabled: bool,
    balance_method: str,
    seed: int,
    folder: Union[str, Path],
    y_holdout: np.ndarray,
    holdout_ids: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 400,
    top_n: int = 20,
    include_outliers_in_top: bool = True,
    save_figures: bool = True
) -> Dict[str, Any]:
    """Execute cross-fold stability analyses and holdout bootstrap evaluation.

    Args:
        exp: Experiment tracking runner providing file resolution methods.
        models: List of five pre-fitted fold models.
        feature_names_per_model: List of feature names per fold.
        X_train_per_model: Training features per fold.
        y_train_per_model: Training targets per fold.
        groups_per_model: Patient clustering identifiers per fold.
        calibration_enabled: Flag indicating if internal calibration split was used.
        balance_method: Class re-balancing identifier.
        seed: Random seed for deterministic reproducibility.
        folder: Output directory for saving reports and figures.
        y_holdout: Holdout ground truth labels.
        holdout_ids: Clustered identifiers for holdout samples.
        probabilities: Predicted probabilities across folds (shape: 5, N).
        n_bootstrap: Number of bootstrap iterations (set 0 to skip).
        top_n: Number of leading features in caterpillar plots.
        include_outliers_in_top: Append extreme outliers in top view plots.
        save_figures: Whether to render and save PDF, PNG, and SVG plots.

    Returns:
        Dictionary containing summary tables, DataFrames, and output paths.

    Raises:
        ValueError: If folds != 5, or data dimension mismatches occur.
    """
    import polars as pl

    if not all(
        len(a) == 5
        for a in [
            models,
            feature_names_per_model,
            X_train_per_model,
            y_train_per_model,
            groups_per_model,
        ]
    ):
        raise ValueError("Stability evaluation requires exactly 5 cross-validation folds.")
    if n_bootstrap != 0 and n_bootstrap < 2:
        raise ValueError("n_bootstrap must be either 0 or >= 2.")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    universe = list(dict.fromkeys(f for names in feature_names_per_model for f in names))
    prepared, scales, betas = [], [], []

    for i, (wrapped, names) in enumerate(zip(models, feature_names_per_model)):
        model = root_lr(wrapped)
        if model.coef_.shape != (1, len(names)):
            raise ValueError(f"Fold {i+1}: coefficient/feature dimension mismatch.")

        X = as_frame(X_train_per_model[i], names)
        Xfit, yfit, ids = exact_training_subset(
            X,
            y_train_per_model[i],
            groups_per_model[i],
            calibration_enabled=calibration_enabled,
            balance_method=balance_method,
            seed=seed,
        )

        saved_X = as_frame(pl.read_parquet(exp.get_lasso_path("X", i, "parquet")), names)
        saved_y = np.load(exp.get_lasso_path("y", i, "npy"), allow_pickle=False).reshape(-1)

        if (
            saved_X.shape != Xfit.shape
            or not np.array_equal(saved_y, yfit)
            or not np.allclose(saved_X.to_numpy(), Xfit.to_numpy(), rtol=1e-9, atol=1e-10)
        ):
            raise ValueError(
                f"Fold {i+1}: Cached parquet arrays do not match reconstructed subset."
            )

        with np.load(exp.get_lasso_path("stability_scaling", i, "npz"), allow_pickle=False) as meta:
            if list(meta["feature_names"]) != list(names):
                raise ValueError(f"Fold {i+1}: scaling feature ordering mismatch.")
            scale = meta["scale"].copy()

        if scale.shape != (len(names),) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid standard scaling vector.")

        scales.append(pd.Series(scale, index=names))
        betas.append(pd.Series(model.coef_[0] / scale, index=names))
        prepared.append((model, Xfit, yfit, ids, scale))

    reference = pd.concat(scales, axis=1).median(axis=1).reindex(universe)
    reference.rename("reference_increment").to_csv(folder / "reference_increments.csv")

    fixed, fixed_draws = compare_fixed_models(betas, universe, effect_units=reference)
    fixed.to_csv(folder / "fixed_models_summary.csv", index=False)
    pd.DataFrame(fixed_draws, columns=universe).to_csv(folder / "fixed_raw_betas.csv", index=False)

    if save_figures:
        figs = caterpillar_pair(
            fixed,
            folder / "fixed_models",
            prefix="fixed",
            top_n=top_n,
            include_outliers_in_top=include_outliers_in_top,
            title="Five Fixed LR Models · OR Dispersion Prior to Calibration",
            label_transform = short_feature_name
        )
        for fig in figs.values():
            plt.close(fig)

    summaries = []
    all_draws = []
    for i, (model, Xfit, yfit, ids, scale) in enumerate(prepared):
        if not n_bootstrap:
            break

        fold_folder = folder / f"bootstrap_fold_{i+1}"
        fold_folder.mkdir(parents=True, exist_ok=True)
        stamp = _fingerprint(Xfit, yfit, ids, scale, model, n_bootstrap, seed + i)
        cache = fold_folder / f"draws_{stamp}.npz"
        log_path = fold_folder / f"diagnostics_{stamp}.csv"

        if cache.exists() and log_path.exists():
            with np.load(cache, allow_pickle=False) as data:
                draws = data["betas"].copy()
            log = pd.read_csv(log_path)
            if (log.status != "ok").any():
                warnings.warn(f"Fold {i+1}: cached bootstrap log contains failures: {log_path}")
            print(f"[STABILITY] Fold {i+1}: bootstrap loaded from cache.", flush=True)
        else:
            def fit_extract(Xb: pd.DataFrame, yb: np.ndarray, algorithm_seed: int) -> pd.Series:
                lr = clone(model)
                lr.set_params(random_state=algorithm_seed, warm_start=False)
                lr.fit(Xb.to_numpy(), yb)
                return pd.Series(lr.coef_[0] / scale, index=Xb.columns)

            print(f"[STABILITY] Fold {i+1}: {n_bootstrap} patient draws, C={model.C}.", flush=True)
            draws, log = bootstrap_coefficients(
                Xfit,
                yfit,
                universe,
                fit_extract,
                n_bootstrap=n_bootstrap,
                seed=seed + i,
                patient_ids=ids,
            )
            log.to_csv(log_path, index=False)
            np.savez_compressed(cache, betas=draws, feature_names=np.asarray(universe, dtype=str))

        all_draws.append(draws)
        summary = coefficient_summary(draws, universe, effect_units=reference)
        summary.to_csv(fold_folder / "summary.csv", index=False)
        summary["fold"] = i + 1
        summaries.append(summary)

        if save_figures:
            figs = caterpillar_pair(
                summary,
                fold_folder,
                prefix="bootstrap",
                top_n=top_n,
                include_outliers_in_top=include_outliers_in_top,
                title=f"LR Fold {i+1} · Conditional Bootstrap · C={model.C:.3g}",
                label_transform = short_feature_name
            )
            for fig in figs.values():
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Pooled Cross-Fold Mega-Bootstrap (5 folds x N draws = 2000 draws)
    # -------------------------------------------------------------------------
    pooled_summary = None
    if n_bootstrap and len(all_draws) == len(prepared):
        pooled_draws = np.vstack(all_draws)
        pooled_folder = folder / "bootstrap_pooled_all_folds"
        pooled_folder.mkdir(parents=True, exist_ok=True)

        pooled_summary = coefficient_summary(
            pooled_draws,
            universe,
            effect_units=reference,
            outlier_k=3.0,
        )
        pooled_summary.to_csv(pooled_folder / "summary.csv", index=False)
        np.savez_compressed(
            pooled_folder / "pooled_draws.npz",
            betas=pooled_draws,
            feature_names=np.asarray(universe, dtype=str),
        )

        if save_figures:
            figs = caterpillar_pair(
                pooled_summary,
                pooled_folder,
                prefix="mega_bootstrap",
                top_n=top_n,
                include_outliers_in_top=include_outliers_in_top,
                title=f"L1-LR Pooled Multi-Fold Bootstrap ({len(pooled_draws)} Draws) · OR Stability",
                label_transform = short_feature_name
            )
            for fig in figs.values():
                plt.close(fig)
            print(f"[STABILITY] Mega-bootstrap caterpillar ({len(pooled_draws)} draws) saved to {pooled_folder}.", flush=True)

    p = np.asarray(probabilities, dtype=float)
    y_holdout = np.asarray(y_holdout).reshape(-1)
    holdout_ids = np.asarray(holdout_ids).reshape(-1)

    if (
        p.shape != (5, len(y_holdout))
        or holdout_ids.shape != y_holdout.shape
        or pd.isna(holdout_ids).any()
        or len(pd.unique(holdout_ids)) != len(y_holdout)
    ):
        raise ValueError("Expected five prediction columns aligned with holdout patients.")

    predictions = pd.DataFrame(p.T, columns=[f"fold_{i+1}" for i in range(5)])
    predictions["ensemble"] = predictions.mean(axis=1)

    perf, perf_draws = bootstrap_predictions(
        y_holdout, predictions, seed=seed, n_bootstrap=2000
    )
    perf.to_csv(folder / "performance_summary.csv", index=False)
    np.savez_compressed(
        folder / "performance_draws.npz",
        scores=perf_draws,
        models=predictions.columns.to_numpy(dtype=str),
        metrics=np.array(["AUC", "Brier"]),
    )

    if save_figures:
        fig = plot_performance(perf, folder)
        plt.close(fig)

    return {
        "analysis": "conditional_stability_v2",
        "fixed_coefficients": fixed,
        "bootstrap_coefficients": summaries,
        "pooled_bootstrap_coefficients": pooled_summary,
        "fixed_performance": perf,
        "reference_increments": reference,
        "folder": str(folder),
    }