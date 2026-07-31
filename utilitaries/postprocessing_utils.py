"""Functions to add to ``utilitaries/new_show_fig_utils.py``.

These helpers evaluate an independent holdout without retraining models and
compute ensemble SHAP values for fold-specific TSFEL representations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, matthews_corrcoef


def f1_at_fixed_threshold(
    y_true,
    probabilities,
    threshold,
):
    """Compute F1 using a threshold selected before holdout evaluation.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.
        threshold:
            Fixed decision threshold, typically selected from pooled OOF
            predictions.

    Returns:
        The F1 score computed at the fixed threshold.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return float(
        f1_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )


def mcc_at_fixed_threshold(
    y_true,
    probabilities,
    threshold,
):
    """Compute MCC using a threshold selected before holdout evaluation.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.
        threshold:
            Fixed decision threshold, typically selected from pooled OOF
            predictions.

    Returns:
        The Matthews correlation coefficient at the fixed threshold.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return float(
        matthews_corrcoef(
            y_true,
            predictions,
        )
    )


def bootstrap_holdout_metrics(
    y_true,
    probabilities,
    metric_functions,
    n_bootstrap=2000,
    confidence_level=0.95,
    seed=42,
):
    """Bootstrap metrics on independent holdout patients only.

    Models are not retrained. Each bootstrap iteration samples holdout patient
    indices with replacement and evaluates the already-computed ensemble
    probabilities.

    Args:
        y_true:
            Binary holdout labels.
        probabilities:
            Final holdout probabilities, usually averaged across fold models.
        metric_functions:
            Mapping from metric names to callables accepting
            ``(y_true, probabilities)``.
        n_bootstrap:
            Number of valid bootstrap resamples.
        confidence_level:
            Percentile confidence level between 0 and 1.
        seed:
            Random seed.

    Returns:
        A dictionary containing point estimates, bootstrap distributions, and
        percentile confidence intervals.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "The holdout arrays must not be empty."
        )

    if np.unique(y_true).size != 2:
        raise ValueError(
            "The complete holdout must contain both binary classes."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be strictly between 0 and 1."
        )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be greater than zero."
        )

    if not metric_functions:
        raise ValueError(
            "metric_functions must contain at least one metric."
        )

    estimates = {
        metric_name: float(
            metric_function(
                y_true,
                probabilities,
            )
        )
        for metric_name, metric_function in metric_functions.items()
    }

    distributions = {
        metric_name: []
        for metric_name in metric_functions
    }

    rng = np.random.default_rng(seed)
    n_patients = len(y_true)
    valid_resamples = 0
    attempted_resamples = 0
    maximum_attempts = max(
        n_bootstrap * 20,
        n_bootstrap + 100,
    )

    while (
        valid_resamples < n_bootstrap
        and attempted_resamples < maximum_attempts
    ):
        attempted_resamples += 1

        sampled_indices = rng.integers(
            0,
            n_patients,
            size=n_patients,
        )

        sampled_y = y_true[sampled_indices]

        # AUC, AUPRC, and several calibration metrics require both classes.
        if np.unique(sampled_y).size < 2:
            continue

        sampled_probabilities = probabilities[
            sampled_indices
        ]

        current_values = {}
        invalid_resample = False

        for metric_name, metric_function in metric_functions.items():
            try:
                metric_value = float(
                    metric_function(
                        sampled_y,
                        sampled_probabilities,
                    )
                )
            except (ValueError, FloatingPointError):
                invalid_resample = True
                break

            if not np.isfinite(metric_value):
                invalid_resample = True
                break

            current_values[metric_name] = metric_value

        if invalid_resample:
            continue

        for metric_name, metric_value in current_values.items():
            distributions[metric_name].append(
                metric_value
            )

        valid_resamples += 1

    if valid_resamples < n_bootstrap:
        raise RuntimeError(
            "Unable to generate the requested number of valid bootstrap "
            f"resamples: {valid_resamples}/{n_bootstrap}."
        )

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    distributions = {
        metric_name: np.asarray(
            values,
            dtype=float,
        )
        for metric_name, values in distributions.items()
    }

    summary = {}

    for metric_name, values in distributions.items():
        summary[metric_name] = {
            "estimate": estimates[metric_name],
            "bootstrap_mean": float(
                np.mean(values)
            ),
            "bootstrap_std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
            "ci_lower": float(
                np.percentile(
                    values,
                    lower_percentile,
                )
            ),
            "ci_upper": float(
                np.percentile(
                    values,
                    upper_percentile,
                )
            ),
        }

    return {
        "summary": summary,
        "distributions": distributions,
        "n_bootstrap": valid_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
    }


def _to_2d_numpy(
    X,
    *,
    allow_nan=False,
):
    """Convert a tabular matrix to a two-dimensional NumPy array.

    Args:
        X:
            Input matrix, possibly represented as a Polars or Pandas
            DataFrame.
        allow_nan:
            Whether missing values are accepted. Missing values should be
            preserved when the underlying model handles them natively, such
            as XGBoost.

    Returns:
        A two-dimensional floating-point NumPy array.

    Raises:
        ValueError:
            If the matrix is not two-dimensional, contains infinite values,
            or contains missing values when ``allow_nan`` is False.
    """
    if hasattr(X, "to_numpy"):
        X_array = X.to_numpy()
    else:
        X_array = np.asarray(X)

    X_array = np.asarray(
        X_array,
        dtype=np.float32,
    )

    if X_array.ndim != 2:
        raise ValueError(
            "Each holdout fold matrix must be two-dimensional. "
            f"Received shape {X_array.shape}."
        )

    # Infinite values are never accepted. They generally indicate a genuine
    # preprocessing error rather than an ordinary missing observation.
    if np.isinf(X_array).any():
        n_positive_inf = int(
            np.isposinf(X_array).sum()
        )

        n_negative_inf = int(
            np.isneginf(X_array).sum()
        )

        raise ValueError(
            "A holdout fold matrix contains infinite values: "
            f"+inf={n_positive_inf}, -inf={n_negative_inf}."
        )

    # Preserve NaN values for estimators that handle missing values natively.
    if not allow_nan and np.isnan(X_array).any():
        raise ValueError(
            "A holdout fold matrix contains NaN values."
        )

    return X_array


def shap_holdout_ensemble(
    models,
    X_test_per_model,
    feature_names_per_model,
    model_name,
    max_background_samples=100,
    max_display=20,
    max_evals=None,
    savefig=True,
    folder="",
    transparent=False,
    seed=42,
):
    """Compute mean holdout SHAP values across fold-specific TSFEL models.

    Each fold model explains the same holdout patients using the feature matrix
    produced by that fold's preprocessing. SHAP values are realigned on the
    union of feature names. A feature absent from a fold receives a zero SHAP
    contribution for that fold because the corresponding model does not use it.

    Args:
        models:
            Fitted fold estimators exposing ``predict_proba``.
        X_test_per_model:
            One transformed holdout matrix per fold model.
        feature_names_per_model:
            Ordered feature names for each fold matrix.
        model_name:
            Model name displayed in the figure.
        max_background_samples:
            Maximum number of holdout rows used as each fold's SHAP background.
        max_display:
            Maximum number of features displayed in the beeswarm.
        max_evals:
            Optional number of permutation evaluations. When omitted, SHAP
            selects its default.
        savefig:
            Whether PNG and PDF files should be saved.
        folder:
            Output directory.
        transparent:
            Whether saved figures use a transparent background.
        seed:
            Random seed used for background sampling.

    Returns:
        A dictionary containing aligned SHAP arrays, their mean, aligned mean
        feature values, global feature names, and ranked feature importance.
    """
    models = list(models)
    X_test_per_model = list(X_test_per_model)
    feature_names_per_model = [
        list(names)
        for names in feature_names_per_model
    ]

    n_models = len(models)

    if n_models == 0:
        raise ValueError(
            "models must contain at least one fitted model."
        )

    if not (
        n_models
        == len(X_test_per_model)
        == len(feature_names_per_model)
    ):
        raise ValueError(
            "models, X_test_per_model, and feature_names_per_model "
            "must have the same length."
        )

    X_arrays = [
        _to_2d_numpy(
            X_fold,
            allow_nan=True,
        )
        for X_fold in X_test_per_model
    ]

    for fold_index, X_fold in enumerate(
        X_arrays
    ):
        print(
            f"[SHAP] Fold {fold_index + 1}: "
            f"shape={X_fold.shape}, "
            f"NaN={int(np.isnan(X_fold).sum())}, "
            f"inf={int(np.isinf(X_fold).sum())}"
        )

    n_patients = X_arrays[0].shape[0]

    for fold_index, (X_fold, names) in enumerate(
        zip(
            X_arrays,
            feature_names_per_model,
        )
    ):
        if X_fold.shape[0] != n_patients:
            raise ValueError(
                "All holdout fold matrices must contain the same patients "
                "in the same order."
            )

        if X_fold.shape[1] != len(names):
            raise ValueError(
                f"Fold {fold_index + 1}: {X_fold.shape[1]} columns for "
                f"{len(names)} feature names."
            )

        if len(set(names)) != len(names):
            raise ValueError(
                f"Fold {fold_index + 1} contains duplicate feature names."
            )

    global_feature_names = sorted(
        set().union(
            *[
                set(names)
                for names in feature_names_per_model
            ]
        )
    )

    global_feature_to_index = {
        feature_name: feature_index
        for feature_index, feature_name in enumerate(
            global_feature_names
        )
    }

    aligned_shap_per_model = []
    aligned_values_per_model = []
    base_values_per_model = []
    rng = np.random.default_rng(seed)

    for fold_index, (model, X_fold, fold_feature_names) in enumerate(
        zip(
            models,
            X_arrays,
            feature_names_per_model,
        )
    ):
        print(
            "[SHAP] Computing holdout values for fold model "
            f"{fold_index + 1}/{n_models}."
        )

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"Fold model {fold_index + 1} does not expose predict_proba."
            )

        if len(X_fold) > max_background_samples:
            background_indices = rng.choice(
                len(X_fold),
                size=max_background_samples,
                replace=False,
            )
            background = X_fold[background_indices]
        else:
            background = X_fold

        def predict_positive_probability(data):
            probabilities = np.asarray(
                model.predict_proba(
                    np.asarray(
                        data,
                        dtype=np.float32,
                    )
                ),
                dtype=float,
            )

            if probabilities.ndim == 1:
                return probabilities

            if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                raise ValueError(
                    "Only binary predict_proba outputs are supported."
                )

            return probabilities[:, 1]

        explainer = shap.Explainer(
            predict_positive_probability,
            shap.maskers.Independent(background),
            feature_names=fold_feature_names,
            algorithm="permutation",
        )

        explanation_kwargs = {}

        if max_evals is not None:
            explanation_kwargs["max_evals"] = max_evals

        explanation = explainer(
            X_fold,
            **explanation_kwargs,
        )

        fold_shap = np.asarray(
            explanation.values,
            dtype=float,
        )

        if fold_shap.ndim == 3 and fold_shap.shape[-1] == 2:
            fold_shap = fold_shap[..., 1]

        if fold_shap.shape != X_fold.shape:
            raise ValueError(
                f"Fold {fold_index + 1}: unexpected SHAP shape "
                f"{fold_shap.shape}; expected {X_fold.shape}."
            )

        aligned_shap = np.zeros(
            (
                n_patients,
                len(global_feature_names),
            ),
            dtype=float,
        )

        # NaN means that the feature was absent from the fold. This prevents
        # zero padding from falsely coloring absent variables as low values.
        aligned_values = np.full(
            (
                n_patients,
                len(global_feature_names),
            ),
            np.nan,
            dtype=float,
        )

        for local_feature_index, feature_name in enumerate(
            fold_feature_names
        ):
            global_feature_index = global_feature_to_index[
                feature_name
            ]

            aligned_shap[:, global_feature_index] = (
                fold_shap[:, local_feature_index]
            )

            aligned_values[:, global_feature_index] = (
                X_fold[:, local_feature_index]
            )

        fold_base_values = np.asarray(
            explanation.base_values,
            dtype=float,
        ).reshape(-1)

        if fold_base_values.size == 1:
            fold_base_values = np.repeat(
                fold_base_values,
                n_patients,
            )

        if fold_base_values.size != n_patients:
            raise ValueError(
                f"Fold {fold_index + 1}: unexpected base-value shape."
            )

        aligned_shap_per_model.append(
            aligned_shap
        )
        aligned_values_per_model.append(
            aligned_values
        )
        base_values_per_model.append(
            fold_base_values
        )

    aligned_shap_per_model = np.stack(
        aligned_shap_per_model,
        axis=0,
    )

    aligned_values_per_model = np.stack(
        aligned_values_per_model,
        axis=0,
    )

    base_values_per_model = np.stack(
        base_values_per_model,
        axis=0,
    )

    mean_shap_values = np.mean(
        aligned_shap_per_model,
        axis=0,
    )

    with np.errstate(invalid="ignore"):
        mean_feature_values = np.nanmean(
            aligned_values_per_model,
            axis=0,
        )

    # This should only occur for an impossible union feature, but replacing it
    # keeps the SHAP Explanation finite and explicit.
    mean_feature_values = np.nan_to_num(
        mean_feature_values,
        nan=0.0,
    )

    mean_base_values = np.mean(
        base_values_per_model,
        axis=0,
    )

    mean_absolute_shap = np.mean(
        np.abs(mean_shap_values),
        axis=0,
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": global_feature_names,
                "mean_abs_shap": mean_absolute_shap,
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ensemble_explanation = shap.Explanation(
        values=mean_shap_values,
        base_values=mean_base_values,
        data=mean_feature_values,
        feature_names=global_feature_names,
    )

    plt.figure(figsize=(10, 8))

    shap.plots.beeswarm(
        ensemble_explanation,
        max_display=min(
            max_display,
            len(global_feature_names),
        ),
        show=False,
    )

    plt.title(
        "Mean holdout SHAP values across fold models\n"
        f"{model_name}"
    )

    plt.tight_layout()

    output_directory = Path(folder)

    if savefig:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_directory
            / "holdout_ensemble_shap_beeswarm.png",
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )

        plt.savefig(
            output_directory
            / "holdout_ensemble_shap_beeswarm.pdf",
            bbox_inches="tight",
            transparent=transparent,
        )

        feature_importance.to_csv(
            output_directory
            / "holdout_ensemble_shap_importance.csv",
            index=False,
        )

    plt.show()
    plt.close()

    return {
        "mean_shap_values": mean_shap_values,
        "aligned_shap_per_model": aligned_shap_per_model,
        "mean_feature_values": mean_feature_values,
        "mean_base_values": mean_base_values,
        "feature_names": global_feature_names,
        "feature_importance": feature_importance,
    }

