"""Functions to add to ``utilitaries/new_show_fig_utils.py``.

These helpers evaluate an independent holdout without retraining models and
compute ensemble SHAP values for fold-specific TSFEL representations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import os
from typing import *
import re
import textwrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, matthews_corrcoef
import utilitaries.training_utils as utils
import utilitaries.features_extraction_utils as feat_utils

def calibration_intercept(
    y_true,
    probabilities,
):
    """Compute calibration intercept via logistic regression on logits.

    The intercept is estimated by fitting a logistic regression where the
    feature is the logit of the predicted probabilities and the target is
    the binary ground truth.  A well-calibrated model has an intercept
    close to zero.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.

    Returns:
        The calibration intercept as a floating-point value.

    Raises:
        ValueError:
            If the arrays are empty, have mismatched lengths, or contain
            only a single class.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if np.unique(y_true).size < 2:
        raise ValueError(
            "Both classes are required to estimate calibration statistics."
        )

    eps = 1e-7
    probas_clipped = np.clip(probabilities, eps, 1.0 - eps)
    logits = np.log(probas_clipped / (1.0 - probas_clipped)).reshape(-1, 1)

    calib_model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1000,
    )
    calib_model.fit(logits, y_true)

    return float(calib_model.intercept_[0])


def calibration_slope(
    y_true,
    probabilities,
):
    """Compute calibration slope via logistic regression on logits.

    The slope is estimated by fitting a logistic regression where the
    feature is the logit of the predicted probabilities and the target is
    the binary ground truth.  A well-calibrated model has a slope close
    to one.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.

    Returns:
        The calibration slope as a floating-point value.

    Raises:
        ValueError:
            If the arrays are empty, have mismatched lengths, or contain
            only a single class.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if np.unique(y_true).size < 2:
        raise ValueError(
            "Both classes are required to estimate calibration statistics."
        )

    eps = 1e-7
    probas_clipped = np.clip(probabilities, eps, 1.0 - eps)
    logits = np.log(probas_clipped / (1.0 - probas_clipped)).reshape(-1, 1)

    calib_model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1000,
    )
    calib_model.fit(logits, y_true)

    return float(calib_model.coef_[0, 0])


def _bin_errors_fixed_10pct(
    probabilities,
    y_true,
):
    """Compute per-bin absolute errors using fixed 10% risk brackets.

    Args:
        probabilities:
            Positive-class predicted probabilities, shape ``(n,)``.
        y_true:
            Binary ground-truth labels, shape ``(n,)``.

    Returns:
        A NumPy array of absolute errors for each non-empty bin.

    Raises:
        ValueError:
            If the inputs are empty or have mismatched lengths.
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if probabilities.size == 0:
        raise ValueError(
            "probabilities must not be empty."
        )

    # Assign each sample to a fixed 10% bracket [0-10%, 10-20%, …, 90-100%[
    bin_indices = np.clip(
        (probabilities * 10).astype(np.int64),
        0,
        9,
    )

    errors = np.zeros(10, dtype=float)
    counts = np.zeros(10, dtype=np.int64)

    for bin_idx in range(10):
        mask = bin_indices == bin_idx
        n_in_bin = np.count_nonzero(mask)

        if n_in_bin == 0:
            continue

        obs_rate = np.mean(y_true[mask])
        pred_mean = np.mean(probabilities[mask])
        errors[bin_idx] = abs(obs_rate - pred_mean)
        counts[bin_idx] = n_in_bin

    # Return only errors for non-empty bins
    return errors[counts > 0]


def ici_score(
    y_true,
    probabilities,
):
    """Compute the Integrated Calibration Index (ICI) with fixed 10% bins.

    The ICI is the weighted mean of absolute calibration errors across
    fixed 10% risk brackets.  Empty bins are ignored.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.

    Returns:
        The ICI as a floating-point value between 0 and 1.

    Raises:
        ValueError:
            If the arrays are empty or have mismatched lengths.
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if probabilities.size == 0:
        raise ValueError(
            "probabilities must not be empty."
        )

    # Fixed 10% bracket assignment
    bin_indices = np.clip(
        (probabilities * 10).astype(np.int64),
        0,
        9,
    )

    n_total = len(probabilities)
    ici = 0.0

    for bin_idx in range(10):
        mask = bin_indices == bin_idx
        n_in_bin = np.count_nonzero(mask)

        if n_in_bin == 0:
            continue

        obs_rate = np.mean(y_true[mask])
        pred_mean = np.mean(probabilities[mask])
        bin_weight = n_in_bin / n_total
        ici += abs(obs_rate - pred_mean) * bin_weight

    return float(ici)


def e90_score(
    y_true,
    probabilities,
):
    """Compute E90: 90th percentile of per-bin absolute calibration errors.

    Errors are computed on fixed 10% risk brackets.  Empty bins are
    excluded before the percentile is evaluated.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.

    Returns:
        The E90 value as a floating-point number.

    Raises:
        ValueError:
            If the arrays are empty or have mismatched lengths.
    """
    bin_errs = _bin_errors_fixed_10pct(probabilities, y_true)

    if bin_errs.size == 0:
        return 0.0

    return float(np.percentile(bin_errs, 90))


def eMax_score(
    y_true,
    probabilities,
):
    """Compute EMax: maximum absolute calibration error across fixed 10% bins.

    Empty bins are excluded before the maximum is evaluated.

    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Positive-class predicted probabilities.

    Returns:
        The EMax value as a floating-point number.

    Raises:
        ValueError:
            If the arrays are empty or have mismatched lengths.
    """
    bin_errs = _bin_errors_fixed_10pct(probabilities, y_true)

    if bin_errs.size == 0:
        return 0.0

    return float(np.max(bin_errs))


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

def _wrap_feature_name(name: str, width: int = 32) -> str:
    """Insère des retours à la ligne (\n) propre sans créer de blocs de 3+ lignes."""
    if len(name) <= width:
        return name
    clean_name = name.replace("_", " ")
    wrapped = textwrap.fill(clean_name, width=width)
    return wrapped.replace(" ", "_")


def build_tsfel_pattern(scores_dict: Dict[str, int]) -> re.Pattern:
    """
    Construit une Regex compilée et ordonnée par longueur décroissante
    à partir des clés du dictionnaire TSFEL.
    """
    # 1. Récupère toutes les features du dictionnaire
    features = list(scores_dict.keys())
    
    # 2. Tri par longueur décroissante pour éviter qu'un motif court (ex: 'mean')
    #    ne mange un motif long (ex: 'mean absolute diff')
    features.sort(key=len, reverse=True)
    
    # 3. Échappe les caractères spéciaux et tolère espaces/underscores
    patterns = [re.escape(f).replace(r'\ ', r'[\s_]+') for f in features]
    
    # 4. Inclus aussi quelques variations classiques non présentes dans le dico
    additional_suffixes = [r'first', r'last', r'auc', r'sum']
    patterns.extend(additional_suffixes)
    
    # Motif global : matches un underscore suivi du nom de la feature
    # et de TOUT ce qui peut suivre (paramètres TSFEL, chiffres, etc.)
    combined_pattern = r'_(?:' + '|'.join(patterns) + r')(?:[\s_].*)?$'
    
    return re.compile(combined_pattern, flags=re.IGNORECASE)

# Compilation unique pour des performances maximales
TSFEL_REGEX = build_tsfel_pattern(feat_utils.explainability_scores)


def _clean_feature_name(name: str) -> str:
    """
    Extrait la racine d'une variable en supprimant les suffixes TSFEL 
    de manière extrêmement robuste.
    """
    if not isinstance(name, str):
        return name
        
    # Nettoyage basé sur la Regex TSFEL
    clean_name = TSFEL_REGEX.sub('', name)
    
    # Suppression d'éventuels trailing underscores restants
    return clean_name.rstrip('_').strip()

def clean_feature_aggregated(name: str, known_categorical_features: Optional[List[str]] = None) -> str:
    """Applique le nettoyage TSFEL puis le nettoyage OHE pour l'agrégation."""
    # Étape 1 : nettoyage TSFEL / temporel
    name_clean = _clean_feature_name(name)
    
    # Étape 2 : nettoyage One-Hot Encoding
    name_clean = _clean_ohe_name(name_clean, known_categorical_features)
    
    return name_clean

def _clean_ohe_name(name: str, known_categorical_features: Optional[List[str]] = None) -> str:
    """Nettoie le suffixe One-Hot Encoding si le début de la chaîne correspond 

    à une variable catégorielle connue.
    """
    if not known_categorical_features:
        return name

    # On trie par longueur décroissante pour éviter qu'un nom court ne préempte un nom long
    # ex: 'type' vs 'admission_type'
    sorted_known = sorted(known_categorical_features, key=len, reverse=True)

    for cat_feat in sorted_known:
        # Si la feature commence par la variable catégorielle suivie d'un underscore
        # ex: "admission_type_2" ou "admission_type_Emergency"
        if name.startswith(f"{cat_feat}_"):
            return cat_feat

    return name



def shap_tree_holdout_ensemble(
    models,
    X_test_per_model,
    feature_names_per_model,
    model_name,
    max_display=20,
    savefig=True,
    folder="",
    transparent=False,
):
    models = list(models)
    X_test_per_model = list(X_test_per_model)
    feature_names_per_model = [list(names) for names in feature_names_per_model]
    n_models = len(models)

    if n_models == 0:
        raise ValueError("models must contain at least one fitted model.")

    if not (n_models == len(X_test_per_model) == len(feature_names_per_model)):
        raise ValueError("models, X_test_per_model, and feature_names_per_model must have the same length.")

    X_arrays = [_to_2d_numpy(X_fold, allow_nan=True) for X_fold in X_test_per_model]
    n_patients = X_arrays[0].shape[0]

    global_feature_names = sorted(
        set().union(*[set(names) for names in feature_names_per_model])
    )
    global_feature_to_index = {name: idx for idx, name in enumerate(global_feature_names)}

    aligned_shap_per_model = []
    aligned_values_per_model = []
    base_values_per_model = []

    for fold_index, (model, X_fold, fold_feature_names) in enumerate(
        zip(models, X_arrays, feature_names_per_model)
    ):
        tree_model = utils.get_root_estimator(model)
        explainer = shap.TreeExplainer(tree_model)
        explanation = explainer(X_fold)

        fold_shap = np.asarray(explanation.values, dtype=float)
        if fold_shap.ndim == 3:
            fold_shap = fold_shap[..., 1]

        aligned_shap = np.zeros((n_patients, len(global_feature_names)), dtype=float)
        aligned_values = np.full((n_patients, len(global_feature_names)), np.nan, dtype=float)

        for local_idx, name in enumerate(fold_feature_names):
            global_idx = global_feature_to_index[name]
            aligned_shap[:, global_idx] = fold_shap[:, local_idx]
            aligned_values[:, global_idx] = X_fold[:, local_idx]

        fold_base_values = np.asarray(explanation.base_values, dtype=float)
        if fold_base_values.ndim == 2:
            fold_base_values = fold_base_values[:, 1]
        fold_base_values = fold_base_values.reshape(-1)
        if fold_base_values.size == 1:
            fold_base_values = np.repeat(fold_base_values, n_patients)

        aligned_shap_per_model.append(aligned_shap)
        aligned_values_per_model.append(aligned_values)
        base_values_per_model.append(fold_base_values)

    mean_shap_values = np.mean(np.stack(aligned_shap_per_model, axis=0), axis=0)
    with np.errstate(invalid="ignore"):
        mean_feature_values = np.nanmean(np.stack(aligned_values_per_model, axis=0), axis=0)
    mean_feature_values = np.nan_to_num(mean_feature_values, nan=0.0)
    mean_base_values = np.mean(np.stack(base_values_per_model, axis=0), axis=0)

    output_directory = Path(folder)
    if savefig:
        output_directory.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # VERSION 1 : DÉTAILLÉE
    # =========================================================================
    wrapped_detailed_names = [_wrap_feature_name(name, width=32) for name in global_feature_names]

    mean_abs_shap_detailed = np.mean(np.abs(mean_shap_values), axis=0)
    importance_detailed = (
        pd.DataFrame({"feature": global_feature_names, "mean_abs_shap": mean_abs_shap_detailed})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    exp_detailed = shap.Explanation(
        values=mean_shap_values,
        base_values=mean_base_values,
        data=mean_feature_values,
        feature_names=wrapped_detailed_names,
    )

    # plot_size augmente la hauteur physique verticale pour espacer les labels Y
    plt.figure()
    shap.plots.beeswarm(
        exp_detailed, 
        max_display=min(max_display, len(wrapped_detailed_names)), 
        plot_size=(10, 13),  # <-- Hauteur augmentée (13 au lieu de 8/10 par défaut)
        show=False
    )
    plt.yticks(fontsize= 12)  # Police légèrement plus petite pour éviter les chevauchements
    # plt.title(f"TreeSHAP Detailed - {model_name}", fontsize=12, pad=15)
    plt.tight_layout()

    if savefig:
        plt.savefig(output_directory / "holdout_ensemble_treeshap_beeswarm_detailed.png", dpi=300, bbox_inches="tight", transparent=transparent)
        plt.savefig(output_directory / "holdout_ensemble_treeshap_beeswarm_detailed.pdf", bbox_inches="tight", transparent=transparent)
        importance_detailed.to_csv(output_directory / "holdout_ensemble_treeshap_importance_detailed.csv", index=False)
    
    plt.show()
    plt.close()

    # =========================================================================
    # VERSION 2 : AGRÉGÉE PAR FEATURE RACINE (Inclut TSFEL + One-Hot Encoded)
    # =========================================================================
    KNOWN_CATEGORICALS = ['admission_type', 'icu_ghm', 'icu_mode_entree', 'hx']
    root_mapping = {}
    for idx, name in enumerate(global_feature_names):
        root = clean_feature_aggregated(name, known_categorical_features=KNOWN_CATEGORICALS)
        root_mapping.setdefault(root, []).append(idx)

    unique_roots = list(root_mapping.keys())
    agg_shap_values = np.zeros((n_patients, len(unique_roots)), dtype=float)
    agg_feature_values = np.zeros((n_patients, len(unique_roots)), dtype=float)

    for r_idx, (root, col_indices) in enumerate(root_mapping.items()):
        # Somme des SHAP values pour toutes les sous-variables OHE / TSFEL
        agg_shap_values[:, r_idx] = np.sum(mean_shap_values[:, col_indices], axis=1)
        # Moyenne des valeurs réelles de la variable pour l'affichage couleur de SHAP
        agg_feature_values[:, r_idx] = np.mean(mean_feature_values[:, col_indices], axis=1)

    wrapped_aggregated_names = [_wrap_feature_name(name, width=32) for name in unique_roots]

    mean_abs_shap_agg = np.mean(np.abs(agg_shap_values), axis=0)
    importance_aggregated = (
        pd.DataFrame({"feature": unique_roots, "mean_abs_shap": mean_abs_shap_agg})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    exp_aggregated = shap.Explanation(
        values=agg_shap_values,
        base_values=mean_base_values,
        data=agg_feature_values,
        feature_names=wrapped_aggregated_names,
    )

    plt.figure()
    shap.plots.beeswarm(
        exp_aggregated, 
        max_display=min(max_display, len(wrapped_aggregated_names)), 
        plot_size=(10, 13),
        show=False
    )

    # Renommage de la barre du bas si nécessaire
    ax = plt.gca()
    labels = [label.get_text() for label in ax.get_yticklabels()]
    n_total_agg = len(wrapped_aggregated_names)
    if n_total_agg > max_display:
        n_others_agg = n_total_agg - max_display + 1
        labels[0] = f"Sum of {n_others_agg} features"
        ax.set_yticklabels(labels)

    plt.yticks(fontsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(output_directory / "holdout_ensemble_treeshap_beeswarm_aggregated.png", dpi=300, bbox_inches="tight", transparent=transparent)
        plt.savefig(output_directory / "holdout_ensemble_treeshap_beeswarm_aggregated.pdf", bbox_inches="tight", transparent=transparent)
        importance_aggregated.to_csv(output_directory / "holdout_ensemble_treeshap_importance_aggregated.csv", index=False)

    plt.show()
    plt.close()

    return {
        "detailed": {
            "mean_shap_values": mean_shap_values,
            "feature_names": global_feature_names,
            "feature_importance": importance_detailed,
        },
        "aggregated": {
            "mean_shap_values": agg_shap_values,
            "feature_names": unique_roots,
            "feature_importance": importance_aggregated,
        },
    }

def linear_coefficients_holdout_ensemble(
    models,
    feature_names_per_model,
    savefig=True,
    folder="",
):
    """Aggregate coefficients across fold-specific linear models."""

    models = list(models)
    feature_names_per_model = [
        list(names)
        for names in feature_names_per_model
    ]

    if len(models) != len(feature_names_per_model):
        raise ValueError(
            "models and feature_names_per_model must have the same length."
        )

    global_feature_names = sorted(
        set().union(
            *[
                set(names)
                for names in feature_names_per_model
            ]
        )
    )

    feature_to_index = {
        name: index
        for index, name in enumerate(global_feature_names)
    }

    aligned_coefficients = np.zeros(
        (
            len(models),
            len(global_feature_names),
        ),
        dtype=float,
    )

    for fold_index, (model, feature_names) in enumerate(
        zip(models, feature_names_per_model)
    ):
        if not hasattr(model, "coef_"):
            raise TypeError(
                f"Fold model {fold_index + 1} does not expose coef_."
            )

        coefficients = np.asarray(
            model.coef_,
            dtype=float,
        ).reshape(-1)

        if len(coefficients) != len(feature_names):
            raise ValueError(
                f"Fold {fold_index + 1}: "
                f"{len(coefficients)} coefficients for "
                f"{len(feature_names)} features."
            )

        for coefficient, feature_name in zip(
            coefficients,
            feature_names,
        ):
            aligned_coefficients[
                fold_index,
                feature_to_index[feature_name],
            ] = coefficient

    mean_coefficients = np.mean(
        aligned_coefficients,
        axis=0,
    )

    mean_absolute_coefficients = np.mean(
        np.abs(aligned_coefficients),
        axis=0,
    )

    coefficient_std = np.std(
        aligned_coefficients,
        axis=0,
        ddof=1,
    )

    coefficient_table = (
        pd.DataFrame(
            {
                "feature": global_feature_names,
                "mean_coefficient": mean_coefficients,
                "mean_abs_coefficient": mean_absolute_coefficients,
                "std_coefficient": coefficient_std,
            }
        )
        .sort_values(
            "mean_abs_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    output_directory = Path(folder)

    if savefig:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        coefficient_table.to_csv(
            output_directory
            / "holdout_ensemble_linear_coefficients.csv",
            index=False,
        )

    return {
        "coefficients_per_model": aligned_coefficients,
        "mean_coefficients": mean_coefficients,
        "mean_abs_coefficients": mean_absolute_coefficients,
        "feature_names": global_feature_names,
        "coefficient_table": coefficient_table,
    }


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. GRAPHICAL PLOT AVEC "Sum of X features"
# -----------------------------------------------------------------------------
def plot_odds_ratios_with_others(
    csv_file: str,
    save: bool = True,
    folder: str = "",
    transparent: bool = False,
    top_n: int = 20,
    title: str = "Analyse multivariée - Odds Ratios"
):
    df = pd.read_csv(csv_file)

    # Selection of top N features by absolute importance
    df_sorted = df.sort_values(by="mean_abs_coefficient", ascending=False)
    df_top = df_sorted.head(top_n).copy()
    df_others = df_sorted.iloc[top_n:]

    # Compute OR for top N features and sort by OR ascending
    df_top['OR'] = np.exp(df_top['mean_coefficient'])
    df_top['OR_lower'] = np.exp(df_top['mean_coefficient'] - 1.96 * df_top['std_coefficient'])
    df_top['OR_upper'] = np.exp(df_top['mean_coefficient'] + 1.96 * df_top['std_coefficient'])
    df_top = df_top.sort_values(by="OR").reset_index(drop=True)

    # Aggregate remaining features into 'Sum of X features'
    if not df_others.empty:
        sum_coef_others = df_others['mean_coefficient'].sum()
        std_coef_others = np.sqrt((df_others['std_coefficient'] ** 2).sum())
        n_others = len(df_others)  # Nombre de features restantes

        others_label = f"Sum of {n_others} features"

        row_others = pd.DataFrame([{
            'feature': others_label,
            'mean_coefficient': sum_coef_others,
            'std_coefficient': std_coef_others,
        }])
        row_others['OR'] = np.exp(row_others['mean_coefficient'])
        row_others['OR_lower'] = np.exp(row_others['mean_coefficient'] - 1.96 * row_others['std_coefficient'])
        row_others['OR_upper'] = np.exp(row_others['mean_coefficient'] + 1.96 * row_others['std_coefficient'])

        # Append 'Sum of X features' at the end
        df_plot = pd.concat([df_top, row_others], ignore_index=True)
    else:
        df_plot = df_top

    fig, ax = plt.subplots(figsize=(13, 7))

    yerr_lower = (df_plot['OR'] - df_plot['OR_lower']).to_numpy()
    yerr_upper = (df_plot['OR_upper'] - df_plot['OR']).to_numpy()

    ax.axhline(y=1, color='red', linestyle='--', label='OR = 1')

    # Distinct color for the aggregated row
    colors = ['black' if not feat.startswith('Sum of ') else 'darkblue' for feat in df_plot['feature']]

    for idx in range(len(df_plot)):
        ax.errorbar(
            x=idx,
            y=df_plot.iloc[idx]['OR'],
            yerr=[[yerr_lower[idx]], [yerr_upper[idx]]],
            fmt='o',
            color=colors[idx],
            ecolor='gray',
            capsize=3
        )

    # Dummy entry for the legend
    ax.plot([], [], 'o', color='black', label='OR (95% CI)')

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot['feature'], rotation=45, ha='right')
    ax.set_ylabel('Odds Ratio')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save:
        filename = "odds_ratio_plot_with_others.png"
        save_path = os.path.join(folder, filename) if folder else filename
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, transparent=transparent, bbox_inches="tight")
        print(f"[plot_odds_ratios_with_others] Figure saved to: {save_path}", flush=True)

    plt.close()

    return fig, ax


# -----------------------------------------------------------------------------
# 2. GRAPHICAL PLOT PAR CONTRIBUTION CUMULÉE DES VARIABLES
# -----------------------------------------------------------------------------
def plot_aggregated_odds_ratios(
    csv_file: str,
    save: bool = True,
    folder: str = "",
    transparent: bool = False,
    top_n: int = 15,
    title: str = "Importance Cumulée par Variable (Odds Ratios)"
):
    df = pd.read_csv(csv_file)

    KNOWN_CATEGORICALS = ['admission_type', 'icu_ghm', 'icu_mode_entree', 'hx']
    # Extraction de la racine nettoyée (TSFEL + OHE)
    df['root_feature'] = df['feature'].apply(
        clean_feature_aggregated, 
        known_categorical_features=KNOWN_CATEGORICALS
    )

    # Agrégation par variable racine (somme des coefficients, variance combinée)
    agg_df = df.groupby('root_feature').agg(
        cum_coef=('mean_coefficient', 'sum'),
        std_coef=('std_coefficient', lambda x: np.sqrt((x**2).sum()))
    ).reset_index()

    agg_df['abs_cum_coef'] = agg_df['cum_coef'].abs()

    # Sélection du Top N
    agg_sorted = agg_df.sort_values(by="abs_cum_coef", ascending=False)
    df_top = agg_sorted.head(top_n).copy()
    df_others = agg_sorted.iloc[top_n:]

    # Calcul des Odds Ratios pour le top N
    df_top['OR'] = np.exp(df_top['cum_coef'])
    df_top['OR_lower'] = np.exp(df_top['cum_coef'] - 1.96 * df_top['std_coef'])
    df_top['OR_upper'] = np.exp(df_top['cum_coef'] + 1.96 * df_top['std_coef'])
    df_top = df_top.sort_values(by="OR").reset_index(drop=True)

    # Regroupement du reste dans "Sum of X root features"
    if not df_others.empty:
        sum_others = df_others['cum_coef'].sum()
        std_others = np.sqrt((df_others['std_coef'] ** 2).sum())
        n_others = len(df_others)

        others_label = f"Sum of {n_others} root features"

        row_others = pd.DataFrame([{
            'root_feature': others_label,
            'cum_coef': sum_others,
            'std_coef': std_others,
        }])
        row_others['OR'] = np.exp(row_others['cum_coef'])
        row_others['OR_lower'] = np.exp(row_others['cum_coef'] - 1.96 * row_others['std_coef'])
        row_others['OR_upper'] = np.exp(row_others['cum_coef'] + 1.96 * row_others['std_coef'])

        df_plot = pd.concat([df_top, row_others], ignore_index=True)
    else:
        df_plot = df_top

    fig, ax = plt.subplots(figsize=(12, 7))

    yerr_lower = (df_plot['OR'] - df_plot['OR_lower']).to_numpy()
    yerr_upper = (df_plot['OR_upper'] - df_plot['OR']).to_numpy()

    ax.axhline(y=1, color='red', linestyle='--', label='OR = 1')

    colors = ['black' if not feat.startswith('Sum of ') else 'darkblue' for feat in df_plot['root_feature']]

    for idx in range(len(df_plot)):
        ax.errorbar(
            x=idx,
            y=df_plot.iloc[idx]['OR'],
            yerr=[[yerr_lower[idx]], [yerr_upper[idx]]],
            fmt='o',
            color=colors[idx],
            ecolor='gray',
            capsize=3
        )

    ax.plot([], [], 'o', color='black', label='Aggregated OR (95% CI)')

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot['root_feature'], rotation=45, ha='right')
    ax.set_ylabel('Aggregated Odds Ratio')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save:
        filename = "odds_ratio_plot_aggregated.png"
        save_path = os.path.join(folder, filename) if folder else filename
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, transparent=transparent, bbox_inches="tight")
        print(f"[plot_aggregated_odds_ratios] Figure saved to: {save_path}", flush=True)

    plt.close()

    return fig, ax