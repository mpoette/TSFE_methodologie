"""Explainability figures for SHAP values and linear-model coefficients."""

from utilitaries.figures.output import save_figure as save_figure_file
from utilitaries.figures.feature_names import short_feature_name
from utilitaries.feature_trace import load_feature_trace

from pathlib import Path
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import os
from typing import *
import re
import textwrap
import tsfel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, matthews_corrcoef
import utilitaries.training_utils as utils
import utilitaries.features_extraction_utils as feat_utils

def _to_2d_numpy(
    X,
    *,
    allow_nan=False,
):
    """Convert a tabular matrix to a two-dimensional NumPy array.
    
    Args:
        X:
            Input matrix, potentially represented as a Pandas or Polars DataFrame.
        allow_nan:
            Whether missing values are allowed.
    
    Returns:
        numpy.ndarray:
            Two-dimensional floating-point NumPy array.
    
    Raises:
        ValueError:
            If the matrix is not two-dimensional, contains infinite values, or
            contains missing values when ``allow_nan`` is False.
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
    """Wrap a feature name over multiple lines for display.
    
    Args:
        name:
            Feature name to wrap.
        width:
            Maximum target line width.
    
    Returns:
        str:
            Wrapped feature name.
    """
    display_name = short_feature_name(name).replace("_", " ")
    return textwrap.fill(display_name, width=width)


def build_tsfel_pattern(scores_dict: Dict[str, int]) -> re.Pattern:
    """Build a compiled regular expression for TSFEL feature suffixes.
    
    Patterns are sorted by decreasing length so that longer descriptor names are
    matched before shorter overlapping names.
    
    Args:
        scores_dict:
            Mapping whose keys are TSFEL descriptor names.
    
    Returns:
        re.Pattern:
            Compiled case-insensitive regular expression matching TSFEL suffixes.
    """
    # 1. Retrieve all features from the dictionary
    features = list(scores_dict.keys())
    
    # 2. Sort by decreasing length to prevent a short pattern (e.g., 'mean')
    #    from matching before a longer pattern (e.g., 'mean absolute diff')
    features.sort(key=len, reverse=True)
    
    # 3. Escape special characters and allow spaces/underscores
    patterns = [re.escape(f).replace(r'\ ', r'[\s_]+') for f in features]
    
    # 4. Also include a few common variants not present in the dictionary
    additional_suffixes = [r'first', r'last', r'auc', r'sum']
    patterns.extend(additional_suffixes)
    
    # Global pattern: match an underscore followed by the feature name
    # and EVERYTHING that may follow (TSFEL parameters, digits, etc.)
    combined_pattern = r'_(?:' + '|'.join(patterns) + r')(?:[\s_].*)?$'
    
    return re.compile(combined_pattern, flags=re.IGNORECASE)

# Compile once for maximum performance
TSFEL_REGEX = build_tsfel_pattern(feat_utils.explainability_scores)


def _clean_feature_name(name: str) -> str:
    """Extract the root feature name by removing TSFEL suffixes.
    
    Args:
        name:
            Feature name to clean.
    
    Returns:
        str:
            Cleaned root feature name.
    """
    if not isinstance(name, str):
        return name
        
    # Clean using the TSFEL regex
    clean_name = TSFEL_REGEX.sub('', name)
    
    # Remove any remaining trailing underscores
    return clean_name.rstrip('_').strip()

def clean_feature_aggregated(name: str, known_categorical_features: Optional[List[str]] = None) -> str:
    """Clean a feature name for aggregation.
    
    TSFEL-derived suffixes are removed first, followed by optional One-Hot
    Encoding suffix cleanup.
    
    Args:
        name:
            Feature name to clean.
        known_categorical_features:
            Optional list of known categorical root feature names.
    
    Returns:
        str:
            Cleaned feature name used for aggregation.
    """
    # Step 1: TSFEL / temporal cleaning
    name_clean = _clean_feature_name(name)
    
    # Step 2: One-Hot Encoding cleaning
    name_clean = _clean_ohe_name(name_clean, known_categorical_features)
    
    return name_clean

def _clean_ohe_name(name: str, known_categorical_features: Optional[List[str]] = None) -> str:
    """Remove a One-Hot Encoding suffix from a known categorical feature.
    
    Args:
        name:
            Feature name to clean.
        known_categorical_features:
            Optional list of known categorical root feature names.
    
    Returns:
        str:
            Categorical root name when a match is found; otherwise, the original
            input name.
    """
    if not known_categorical_features:
        return name

    # Sort by decreasing length to prevent a short name from preempting a longer one
    # ex: 'type' vs 'admission_type'
    sorted_known = sorted(known_categorical_features, key=len, reverse=True)

    for cat_feat in sorted_known:
        # If the feature starts with the categorical variable followed by an underscore
        # e.g.: "admission_type_2" or "admission_type_Emergency"
        if name.startswith(f"{cat_feat}_"):
            return cat_feat

    return name



# ============================================================================
# TREE-MODEL INTERPRETABILITY
# ============================================================================

def shap_tree_holdout_ensemble(
    models,
    X_test_per_model,
    feature_names_per_model,
    model_name,
    max_display=20,
    savefig=True,
    folder="",
    transparent=False,

    output_format: str = "pdf",
):
    """Aggregate and visualize SHAP values from fold-specific tree models.

    Each model may use a different feature subset. The function aligns SHAP
    values on the union of feature names, averages them across fold models,
    and produces detailed and source-variable-level importance artifacts.

    Args:
        models: Fitted tree estimators, one per fold.
        X_test_per_model: Holdout feature matrices aligned with ``models``.
        feature_names_per_model: Feature names for each holdout matrix.
        model_name: Name used in plot titles and output artifacts.
        max_display: Maximum number of features displayed in each plot.
        savefig: Whether to save figures and CSV summaries.
        folder: Directory in which generated artifacts are stored.
        transparent: Whether saved figures use a transparent background.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A dictionary containing detailed and source-aggregated SHAP matrices,
        feature names, and importance tables.

    Raises:
        ValueError: If no models are supplied, collection lengths differ, or
            holdout matrices contain inconsistent patient counts.
    """
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
    # VERSION 1: DETAILED
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

    # plot_size increases the physical vertical height to space out Y labels
    plt.figure()
    shap.plots.beeswarm(
        exp_detailed, 
        max_display=min(max_display, len(wrapped_detailed_names)), 
        plot_size=(10, 13),  # <-- Increased height (13 instead of the default 8/10)
        show=False
    )
    plt.yticks(fontsize= 12)  # Slightly smaller font to avoid overlap
    # plt.title(f"TreeSHAP Detailed - {model_name}", fontsize=12, pad=15)
    plt.tight_layout()

    if savefig:
        save_figure_file(plt, output_directory / "holdout_ensemble_treeshap_beeswarm_detailed", output_format, bbox_inches="tight", transparent=transparent)
        importance_detailed.to_csv(output_directory / "holdout_ensemble_treeshap_importance_detailed.csv", index=False)
    
    plt.show()
    plt.close()

    # =========================================================================
    # VERSION 2: AGGREGATED BY ROOT FEATURE (Includes TSFEL + One-Hot Encoded)
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
        # Sum SHAP values across all OHE / TSFEL sub-features
        agg_shap_values[:, r_idx] = np.sum(mean_shap_values[:, col_indices], axis=1)
        # Mean actual feature values for SHAP color display
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

    # Rename the bottom bar if necessary
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
        save_figure_file(plt, output_directory / "holdout_ensemble_treeshap_beeswarm_aggregated", output_format, bbox_inches="tight", transparent=transparent)
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

# ============================================================================
# LINEAR-MODEL INTERPRETABILITY
# ============================================================================

def linear_coefficients_holdout_ensemble(
    models,
    feature_names_per_model,
    savefig=True,
    folder="",
):
    """Aggregate coefficients from fold-specific linear models.
    
    The function produces two aggregation levels. At the individual-feature
    level, coefficients are aligned across models and summarized by their mean,
    mean absolute value, and standard deviation. At the clinical-variable level,
    TSFEL and One-Hot Encoded features are grouped by their root variable and
    their cumulative absolute coefficient importance is summarized across models.
    
    Args:
        models:
            Iterable of fitted linear models exposing a ``coef_`` attribute.
        feature_names_per_model:
            Feature names corresponding to each fitted model.
        savefig:
            Whether to save the generated CSV tables.
        folder:
            Output directory.
    
    Returns:
        dict:
            Aligned coefficients, feature-level summaries, root-feature names,
            cumulative importances, and aggregated importance tables.
    
    Raises:
        ValueError:
            If model and feature-name counts are inconsistent or no model is
            provided.
        TypeError:
            If a model does not expose a ``coef_`` attribute.
    """

    # -------------------------------------------------------------------------
    # Preparation
    # -------------------------------------------------------------------------

    models = list(models)

    feature_names_per_model = [
        list(names)
        for names in feature_names_per_model
    ]

    if len(models) != len(feature_names_per_model):
        raise ValueError(
            "models and feature_names_per_model must have the same length."
        )

    if len(models) == 0:
        raise ValueError(
            "models must contain at least one fitted model."
        )

    # -------------------------------------------------------------------------
    # Global set of features present in at least one model
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Coefficient matrix:
    #
    # rows    = models
    # columns = global features
    #
    # A feature absent from a model receives a zero coefficient.
    # -------------------------------------------------------------------------

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

    # =========================================================================
    # 1. FEATURE LEVEL
    # =========================================================================

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

    coefficient_table = pd.DataFrame({
        "feature": global_feature_names,
        "mean_coefficient": mean_coefficients,
        "mean_abs_coefficient": mean_absolute_coefficients,
        "std_coefficient": coefficient_std,
    })

    # Add the individual coefficients from each model
    # to preserve full traceability.
    for model_idx in range(len(models)):
        coefficient_table[
            f"model_{model_idx + 1}_coefficient"
        ] = aligned_coefficients[model_idx]

    coefficient_table = (
        coefficient_table
        .sort_values(
            "mean_abs_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # 2. AGGREGATION BY CLINICAL VARIABLE
    # =========================================================================

    KNOWN_CATEGORICALS = [
        "admission_type",
        "icu_ghm",
        "icu_mode_entree",
        "hx",
    ]

    # Root clinical variable corresponding to each TSFEL/OHE feature
    root_features = [
        clean_feature_aggregated(
            feature_name,
            known_categorical_features=KNOWN_CATEGORICALS,
        )
        for feature_name in global_feature_names
    ]

    # Preserve order of appearance
    unique_roots = list(
        dict.fromkeys(root_features)
    )

    # Matrix:
    #
    # rows    = models
    # columns = root clinical variables
    #
    # importance(root, model) = sum of |beta| across features
    # derived from this clinical variable.
    cumulative_importance_per_model = np.zeros(
        (
            len(models),
            len(unique_roots),
        ),
        dtype=float,
    )

    n_features_per_root = np.zeros(
        len(unique_roots),
        dtype=int,
    )

    for root_idx, root in enumerate(unique_roots):

        feature_indices = [
            feature_idx
            for feature_idx, feature_root in enumerate(root_features)
            if feature_root == root
        ]

        n_features_per_root[root_idx] = len(
            feature_indices
        )

        cumulative_importance_per_model[
            :,
            root_idx,
        ] = np.sum(
            np.abs(
                aligned_coefficients[
                    :,
                    feature_indices,
                ]
            ),
            axis=1,
        )

    # -------------------------------------------------------------------------
    # Mean and variability across models
    # -------------------------------------------------------------------------

    cumulative_importance_mean = np.mean(
        cumulative_importance_per_model,
        axis=0,
    )

    cumulative_importance_std = np.std(
        cumulative_importance_per_model,
        axis=0,
        ddof=1,
    )

    cumulative_importance_table = pd.DataFrame({
        "root_feature": unique_roots,
        "n_features": n_features_per_root,
        "mean_cumulative_importance": cumulative_importance_mean,
        "std_cumulative_importance": cumulative_importance_std,
    })

    # Individual values for the five models
    for model_idx in range(len(models)):
        cumulative_importance_table[
            f"model_{model_idx + 1}_cumulative_importance"
        ] = cumulative_importance_per_model[
            model_idx
        ]

    cumulative_importance_table = (
        cumulative_importance_table
        .sort_values(
            "mean_cumulative_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # 3. SAVE
    # =========================================================================

    output_directory = Path(folder)

    if savefig:

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Detailed coefficients
        coefficient_table.to_csv(
            output_directory
            / "holdout_ensemble_linear_coefficients.csv",
            index=False,
        )

        # Cumulative importance by clinical variable
        cumulative_importance_table.to_csv(
            output_directory
            / "holdout_ensemble_linear_cumulative_importance.csv",
            index=False,
        )

    # =========================================================================
    # 4. RETURN
    # =========================================================================

    return {
        # ---------------------------------------------------------------------
        # Individual-feature level
        # ---------------------------------------------------------------------
        "coefficients_per_model":
            aligned_coefficients,

        "mean_coefficients":
            mean_coefficients,

        "mean_abs_coefficients":
            mean_absolute_coefficients,

        "std_coefficients":
            coefficient_std,

        "feature_names":
            global_feature_names,

        "coefficient_table":
            coefficient_table,

        # ---------------------------------------------------------------------
        # Aggregated clinical-variable level
        # ---------------------------------------------------------------------
        "root_feature_names":
            unique_roots,

        "cumulative_importance_per_model":
            cumulative_importance_per_model,

        "cumulative_importance_mean":
            cumulative_importance_mean,

        "cumulative_importance_std":
            cumulative_importance_std,

        "cumulative_importance_table":
            cumulative_importance_table,
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
# 1. DETAILED ODDS RATIO PLOT
# -----------------------------------------------------------------------------
# ============================================================================
# ODDS-RATIO VISUALIZATION
# ============================================================================

def plot_odds_ratios_with_others(
    csv_file: str,
    save: bool = True,
    folder: str = "",
    transparent: bool = False,
    top_n: int = 20,
    title: str = "Multivariable Analysis - Odds Ratios",
    underscore_groups_per_line: int = 3,

    output_format: str = "pdf",
):
    """Plot detailed odds ratios and confidence intervals from a CSV file.

    Args:
        csv_file: Path to the coefficient summary CSV file.
        save: Whether to save the generated figure.
        folder: Destination directory for saved figures.
        transparent: Whether saved figures use a transparent background.
        top_n: Number of highest-importance features to display.
        title: Figure title.
        underscore_groups_per_line: Number of underscore-delimited label
            groups displayed on each line.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the Matplotlib figure and axes.
    """
    df = pd.read_csv(csv_file)

    # ------------------------------------------------------------------
    # Utility function:
    # split labels every N groups separated by "_"
    #
    # Example with n=3:
    # feature_with_many_different_groups_here
    # ->
    # feature_with_many
    # different_groups_here
    # ------------------------------------------------------------------
    def wrap_label_by_underscores(label, n=3):
        """Wrap an underscore-delimited label after a fixed number of groups.

        Args:
            label: Label to wrap.
            n: Maximum number of underscore-delimited groups per line.

        Returns:
            The wrapped label.
        """
        parts = str(label).split("_")

        return "\n".join(
            "_".join(parts[i:i + n])
            for i in range(0, len(parts), n)
        )

    # Select the top N features according to their mean absolute importance
    df_plot = (
        df.sort_values(
            by="mean_abs_coefficient",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # OR computed from the mean coefficient across the 5 models
    df_plot["OR"] = np.exp(
        df_plot["mean_coefficient"]
    )

    # Descriptive inter-model dispersion:
    # exp(mean_beta ± 1.96 * SD_beta)
    #
    # WARNING:
    # this is NOT a statistical 95% confidence interval.
    df_plot["OR_lower"] = np.exp(
        df_plot["mean_coefficient"]
        - 1.96 * df_plot["std_coefficient"]
    )

    df_plot["OR_upper"] = np.exp(
        df_plot["mean_coefficient"]
        + 1.96 * df_plot["std_coefficient"]
    )

    # For the odds-ratio plot, sort by OR:
    # associations < 1 followed by > 1
    df_plot = (
        df_plot.sort_values(
            by="OR",
            ascending=True
        )
        .reset_index(drop=True)
    )

    # Slightly wider to give tick labels more room
    fig_width = max(
        13,
        0.7 * len(df_plot)
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 7)
    )

    yerr_lower = (
        df_plot["OR"]
        - df_plot["OR_lower"]
    ).to_numpy()

    yerr_upper = (
        df_plot["OR_upper"]
        - df_plot["OR"]
    ).to_numpy()

    # Reference value
    ax.axhline(
        y=1,
        color="red",
        linestyle="--",
        label="OR = 1"
    )

    # Points + inter-model dispersion
    for idx in range(len(df_plot)):
        ax.errorbar(
            x=idx,
            y=df_plot.iloc[idx]["OR"],
            yerr=[
                [yerr_lower[idx]],
                [yerr_upper[idx]]
            ],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=3
        )

    # Descriptive legend entry
    ax.plot(
        [],
        [],
        "o",
        color="black",
        label=r"OR (inter-model variability: $\pm 1.96\,SD$)"
    )

    # ------------------------------------------------------------------
    # X-axis labels
    # ------------------------------------------------------------------
    ax.set_xticks(
        range(len(df_plot))
    )

    wrapped_labels = [
        wrap_label_by_underscores(
            short_feature_name(label),
            n=underscore_groups_per_line
        )
        for label in df_plot["feature"]
    ]

    ax.set_xticklabels(
        wrapped_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Add a little extra space between ticks and labels
    ax.tick_params(
        axis="x",
        pad=4
    )

    ax.set_ylabel(
        "Odds Ratio"
    )

    if title is not None:
        ax.set_title(title)

    ax.legend(
        loc="upper left"
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save:
        filename = "odds_ratio_plot_with_others.png"

        save_path = (
            os.path.join(folder, filename)
            if folder
            else filename
        )

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        save_path = save_figure_file(plt, save_path, output_format, transparent=transparent, bbox_inches="tight")

        print(
            f"[plot_odds_ratios_with_others] "
            f"Figure saved to: {save_path}",
            flush=True
        )

    plt.close()

    return fig, ax


# -----------------------------------------------------------------------------
# 2. AGGREGATED IMPORTANCE BY CLINICAL VARIABLE
#
# Importance of a root variable =
#     sum of mean_abs_coefficient
#
# We deliberately do not compute:
#   - an aggregated OR,
#   - a pseudo-interval,
#   - an uncertainty bar.
# --------------------------------------------------------------------------------
def plot_aggregated_odds_ratios(
    csv_file: str,
    save: bool = True,
    folder: str = "",
    transparent: bool = False,
    top_n: int = 15,
    title: str = "Cumulative Importance by Clinical Variable"
,
    output_format: str = "pdf",
):
    """Plot cumulative coefficient importance by source clinical variable.

    Args:
        csv_file: Path to the aggregated coefficient summary CSV file.
        save: Whether to save the generated figure.
        folder: Destination directory for saved figures.
        transparent: Whether saved figures use a transparent background.
        top_n: Number of highest-importance variables to display.
        title: Figure title.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the Matplotlib figure and axes.
    """
    df = pd.read_csv(csv_file)

    # Top N by mean cumulative importance
    df_plot = (
        df.sort_values(
            "mean_cumulative_importance",
            ascending=False,
        )
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    x = np.arange(len(df_plot))

    y = df_plot[
        "mean_cumulative_importance"
    ].to_numpy()

    yerr = df_plot[
        "std_cumulative_importance"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=3,
        linestyle="none",
        label=r"Mean cumulative importance $\pm$ inter-model SD",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [short_feature_name(name) for name in df_plot["root_feature"]],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        r"Cumulative importance ($\sum |\beta|$)"
    )

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="upper right")

    plt.tight_layout()

    if save:
        filename = "odds_ratio_plot_aggregated.png"

        save_path = (
            os.path.join(folder, filename)
            if folder
            else filename
        )

        if folder:
            os.makedirs(folder, exist_ok=True)

        save_path = save_figure_file(plt, save_path, output_format, transparent=transparent, bbox_inches="tight")

        print(
            f"[plot_aggregated_odds_ratios] "
            f"Figure saved to: {save_path}",
            flush=True,
        )

    plt.close()

    return fig, ax





# ============================================================================
# TSFEL INTERPRETABILITY
# ============================================================================

def _extract_tsfel_descriptor(
    feature_name: str,
    root_name: str,
    scores_dict: Dict[str, int],
) -> Optional[str]:
    """Extract the TSFEL descriptor associated with a detailed feature name.
    
    Variants using spaces or underscores are accepted. Parameters occurring after
    a descriptor name remain associated with that descriptor.
    
    Args:
        feature_name:
            Detailed feature name.
        root_name:
            Root variable name associated with the feature.
        scores_dict:
            Mapping whose keys are valid TSFEL descriptor names.
    
    Returns:
        Optional[str]:
            Matching TSFEL descriptor name, or ``None`` if no descriptor is
            identified.
    """

    if not isinstance(feature_name, str) or not isinstance(root_name, str):
        return None

    # --------------------------------------------------------------
    # Remove the root variable to keep only the suffix
    # corresponding to the TSFEL descriptor.
    # --------------------------------------------------------------

    prefix = f"{root_name}_"

    if feature_name.lower().startswith(prefix.lower()):

        suffix = feature_name[len(prefix):]

    else:

        # Fallback based on the existing function.
        cleaned = _clean_feature_name(
            feature_name
        )

        if cleaned == feature_name:
            return None

        suffix = feature_name[
            len(cleaned):
        ].lstrip("_")

    if not suffix:
        return None

    # --------------------------------------------------------------
    # Longest descriptors first.
    #
    # Important:
    # for example, prevents "mean" from being detected before
    # "mean absolute diff".
    # --------------------------------------------------------------

    descriptors = sorted(
        scores_dict.keys(),
        key=len,
        reverse=True,
    )

    # Normalize spaces / underscores.
    suffix_norm = re.sub(
        r"[\s_]+",
        " ",
        suffix,
    ).strip().lower()

    # --------------------------------------------------------------
    # Search for the TSFEL descriptor
    # --------------------------------------------------------------

    for descriptor in descriptors:

        descriptor_norm = re.sub(
            r"[\s_]+",
            " ",
            str(descriptor),
        ).strip().lower()

        if (
            suffix_norm == descriptor_norm
            or suffix_norm.startswith(
                descriptor_norm + " "
            )
        ):
            return str(
                descriptor
            )

    # --------------------------------------------------------------
    # Compatibility with the additional suffixes already present
    # in build_tsfel_pattern().
    # --------------------------------------------------------------

    for descriptor in (
        "first",
        "last",
        "auc",
        "sum",
    ):

        if (
            suffix_norm == descriptor
            or suffix_norm.startswith(
                descriptor + " "
            )
        ):
            return descriptor

    return None




def _get_explainability_tier(score):
    """Convert a TSFEL explainability score into an explainability tier.
    
    Args:
        score:
            Explainability score associated with a TSFEL descriptor.
    
    Returns:
        int:
            Explainability tier from 1 to 6.
    
    Notes:
        Tier 1 corresponds to scores greater than or equal to 90; Tier 2 to
        scores from 70 to less than 90; Tier 3 to scores from 50 to less than 70;
        Tier 4 to scores from 30 to less than 50; Tier 5 to scores from 10 to
        less than 30; and Tier 6 to scores below 10.
    """

    if score >= 90:
        return 1

    elif score >= 70:
        return 2

    elif score >= 50:
        return 3

    elif score >= 30:
        return 4

    elif score >= 10:
        return 5

    else:
        return 6


def _get_active_standard_tsfel_descriptors():
    """Return the standard TSFEL descriptors used by the extraction pipeline.
    
    The function reproduces the exclusions applied by
    ``extract_tsfel_per_patient``: fractal descriptors, ``Histogram mode``, and
    ``MFCC`` are removed.
    
    Returns:
        dict:
            Mapping from normalized descriptor names to canonical TSFEL names.
    """

    config = tsfel.get_features_by_domain()

    # Same configuration as extraction
    config.pop(
        "fractal",
        None,
    )

    if "statistical" in config:
        config["statistical"].pop(
            "Histogram mode",
            None,
        )

    if "spectral" in config:
        config["spectral"].pop(
            "MFCC",
            None,
        )

    standard_descriptors = {}

    for domain_features in config.values():

        for descriptor in domain_features.keys():

            normalized = re.sub(
                r"[_\s]+",
                " ",
                str(descriptor).lower(),
            ).strip()

            standard_descriptors[
                normalized
            ] = descriptor

    return standard_descriptors

def _get_tsfel_display_group(descriptor: str) -> Optional[str]:
    """Return the display group associated with a TSFEL descriptor.
    
    The Statistical, Temporal, and Spectral groups follow the official TSFEL
    feature definitions. Wavelet descriptors are intentionally displayed in a
    separate group, while fractal descriptors are excluded.
    
    Args:
        descriptor:
            TSFEL descriptor name.
    
    Returns:
        Optional[str]:
            ``"Statistical"``, ``"Temporal"``, ``"Spectral"``, ``"Wavelet"``,
            or ``None`` when the descriptor is excluded or unknown.
    """

    if not isinstance(descriptor, str):
        return None

    name = re.sub(
        r"[\s_]+",
        " ",
        descriptor.strip().lower(),
    )

    # ==================================================================
    # WAVELET
    #
    # Subgroup deliberately separated from the official Spectral domain.
    # ==================================================================

    if name.startswith("wavelet "):
        return "Wavelet"

    # ==================================================================
    # STATISTICAL
    # ==================================================================

    statistical = {
        "absolute energy",
        "average power",
        "ecdf",
        "ecdf percentile",
        "ecdf percentile count",
        "entropy",
        "histogram mode",
        "interquartile range",
        "kurtosis",
        "max",
        "mean",
        "mean absolute deviation",
        "median",
        "median absolute deviation",
        "min",
        "peak to peak distance",
        "root mean square",
        "skewness",
        "standard deviation",
        "std",
        "variance",
    }

    # ==================================================================
    # TEMPORAL
    # ==================================================================

    temporal = {
        "area under the curve",
        "autocorrelation",
        "centroid",
        "lempel-ziv complexity",
        "mean absolute diff",
        "mean diff",
        "median absolute diff",
        "median diff",
        "negative turning points",
        "positive turning points",
        "signal distance",
        "distance",
        "slope",
        "sum absolute diff",
        "zero crossing rate",
        "neighbourhood peaks",
    }

    # ==================================================================
    # SPECTRAL
    # ==================================================================

    spectral = {
        "fft mean coefficient",
        "fft mean coeff",
        "spectrogram mean coefficient",
        "spectrogram mean coeff",
        "fundamental frequency",
        "human range energy",
        "lpcc",
        "mfcc",
        "max power spectrum",
        "maximum frequency",
        "median frequency",
        "power bandwidth",
        "spectral centroid",
        "spectral decrease",
        "spectral distance",
        "spectral entropy",
        "spectral kurtosis",
        "spectral positive turning points",
        "spectral positive turning",
        "spectral roll-off",
        "spectral roll-on",
        "spectral skewness",
        "spectral slope",
        "spectral spread",
        "spectral variation",
    }

    # ==================================================================
    # FRACTAL
    #
    # Deliberately excluded from this figure.
    # ==================================================================

    fractal = {
        "detrended fluctuation analysis",
        "dfa",
        "higuchi fractal dimension",
        "hurst exponent",
        "maximum fractal length",
        "multiscale entropy",
        "mse",
        "petrosian fractal dimension",
    }

    if name in statistical:
        return "Statistical"

    if name in temporal:
        return "Temporal"

    if name in spectral:
        return "Spectral"

    if name in fractal:
        return None

    # Unknown descriptor:
    # do not assign it to a domain arbitrarily.
    return None


def shap_tsfel_importance_matrix(
    shap_results,
    *,
    savefig=True,
    folder="",
    transparent=False,
    filename="holdout_ensemble_treeshap_tsfel_matrix",
    title=None,
    top_raw_variables=None,
    top_descriptors=None,
    min_importance=0.0,
    normalize=False,
    all_descriptors=False,
    transpose=False,
    figsize=None,
    cmap="Reds",
    show=True,
    feature_trace=None,

    output_format: str = "pdf",
):
    """Plot a TSFEL SHAP importance heatmap.
    
    The matrix relates raw variables to TSFEL descriptors. Raw variables are
    ordered by cumulative SHAP importance, while descriptors are grouped by their
    display family and ranked according to explainability and SHAP importance.
    
    Args:
        shap_results:
            SHAP results dictionary containing detailed feature-level values.
        savefig:
            Whether to save the generated figure.
        folder:
            Output directory.
        transparent:
            Whether saved figures should use a transparent background.
        filename:
            Base filename used when saving the figure.
        title:
            Optional figure title.
        top_raw_variables:
            Optional maximum number of raw variables to display.
        top_descriptors:
            Optional maximum number of TSFEL descriptors to display.
        min_importance:
            Minimum SHAP importance required for inclusion.
        normalize:
            Whether to normalize the displayed importance matrix.
        all_descriptors:
            Whether to include all active standard TSFEL descriptors, including
            descriptors absent after preprocessing.
        transpose:
            Whether to transpose the matrix.
        figsize:
            Optional Matplotlib figure size.
        cmap:
            Matplotlib colormap used for the heatmap.
        show:
            Whether to display the figure.
        feature_trace:
            Optional feature-removal mapping or JSON trace path. Removed TSFEL
            combinations receive stage-specific gray shades for zero variance,
            correlation, and Boruta filtering.
    
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        dict:
            Displayed importance matrix, names, ordering information, and related
            metadata.
    
    Raises:
        ValueError:
            If the SHAP matrix is invalid, no compatible TSFEL feature is found,
            or display limits are invalid.
    """

    # ==================================================================
    # 1. DETAILED SHAP
    # ==================================================================

    if "detailed" not in shap_results:
        raise KeyError(
            "shap_results must contain the 'detailed' key."
        )

    detailed = shap_results["detailed"]

    mean_shap_values = np.asarray(
        detailed["mean_shap_values"],
        dtype=float,
    )

    feature_names = list(
        detailed["feature_names"]
    )

    if mean_shap_values.ndim != 2:
        raise ValueError(
            "detailed['mean_shap_values'] must be a 2D matrix."
        )

    if mean_shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "The number of SHAP columns does not match "
            "the number of feature_names."
        )

    # ==================================================================
    # 2. STANDARD TSFEL REFERENCE
    # ==================================================================

    standard_tsfel_descriptors = (
        _get_active_standard_tsfel_descriptors()
    )

    if isinstance(feature_trace, Mapping):
        feature_removal_trace = {
            str(name): str(stage)
            for name, stage in feature_trace.items()
        }
    else:
        feature_removal_trace = load_feature_trace(feature_trace)

    removal_stage_priority = {
        "zero_variance": 1,
        "correlation": 2,
        "boruta": 3,
    }
    pair_removal_stages = {}
    for removed_feature, removal_stage in feature_removal_trace.items():
        root_name = _clean_feature_name(removed_feature)
        descriptor = _extract_tsfel_descriptor(
            removed_feature,
            root_name,
            feat_utils.explainability_scores,
        )
        if descriptor is None:
            continue
        normalized_descriptor = re.sub(
            r"[_\s]+", " ", str(descriptor).lower()
        ).strip()
        if normalized_descriptor not in standard_tsfel_descriptors:
            continue
        if _get_tsfel_display_group(descriptor) is None:
            continue
        pair = (descriptor, root_name)
        previous_stage = pair_removal_stages.get(pair)
        if previous_stage is None or (
            removal_stage_priority.get(removal_stage, 0)
            > removal_stage_priority.get(previous_stage, 0)
        ):
            pair_removal_stages[pair] = removal_stage

    # ==================================================================
    # 3. DETAILED IMPORTANCE
    #
    # Each individual feature:
    #
    #     mean(|SHAP|)
    # ==================================================================

    mean_abs_shap = np.mean(
        np.abs(mean_shap_values),
        axis=0,
    )

    rows = []

    for feature_name, importance in zip(
        feature_names,
        mean_abs_shap,
    ):

        # --------------------------------------------------------------
        # Raw variable
        # --------------------------------------------------------------

        root_name = _clean_feature_name(
            feature_name
        )

        # --------------------------------------------------------------
        # TSFEL descriptor
        # --------------------------------------------------------------

        descriptor = _extract_tsfel_descriptor(
            feature_name,
            root_name,
            feat_utils.explainability_scores,
        )

        if descriptor is None:
            continue

        # --------------------------------------------------------------
        # Keep only genuine standard TSFEL descriptors.
        #
        # Avoid internal aliases such as:
        #   std
        #   fft mean coeff
        #   spectrogram mean coeff
        # --------------------------------------------------------------

        normalized_descriptor = re.sub(
            r"[_\s]+",
            " ",
            str(descriptor).lower(),
        ).strip()

        if (
            normalized_descriptor
            not in standard_tsfel_descriptors
        ):
            continue

        # --------------------------------------------------------------
        # Display group
        # --------------------------------------------------------------

        group = _get_tsfel_display_group(
            descriptor
        )

        # Fractal / unknown
        if group is None:
            continue

        importance = float(
            importance
        )

        if not np.isfinite(importance):
            continue

        explainability_score = float(
            feat_utils.explainability_scores.get(
                descriptor,
                0,
            )
        )

        rows.append({
            "feature": feature_name,
            "raw_variable": root_name,
            "descriptor": descriptor,
            "group": group,
            "explainability_score": explainability_score,
            "mean_abs_shap": importance,
        })

    if not rows:
        raise ValueError(
            "No compatible TSFEL feature was identified."
        )

    long_table = pd.DataFrame(
        rows
    )

    # ==================================================================
    # 4. COMBINATIONS ACTUALLY PRESENT AFTER PREPROCESSING
    #
    # Distinguishes between:
    #
    #   absent combination              -> gray
    #   present combination + SHAP == 0 -> white
    # ==================================================================

    selected_pairs = set(
        zip(
            long_table["descriptor"],
            long_table["raw_variable"],
        )
    )

    # ==================================================================
    # 5. MATRIX
    #
    # Multiple sub-features from the same pair are summed:
    #
    #     sum(mean(|SHAP|))
    #
    # Example:
    #
    # Wavelet variance scale 1 \
    # Wavelet variance scale 2  -> same cell
    # Wavelet variance scale 3 /
    # ==================================================================

    matrix = long_table.pivot_table(
        index="descriptor",
        columns="raw_variable",
        values="mean_abs_shap",
        aggfunc="sum",
        fill_value=0.0,
    )

    traced_descriptors = list(
        dict.fromkeys(pair[0] for pair in pair_removal_stages)
    )
    traced_variables = list(
        dict.fromkeys(pair[1] for pair in pair_removal_stages)
    )
    matrix = matrix.reindex(
        index=[*matrix.index, *[name for name in traced_descriptors if name not in matrix.index]],
        columns=[*matrix.columns, *[name for name in traced_variables if name not in matrix.columns]],
        fill_value=0.0,
    )

    # ==================================================================
    # 6. ADD ALL OPTIONAL DESCRIPTORS
    # ==================================================================

    if all_descriptors:

        all_tsfel_descriptors = []

        for descriptor in (
            feat_utils.explainability_scores.keys()
        ):

            normalized_descriptor = re.sub(
                r"[_\s]+",
                " ",
                str(descriptor).lower(),
            ).strip()

            # Internal alias -> ignored
            if (
                normalized_descriptor
                not in standard_tsfel_descriptors
            ):
                continue

            group = _get_tsfel_display_group(
                descriptor
            )

            if group is None:
                continue

            all_tsfel_descriptors.append(
                descriptor
            )

        missing_descriptors = [
            descriptor
            for descriptor in all_tsfel_descriptors
            if descriptor not in matrix.index
        ]

        if missing_descriptors:

            missing_matrix = pd.DataFrame(
                0.0,
                index=missing_descriptors,
                columns=matrix.columns,
            )

            matrix = pd.concat(
                [
                    matrix,
                    missing_matrix,
                ],
                axis=0,
            )

    # ==================================================================
    # 7. OPTIONAL THRESHOLD
    # ==================================================================

    if min_importance > 0:

        matrix = matrix.mask(
            matrix < min_importance,
            0.0,
        )

    # ==================================================================
    # 8. RAW VARIABLE ORDER
    #
    # Importance:
    #
    #     sum_descripteurs mean(|SHAP|)
    #
    # No positive / negative cancellation.
    # ==================================================================

    variable_importance = (
        matrix
        .sum(axis=0)
        .sort_values(
            ascending=False
        )
    )

    ordered_columns = (
        variable_importance
        .index
        .tolist()
    )

    # ==================================================================
    # 9. TOP VARIABLES
    # ==================================================================

    if top_raw_variables is not None:

        if top_raw_variables <= 0:
            raise ValueError(
                "top_raw_variables must be greater than 0 or None."
            )

        ordered_columns = ordered_columns[
            :top_raw_variables
        ]

    matrix = matrix.loc[
        :,
        ordered_columns,
    ]

    # ==================================================================
    # 10. DESCRIPTOR IMPORTANCE
    # ==================================================================

    descriptor_importance = matrix.sum(
        axis=1
    )

    descriptor_infos = []

    for descriptor in matrix.index:

        group = _get_tsfel_display_group(
            descriptor
        )

        if group is None:
            continue

        score = float(
            feat_utils.explainability_scores.get(
                descriptor,
                0,
            )
        )

        descriptor_infos.append({
            "descriptor": descriptor,
            "group": group,
            "explainability_score": score,
            "importance": float(
                descriptor_importance.loc[
                    descriptor
                ]
            ),
        })

    descriptor_info_df = pd.DataFrame(
        descriptor_infos
    )

    # ==================================================================
    # 11. GROUP ORDER
    # ==================================================================

    group_order = [
        "Statistical",
        "Temporal",
        "Spectral",
        "Wavelet",
    ]

    ordered_descriptors = []

    for group in group_order:

        group_df = descriptor_info_df[
            descriptor_info_df["group"] == group
        ].copy()

        if group_df.empty:
            continue

        # --------------------------------------------------------------
        # Within each family:
        #
        # 1. decreasing explainability
        # 2. decreasing SHAP importance
        # 3. alphabetical name
        # --------------------------------------------------------------

        group_df = group_df.sort_values(
            by=[
                "explainability_score",
                "importance",
                "descriptor",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        ordered_descriptors.extend(
            group_df[
                "descriptor"
            ].tolist()
        )

    # ==================================================================
    # 12. TOP DESCRIPTORS
    # ==================================================================

    if top_descriptors is not None:

        if top_descriptors <= 0:
            raise ValueError(
                "top_descriptors must be greater than 0 or None."
            )

        ordered_descriptors = ordered_descriptors[
            :top_descriptors
        ]

    matrix = matrix.loc[
        ordered_descriptors
    ]

    # ==================================================================
    # 13. GROUP POSITIONS
    #
    # These positions correspond to the descriptors.
    #
    # Normal:
    #     positions on Y
    #
    # Transposed:
    #     same positions on X
    # ==================================================================

    displayed_groups = {}

    for descriptor_idx, descriptor in enumerate(
        matrix.index
    ):

        group = _get_tsfel_display_group(
            descriptor
        )

        if group is None:
            continue

        displayed_groups.setdefault(
            group,
            [],
        ).append(
            descriptor_idx
        )

    # ==================================================================
    # 14. NORMALIZATION
    # ==================================================================

    if normalize:

        total = float(
            matrix.to_numpy().sum()
        )

        if total > 0:

            matrix = (
                matrix / total
            )

    # ==================================================================
    # 15. OPTIONAL TRANSPOSITION
    #
    # Transpose AFTER establishing:
    #
    #   - variable order
    #   - descriptor order
    #   - family positions
    #
    # so that all logic remains identical.
    # ==================================================================

    if transpose:

        matrix = matrix.T

        effective_filename = (
            f"{filename}_transpose"
        )

    else:

        effective_filename = filename

    # ==================================================================
    # 16. DIMENSIONS
    # ==================================================================

    n_rows, n_cols = matrix.shape

    if figsize is None:

        if not transpose:

            fig_width = max(
                10.0,
                min(
                    24.0,
                    0.48 * n_cols + 5.0,
                ),
            )

            fig_height = max(
                7.0,
                min(
                    30.0,
                    0.38 * n_rows + 3.5,
                ),
            )

        else:

            # In transposed mode:
            #
            # more width because the descriptors
            # TSFEL are now on X.

            fig_width = max(
                12.0,
                min(
                    30.0,
                    0.42 * n_cols + 5.0,
                ),
            )

            fig_height = max(
                7.0,
                min(
                    24.0,
                    0.45 * n_rows + 4.0,
                ),
            )

        figsize = (
            fig_width,
            fig_height,
        )

    # ==================================================================
    # 17. FIGURE
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    values = matrix.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------------
    # Normal heatmap:
    #
    # SHAP = 0 -> white
    # SHAP > 0 -> red
    # --------------------------------------------------------------

    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
    )

    # ==================================================================
    # 18. MASK OF UNSELECTED COMBINATIONS
    #
    # IMPORTANT :
    #
    # Mask only a combination absent from preprocessing.
    #
    # A present combination with SHAP == 0 remains white.
    # ==================================================================

    if not transpose:

        # --------------------------------------------------------------
        # Normal:
        #
        # rows    = descriptors
        # columns = variables
        # --------------------------------------------------------------

        selected_mask = np.array(
            [
                [
                    (
                        descriptor,
                        raw_variable,
                    )
                    in selected_pairs

                    for raw_variable
                    in matrix.columns
                ]

                for descriptor
                in matrix.index
            ],
            dtype=bool,
        )

    else:

        # --------------------------------------------------------------
        # Transposed:
        #
        # rows    = variables
        # columns = descriptors
        # --------------------------------------------------------------

        selected_mask = np.array(
            [
                [
                    (
                        descriptor,
                        raw_variable,
                    )
                    in selected_pairs

                    for descriptor
                    in matrix.columns
                ]

                for raw_variable
                in matrix.index
            ],
            dtype=bool,
        )

    # --------------------------------------------------------------
    # Stage-specific gray overlay on cells without a selected feature.
    # --------------------------------------------------------------

    from matplotlib.colors import ListedColormap

    stage_codes = {
        "zero_variance": 1,
        "correlation": 2,
        "boruta": 3,
        "absent": 4,
    }

    if not transpose:
        removal_stage_matrix = np.array(
            [
                [
                    "selected" if selected_mask[row_index, column_index]
                    else pair_removal_stages.get(
                        (descriptor, raw_variable), "absent"
                    )
                    for column_index, raw_variable in enumerate(matrix.columns)
                ]
                for row_index, descriptor in enumerate(matrix.index)
            ],
            dtype=object,
        )
    else:
        removal_stage_matrix = np.array(
            [
                [
                    "selected" if selected_mask[row_index, column_index]
                    else pair_removal_stages.get(
                        (descriptor, raw_variable), "absent"
                    )
                    for column_index, descriptor in enumerate(matrix.columns)
                ]
                for row_index, raw_variable in enumerate(matrix.index)
            ],
            dtype=object,
        )

    gray_values = np.array(
        [
            [stage_codes.get(stage, 4) for stage in row]
            for row in removal_stage_matrix
        ],
        dtype=float,
    )
    gray_overlay = np.ma.masked_where(selected_mask, gray_values)

    gray_cmap = ListedColormap(
        [
            "#4d4d4d",
            "#858585",
            "#bdbdbd",
            "#e3e3e3",
        ]
    )

    ax.imshow(
        gray_overlay,
        aspect="auto",
        interpolation="nearest",
        cmap=gray_cmap,
        vmin=0.5,
        vmax=4.5,
    )

    # ==================================================================
    # 19. AXES
    # ==================================================================

    ax.set_xticks(
        np.arange(
            n_cols
        )
    )

    ax.set_yticks(
        np.arange(
            n_rows
        )
    )

    if not transpose:

        # --------------------------------------------------------------
        # NORMAL
        #
        # X = variables
        # Y = descriptors
        # --------------------------------------------------------------

        ax.set_xticklabels(
            [
                _wrap_feature_name(
                    str(name),
                    width=18,
                )
                for name in matrix.columns
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )

        ax.set_yticklabels(
            [
                _wrap_feature_name(
                    str(name),
                    width=28,
                )
                for name in matrix.index
            ],
            fontsize=9,
        )

        ax.set_xlabel(
            "Raw variables"
        )

        ax.set_ylabel(
            "Extracted TSFEL features"
        )

    else:

        # --------------------------------------------------------------
        # TRANSPOSE
        #
        # X = descriptors
        # Y = variables
        # --------------------------------------------------------------

        ax.set_xticklabels(
            [
                _wrap_feature_name(
                    str(name),
                    width=22,
                )
                for name in matrix.columns
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )

        ax.set_yticklabels(
            [
                _wrap_feature_name(
                    str(name),
                    width=24,
                )
                for name in matrix.index
            ],
            fontsize=9,
        )

        ax.set_xlabel(
            "Extracted TSFEL features"
        )

        ax.set_ylabel(
            "Raw variables"
        )

    # ==================================================================
    # 20. GRID BETWEEN CELLS
    # ==================================================================

    ax.set_xticks(
        np.arange(
            -0.5,
            n_cols,
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            n_rows,
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.25,
        alpha=0.25,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # ==================================================================
    # 21. TSFEL GROUPS
    # ==================================================================

    if not transpose:

        # --------------------------------------------------------------
        # NORMAL
        #
        # Descriptors on Y:
        #   - horizontal separators
        #   - vertical family names in the figure
        # --------------------------------------------------------------

        for group in group_order:

            if group not in displayed_groups:
                continue

            positions = displayed_groups[
                group
            ]

            first_position = min(
                positions
            )

            last_position = max(
                positions
            )

            # ----------------------------------------------------------
            # Horizontal separation
            # ----------------------------------------------------------

            if first_position > 0:

                ax.axhline(
                    y=first_position - 0.5,
                    linewidth=1.5,
                    color="black",
                    alpha=0.75,
                )

            middle_position = (
                first_position
                + last_position
            ) / 2.0

            # ----------------------------------------------------------
            # Family name
            # ----------------------------------------------------------

            ax.text(
                0.01,
                middle_position,
                group,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                rotation=90,
                clip_on=False,
            )

    else:

        # --------------------------------------------------------------
        # TRANSPOSE
        #
        # Descriptors on X:
        #   - vertical separators
        #   - horizontal family names
        # --------------------------------------------------------------

        for group in group_order:

            if group not in displayed_groups:
                continue

            positions = displayed_groups[
                group
            ]

            first_position = min(
                positions
            )

            last_position = max(
                positions
            )

            # ----------------------------------------------------------
            # Vertical separation
            # ----------------------------------------------------------

            if first_position > 0:

                ax.axvline(
                    x=first_position - 0.5,
                    linewidth=1.5,
                    color="black",
                    alpha=0.75,
                )

            middle_position = (
                first_position
                + last_position
            ) / 2.0

            # ----------------------------------------------------------
            # Transposed family name
            #
            # The text is now horizontal and placed at the top
            # of the corresponding area.
            # ----------------------------------------------------------

            ax.text(
                middle_position,
                0.01,
                group,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                rotation=0,
                clip_on=False,
            )

    # ==================================================================
    # 22. COLORBAR
    # ==================================================================

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    if normalize:

        cbar.set_label(
            "Relative cumulative SHAP importance"
        )

    else:

        cbar.set_label(
            r"Cumulative SHAP importance "
            r"($\sum \mathrm{mean}(|SHAP|)$)"
        )

    from matplotlib.patches import Patch

    removal_legend = [
        Patch(facecolor="#4d4d4d", label="Removed: zero variance"),
        Patch(facecolor="#858585", label="Removed: correlation"),
        Patch(facecolor="#bdbdbd", label="Removed: Boruta"),
        Patch(facecolor="#e3e3e3", label="Unavailable combination"),
    ]
    cbar.ax.legend(
        handles=removal_legend,
        title="Feature status",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    # ==================================================================
    # 23. TITLE
    # ==================================================================

    if title is not None:

        ax.set_title(
            title,
            pad=15,
        )

    # ==================================================================
    # 24. MARGINS
    # ==================================================================

    if not transpose:

        fig.subplots_adjust(
            left=0.27,
            bottom=0.20,
            right=0.92,
            top=0.94,
        )

    else:

        # More space at the bottom for descriptor names.
        # The top is also given slightly more space for families.

        fig.subplots_adjust(
            left=0.18,
            bottom=0.28,
            right=0.92,
            top=0.94,
        )

    # ==================================================================
    # 25. SAVE
    # ==================================================================

    output_directory = Path(
        folder
    )

    if savefig:

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_figure_file(
            fig,
            output_directory / effective_filename,
            output_format,
            bbox_inches="tight",
            transparent=transparent,
        )

        matrix.to_csv(
            output_directory
            / f"{effective_filename}.csv"
        )

        long_table.to_csv(
            output_directory
            / f"{effective_filename}_details.csv",
            index=False,
        )

        pd.DataFrame(
            removal_stage_matrix,
            index=matrix.index,
            columns=matrix.columns,
        ).to_csv(
            output_directory
            / f"{effective_filename}_removal_stages.csv"
        )

    # ==================================================================
    # 26. DISPLAY
    # ==================================================================

    if show:

        plt.show()

    else:

        plt.close(
            fig
        )

    # ==================================================================
    # 27. RETURN
    # ==================================================================

    return {
        "matrix": matrix,
        "long_table": long_table,
        "variable_importance": variable_importance,
        "descriptor_info": descriptor_info_df,
        "removal_stage_matrix": pd.DataFrame(
            removal_stage_matrix,
            index=matrix.index,
            columns=matrix.columns,
        ),
        "transposed": transpose,
        "fig": fig,
        "ax": ax,
    }
