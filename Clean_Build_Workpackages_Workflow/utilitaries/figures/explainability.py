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

import scipy.stats as st

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
    model_name: str,
    max_display: int = 20,
    savefig: bool = True,
    folder: str = "",
    transparent: bool = False,
    output_format: str = "pdf",
    aggregation_mode: str = "sum",
    precomputed_shap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate and visualize SHAP values from fold-specific tree models.

    Aligns SHAP explanations across multiple models on the union of features,
    averages them, and renders both detailed and source-variable-level beeswarm
    plots. An optional caching mechanism avoids recomputing TreeExplainer
    instances across multiple aggregation modes.

    Args:
        models: Fitted tree estimators, one per fold.
        X_test_per_model: Holdout feature matrices aligned with ``models``.
        feature_names_per_model: Feature names for each holdout matrix.
        model_name: Model identifier used for display and filenames.
        max_display: Maximum number of features displayed in beeswarm plots.
        savefig: Whether to save generated figures and CSV summaries.
        folder: Output directory path where artifacts will be saved.
        transparent: Whether saved figures use a transparent background.
        output_format: Figure output format (e.g., ``"pdf"`` or ``"png"``).
        aggregation_mode: Metric for aggregating sub-features into root variables.
            Supported modes: ``"sum"``, ``"signed_l2"``, ``"signed_max"``,
            ``"mean"``.
        precomputed_shap: Optional dictionary containing precomputed SHAP arrays
            to bypass recalculation.

    Returns:
        A dictionary containing:
            - ``"raw_fold_data"``: Unaggregated aligned SHAP arrays for caching.
            - ``"detailed"``: Detailed feature-level values and importance table.
            - ``"aggregated"``: Root-variable-level aggregated SHAP values and
              importance table.

    Raises:
        ValueError: If ``models`` is empty, inputs have mismatched lengths, or
            patient counts are inconsistent across folds.
    """
    models = list(models)
    X_test_per_model = list(X_test_per_model)
    feature_names_per_model = [list(names) for names in feature_names_per_model]
    n_models = len(models)

    if n_models == 0:
        raise ValueError("models must contain at least one fitted model.")

    if not (n_models == len(X_test_per_model) == len(feature_names_per_model)):
        raise ValueError(
            "models, X_test_per_model, and feature_names_per_model must have the same length."
        )

    # -------------------------------------------------------------------------
    # 1. COMPUTE OR RETRIEVE PRECOMPUTED SHAP VALUES
    # -------------------------------------------------------------------------
    if precomputed_shap is not None:
        mean_shap_values = precomputed_shap["mean_shap_values"]
        mean_feature_values = precomputed_shap["mean_feature_values"]
        mean_base_values = precomputed_shap["mean_base_values"]
        global_feature_names = precomputed_shap["global_feature_names"]
        n_patients = precomputed_shap["n_patients"]
    else:
        X_arrays = [_to_2d_numpy(X_fold, allow_nan=True) for X_fold in X_test_per_model]
        n_patients = X_arrays[0].shape[0]

        global_feature_names = sorted(
            set().union(*[set(names) for names in feature_names_per_model])
        )
        global_feature_to_index = {
            name: idx for idx, name in enumerate(global_feature_names)
        }

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

            aligned_shap = np.zeros(
                (n_patients, len(global_feature_names)), dtype=float
            )
            aligned_values = np.full(
                (n_patients, len(global_feature_names)), np.nan, dtype=float
            )

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
            mean_feature_values = np.nanmean(
                np.stack(aligned_values_per_model, axis=0), axis=0
            )
        mean_feature_values = np.nan_to_num(mean_feature_values, nan=0.0)
        mean_base_values = np.mean(np.stack(base_values_per_model, axis=0), axis=0)

    output_directory = Path(folder)
    if savefig:
        output_directory.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 2. VERSION 1: DETAILED FEATURE-LEVEL BEESWARM
    # =========================================================================
    mean_abs_shap_detailed = np.mean(np.abs(mean_shap_values), axis=0)
    importance_detailed = (
        pd.DataFrame(
            {
                "feature": global_feature_names,
                "mean_abs_shap": mean_abs_shap_detailed,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Render detailed beeswarm only on the initial run
    if precomputed_shap is None:
        wrapped_detailed_names = [
            _wrap_feature_name(name, width=32) for name in global_feature_names
        ]
        exp_detailed = shap.Explanation(
            values=mean_shap_values,
            base_values=mean_base_values,
            data=mean_feature_values,
            feature_names=wrapped_detailed_names,
        )

        plt.figure()
        shap.plots.beeswarm(
            exp_detailed,
            max_display=min(max_display, len(wrapped_detailed_names)),
            plot_size=(10, 13),
            show=False,
        )
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if savefig:
            save_figure_file(
                plt,
                output_directory / "holdout_ensemble_treeshap_beeswarm_detailed",
                output_format,
                bbox_inches="tight",
                transparent=transparent,
            )
            importance_detailed.to_csv(
                output_directory
                / "holdout_ensemble_treeshap_importance_detailed.csv",
                index=False,
            )

        plt.show()
        plt.close()

    # =========================================================================
    # 3. VERSION 2: AGGREGATED BY ROOT CLINICAL VARIABLE
    # =========================================================================
    known_categoricals = ["admission_type", "icu_ghm", "icu_mode_entree", "hx"]
    root_mapping = {}
    for idx, name in enumerate(global_feature_names):
        root = clean_feature_aggregated(
            name, known_categorical_features=known_categoricals
        )
        root_mapping.setdefault(root, []).append(idx)

    unique_roots = list(root_mapping.keys())
    agg_shap_values = np.zeros((n_patients, len(unique_roots)), dtype=float)
    agg_feature_values = np.zeros((n_patients, len(unique_roots)), dtype=float)

    for r_idx, (root, col_indices) in enumerate(root_mapping.items()):
        sub_shap = mean_shap_values[:, col_indices]

        if aggregation_mode == "signed_l2":
            net_sign = np.sign(np.sum(sub_shap, axis=1))
            agg_shap_values[:, r_idx] = net_sign * np.sqrt(
                np.sum(sub_shap**2, axis=1)
            )
        elif aggregation_mode == "signed_max":
            max_idx = np.argmax(np.abs(sub_shap), axis=1)
            agg_shap_values[:, r_idx] = sub_shap[np.arange(n_patients), max_idx]
        elif aggregation_mode == "mean":
            agg_shap_values[:, r_idx] = np.mean(sub_shap, axis=1)
        else:  # "sum"
            agg_shap_values[:, r_idx] = np.sum(sub_shap, axis=1)

        agg_feature_values[:, r_idx] = np.mean(
            mean_feature_values[:, col_indices], axis=1
        )

    wrapped_aggregated_names = [
        _wrap_feature_name(name, width=32) for name in unique_roots
    ]
    mean_abs_shap_agg = np.mean(np.abs(agg_shap_values), axis=0)
    importance_aggregated = (
        pd.DataFrame(
            {"feature": unique_roots, "mean_abs_shap": mean_abs_shap_agg}
        )
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
        show=False,
    )

    # Adjust truncated feature summary label
    ax = plt.gca()
    labels = [label.get_text() for label in ax.get_yticklabels()]
    n_total_agg = len(wrapped_aggregated_names)
    if n_total_agg > max_display:
        n_others_agg = n_total_agg - max_display + 1
        labels[0] = f"Sum of {n_others_agg} features"
        ax.set_yticklabels(labels)

    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Mode-specific filename export
    agg_suffix = f"_{aggregation_mode}"
    if savefig:
        save_figure_file(
            plt,
            output_directory
            / f"holdout_ensemble_treeshap_beeswarm_aggregated{agg_suffix}",
            output_format,
            bbox_inches="tight",
            transparent=transparent,
        )
        importance_aggregated.to_csv(
            output_directory
            / f"holdout_ensemble_treeshap_importance_aggregated{agg_suffix}.csv",
            index=False,
        )

    plt.show()
    plt.close()

    return {
        "raw_fold_data": {
            "mean_shap_values": mean_shap_values,
            "mean_feature_values": mean_feature_values,
            "mean_base_values": mean_base_values,
            "global_feature_names": global_feature_names,
            "n_patients": n_patients,
        },
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
        # "lempel-ziv complexity",
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


import os
import re
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def shap_tsfel_importance_matrix(
    shap_results: Dict[str, Any],
    *,
    savefig: bool = True,
    folder: str = "",
    transparent: bool = False,
    filename: str = "holdout_ensemble_treeshap_tsfel_matrix",
    metric_mode: str = "l2_norm",
    title: Optional[str] = None,
    top_raw_variables: Optional[int] = None,
    top_descriptors: Optional[int] = None,
    min_importance: float = 0.0,
    normalize: bool = False,
    all_descriptors: bool = False,
    single_gray_removed: bool = False,
    transpose: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "Reds",
    show: bool = True,
    feature_trace: Any = None,
    output_format: str = "pdf",
) -> Dict[str, Any]:
    """Plot a structured 2D SHAP importance matrix across TSFEL descriptors.

    Constructs a matrix crossing TSFEL descriptors against raw physiological
    variables. Aggregates multi-descriptor cells according to ``metric_mode``,
    overlays pipeline feature removal stages in grayscale, and groups descriptors
    into functional domains.

    Args:
        shap_results: SHAP evaluation dictionary containing the ``"detailed"``
            key.
        savefig: Whether to persist generated figures and tabular CSVs.
        folder: Target directory path for exported files.
        transparent: Whether saved figures have a transparent canvas.
        filename: Base export filename prefix.
        metric_mode: Cell aggregation function. Choices: ``"l2_norm"``,
            ``"sum"``, ``"mean"``, ``"max"``.
        title: Optional custom figure title.
        top_raw_variables: Maximum number of raw variables to display.
        top_descriptors: Maximum number of descriptors to display.
        min_importance: Minimum SHAP importance threshold for inclusion.
        normalize: Whether to normalize matrix values to sum to 1.
        all_descriptors: Whether to display all standard TSFEL descriptors.
        single_gray_removed: If True, renders all filtered features in a single
            gray tone instead of stage-specific tones.
        transpose: Whether to swap axes (raw variables vs descriptors).
        figsize: Optional Matplotlib figure size tuple.
        cmap: Colormap applied to active feature importance.
        show: Whether to render the plot interactively.
        feature_trace: Path or dictionary mapping removed features to their
            filtering stage.
        output_format: File extension for saved figures (e.g., ``"pdf"``, ``"png"``).

    Returns:
        A dictionary containing:
            - ``"matrix"``: The final pivoted DataFrame.
            - ``"long_table"``: Tidy-format DataFrame of active features.
            - ``"variable_importance"``: Series of column importance scores.
            - ``"descriptor_info"``: Metadata DataFrame for TSFEL descriptors.
            - ``"removal_stage_matrix"``: Categorical matrix of removal stages.
            - ``"transposed"``: Boolean flag indicating orientation.
            - ``"fig"``: Matplotlib Figure object.
            - ``"ax"``: Matplotlib Axes object.

    Raises:
        KeyError: If ``shap_results`` does not contain the ``"detailed"`` key.
        ValueError: If input dimensions are inconsistent or no valid TSFEL features
            are identified.
    """
    # ==================================================================
    # 1. EXTRACT DETAILED SHAP DATA
    # ==================================================================
    if "detailed" not in shap_results:
        raise KeyError("shap_results must contain the 'detailed' key.")

    detailed = shap_results["detailed"]
    mean_shap_values = np.asarray(detailed["mean_shap_values"], dtype=float)
    feature_names = list(detailed["feature_names"])

    if mean_shap_values.ndim != 2:
        raise ValueError("detailed['mean_shap_values'] must be a 2D matrix.")
    if mean_shap_values.shape[1] != len(feature_names):
        raise ValueError("The number of SHAP columns does not match feature_names.")

    # ==================================================================
    # 2. FEATURE SELECTION REMOVAL TRACE
    # ==================================================================
    standard_tsfel_descriptors = _get_active_standard_tsfel_descriptors()

    if isinstance(feature_trace, Mapping):
        feature_removal_trace = {
            str(name): str(stage) for name, stage in feature_trace.items()
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
    # 3. DETAILED IMPORTANCE TABLE
    # ==================================================================
    mean_abs_shap = np.mean(np.abs(mean_shap_values), axis=0)
    rows = []

    for feature_name, importance in zip(feature_names, mean_abs_shap):
        root_name = _clean_feature_name(feature_name)
        descriptor = _extract_tsfel_descriptor(
            feature_name,
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

        group = _get_tsfel_display_group(descriptor)
        if group is None:
            continue

        importance = float(importance)
        if not np.isfinite(importance):
            continue

        explainability_score = float(
            feat_utils.explainability_scores.get(descriptor, 0)
        )

        rows.append(
            {
                "feature": feature_name,
                "raw_variable": root_name,
                "descriptor": descriptor,
                "group": group,
                "explainability_score": explainability_score,
                "mean_abs_shap": importance,
            }
        )

    if not rows:
        raise ValueError("No compatible TSFEL feature was identified.")

    long_table = pd.DataFrame(rows)

    # ==================================================================
    # 4. ACTIVE FEATURE PAIRS
    # ==================================================================
    selected_pairs = set(zip(long_table["descriptor"], long_table["raw_variable"]))

    # ==================================================================
    # 5. MATRIX PIVOT WITH METRIC AGGREGATION
    # ==================================================================
    if metric_mode == "l2_norm":
        agg_func = lambda s: float(np.sqrt(np.sum(s**2)))
    elif metric_mode == "mean":
        agg_func = "mean"
    elif metric_mode == "max":
        agg_func = "max"
    else:  # "sum"
        agg_func = "sum"

    matrix = long_table.pivot_table(
        index="descriptor",
        columns="raw_variable",
        values="mean_abs_shap",
        aggfunc=agg_func,
        fill_value=0.0,
    )

    traced_descriptors = list(dict.fromkeys(pair[0] for pair in pair_removal_stages))
    traced_variables = list(dict.fromkeys(pair[1] for pair in pair_removal_stages))
    matrix = matrix.reindex(
        index=[
            *matrix.index,
            *[name for name in traced_descriptors if name not in matrix.index],
        ],
        columns=[
            *matrix.columns,
            *[name for name in traced_variables if name not in matrix.columns],
        ],
        fill_value=0.0,
    )

    # ==================================================================
    # 6. ALL DESCRIPTORS DISPLAY OPTION
    # ==================================================================
    if all_descriptors:
        all_tsfel_descriptors = []
        for descriptor in feat_utils.explainability_scores.keys():
            normalized_descriptor = re.sub(
                r"[_\s]+", " ", str(descriptor).lower()
            ).strip()
            if normalized_descriptor not in standard_tsfel_descriptors:
                continue
            if _get_tsfel_display_group(descriptor) is None:
                continue
            all_tsfel_descriptors.append(descriptor)

        missing_descriptors = [
            d for d in all_tsfel_descriptors if d not in matrix.index
        ]
        if missing_descriptors:
            missing_matrix = pd.DataFrame(
                0.0, index=missing_descriptors, columns=matrix.columns
            )
            matrix = pd.concat([matrix, missing_matrix], axis=0)

    # ==================================================================
    # 7. MINIMUM IMPORTANCE FILTERING
    # ==================================================================
    if min_importance > 0:
        matrix = matrix.mask(matrix < min_importance, 0.0)

    # ==================================================================
    # 8. COLUMN (VARIABLE) SORTING
    # ==================================================================
    variable_importance = matrix.sum(axis=0).sort_values(ascending=False)
    ordered_columns = variable_importance.index.tolist()

    if top_raw_variables is not None:
        if top_raw_variables <= 0:
            raise ValueError("top_raw_variables must be greater than 0 or None.")
        ordered_columns = ordered_columns[:top_raw_variables]

    matrix = matrix.loc[:, ordered_columns]

    # ==================================================================
    # 9. ROW (DESCRIPTOR) SORTING & GROUPING
    # ==================================================================
    descriptor_importance = matrix.sum(axis=1)
    descriptor_infos = []

    for descriptor in matrix.index:
        group = _get_tsfel_display_group(descriptor)
        if group is None:
            continue
        score = float(feat_utils.explainability_scores.get(descriptor, 0))
        descriptor_infos.append(
            {
                "descriptor": descriptor,
                "group": group,
                "explainability_score": score,
                "importance": float(descriptor_importance.loc[descriptor]),
            }
        )

    descriptor_info_df = pd.DataFrame(descriptor_infos)
    group_order = ["Statistical", "Temporal", "Spectral", "Wavelet"]
    ordered_descriptors = []

    for group in group_order:
        group_df = descriptor_info_df[descriptor_info_df["group"] == group].copy()
        if group_df.empty:
            continue
        group_df = group_df.sort_values(
            by=["explainability_score", "importance", "descriptor"],
            ascending=[False, False, True],
        )
        ordered_descriptors.extend(group_df["descriptor"].tolist())

    if top_descriptors is not None:
        if top_descriptors <= 0:
            raise ValueError("top_descriptors must be greater than 0 or None.")
        ordered_descriptors = ordered_descriptors[:top_descriptors]

    matrix = matrix.loc[ordered_descriptors]

    displayed_groups = {}
    for descriptor_idx, descriptor in enumerate(matrix.index):
        group = _get_tsfel_display_group(descriptor)
        if group is not None:
            displayed_groups.setdefault(group, []).append(descriptor_idx)

    if normalize:
        total = float(matrix.to_numpy().sum())
        if total > 0:
            matrix = matrix / total

    if transpose:
        matrix = matrix.T
        effective_filename = f"{filename}_transpose"
    else:
        effective_filename = filename

    # ==================================================================
    # 10. PLOT INITIALIZATION
    # ==================================================================
    n_rows, n_cols = matrix.shape

    if figsize is None:
        if not transpose:
            fig_width = max(11.0, min(26.0, 0.48 * n_cols + 6.5))
            fig_height = max(7.0, min(30.0, 0.38 * n_rows + 3.5))
        else:
            fig_width = max(13.0, min(32.0, 0.42 * n_cols + 6.5))
            fig_height = max(7.0, min(24.0, 0.45 * n_rows + 4.0))
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    values = matrix.to_numpy(dtype=float)

    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
    )

    # ==================================================================
    # 11. REMOVAL OVERLAY
    # ==================================================================
    if not transpose:
        selected_mask = np.array(
            [
                [(d, v) in selected_pairs for v in matrix.columns]
                for d in matrix.index
            ],
            dtype=bool,
        )

        removal_stage_matrix = np.array(
            [
                [
                    "selected"
                    if selected_mask[r, c]
                    else pair_removal_stages.get((d, v), "absent")
                    for c, v in enumerate(matrix.columns)
                ]
                for r, d in enumerate(matrix.index)
            ],
            dtype=object,
        )
    else:
        selected_mask = np.array(
            [
                [(d, v) in selected_pairs for d in matrix.columns]
                for v in matrix.index
            ],
            dtype=bool,
        )

        removal_stage_matrix = np.array(
            [
                [
                    "selected"
                    if selected_mask[r, c]
                    else pair_removal_stages.get((d, v), "absent")
                    for c, d in enumerate(matrix.columns)
                ]
                for r, v in enumerate(matrix.index)
            ],
            dtype=object,
        )

    if single_gray_removed:
        stage_codes = {
            "zero_variance": 1,
            "correlation": 1,
            "boruta": 1,
            "absent": 2,
        }
        gray_cmap = ListedColormap(["#a0a0a0", "#000000"])
        vmax_gray = 2.5
    else:
        stage_codes = {
            "zero_variance": 1,
            "correlation": 2,
            "boruta": 3,
            "absent": 4,
        }
        gray_cmap = ListedColormap(
            ["#d3d3d3", "#858585", "#535353", "#000000"]
        )
        vmax_gray = 4.5

    gray_values = np.array(
        [
            [stage_codes.get(stage, stage_codes["absent"]) for stage in row]
            for row in removal_stage_matrix
        ],
        dtype=float,
    )

    gray_overlay = np.ma.masked_where(selected_mask, gray_values)

    ax.imshow(
        gray_overlay,
        aspect="auto",
        interpolation="nearest",
        cmap=gray_cmap,
        vmin=0.5,
        vmax=vmax_gray,
    )

    # Axis ticks and orientation labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    if not transpose:
        ax.set_xticklabels(
            [
                _wrap_feature_name(str(name), width=18)
                for name in matrix.columns
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )
        ax.set_yticklabels(
            [_wrap_feature_name(str(name), width=28) for name in matrix.index],
            fontsize=9,
        )
        ax.set_xlabel("Raw variables")
        ax.set_ylabel("Extracted TSFEL features")
    else:
        ax.set_xticklabels(
            [
                _wrap_feature_name(str(name), width=22)
                for name in matrix.columns
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )
        ax.set_yticklabels(
            [_wrap_feature_name(str(name), width=24) for name in matrix.index],
            fontsize=9,
        )
        ax.set_xlabel("Extracted TSFEL features")
        ax.set_ylabel("Raw variables")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Domain separator rules and labels
    if not transpose:
        for group in group_order:
            if group not in displayed_groups:
                continue
            pos = displayed_groups[group]
            f_pos, l_pos = min(pos), max(pos)
            if f_pos > 0:
                ax.axhline(y=f_pos - 0.5, linewidth=1.5, color="black", alpha=0.75)
            ax.text(
                0.01,
                (f_pos + l_pos) / 2.0,
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
        for group in group_order:
            if group not in displayed_groups:
                continue
            pos = displayed_groups[group]
            f_pos, l_pos = min(pos), max(pos)
            if f_pos > 0:
                ax.axvline(x=f_pos - 0.5, linewidth=1.5, color="black", alpha=0.75)
            ax.text(
                (f_pos + l_pos) / 2.0,
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
    # 12. COLORBAR AND REMOVAL LEGEND
    # ==================================================================
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)

    if normalize:
        cbar.set_label("Relative SHAP importance")
    else:
        metric_labels = {
            "l2_norm": r"SHAP importance ($L_2$ norm: $\sqrt{\sum \mathrm{mean}(|\mathrm{SHAP}|)^2}$)",
            "mean": r"Mean SHAP importance ($\mathrm{mean}(|\mathrm{SHAP}|)$)",
            "max": r"Peak SHAP importance ($\max |\mathrm{SHAP}|)$",
            "sum": r"Cumulative SHAP importance ($\sum \mathrm{mean}(|\mathrm{SHAP}|)$)",
        }
        cbar.set_label(
            metric_labels.get(
                metric_mode,
                r"Cumulative SHAP importance ($\sum \mathrm{mean}(|\mathrm{SHAP}|)$)",
            )
        )

    if single_gray_removed:
        removal_legend = [
            Patch(facecolor="#a0a0a0", label="Removed during selection"),
        ]
    else:
        removal_legend = [
            Patch(facecolor="#d3d3d3", label="Removed: zero variance"),
            Patch(facecolor="#858585", label="Removed: correlation"),
            Patch(facecolor="#535353", label="Removed: Boruta"),
        ]

    cbar.ax.legend(
        handles=removal_legend,
        title="Feature status",
        loc="upper center",
        bbox_to_anchor=(2.0, -0.05),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        alignment="left",
    )

    if title is not None:
        ax.set_title(title, pad=15)

    if not transpose:
        fig.subplots_adjust(left=0.25, bottom=0.20, right=0.80, top=0.94)
    else:
        fig.subplots_adjust(left=0.18, bottom=0.26, right=0.80, top=0.94)

    # Save artifacts
    output_directory = Path(folder)
    if savefig:
        output_directory.mkdir(parents=True, exist_ok=True)
        save_figure_file(
            fig,
            output_directory / effective_filename,
            output_format,
            bbox_inches="tight",
            transparent=transparent,
        )
        matrix.to_csv(output_directory / f"{effective_filename}.csv")
        long_table.to_csv(
            output_directory / f"{effective_filename}_details.csv", index=False
        )
        pd.DataFrame(
            removal_stage_matrix, index=matrix.index, columns=matrix.columns
        ).to_csv(output_directory / f"{effective_filename}_removal_stages.csv")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "matrix": matrix,
        "long_table": long_table,
        "variable_importance": variable_importance,
        "descriptor_info": descriptor_info_df,
        "removal_stage_matrix": pd.DataFrame(
            removal_stage_matrix, index=matrix.index, columns=matrix.columns
        ),
        "transposed": transpose,
        "fig": fig,
        "ax": ax,
    }