"""Model, dataset, and subgroup comparison visualization utilities."""

from utilitaries.figures.output import save_figure as save_figure_file

import functools
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import polars as pl
import math
import seaborn as sns
import os
import shap
import numpy as np
from scipy.stats import gaussian_kde
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc
    )
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

import utilitaries.features_extraction_utils as feu
from pathlib import Path

import utilitaries.evaluate_utils as evaluate
from utilitaries.postprocessing_utils import (
    bootstrap_holdout_metrics,
    f1_at_fixed_threshold,
    mcc_at_fixed_threshold,
)


# ============================================================================
# SHARED PLOTTING HELPERS
# ============================================================================

from utilitaries.figures.performance import *
from utilitaries.figures.performance import _show_title

def compare_models_figure(figname, max_cols=3, savefig = False, folder = "", output_format: str = "pdf", **paths):
    """Display the same saved figure for several models in a dynamic grid.
    
    Args:
        figname:
            Name of the image file to load from each model directory.
        max_cols:
            Maximum number of columns in the comparison grid.
        savefig:
            Whether the assembled comparison figure should be saved.
        folder:
            Directory in which the comparison figure should be written.
        **paths:
            Mapping from model names to directories containing ``figname``.
    
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        ``None``. If no path is provided, the function prints an error message
        and returns immediately.
    """
    path_count = len(paths)
    
    if path_count == 0:
        print("Error: You must provide at least one path.")
        return
    
    column_count = min(path_count, max_cols)
    row_count = math.ceil(path_count / column_count)
    # Create a dynamic subplot grid
    fig, axes = plt.subplots(row_count, column_count, figsize=(5 * column_count, 5 * row_count))
    
    # For a single path, Matplotlib returns one axis instead of an array.
    # Wrap it in a list so the iteration works consistently.
    if path_count == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # Iterate over the provided directories
    for i, (model_name, dir_path) in enumerate(paths.items()):
        ax = axes[i]
        image_path = Path(dir_path) / figname        
        
        if image_path.exists():
            img = plt.imread(image_path)
            ax.imshow(img)
            ax.set_title(model_name)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"Image not found\n{model_name}", 
                    ha='center', va='center', color='red')
            ax.set_title(model_name)
            ax.axis('off')
    
    for j in range(path_count, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if savefig:
        save_name = f"comparison_{figname}"
        if not save_name.lower().endswith(".png"):
            save_name += ".png"
        save_figure_file(plt, f"{folder}/{save_name}", output_format, bbox_inches="tight")
    plt.show()

# ============================================================================
# HOLDOUT SUBGROUP METRICS
# ============================================================================

def _build_holdout_metric_functions(
    threshold=0.5,
):
    """Return metric functions compatible with bootstrap_holdout_metrics.

    Includes standard prediction metrics (AUC, AUPRC, F1, MCC, Brier) and
    calibration metrics (intercept, slope, ICI, E90, EMax).

    Args:
        threshold:
            Fixed decision threshold used for F1 and MCC computation.
            Defaults to ``0.5``.

    Returns:
        A dictionary mapping metric names to callables accepting
        ``(y_true, probabilities)``.
    """
    from sklearn.metrics import (
        brier_score_loss,
        roc_auc_score,
    )

    from utilitaries.postprocessing_utils import (
        calibration_intercept,
        calibration_slope,
        e90_score,
        # eMax_score,
        ici_score,
    )

    def _auprc(y_true, probas):
        """Compute the area under the precision-recall curve.

        Args:
            y_true: Binary target labels.
            probas: Predicted positive-class probabilities.

        Returns:
            The area under the precision-recall curve.
        """
        p, r, _ = precision_recall_curve(y_true, probas)
        return float(auc(r, p))

    return {
        "auc": roc_auc_score,
        "auprc": _auprc,
        "brier": brier_score_loss,
        "f1": functools.partial(
            f1_at_fixed_threshold,
            threshold=threshold,
        ),
        "mcc": functools.partial(
            mcc_at_fixed_threshold,
            threshold=threshold,
        ),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "ici": ici_score,
        "e90": e90_score,
        # "eMax": eMax_score,
    }


def compute_subgroup_holdout_metrics(
    probas,
    y_true,
    subgroup_labels,
    subgroup_names=None,
    n_bootstrap=2000,
    confidence_level=0.95,
    seed=42,
):
    """Compute bootstrap holdout metrics for each subgroup independently.

    Given a full set of holdout predictions and a categorical subgroup
    assignment, this function filters the predictions per subgroup and runs
    :func:`bootstrap_holdout_metrics` from ``postprocessing_utils`` on each
    subset.

    The returned list of dictionaries is directly compatible with
    :func:`generate_comparative_report` in **"holdout"** mode.

    Args:
        probas:
            Array-like of predicted probabilities for the positive class
            (length ``n_samples``).
        y_true:
            Array-like of binary ground-truth labels (length ``n_samples``).
        subgroup_labels:
            Array-like of categorical subgroup identifiers (length
            ``n_samples``).  Each unique value defines one subgroup.
        subgroup_names:
            Optional mapping from raw subgroup label to a human-readable name.
            When omitted, the string representation of each label is used.
        n_bootstrap:
            Number of bootstrap resamples per subgroup.
        confidence_level:
            Confidence level for percentile intervals (0 < p < 1).
        seed:
            Random seed for reproducibility.

    Returns:
        A list of ``(display_name, config_dict)`` tuples, one per subgroup,
        sorted alphabetically by display name.  Each ``config_dict`` contains:

        - ``"probas_holdout"``: subgroup predicted probabilities.
        - ``"y_true_holdout"``: subgroup ground-truth labels.
        - ``"bootstrap_holdout"``: dictionary returned by
          :func:`bootstrap_holdout_metrics`.
        - ``"metrics"``: point-estimate dictionary from
          :func:`compute_binary_metrics`.

    Raises:
        ValueError:
            If inputs have mismatched lengths or a subgroup contains only one
            class.
    """
    probas = np.asarray(probas, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    subgroup_labels = np.asarray(subgroup_labels, dtype=object).ravel()

    if probas.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"probas and y_true must have the same length: "
            f"{probas.shape[0]} != {y_true.shape[0]}."
        )

    if probas.shape[0] != subgroup_labels.shape[0]:
        raise ValueError(
            f"probas and subgroup_labels must have the same length: "
            f"{probas.shape[0]} != {subgroup_labels.shape[0]}."
        )

    if subgroup_names is None:
        subgroup_names = {}

    unique_labels = np.unique(subgroup_labels)
    results = []

    for label in unique_labels:
        if subgroup_names and label not in subgroup_names:
            continue
        mask = subgroup_labels == label
        sub_probas = probas[mask]
        sub_y_true = y_true[mask]

        if sub_probas.size == 0:
            continue

        display_name = subgroup_names.get(label, str(label))

        # Skip subgroups with only one class
        if np.unique(sub_y_true).size < 2:
            print(
                f"[compute_subgroup_holdout_metrics] Skipping subgroup "
                f"'{display_name}': only one class present "
                f"({len(sub_probas)} samples)."
            )
            continue

        # Point estimates via compute_binary_metrics (provides best_threshold)
        try:
            metrics = compute_binary_metrics(sub_probas, sub_y_true)
        except ValueError as exc:
            print(
                f"[compute_subgroup_holdout_metrics] Skipping subgroup "
                f"'{display_name}': {exc}"
            )
            continue

        # Use the threshold optimized on this subgroup for bootstrap metrics
        best_threshold = metrics["best_threshold"]

        # Build metric functions with the subgroup-specific threshold
        metric_functions = _build_holdout_metric_functions(
            threshold=best_threshold
        )

        # Bootstrap confidence intervals
        try:
            bootstrap_result = bootstrap_holdout_metrics(
                sub_y_true,
                sub_probas,
                metric_functions,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
        except RuntimeError as exc:
            print(
                f"[compute_subgroup_holdout_metrics] Bootstrap failed for "
                f"subgroup '{display_name}': {exc}"
            )
            continue

        config = {
            "probas_holdout": sub_probas,
            "y_true_holdout": sub_y_true,
            "bootstrap_holdout": bootstrap_result,
            "metrics": metrics,
        }

        results.append((display_name, config))

    # Sort alphabetically for deterministic output
    results.sort(key=lambda item: item[0])
    return results


def generate_subgroup_comparative_report(
    probas,
    y_true,
    subgroup_labels,
    subgroup_names=None,
    n_bootstrap=2000,
    confidence_level=0.95,
    seed=42,
    save_dir=None,
    table_format="fancy_grid",
    model_names_map=None,
    title = ...
):
    """Generate a comparative holdout report across patient subgroups."""
    subgroup_results = compute_subgroup_holdout_metrics(
        probas=probas,
        y_true=y_true,
        subgroup_labels=subgroup_labels,
        subgroup_names=subgroup_names,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    return generate_comparative_report(
        configurations=subgroup_results,
        save_dir=save_dir,
        table_format=table_format,
        evaluation_mode="holdout",
        model_names_map=model_names_map,
        title = title,
    )


# ============================================================================
# HOLDOUT COMPARISON CHARTS
# ============================================================================

def _plot_metrics_bar_group(
    ax,
    metrics,
    data_list,
    metric_display_names,
    width,
    default_colors,
    color_offset=0,
    title=None,
):
    """Plot a single group of metrics (discrimination or calibration) as grouped bars.

    Args:
        ax:
            Matplotlib Axes to draw onto.
        metrics:
            List of metric keys to display on the x-axis.
        data_list:
            List of dictionaries with keys ``"name"``, ``"estimates"``,
            ``"ci_lower"``, and ``"ci_upper"`` (all mapping metric names to
            float values).
        metric_display_names:
            Mapping from metric key to human-readable label.
        width:
            Width of a single bar.
        default_colors:
            Color palette to draw from.
        color_offset:
            Index offset into ``default_colors``.
        title:
            Optional subplot title.
    """
    x = np.arange(len(metrics))
    label_offset = width * (len(data_list) - 1) / 2 if len(data_list) > 1 else 0

    for idx, entry in enumerate(data_list):
        color_idx = (idx + color_offset) % len(default_colors)
        color = default_colors[color_idx]

        estimates = np.array([entry["estimates"][m] for m in metrics])
        ci_lower = np.array([entry["ci_lower"][m] for m in metrics])
        ci_upper = np.array([entry["ci_upper"][m] for m in metrics])

        lower_err = np.maximum(0, estimates - ci_lower)
        upper_err = np.maximum(0, ci_upper - estimates)

        ax.bar(
            x + idx * width,
            estimates,
            width,
            label=entry["name"],
            color=color,
            yerr=[lower_err, upper_err],
            capsize=3,
        )

    ax.set_xticks(x + label_offset)
    ax.set_xticklabels(
        [metric_display_names[m] for m in metrics],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(bottom=0)


def plot_holdout_metrics_barplot(
    configurations,
    save_dir=None,
    model_names_map=None,
    color_offset=0,

    output_format: str = "pdf",
):
    """Plot grouped bar charts with 95% CI error bars split into discrimination
    and calibration metrics.
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.
    """
    configurations = list(configurations)

    if not configurations:
        return

    default_names_map = {
        "IGS2": "IGS2",
        "Logistic_Regression_Lasso_TSFEL": "L1-LR",
        "SVC_TSFEL": "SVC",
        "RandomForest_TSFEL": "Random Forest",
        "XGBoost_TSFEL": "XGBoost",
        "InceptionTimeModified": "Inception Time",
        "LstmTimeModified": "LSTM",
        "Transformer Encoder": "Vanilla Transformer",
    }
    names_map = default_names_map.copy()
    if model_names_map is not None:
        names_map.update(model_names_map)

    discrimination_metrics = ["auc", "auprc", "f1", "mcc"]
    calibration_metrics = [
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "ici",
        "e90",
    ]

    metric_display_names = {
        "auc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1-Score",
        "mcc": "MCC",
        "brier": "Brier",
        "calibration_intercept": "|Intercept|",
        "calibration_slope": "|Slope - 1|",
        "ici": "ICI",
        "e90": "E90",
    }

    all_metrics = discrimination_metrics + calibration_metrics
    model_data = []

    for name, config in configurations:

        bootstrap = config.get("bootstrap_holdout")
        if bootstrap is None:
            continue

        display_name = names_map.get(name, name)
        summary = bootstrap.get("summary", {})
        data = {
            "name": display_name,
            "estimates": {},
            "ci_lower": {},
            "ci_upper": {},
        }

        for metric in all_metrics:
            entry = summary.get(metric, {})
            est = float(entry.get("estimate", 0.0))
            low = float(entry.get("ci_lower", 0.0))
            upp = float(entry.get("ci_upper", 0.0))

            if metric == "calibration_intercept":
                est = abs(est)
                low, upp = min(abs(low), abs(upp)), max(abs(low), abs(upp))
            elif metric == "calibration_slope":
                est = abs(est - 1.0)
                d1, d2 = abs(low - 1.0), abs(upp - 1.0)
                low, upp = min(d1, d2), max(d1, d2)

            data["estimates"][metric] = est
            data["ci_lower"][metric] = low
            data["ci_upper"][metric] = upp

        model_data.append(data)

    if not model_data:
        return

    n_models = len(model_data)
    default_colors = sns.color_palette("tab10", 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")

    width = 0.8 / max(n_models, 1)

    _plot_metrics_bar_group(
        ax=ax1,
        metrics=discrimination_metrics,
        data_list=model_data,
        metric_display_names=metric_display_names,
        width=width,
        default_colors=default_colors,
        color_offset=color_offset,
    )
    _plot_metrics_bar_group(
        ax=ax2,
        metrics=calibration_metrics,
        data_list=model_data,
        metric_display_names=metric_display_names,
        width=width,
        default_colors=default_colors,
        color_offset=color_offset,
    )

    ax1.set_ylabel("Score (Higher is better)")
    ax2.set_ylabel("Distance / Error (Lower is better)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(n_models, 5),
        fontsize=10,
    )

    if save_dir is not None:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_figure_file(fig, output_path / "holdout_metrics_barplot.png", output_format, bbox_inches="tight")

    return fig

def plot_dataset_comparison_barplot(
    model_name,
    dataset_configurations,
    save_dir=None,
    title=...,

    output_format: str = "pdf",
):
    """For a single model, plot grouped bar charts with 95% CI error bars split
    into discrimination and calibration metrics, comparing datasets.
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.
    """
    dataset_configurations = list(dataset_configurations)

    if not dataset_configurations:
        return

    display_model_name = model_name

    discrimination_metrics = ["auc", "auprc", "f1", "mcc"]
    calibration_metrics = [
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "ici",
        "e90",
    ]

    metric_display_names = {
        "auc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1-Score",
        "mcc": "MCC",
        "brier": "Brier",
        "calibration_intercept": "|Intercept|",
        "calibration_slope": "|Slope - 1|",
        "ici": "ICI",
        "e90": "E90",
    }

    all_metrics = discrimination_metrics + calibration_metrics

    dataset_data = []

    for ds_name, config in dataset_configurations:
        bootstrap = config.get("bootstrap_holdout")
        if bootstrap is None:
            continue

        summary = bootstrap.get("summary", {})
        data = {
            "name": ds_name,
            "estimates": {},
            "ci_lower": {},
            "ci_upper": {},
        }

        for metric in all_metrics:
            entry = summary.get(metric, {})
            est = float(entry.get("estimate", 0.0))
            low = float(entry.get("ci_lower", 0.0))
            upp = float(entry.get("ci_upper", 0.0))

            if metric == "calibration_intercept":
                est = abs(est)
                low, upp = min(abs(low), abs(upp)), max(abs(low), abs(upp))
            elif metric == "calibration_slope":
                est = abs(est - 1.0)
                d1, d2 = abs(low - 1.0), abs(upp - 1.0)
                low, upp = min(d1, d2), max(d1, d2)

            data["estimates"][metric] = est
            data["ci_lower"][metric] = low
            data["ci_upper"][metric] = upp

        dataset_data.append(data)

    if not dataset_data:
        return

    n_datasets = len(dataset_data)
    default_colors = sns.color_palette("tab10", max(n_datasets, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")

    width = 0.8 / max(n_datasets, 1)

    _plot_metrics_bar_group(
        ax=ax1,
        metrics=discrimination_metrics,
        data_list=dataset_data,
        metric_display_names=metric_display_names,
        width=width,
        default_colors=default_colors,
    )
    _plot_metrics_bar_group(
        ax=ax2,
        metrics=calibration_metrics,
        data_list=dataset_data,
        metric_display_names=metric_display_names,
        width=width,
        default_colors=default_colors,
    )

    ax1.set_ylabel("Score (Higher is better)")
    ax2.set_ylabel("Distance / Error (Lower is better)")

    _show_title(title, f"Dataset Comparison - {display_model_name}", ax1)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(n_datasets, 5),
        fontsize=10,
    )

    if save_dir is not None:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-", " ") else "_"
            for c in display_model_name
        )
        save_figure_file(fig, output_path / f"{safe_name}_dataset_comparison.png", output_format, bbox_inches="tight")

    return fig

def old_generate_comparative_report(
    configurations,
    y_true_base=None,
    save_dir=None,
    table_format="fancy_grid",
    evaluation_mode="oof",
    model_names_map=None,
    title=...,
    color_offset = 0,

    output_format: str = "pdf",
):
    """Generate comparison tables and collective OOF or Holdout figures.

    The summary table contains fold-level ``mean ± std`` values for OOF mode,
    or bootstrap ``estimate [CI95%]`` values for holdout mode.
    ROC, precision-recall, and calibration figures use pooled predictions
    and the corresponding stored metrics.

    Args:
        configurations:
            Iterable of ``(model_name, all_results)`` pairs.
        y_true_base:
            Optional common labels. When omitted, labels are read from each
            configuration (``y_true_oof`` or ``y_true_holdout``).
        save_dir:
            Optional output directory.
        table_format:
            Console format passed to :func:`tabulate`.
        evaluation_mode:
            Either ``"oof"`` to use out-of-fold results (default) or
            ``"holdout"`` to use independent holdout results.
        model_names_map:
            Optional dictionary mapping technical model names to formatted
            display names (e.g. {"Logistic_Regression_Lasso_TSFEL": "L1-LR"}).
        title:
            Optional figure title prefix. Omit (default ``...``) to keep the
            automatic titles, pass ``None`` to remove them, or provide a custom
            string prepended to each subtitle.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A pandas DataFrame containing formatted metric values.
    """
    if evaluation_mode not in ("oof", "holdout"):
        raise ValueError(
            f"evaluation_mode must be 'oof' or 'holdout', got {evaluation_mode!r}."
        )

    configurations = list(configurations)

    # ---- Display Names Mapping ----
    # Default mapping when no explicit dictionary is provided
    default_names_map = {
        "IGS2": "IGS2",
        "Logistic_Regression_Lasso_TSFEL": "L1-LR",  # or "Logistic Regression"
        "SVC_TSFEL": "SVC",
        "RandomForest_TSFEL": "Random Forest",
        "XGBoost_TSFEL": "XGBoost",
        "InceptionTimeModified": "Inception Time",
        "LstmTimeModified": "LSTM",
        "Transformer Encoder": "Vanilla Transformer",
    }
    
    # Merge any user-provided overrides
    names_map = default_names_map.copy()
    if model_names_map is not None:
        names_map.update(model_names_map)

    predefined_order = [
        "IGS2",
        "Logistic_Regression_Lasso_TSFEL",
        "SVC_TSFEL",
        "RandomForest_TSFEL",
        "XGBoost_TSFEL",
        "InceptionTimeModified",
        "LstmTimeModified",
    ]

    def get_sort_key(item):
        """Build a stable model-ordering key.

        Args:
            item: A ``(model_name, configuration)`` pair.

        Returns:
            A tuple that places predefined models first and sorts all others
            alphabetically.
        """
        model_name = item[0]
        if model_name in predefined_order:
            return 0, predefined_order.index(model_name)
        return 1, model_name

    def require_key(config, key, model_name):
        """Retrieve a required result value with model-specific diagnostics.

        Args:
            config: Model result mapping.
            key: Required mapping key.
            model_name: Model name included in error messages.

        Returns:
            The value stored under ``key``.

        Raises:
            KeyError: If ``key`` is absent from ``config``.
        """
        if key not in config:
            raise KeyError(
                f"{model_name}: required result key '{key}' is missing."
            )
        return config[key]

    # ---- Key mapping depending on mode ----
    if evaluation_mode == "oof":
        y_true_key = "y_true_oof"
        probas_key = "probas_oof"
        metric_suffix = ""  # e.g. "auc_oof", "auc_mean"
    else:
        y_true_key = "y_true_holdout"
        probas_key = "probas_holdout"
        metric_suffix = "_holdout"  # not used for bootstrap, see below

    def format_metric(config, metric_name, model_name):
        """Format metric value: mean±std for OOF, estimate[CI] for holdout."""
        if evaluation_mode == "oof":
            mean_value = float(
                require_key(config, f"{metric_name}_mean", model_name)
            )
            std_value = float(
                require_key(config, f"{metric_name}_std", model_name)
            )
            return f"{mean_value:.3f} ± {std_value:.3f}"
        else:
            bootstrap = require_key(config, "bootstrap_holdout", model_name)
            summary = bootstrap["summary"]
            entry = summary[metric_name]
            estimate = float(entry["estimate"])
            ci_lower = float(entry["ci_lower"])
            ci_upper = float(entry["ci_upper"])
            return f"{estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

    configurations = sorted(configurations, key=get_sort_key)
    default_colors = sns.color_palette(
        "tab10",
        n_colors=max(len(configurations), 10),
    )

    results = {}
    plot_data_list = []
    for idx, (name, config) in enumerate(configurations):
        # Retrieve the normalized display name
        display_name = names_map.get(name, name)

        probas = np.asarray(
            require_key(config, probas_key, name),
            dtype=float,
        ).ravel()

        current_y = (
            y_true_base
            if y_true_base is not None
            else require_key(config, y_true_key, name)
        )

        y_true = np.asarray(current_y, dtype=int).ravel()

        if probas.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"{name}: probas and y_true have different "
                f"lengths: {len(probas)} != {len(y_true)}."
            )

        color = config.get(
            "color",
            default_colors[idx % len(default_colors)],
        )

        constant_predictions = np.all(probas == probas[0])

        if constant_predictions:
            prevalence = float(np.mean(y_true))
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            precision = np.array([1.0, prevalence, prevalence])
            recall = np.array([0.0, 0.0, 1.0])
            fop = np.array([prevalence])
            mpv = np.array([float(probas[0])])
        else:
            fpr, tpr, _ = roc_curve(y_true, probas)
            precision, recall, _ = precision_recall_curve(y_true, probas)
            fop, mpv = calibration_curve(
                y_true, probas, n_bins=10, strategy="uniform"
            )

        # ---- Get AUC/AUPRC for labels ----
        if evaluation_mode == "oof":
            auc_val = float(require_key(config, "auc_oof", name))
            auprc_val = float(require_key(config, "auprc_oof", name))
            intercept_val = float(require_key(config, "calibration_intercept_oof", name))
            slope_val = float(require_key(config, "calibration_slope_oof", name))
            ici_val = float(require_key(config, "ici_oof", name))
        else:
            bootstrap = require_key(config, "bootstrap_holdout", name)
            summary = bootstrap["summary"]
            auc_val = float(summary["auc"]["estimate"])
            auprc_val = float(summary["auprc"]["estimate"])
            cal_stats = get_calibration_stats(probas, y_true)
            intercept_val = cal_stats["intercept"]
            slope_val = cal_stats["slope"]
            ici_val = cal_stats["ici"]

        # ---- Results table ----
        metrics_dict = {
            "AUROC": format_metric(config, "auc", name),
            "AUPRC": format_metric(config, "auprc", name),
            "F1-Score": format_metric(config, "f1_score" if evaluation_mode == "oof" else "f1", name),
            "MCC": format_metric(config, "mcc", name),
            "Brier": format_metric(config, "brier", name),
            "Intercept": format_metric(config, "calibration_intercept", name),
            "Slope": format_metric(config, "calibration_slope", name),
            "ICI": format_metric(config, "ici", name),
            "E90": format_metric(config, "e90", name),
        }

        # Use display_name as the key in the final table
        results[display_name] = metrics_dict

        label_prefix = "OOF" if evaluation_mode == "oof" else "Holdout"
        prevalence = float(np.mean(y_true))
        plot_data_list.append({
            "name": display_name,  # Use the display name in plot legends
            "color": color,
            "fpr": fpr,
            "tpr": tpr,
            "auc_val": auc_val,
            "recall": recall,
            "precision": precision,
            "auprc_val": auprc_val,
            "fop": fop,
            "mpv": mpv,
            "prevalence" : prevalence,
            "intercept_val": intercept_val,
            "slope_val": slope_val,
            "ici_val": ici_val,
            "label_prefix": label_prefix,
        })

    # ---- Mode label (used for titles) ----
    mode_label = "OOF" if evaluation_mode == "oof" else "Holdout"

    # ---- Figures ----
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        layout="constrained",
    )

    plot_comparative_axes(
        plot_data_list=plot_data_list,
        axes=axes,
        show_legend=True,
    )

    if title is not None:
        fig.suptitle(
            f"{title} — {mode_label} comparison",
            fontsize=14,
            fontweight="bold",
        )

    # ---- Table ----
    results_df = pd.DataFrame(results).T
    print(f"\n=== {mode_label} PERFORMANCE COMPARISON TABLE ===")
    print(tabulate(results_df, headers="keys", tablefmt=table_format, showindex=True))

    if save_dir is not None:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_figure_file(
            fig,
            output_path / f"collective_performance_{mode_label.lower()}",
            output_format,
            bbox_inches="tight",
        )

        with open(
            output_path / f"results_table_{mode_label.lower()}.tex",
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(
                tabulate(
                    results_df,
                    headers="keys",
                    tablefmt="latex_booktabs",
                    showindex=True,
                )
            )

    # ---- Holdout barplot with CI ----
    fig_bar = None
    if evaluation_mode == "holdout":
        fig_bar = plot_holdout_metrics_barplot(
            configurations=configurations,
            save_dir=save_dir,
            model_names_map = model_names_map,
            color_offset = color_offset
        , output_format=output_format)

    plt.show()
    plt.close(fig)

    if fig_bar is not None:
        plt.close(fig_bar)
    
    return results_df


# ============================================================================
# CURRENT COMPARATIVE REPORTING
# ============================================================================

def plot_comparative_axes(
    plot_data_list,
    axes,
    show_legend=False,
):
    """Draw ROC, precision-recall, and calibration data on existing axes.

    Args:
        plot_data_list: Prepared model or subgroup curve data. Each item must
            contain curve coordinates, prevalence, display name, and color.
        axes: Three Matplotlib axes ordered as ROC, precision-recall, and
            calibration axes.
        show_legend: Whether to display the shared ROC legend.

    Returns:
        None.
    """
    ax_roc, ax_prc, ax_cal = axes

    for item in plot_data_list:
        name = item["name"]
        color = item["color"]

        # ROC
        ax_roc.plot(
            item["fpr"],
            item["tpr"],
            color=color,
            linewidth=1.8,
            label=name,
        )

        # PRC
        ax_prc.plot(
            item["recall"],
            item["precision"],
            color=color,
            linewidth=1.8,
        )

        ax_prc.axhline(
            y=item["prevalence"],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        # Calibration
        ax_cal.plot(
            item["mpv"],
            item["fop"],
            marker="o",
            markersize=3.5,
            color=color,
            linewidth=1.8,
        )

    # Chance ROC
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.0,
    )

    # Perfect calibration
    ax_cal.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        color="black",
        linewidth=1.0,
    )

    ax_roc.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )

    ax_prc.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )

    ax_cal.set(
        xlabel="Mean Predicted Probability",
        ylabel="Observed Fraction of Positives",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )

    for ax in axes:
        ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            alpha=0.45,
        )

    # Use one legend to map colors to subgroups
    # only in the ROC/AUC column
    if show_legend:
        ax_roc.legend(
            loc="lower right",
            fontsize=8,
            frameon=False,
            handlelength=2.2,
        )

def generate_comparative_report(
    configurations,
    y_true_base=None,
    save_dir=None,
    table_format="fancy_grid",
    evaluation_mode="oof",
    model_names_map=None,
    title=...,
    color_offset=0,
    axes=None,
    show=True,

    output_format: str = "pdf",
):
    """Generate comparative metrics, tables, and curves for several models.

    The function supports pooled out-of-fold results and independent holdout
    results. It can create its own figure or render into three caller-provided
    axes, which is used by subgroup comparison grids.

    Args:
        configurations: Iterable of ``(model_name, results)`` pairs.
        y_true_base: Optional target array shared by all configurations.
        save_dir: Optional destination directory for tables and figures.
        table_format: Tabulate output format used for console tables.
        evaluation_mode: Evaluation source, either ``"oof"`` or ``"holdout"``.
        model_names_map: Optional mapping from internal model names to display
            names.
        title: Optional overall figure title. An ellipsis requests the default
            title and ``None`` disables it.
        color_offset: Offset applied to the default color palette.
        axes: Optional existing ROC, precision-recall, and calibration axes.
        show: Whether to display a figure created by this function.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A DataFrame containing the formatted comparative metrics.

    Raises:
        ValueError: If the evaluation mode or supplied arrays are invalid.
        KeyError: If a required result field is missing.
    """
    if evaluation_mode not in ("oof", "holdout"):
        raise ValueError(
            f"evaluation_mode must be 'oof' or 'holdout', "
            f"got {evaluation_mode!r}."
        )

    configurations = list(configurations)

    default_names_map = {
        "IGS2": "IGS2",
        "Logistic_Regression_Lasso_TSFEL": "L1-LR",
        "SVC_TSFEL": "SVC",
        "RandomForest_TSFEL": "Random Forest",
        "XGBoost_TSFEL": "XGBoost",
        "InceptionTimeModified": "Inception Time",
        "LstmTimeModified": "LSTM",
        "Transformer Encoder": "Vanilla Transformer",
    }

    names_map = default_names_map.copy()

    if model_names_map is not None:
        names_map.update(model_names_map)

    predefined_order = [
        "IGS2",
        "Logistic_Regression_Lasso_TSFEL",
        "SVC_TSFEL",
        "RandomForest_TSFEL",
        "XGBoost_TSFEL",
        "InceptionTimeModified",
        "LstmTimeModified",
    ]

    def get_sort_key(item):
        """Build a stable model-ordering key.

        Args:
            item: A ``(model_name, configuration)`` pair.

        Returns:
            A tuple that places predefined models first and sorts all others
            alphabetically.
        """
        model_name = item[0]

        if model_name in predefined_order:
            return 0, predefined_order.index(model_name)

        return 1, model_name

    def require_key(config, key, model_name):
        """Retrieve a required model result.

        Args:
            config: Model result mapping.
            key: Required mapping key.
            model_name: Model name included in error messages.

        Returns:
            The value stored under ``key``.

        Raises:
            KeyError: If ``key`` is absent from ``config``.
        """
        if key not in config:
            raise KeyError(
                f"{model_name}: required result key '{key}' is missing."
            )

        return config[key]

    if evaluation_mode == "oof":
        y_true_key = "y_true_oof"
        probas_key = "probas_oof"
    else:
        y_true_key = "y_true_holdout"
        probas_key = "probas_holdout"

    def format_metric(config, metric_name, model_name):
        """Format one metric for the selected evaluation mode.

        Args:
            config: Model result mapping.
            metric_name: Base metric name.
            model_name: Model name included in missing-key diagnostics.

        Returns:
            A mean-and-standard-deviation string for OOF evaluation or an
            estimate-and-confidence-interval string for holdout evaluation.
        """
        if evaluation_mode == "oof":
            mean_value = float(
                require_key(
                    config,
                    f"{metric_name}_mean",
                    model_name,
                )
            )

            std_value = float(
                require_key(
                    config,
                    f"{metric_name}_std",
                    model_name,
                )
            )

            return f"{mean_value:.3f} ± {std_value:.3f}"

        bootstrap = require_key(
            config,
            "bootstrap_holdout",
            model_name,
        )

        summary = bootstrap["summary"]
        entry = summary[metric_name]

        estimate = float(entry["estimate"])
        ci_lower = float(entry["ci_lower"])
        ci_upper = float(entry["ci_upper"])

        return f"{estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

    configurations = sorted(
        configurations,
        key=get_sort_key,
    )

    default_colors = sns.color_palette(
        "tab10",
        n_colors=max(len(configurations), 10),
    )

    results = {}
    plot_data_list = []
    label_prefix = "OOF" if evaluation_mode == "oof" else "Holdout"

    for idx, (name, config) in enumerate(configurations):
        display_name = names_map.get(name, name)

        probas = np.asarray(
            require_key(config, probas_key, name),
            dtype=float,
        ).ravel()

        current_y = (
            y_true_base
            if y_true_base is not None
            else require_key(config, y_true_key, name)
        )

        y_true = np.asarray(current_y, dtype=int).ravel()

        if probas.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"{name}: probas and y_true have different lengths: "
                f"{len(probas)} != {len(y_true)}."
            )

        prevalence = float(np.mean(y_true))

        constant_predictions = np.all(probas == probas[0])

        if constant_predictions:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])

            precision = np.array([
                1.0,
                prevalence,
                prevalence,
            ])

            recall = np.array([
                0.0,
                0.0,
                1.0,
            ])

            fop = np.array([prevalence])
            mpv = np.array([float(probas[0])])

        else:
            fpr, tpr, _ = roc_curve(
                y_true,
                probas,
            )

            precision, recall, _ = precision_recall_curve(
                y_true,
                probas,
            )

            fop, mpv = calibration_curve(
                y_true,
                probas,
                n_bins=10,
                strategy="uniform",
            )

        if evaluation_mode == "oof":
            auc_val = float(
                require_key(config, "auc_oof", name)
            )

            auprc_val = float(
                require_key(config, "auprc_oof", name)
            )

            intercept_val = float(
                require_key(
                    config,
                    "calibration_intercept_oof",
                    name,
                )
            )

            slope_val = float(
                require_key(
                    config,
                    "calibration_slope_oof",
                    name,
                )
            )

            ici_val = float(
                require_key(config, "ici_oof", name)
            )

        else:
            bootstrap = require_key(
                config,
                "bootstrap_holdout",
                name,
            )

            summary = bootstrap["summary"]

            auc_val = float(
                summary["auc"]["estimate"]
            )

            auprc_val = float(
                summary["auprc"]["estimate"]
            )

            cal_stats = get_calibration_stats(
                probas,
                y_true,
            )

            intercept_val = float(
                cal_stats["intercept"]
            )

            slope_val = float(
                cal_stats["slope"]
            )

            ici_val = float(
                cal_stats["ici"]
            )

        metrics_dict = {
            "AUROC": format_metric(
                config,
                "auc",
                name,
            ),
            "AUPRC": format_metric(
                config,
                "auprc",
                name,
            ),
            "F1-Score": format_metric(
                config,
                "f1_score"
                if evaluation_mode == "oof"
                else "f1",
                name,
            ),
            "MCC": format_metric(
                config,
                "mcc",
                name,
            ),
            "Brier": format_metric(
                config,
                "brier",
                name,
            ),
            "Intercept": format_metric(
                config,
                "calibration_intercept",
                name,
            ),
            "Slope": format_metric(
                config,
                "calibration_slope",
                name,
            ),
            "ICI": format_metric(
                config,
                "ici",
                name,
            ),
            "E90": format_metric(
                config,
                "e90",
                name,
            ),
        }

        results[display_name] = metrics_dict

        color = config.get(
            "color",
            default_colors[
                (idx + color_offset) % len(default_colors)
            ],
        )

        plot_data_list.append({
            "name": display_name,
            "color": color,
            "fpr": fpr,
            "tpr": tpr,
            "auc_val": auc_val,
            "recall": recall,
            "precision": precision,
            "auprc_val": auprc_val,
            "fop": fop,
            "mpv": mpv,
            "prevalence": prevalence,
            "intercept_val": intercept_val,
            "slope_val": slope_val,
            "ici_val": ici_val,
            "label_prefix": label_prefix,
        })

    results_df = pd.DataFrame(results).T

    print(
        f"\n=== {label_prefix} PERFORMANCE "
        "COMPARISON TABLE ==="
    )

    print(
        tabulate(
            results_df,
            headers="keys",
            tablefmt=table_format,
            showindex=True,
        )
    )

    owns_figure = axes is None

    if owns_figure:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
            layout="constrained",
        )
    else:
        fig = axes[0].figure

    plot_comparative_axes(
        plot_data_list=plot_data_list,
        axes=axes,
        show_legend=owns_figure,
    )

    if owns_figure:
        if title is ...:
            fig.suptitle(
                f"{label_prefix} Performance Comparison",
                fontsize=14,
                fontweight="bold",
            )
        elif title is not None:
            fig.suptitle(
                f"{title} — {label_prefix}",
                fontsize=14,
                fontweight="bold",
            )

    fig_bar = None

    if save_dir is not None and owns_figure:
        output_path = Path(save_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_figure_file(
            fig,
            output_path / f"collective_performance_{label_prefix.lower()}",
            output_format,
            bbox_inches="tight",
        )

        with open(
            output_path
            / f"results_table_{label_prefix.lower()}.tex",
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(
                tabulate(
                    results_df,
                    headers="keys",
                    tablefmt="latex_booktabs",
                    showindex=True,
                )
            )

    if (
        evaluation_mode == "holdout"
        and owns_figure
    ):
        fig_bar = plot_holdout_metrics_barplot(
            configurations=configurations,
            save_dir=save_dir,
            model_names_map=model_names_map,
            color_offset=color_offset,
        output_format=output_format)

    if owns_figure and show:
        plt.show()

    if owns_figure:
        plt.close(fig)

    if fig_bar is not None:
        plt.close(fig_bar)

    return results_df
# =============================================================================
# SUBGROUP COMPARISON GRIDS
# =============================================================================

def plot_subgroup_comparative_grid(
    subgroup_reports,
    model_title=None,
    model_names_map=None,
    save_path=None,
    figsize_per_row = (14, 4.5),
    figsize=(16, 10),
    dpi=600,
    show=True,

    output_format: str = "pdf",
):
    """Display subgroup performance as a transposed grid.

    - Rows (3): ROC, Precision-Recall, Calibration curves.
    - Columns (N): Subgroup categories (Age, Sex, Admission, Condition, etc.).
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.
    """
    if not subgroup_reports:
        raise ValueError("subgroup_reports cannot be empty.")

    subgroup_reports = dict(subgroup_reports)
    col_names = list(subgroup_reports.keys())
    n_cols = len(col_names)
    n_rows = 3  # ROC, PRC, Calibration

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )

    row_titles = [
        "ROC curves",
        "Precision–recall curves",
        "Calibration curves",
    ]

    # Use panel_letters to index top-to-bottom or left-to-right
    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")

    for col_idx, (subgroup_name, configurations) in enumerate(
        subgroup_reports.items()
    ):
        # Retrieve the axes column for this subgroup : [ax_roc, ax_prc, ax_cal]
        col_axes = axes[:, col_idx]

        generate_comparative_report(
            configurations=configurations,
            evaluation_mode="holdout",
            model_names_map=model_names_map,
            title=None,
            axes=col_axes,
            show=False,
        output_format=output_format)

        # Remove automatically generated titles
        for ax in col_axes:
            _show_title(title=None, default_title="", ax=ax)

        # Subgroup title at the top of each column
        axes[0, col_idx].set_title(
            subgroup_name,
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

        # Use one legend per subgroup on the upper ROC curve
        col_axes[0].legend(
            loc="lower right",
            fontsize=7,
            frameon=False,
            handlelength=2.0,
        )

        # Remove PRC and calibration legends
        for ax in col_axes[1:]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    # Label curve types on the left side of the rows
    for row_idx, row_title in enumerate(row_titles):
        axes[row_idx, 0].annotate(
            row_title,
            xy=(-0.25, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )

    # Add panel letters (a), (b), and so on across the grid
    for row in axes:
        for ax in row:
            ax.text(
                0.02,
                0.98,
                f"({next(panel_letters)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_file(fig, save_path, output_format, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def generate_subgroup_grid_from_configs(
    probas,
    y_true,
    subgroup_configs,
    n_bootstrap=2000,
    confidence_level=0.95,
    seed=42,
    model_title=None,
    model_names_map=None,
    save_path=None,
    figsize_per_row=(14, 4.5),
    dpi=600,
    show=True,

    output_format: str = "pdf",
):
    """Compute subgroup reports and render them as a comparative grid.

    Args:
        probas: Holdout positive-class probabilities.
        y_true: Holdout binary target labels.
        subgroup_configs: Subgroup definitions containing names, labels, and
            optional display-name mappings.
        n_bootstrap: Number of bootstrap iterations per subgroup.
        confidence_level: Confidence level used for bootstrap intervals.
        seed: Random seed used during bootstrap sampling.
        model_title: Optional title describing the evaluated model.
        model_names_map: Optional internal-to-display model name mapping.
        save_path: Optional output path for the grid.
        figsize_per_row: Legacy per-row size forwarded for compatibility.
        dpi: Resolution of the saved PNG figure.
        show: Whether to display the generated figure.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        The Matplotlib figure returned by ``plot_subgroup_comparative_grid``.
    """
    subgroup_reports = {}

    for subgroup_config in subgroup_configs:
        subgroup_name = subgroup_config["name"]
        subgroup_labels = subgroup_config["labels"]
        subgroup_names = subgroup_config.get("names")

        subgroup_reports[subgroup_name] = (
            compute_subgroup_holdout_metrics(
                probas=probas,
                y_true=y_true,
                subgroup_labels=subgroup_labels,
                subgroup_names=subgroup_names,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
        )

    return plot_subgroup_comparative_grid(
        subgroup_reports=subgroup_reports,
        model_title=model_title,
        model_names_map=model_names_map,
        save_path=save_path,
        figsize_per_row=figsize_per_row,
        dpi=dpi,
        show=show,
    output_format=output_format)
