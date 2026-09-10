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

import itertools
from typing import Dict, List, Optional, Tuple, Union

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


DEFAULT_NAMES_MAP = {
    "IGS2": "IGS2",
    "Logistic_Regression_Lasso_TSFEL": "L1-LR",
    "SVC_TSFEL": "SVC",
    "RandomForest_TSFEL": "Random Forest",
    "XGBoost_TSFEL": "XGBoost",
    "InceptionTimeModified": "ResNet-1D",
    "LstmTimeModified": "LSTM",
    "Transformer Encoder": "Transformer encoder",
}

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

    unique_labels = [lbl for lbl in np.unique(subgroup_labels) if lbl not in ("IGNORE", "None", "nan")]
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
        "InceptionTimeModified": "ResNet-1D",
        "LstmTimeModified": "LSTM",
        "Transformer Encoder": "Transformer encoder",
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
    
def export_custom_latex_table(results_df: pd.DataFrame, evaluation_mode: str) -> str:
    """Generates publication-ready LaTeX tables for OOF or Holdout evaluations.

    Formats metric tables with hierarchical multi-column headers (Discrimination,
    Overall, Calibration) using the tabularx environment. Dynamically identifies
    all top-performing models per column (handling ties based on displayed precision)
    and applies appropriate LaTeX macros (\\ci / \\bestci for holdout, \\meansd /
    \\bestmsd for out-of-fold).

    Args:
        results_df: DataFrame where rows correspond to model names and columns
            correspond to evaluated metrics. String values must follow either
            the 'mean ± std' format (OOF) or the 'estimate [ci_l, ci_u]' format (Holdout).
        evaluation_mode: Evaluation scheme, either 'oof' or 'holdout'. Controls
            table dimensions, spacing, macros, captions, and labels.

    Returns:
        A string containing the complete LaTeX code ready to be written to disk.

    Raises:
        ValueError: If evaluation_mode is not 'oof' or 'holdout'.
    """
    if evaluation_mode not in ("oof", "holdout"):
        raise ValueError(
            f"evaluation_mode must be 'oof' or 'holdout', got {evaluation_mode!r}."
        )

    higher_is_better = {
        "AUROC": True,
        "AUPRC": True,
        "F1-Score": True,
        "MCC": True,
        "Brier": False,
        "Intercept": "zero",
        "Slope": "one",
        "ICI": False,
        "E90": False,
    }

    metric_cols = [
        "AUROC",
        "AUPRC",
        "F1-Score",
        "MCC",
        "Brier",
        "Intercept",
        "Slope",
        "ICI",
        "E90",
    ]
    decimals = 3 if evaluation_mode == "oof" else 2

    def parse_displayed_estimate(val_str: str) -> float:
        val_str = str(val_str).strip()
        if "±" in val_str:
            num = float(val_str.split("±")[0].strip())
        elif "[" in val_str:
            num = float(val_str.split("[")[0].strip())
        else:
            num = float(val_str)
        return float(f"{num:.{decimals}f}")

    parsed_estimates = {}
    for col in metric_cols:
        if col in results_df.columns:
            parsed_estimates[col] = {
                idx: parse_displayed_estimate(str(results_df.loc[idx, col]))
                for idx in results_df.index
            }

    best_models = {}
    for col, target in higher_is_better.items():
        if col not in parsed_estimates:
            continue
        vals = parsed_estimates[col]

        if target is True:
            best_val = max(vals.values())
            best_models[col] = {m for m, v in vals.items() if v == best_val}
        elif target is False:
            best_val = min(vals.values())
            best_models[col] = {m for m, v in vals.items() if v == best_val}
        elif target == "zero":
            min_dist = min(abs(v) for v in vals.values())
            min_dist_rounded = float(f"{min_dist:.{decimals}f}")
            best_models[col] = {
                m
                for m, v in vals.items()
                if float(f"{abs(v):.{decimals}f}") == min_dist_rounded
            }
        elif target == "one":
            min_dist = min(abs(v - 1.0) for v in vals.values())
            min_dist_rounded = float(f"{min_dist:.{decimals}f}")
            best_models[col] = {
                m
                for m, v in vals.items()
                if float(f"{abs(v - 1.0):.{decimals}f}") == min_dist_rounded
            }

    def get_model_category(name: str) -> str:
        clean = name.strip().lower()
        if "igs2" in clean or "saps" in clean:
            return "baseline"
        if any(
            dl in clean
            for dl in ["resnet", "cnn", "lstm", "transformer", "inception"]
        ):
            return "deep_learning"
        return "machine_learning"

    groups = {
        "baseline": ("Clinical Baseline", []),
        "machine_learning": ("Machine Learning (TSFEL features)", []),
        "deep_learning": ("Deep Learning (raw time series)", []),
    }

    for model_name in results_df.index:
        clean_model_name = str(model_name).strip()
        if clean_model_name == "Transformer Encoder":
            model_cell = r"\makecell[{{l}}]{\textbf{Transformer}\\\textbf{Encoder}}"
        elif clean_model_name == "Random Forest":
            model_cell = r"\makecell[{{l}}]{\textbf{Random}\\\textbf{Forest}}"
        else:
            model_cell = f"\\makecell[{{{{l}}}}]{{\\textbf{{{clean_model_name}}}}}"

        row_cells = [model_cell]

        for col in metric_cols:
            if col not in results_df.columns:
                row_cells.append("-")
                continue

            raw_val = (
                str(results_df.loc[model_name, col])
                .strip()
                .replace("\n", "")
                .replace("\r", "")
            )
            is_best = model_name in best_models.get(col, set())

            if evaluation_mode == "oof":
                if "±" in raw_val:
                    mean_val, std_val = [x.strip() for x in raw_val.split("±")]
                    macro = "\\bestmsd" if is_best else "\\meansd"
                    row_cells.append(f"{macro}{{{mean_val}}}{{{std_val}}}")
                else:
                    row_cells.append(raw_val)
            else:
                if "[" in raw_val:
                    est_str, ci_str = raw_val.split("[")
                    ci_l, ci_u = ci_str.replace("]", "").split(",")
                    est_fmt = f"{float(est_str.strip()):.{decimals}f}"
                    ci_fmt = f"{float(ci_l.strip()):.{decimals}f},\\,{float(ci_u.strip()):.{decimals}f}"
                    macro = "\\bestci" if is_best else "\\ci"
                    row_cells.append(f"{macro}{{{est_fmt}}}{{{ci_fmt}}}")
                else:
                    row_cells.append(raw_val)

        cat = get_model_category(clean_model_name)
        groups[cat][1].append(" & ".join(row_cells) + " \\\\")

    section_blocks = []
    for cat_key, (cat_title, rows) in groups.items():
        if not rows:
            continue
        header = f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{cat_title}}}}} \\\\\n\\midrule"
        body = "\n".join(rows)
        section_blocks.append(f"{header}\n{body}")

    joined_rows = "\n\\midrule\n".join(section_blocks)

    if evaluation_mode == "oof":
        margin = "-1.8cm"
        tabcolsep = "2.5pt"
        arraystretch = "1.3"
        caption_text = (
            "Out-of-fold performance metrics (mean $\\pm$ SD) across"
            " cross-validation folds. Bold values indicate best performance"
            " per column."
        )
        label_tag = "tab:model_performance_oof"
    else:
        margin = "-1.8cm"
        tabcolsep = "2.5pt"
        arraystretch = "1.3"
        caption_text = (
            "Performance metrics with 95\\% confidence intervals [95\\% CI] for"
            " the evaluated models. Bold values indicate best performance per"
            " column."
        )
        label_tag = "tab:model_performance"

    template = r"""\FloatBarrier
\begin{table}[p]
\begin{adjustwidth}{__MARGIN__}{__MARGIN__}
\centering
\setlength{\tabcolsep}{__TABCOLSEP__}
\renewcommand{\arraystretch}{__ARRAYSTRETCH__}
\captionsetup{justification=centering}

\scriptsize
\begin{tabularx}{\linewidth}{@{} l *{9}{Y} @{}}
\toprule
& \multicolumn{4}{c}{\textbf{Discrimination}}
& \multicolumn{1}{c}{\textbf{Overall}}
& \multicolumn{4}{c}{\textbf{Calibration}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-10}
\textbf{Model}
& \textbf{AUROC}
& \textbf{AUPRC}
& \textbf{F1-Score}
& \textbf{MCC}
& \textbf{Brier}
& \textbf{Intercept}
& \textbf{Slope}
& \textbf{ICI}
& \textbf{E90}\\
\midrule
__JOINED_ROWS__
\bottomrule
\end{tabularx}

\caption{__CAPTION__}
\label{__LABEL__}
\end{adjustwidth}
\end{table}
\FloatBarrier
"""

    return (
        template.replace("__MARGIN__", margin)
        .replace("__TABCOLSEP__", tabcolsep)
        .replace("__ARRAYSTRETCH__", arraystretch)
        .replace("__JOINED_ROWS__", joined_rows)
        .replace("__CAPTION__", caption_text)
        .replace("__LABEL__", label_tag)
    )

def generate_comparative_report(
    configurations,
    y_true_base=None,
    save_dir=None,
    table_format="fancy_grid",
    evaluation_mode="oof",
    model_names_map=None,
    title=None,  # Disabled by default
    color_offset=0,
    axes=None,
    show=True,
    output_format: str = "pdf",
    save_individual_plots: bool = True,
    order: list = None,
):
    """Generates comparative performance metrics, tables, and curves for models.

    Supports pooled out-of-fold (OOF) predictions and independent holdout sets.
    Renders ROC, Precision-Recall, and calibration curves either into caller-provided
    axes (e.g., subgroup grids) or into dedicated stand-alone figures. When saving,
    it can export both a combined 1x3 overview figure and separate single-plot files.
    """
    if evaluation_mode not in ("oof", "holdout"):
        raise ValueError(
            f"evaluation_mode must be 'oof' or 'holdout', got {evaluation_mode!r}."
        )

    configurations = list(configurations)

    # Predefined model labels fallback mapping (covers all transformer aliases)
    default_names_map = {
        "IGS2": "IGS2",
        "Logistic_Regression_Lasso_TSFEL": "L1-LR",
        "SVC_TSFEL": "SVC",
        "RandomForest_TSFEL": "Random Forest",
        "XGBoost_TSFEL": "XGBoost",
        "InceptionTimeModified": "ResNet-1D",
        "LstmTimeModified": "LSTM",
        "VanillaTransformerModified": "Transformer Encoder",
        "Transformer Encoder": "Transformer Encoder",
        "Transformer_Encoder": "Transformer Encoder",
    }

    names_map = default_names_map.copy()
    if model_names_map is not None:
        names_map.update(model_names_map)

    predefined_order = order if order is not None else [
        "IGS2",
        "Logistic_Regression_Lasso_TSFEL",
        "SVC_TSFEL",
        "RandomForest_TSFEL",
        "XGBoost_TSFEL",
        "InceptionTimeModified",
        "LstmTimeModified",
        "VanillaTransformerModified",
        "Transformer Encoder",
    ]

    def get_sort_key(item):
        """Generates a sorting key to preserve standard benchmark sequence."""
        model_name = item[0]
        if model_name in predefined_order:
            return 0, predefined_order.index(model_name)
        return 1, model_name

    def require_key(config, key, model_name):
        """Safely fetches a required dictionary key with error tracking."""
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
        """Formats mean/std or bootstrap CI into export-ready strings."""
        if evaluation_mode == "oof":
            mean_val = float(require_key(config, f"{metric_name}_mean", model_name))
            std_val = float(require_key(config, f"{metric_name}_std", model_name))
            return f"{mean_val:.3f} ± {std_val:.3f}"

        bootstrap = require_key(config, "bootstrap_holdout", model_name)
        entry = bootstrap["summary"][metric_name]
        estimate = float(entry["estimate"])
        ci_lower = float(entry["ci_lower"])
        ci_upper = float(entry["ci_upper"])
        return f"{estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

    configurations = sorted(configurations, key=get_sort_key)
    default_colors = sns.color_palette("tab10", n_colors=max(len(configurations), 10))

    results = {}
    plot_data_list = []
    label_prefix = "OOF" if evaluation_mode == "oof" else "Holdout"

    for idx, (name, config) in enumerate(configurations):
        display_name = names_map.get(name, name)
        probas = np.asarray(require_key(config, probas_key, name), dtype=float).ravel()
        current_y = (
            y_true_base if y_true_base is not None else require_key(config, y_true_key, name)
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
            precision = np.array([1.0, prevalence, prevalence])
            recall = np.array([0.0, 0.0, 1.0])
            fop = np.array([prevalence])
            mpv = np.array([float(probas[0])])
        else:
            fpr, tpr, _ = roc_curve(y_true, probas)
            precision, recall, _ = precision_recall_curve(y_true, probas)
            fop, mpv = calibration_curve(y_true, probas, n_bins=10, strategy="uniform")

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
            intercept_val = float(cal_stats["intercept"])
            slope_val = float(cal_stats["slope"])
            ici_val = float(cal_stats["ici"])

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
        results[display_name] = metrics_dict

        color = config.get(
            "color",
            default_colors[(idx + color_offset) % len(default_colors)],
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

    print(f"\n=== {label_prefix} PERFORMANCE COMPARISON TABLE ===")
    print(tabulate(results_df, headers="keys", tablefmt=table_format, showindex=True))

    owns_figure = axes is None

    # Step 1: Combined overview figure (ROC, PRC, Calibration)
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    else:
        fig = axes[0].figure

    plot_comparative_axes(
        plot_data_list=plot_data_list,
        axes=axes,
        show_legend=owns_figure,
    )

    # Figure-level supertitles removed

    fig_bar = None

    # Step 2: File persistence and exports
    if save_dir is not None and owns_figure:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_figure_file(
            fig,
            output_path / f"collective_performance_{label_prefix.lower()}",
            output_format,
            bbox_inches="tight",
        )

        # Export individual stand-alone curve files with guaranteed legends
        if save_individual_plots:
            # Récupération des handles et labels existants depuis le premier graphique (ROC)
            handles, labels = axes[0].get_legend_handles_labels()

            # Tuple : (nom_fichier_sans_extension, index_dans_axes)
            single_plots = [
                (f"roc_{label_prefix.lower()}", 0),
                (f"prc_{label_prefix.lower()}", 1),
                (f"calibration_{label_prefix.lower()}", 2),
            ]

            for file_tag, target_idx in single_plots:
                # Création d'une figure dédiée 1x1
                fig_single, ax_single = plt.subplots(
                    figsize=(6, 5), layout="constrained"
                )

                # Liste de 3 axes pour plot_comparative_axes : seul l'axe ciblé reçoit le tracé
                # Les deux autres sont des axes factices créés en mémoire et jetés après
                fig_dummy, dummy_axes = plt.subplots(1, 2)
                trio_axes = [dummy_axes[0], dummy_axes[1]]
                trio_axes.insert(target_idx, ax_single)

                # Tracé sans légende automatique
                plot_comparative_axes(
                    plot_data_list=plot_data_list,
                    axes=trio_axes,
                    show_legend=False,
                )
                plt.close(fig_dummy)

                # Ajout explicite de la légende avec les labels récupérés
                if handles and labels:
                    ax_single.legend(
                        handles,
                        labels,
                        fontsize=7,
                        loc="best",
                        frameon=True,
                    )

                # Sauvegarde directe de la figure dédiée
                save_figure_file(
                    fig_single,
                    output_path / file_tag,
                    output_format,
                    bbox_inches="tight",
                )
                plt.close(fig_single)

        latex_table_code = export_custom_latex_table(
            results_df=results_df,
            evaluation_mode=evaluation_mode,
        )

        with open(
            output_path / f"results_table_{label_prefix.lower()}.tex",
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(latex_table_code)

    # Step 3: Holdout bootstrap barplot (propagates names_map)
    if evaluation_mode == "holdout" and owns_figure:
        fig_bar = plot_holdout_metrics_barplot(
            configurations=configurations,
            save_dir=save_dir,
            model_names_map=names_map,  # Pass the complete dictionary
            color_offset=color_offset,
            output_format=output_format,
        )

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


def plot_paired_bootstrap_forest(
    models_dict: Dict[str, Dict[str, np.ndarray]],
    output_dir: Union[str, Path],
    selected_models: Optional[List[str]] = None,
    custom_pairs: Optional[List[Tuple[str, str]]] = None,
    names_map: Optional[Dict[str, str]] = None,
    save_filename: str = "paired_bootstrap_forest_auc_auprc_brier.pdf",
    metrics: Optional[List[Dict[str, str]]] = None,
    show_plot: bool = True,
) -> Optional[pd.DataFrame]:
    """Generates and exports a publication-ready forest plot comparing paired bootstrap distributions.

    Computes paired deltas across bootstrap resamples for discrimination
    (AUROC, AUPRC) and probabilistic error (Brier score). Delta directions are
    harmonized such that positive values consistently favor the second model (Model B).
    Applies display mapping to model identifiers for publication styling.

    Args:
        models_dict: Nested dictionary mapping model identifiers to their
            bootstrap distributions (e.g., {"auc": np.ndarray, "auprc": np.ndarray,
            "brier": np.ndarray}).
        output_dir: Target directory path where figures and summaries are saved.
        selected_models: List of model substrings, raw keys, or mapped names to compare.
            All pairwise combinations (n choose 2) will be generated. Ignored if
            `custom_pairs` is provided. If both are None, all models present in
            `models_dict` (excluding 'igs2') are evaluated.
        custom_pairs: Explicit list of model identifier tuples `(model_a, model_b)`.
            Accepts raw keys, substrings, or mapped names. Takes precedence over `selected_models`.
        names_map: Dictionary mapping raw model keys to clean publication names.
            Defaults to DEFAULT_NAMES_MAP if None.
        save_filename: Output PDF filename. Defaults to
            "paired_bootstrap_forest_auc_auprc_brier.pdf".
        metrics: Custom configuration list for metrics to evaluate. Each dict must
            contain 'key', 'title', and 'direction' ('b_minus_a' or 'a_minus_b').
            Defaults to AUROC, AUPRC, and Brier score.
        show_plot: Whether to call `plt.show()` during notebook execution.
            Defaults to True.

    Returns:
        pd.DataFrame containing paired delta statistics (mean, 95% CI bounds,
        p-values) across evaluated comparisons, or None if no valid pairs were found.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if names_map is None:
        names_map = DEFAULT_NAMES_MAP

    available_keys = list(models_dict.keys())

    def _get_display_name(raw_key: str) -> str:
        if raw_key in names_map:
            return names_map[raw_key]
        for k, v in names_map.items():
            if k.lower() in raw_key.lower():
                return v
        return raw_key

    def _match_key(query: str) -> Optional[str]:
        # 1. Exact match in raw keys
        if query in available_keys:
            return query
        # 2. Match via reverse lookup from display name
        for raw_k, disp_k in names_map.items():
            if query.lower() == disp_k.lower():
                for real_k in available_keys:
                    if raw_k.lower() in real_k.lower():
                        return real_k
        # 3. Substring match on raw keys
        return next((k for k in available_keys if query.lower() in k.lower()), None)

    # 1. Resolve target pairs
    resolved_pairs: List[Tuple[str, str, str]] = []

    if custom_pairs is not None:
        for a_query, b_query in custom_pairs:
            matched_a = _match_key(a_query)
            matched_b = _match_key(b_query)
            if matched_a and matched_b:
                label_a = _get_display_name(matched_a)
                label_b = _get_display_name(matched_b)
                resolved_pairs.append((matched_a, matched_b, f"{label_a} vs {label_b}"))
            else:
                missing = [q for q, m in [(a_query, matched_a), (b_query, matched_b)] if m is None]
                print(f"[Warning] Could not resolve pair: missing {missing}")
    else:
        if selected_models is not None:
            resolved_keys = []
            for query in selected_models:
                matched = _match_key(query)
                if matched:
                    if matched not in resolved_keys:
                        resolved_keys.append(matched)
                else:
                    print(f"[Warning] Model identifier matching '{query}' not found.")
        else:
            resolved_keys = [k for k in available_keys if "igs2" not in k.lower()]

        for mod_a, mod_b in itertools.combinations(resolved_keys, 2):
            label_a = _get_display_name(mod_a)
            label_b = _get_display_name(mod_b)
            resolved_pairs.append((mod_a, mod_b, f"{label_a} vs {label_b}"))

    if not resolved_pairs:
        print("[Error] No valid model pairs to compare.")
        return None

    # 2. Metric setup
    if metrics is None:
        metrics = [
            {"key": "auc", "title": r"$\Delta$ AUROC $(B - A)$", "direction": "b_minus_a"},
            {"key": "auprc", "title": r"$\Delta$ AUPRC $(B - A)$", "direction": "b_minus_a"},
            {"key": "brier", "title": r"$\Delta$ Brier $(A - B)$", "direction": "a_minus_b"},
        ]

    # 3. Compute paired differences across bootstrap distributions
    records = []
    for mod_a, mod_b, label in resolved_pairs:
        dist_a = models_dict[mod_a]
        dist_b = models_dict[mod_b]
        row = {
            "pair": label,
            "model_A": _get_display_name(mod_a),
            "model_B": _get_display_name(mod_b),
            "raw_model_A": mod_a,
            "raw_model_B": mod_b,
        }

        for m in metrics:
            k = m["key"]
            if k not in dist_a or k not in dist_b:
                continue

            vals_a = np.asarray(dist_a[k], dtype=np.float64)
            vals_b = np.asarray(dist_b[k], dtype=np.float64)

            delta = (vals_b - vals_a) if m["direction"] == "b_minus_a" else (vals_a - vals_b)
            mean_val = float(np.mean(delta))
            ci = np.percentile(delta, [2.5, 97.5])
            pval = float(min(1.0, 2.0 * min(np.mean(delta <= 0), np.mean(delta >= 0))))

            row[f"{k}_mean"] = mean_val
            row[f"{k}_ci_low"] = ci[0]
            row[f"{k}_ci_high"] = ci[1]
            row[f"{k}_pval"] = pval

        records.append(row)

    df_results = pd.DataFrame(records)
    csv_path = output_dir / f"{Path(save_filename).stem}_data.csv"
    df_results.to_csv(csv_path, index=False)

    # 4. Multi-panel Forest Plot
    n_pairs = len(df_results)
    n_metrics = len(metrics)
    fig_height = max(4.5, n_pairs * 0.45 + 1.5)

    fig, axes = plt.subplots(
        1, n_metrics, figsize=(6.0 * n_metrics, fig_height), dpi=300, constrained_layout=True
    )
    if n_metrics == 1:
        axes = [axes]

    y_pos = np.arange(n_pairs)

    for ax, m in zip(axes, metrics):
        k = m["key"]
        means = df_results[f"{k}_mean"].to_numpy()
        ci_lows = df_results[f"{k}_ci_low"].to_numpy()
        ci_highs = df_results[f"{k}_ci_high"].to_numpy()
        pvals = df_results[f"{k}_pval"].to_numpy()

        xerr = np.vstack([means - ci_lows, ci_highs - means])
        colors = ["#1f77b4" if p < 0.05 else "#7f7f7f" for p in pvals]

        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

        for i in range(n_pairs):
            ax.errorbar(
                means[i],
                y_pos[i],
                xerr=[[xerr[0, i]], [xerr[1, i]]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=2.0,
                capsize=4,
                capthick=1.4,
                markersize=6.5,
            )

            p_str = (
                f"$p={pvals[i]:.1e}$"
                if pvals[i] < 0.001
                else (f"$p={pvals[i]:.3f}$" if pvals[i] < 0.05 else "NS")
            )
            annot = f"{means[i]:+.3f} [{ci_lows[i]:+.3f}, {ci_highs[i]:+.3f}] | {p_str}"

            ax.text(
                means[i],
                y_pos[i] - 0.22,
                annot,
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=colors[i],
            )

        ax.set_yticks(y_pos)
        if ax == axes[0]:
            ax.set_yticklabels(df_results["pair"], fontsize=9.0, fontweight="bold")
        else:
            ax.set_yticklabels([])

        ax.invert_yaxis()
        ax.set_ylim(n_pairs - 0.5, -0.6)
        ax.set_xlabel(m["title"], fontsize=9.5)
        ax.set_title(
            f"{k.upper()} Difference\n(Positive favors Model B)",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    fig_path = output_dir / save_filename
    plt.savefig(fig_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

    print(f"[Done] Processed {n_pairs} model comparison pairs.")
    print(f"[Saved] Figure: {fig_path.name} | Data: {csv_path.name}")

    return df_results