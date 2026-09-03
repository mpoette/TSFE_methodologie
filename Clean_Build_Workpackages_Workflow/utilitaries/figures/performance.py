"""Performance, discrimination, and calibration visualization utilities."""

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

def _show_title(title, default_title, ax=None):
    """Display a figure title based on the ``title`` argument.

    Args:
        title:
            The raw ``title`` value passed by the caller.  When it equals
            ``...`` (the sentinel default), ``default_title`` is displayed.
            When it is ``None``, no title is shown.  Any other value is
            treated as a custom string and displayed directly.
        default_title:
            The automatic title used when ``title`` is omitted.
        ax:
            Optional matplotlib Axes. When provided, ``ax.set_title()`` is
            used instead of ``plt.title()`` so subplots are correctly targeted.
    """
    effective_title = default_title if title is ... else title
    if effective_title is None:
        return
    if ax is not None:
        ax.set_title(effective_title)
    else:
        plt.title(effective_title)


# ============================================================================
# LEARNING CURVES
# ============================================================================

def plot_collected_learning_curve(
    sample_sizes,
    train_matrix,
    val_matrix,
    model_name="Model",
    folder="",
    savefig=True,
    transparent=True,
    title=...,

    output_format: str = "pdf",
):
    """Plot a learning curve aggregated across multiple training folds.

    The function computes mean and standard deviation of AUC-ROC scores
    across folds at each training fraction, then plots the training and
    validation curves with a volatility band.

    Args:
        sample_sizes:
            Array-like of sample sizes corresponding to each training
            fraction (e.g., [200, 400, 600, 800, 1000]).
        train_matrix:
            NumPy array of shape ``(n_folds, n_stages)`` containing the
            training AUC-ROC scores at each stage for each fold.
        val_matrix:
            NumPy array of shape ``(n_folds, n_stages)`` containing the
            validation (out-of-fold) AUC-ROC scores at each stage for
            each fold.
        model_name:
            Model name displayed in the figure title.
        folder:
            Output directory for the saved figure.
        savefig:
            Whether to save the generated figure.
        transparent:
            Whether the saved figure should use a transparent background.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        None.
    """
    train_matrix = np.asarray(train_matrix, dtype=float)
    val_matrix = np.asarray(val_matrix, dtype=float)
    sample_sizes = np.asarray(sample_sizes, dtype=int)

    train_mean = np.mean(train_matrix, axis=0)
    val_mean = np.mean(val_matrix, axis=0)
    val_std = np.std(val_matrix, axis=0, ddof=1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        sample_sizes,
        train_mean,
        "o-",
        color="crimson",
        label="Training Score (Mean)",
        linewidth=2,
    )

    plt.plot(
        sample_sizes,
        val_mean,
        "o-",
        color="royalblue",
        label="Validation Score (Mean OOF)",
        linewidth=2,
    )

    plt.fill_between(
        sample_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="royalblue",
        label="OOF Volatility (± 1 STD)",
    )

    _show_title(
        title,
        f"Learning Curve — {model_name} (Embedded Fold Splitting)",
    )
    plt.xlabel("Number of Training Samples (Aggregated)")
    plt.ylabel("AUC-ROC Score")
    plt.ylim(0.6, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if savefig and folder:
        output_path = Path(folder)
        output_path.mkdir(parents=True, exist_ok=True)
        save_figure_file(plt, output_path / "learning_curve.png", output_format, bbox_inches="tight", transparent=transparent)

    plt.show()


# ============================================================================
# CALIBRATION CURVES
# ============================================================================

def _format_calibration_label(
    name,
    stats,
    extra_info=None,
):
    """Build a calibration legend entry from computed statistics.

    Args:
        name:
            Base label (e.g. model name or dataset split).
        stats:
            Dictionary returned by :func:`get_calibration_stats`.
        extra_info:
            Optional suffix appended in parentheses, for example a
            calibration strategy or a dataset split descriptor.

    Returns:
        A formatted string ready to be used as a matplotlib legend label.
    """
    suffix = f" ({extra_info})" if extra_info else ""

    return (
        f"{name}{suffix} "
        f"(Int={stats['intercept']:.2f}, "
        f"Slope={stats['slope']:.2f}, "
        f"ICI={stats['ici']:.3f})"
    )


def plot_calibration_curves(
    curve_specs,
    y_true,
    figsize=(8, 6),
    title=None,
    legend_loc="lower right",
    legend_fontsize=9,
    grid=True,
    grid_alpha=0.6,
    show_perfect=True,
    save_figure=False,
    output_dir=None,
    filename="calibration_curve.png",
    dpi=300,
    transparent=False,
    return_stats=False,

    output_format: str = "pdf",
):
    """Plot one or more binned calibration curves on a single axes.

    Each curve is described by a *specification* dictionary containing at
    least ``"probas"`` and optionally styling hints.  Statistics are computed
    via :func:`get_calibration_stats` using fixed 10% risk brackets, so the
    bin centres align with :func:`calibration_per_risk_brackets`.

    Args:
        curve_specs (list[dict]):
            List of curve specifications.  Each dictionary may contain:

                - ``"probas"`` (array-like, **required**): predicted
                  probabilities for the positive class.
                - ``"name"`` (str, default ``"Curve N"``): base label.
                - ``"extra_info"`` (str, optional): additional text appended
                  to the label (e.g. ``"Platt"`` or ``"train"``).
                - ``"color"`` (str, optional): matplotlib color specifier.
                - ``"marker"`` (str, optional): matplotlib marker (default
                  ``"s"``).
                - ``"linestyle"`` (str, optional): matplotlib linestyle
                  (default ``"-"``).

        y_true (array-like):
            Binary ground-truth labels shared across all curves.
        figsize (tuple, optional):
            Figure size in inches.  Defaults to ``(8, 6)``.
        title (str, optional):
            Figure title.  When omitted, a generic title is used.
        legend_loc (str, optional):
            Matplotlib legend location.  Defaults to ``"lower right"``.
        legend_fontsize (int, optional):
            Legend font size.  Defaults to ``9``.
        grid (bool, optional):
            Whether to display a grid.  Defaults to ``True``.
        grid_alpha (float, optional):
            Grid line transparency.  Defaults to ``0.6``.
        show_perfect (bool, optional):
            Whether to draw the *perfect calibration* reference line.
            Defaults to ``True``.
        save_figure (bool, optional):
            Whether to save the figure to disk.  Defaults to ``False``.
        output_dir (str | Path, optional):
            Output directory.  Created if it does not exist.
        filename (str, optional):
            Output filename.  Defaults to ``"calibration_curve.png"``.
        dpi (int, optional):
            Resolution of the saved figure.  Defaults to ``300``.
        transparent (bool, optional):
            Whether to use a transparent background when saving.
            Defaults to ``False``.
        return_stats (bool, optional):
            Whether to return the list of statistics dictionaries
            (one per curve).  Defaults to ``False``.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        If ``return_stats`` is ``True``, a list of statistics dictionaries
        (in the same order as ``curve_specs``).  Otherwise ``None``.

    Raises:
        ValueError:
            If ``curve_specs`` is empty or any probability array has an
            incompatible length.

    Example:
        Plot *uncalibrated* vs *calibrated* predictions for both a
        **train** and a **test** split::

            curves = [
                {"probas": train_uncalib, "name": "Uncalibrated", "extra_info": "train"},
                {"probas": train_calib,  "name": "Calibrated",   "extra_info": "train"},
                {"probas": test_uncalib,  "name": "Uncalibrated", "extra_info": "test"},
                {"probas": test_calib,    "name": "Calibrated",   "extra_info": "test"},
            ]

            plot_calibration_curves(
                curves,
                y_true=np.concatenate([train_labels, test_labels]),
                title="Multi-split Calibration Comparison",
            )
    """
    if not curve_specs:
        raise ValueError("curve_specs must contain at least one curve.")

    y_true = np.asarray(y_true, dtype=int).ravel()

    # Resolve colors: assign a palette when not explicitly provided
    provided_colors = [
        spec.get("color")
        for spec in curve_specs
        if spec.get("color") is not None
    ]

    needed = len(curve_specs) - len(provided_colors)
    palette = sns.color_palette("tab10", max(needed, 1))
    color_iterator = iter(palette)

    all_stats = []

    fig, ax = plt.subplots(figsize=figsize)

    # Perfect-calibration reference
    if show_perfect:
        ax.plot(
            [0, 1],
            [0, 1],
            "k:",
            label="Perfect calibration",
        )

    for idx, spec in enumerate(curve_specs):
        probas = np.asarray(spec.get("probas"), dtype=float).ravel()

        # Allow per-curve y_true when curves come from different splits
        # (e.g. train vs holdout). Falls back to the global y_true.
        curve_y_true = np.asarray(
            spec.get("y_true", y_true), dtype=int
        ).ravel()

        if probas.shape[0] != curve_y_true.shape[0]:
            raise ValueError(
                f"Curve '{spec.get('name', idx)}': probas length "
                f"({probas.shape[0]}) does not match y_true length "
                f"({curve_y_true.shape[0]})."
            )

        stats = get_calibration_stats(probas, curve_y_true)
        all_stats.append(stats)

        name = spec.get("name", f"Curve {idx + 1}")
        extra_info = spec.get("extra_info")
        color = spec.get("color")
        if color is None:
            color = next(color_iterator)
        marker = spec.get("marker", "s")
        linestyle = spec.get("linestyle", "-")

        label = _format_calibration_label(name, stats, extra_info)

        ax.plot(
            stats["x"],
            stats["obs_rate"],
            marker + linestyle,
            color=color,
            label=label,
        )

    # Axes configuration
    ax.set_xlabel("Predicted Risk (10% brackets)")
    ax.set_ylabel("True fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.1))

    if title:
        ax.set_title(title)

    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    if grid:
        ax.grid(True, linestyle=":", alpha=grid_alpha)

    fig.tight_layout()

    # Save logic
    if save_figure and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_figure_file(fig, output_path / filename, output_format, bbox_inches="tight", transparent=transparent)

    plt.show()
    plt.close(fig)

    return all_stats if return_stats else None


def calibration_curve_homemade(
    probas_uncalib,
    probas_calib,
    y_test_global,
    model_name,
    calibration,
    save_figure,
    output_dir,
    transparent,
    calibration_mode="Platt",

    output_format: str = "pdf",
):
    """Plot a binned calibration curve for raw and calibrated predictions.

    .. deprecated::
        Use :func:`plot_calibration_curves` for new code.  This wrapper
        remains for backward compatibility.

    Points are placed on fixed 10% risk bracket centers (0.05, 0.15, ..., 0.95)
    to match the binning strategy of :func:`calibration_per_risk_brackets`.

    Args:
        probas_uncalib (array-like): Uncalibrated positive-class probabilities.
        probas_calib (array-like): Calibrated positive-class probabilities (or None).
        y_test_global (array-like): Binary ground-truth labels.
        model_name (str): Model name for titles and labels.
        calibration (bool): Whether calibrated predictions should be displayed.
        save_figure (bool): Whether to save the generated figure.
        output_dir (str or Path): Output directory for the figure.
        transparent (bool): Whether to save with a transparent background.
        calibration_mode (str, optional): Name of calibration strategy.
            Defaults to "Platt".

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        None.
    """
    curve_specs = [
        {
            "probas": probas_uncalib,
            "name": f"Raw {model_name}",
            "color": "red",
        },
    ]

    if calibration and probas_calib is not None:
        curve_specs.append(
            {
                "probas": probas_calib,
                "name": f"Calibrated {model_name}",
                "extra_info": calibration_mode,
                "color": "blue",
            }
        )

    plot_calibration_curves(
        curve_specs=curve_specs,
        y_true=y_test_global,
        title=(
            "Global Calibration Curve (5-Fold Cross-Validation)\n"
            f"Model: {model_name}"
        ),
        save_figure=save_figure,
        output_dir=output_dir,
        filename="calibration_curve.png",
        transparent=transparent,
    output_format=output_format)


def plot_collective_calibration_curves(
    model_curves,
    y_true,
    n_bins=10,
    bin_strategy="uniform",
    figsize=(8, 8),
    legend_loc="upper left",
    legend_fontsize=9,
    grid=True,
    grid_alpha=0.6,
    show_perfect=True,
    save_figure=False,
    output_dir=None,
    filename="collective_calibration_curve.png",
    dpi=300,
    transparent=False,

    output_format: str = "pdf",
):
    """Plot calibration curves for multiple models using sklearn's calibration_curve.

    This function is designed for **collective / comparative** visualizations
    where each model contributes its own out-of-fold (or hold-out) predictions
    against a shared ground-truth vector.  Under the hood it relies on
    :func:`sklearn.calibration.calibration_curve` rather than the homemade
    Polars-based binning used by :func:`get_calibration_stats`.

    Args:
        model_curves (list[dict]):
            List of model curve specifications.  Each dictionary may contain:

                - ``"probas"`` (array-like, **required**): predicted
                  probabilities for the positive class.
                - ``"name"`` (str, default ``"Model N"``): display name.
                - ``"color"`` (str, optional): matplotlib color specifier.
                  When omitted, colours are drawn from ``seaborn.tab10``.
                - ``"intercept_oof"`` (float, optional): OOF calibration
                  intercept included in the legend label.
                - ``"slope_oof"`` (float, optional): OOF calibration slope
                  included in the legend label.
                - ``"ici_oof"`` (float, optional): OOF Integrated Calibration
                  Index included in the legend label.

        y_true (array-like):
            Shared binary ground-truth labels.
        n_bins (int, optional):
            Number of bins passed to :func:`sklearn.calibration_curve`.
            Defaults to ``10``.
        bin_strategy (str, optional):
            Binning strategy passed to :func:`sklearn.calibration_curve`.
            Defaults to ``"uniform"``.
        figsize (tuple, optional):
            Figure size in inches.  Defaults to ``(8, 8)``.
        legend_loc (str, optional):
            Matplotlib legend location.  Defaults to ``"upper left"``.
        legend_fontsize (int, optional):
            Legend font size.  Defaults to ``9``.
        grid (bool, optional):
            Whether to display a grid.  Defaults to ``True``.
        grid_alpha (float, optional):
            Grid line transparency.  Defaults to ``0.6``.
        show_perfect (bool, optional):
            Whether to draw the *perfect calibration* reference line.
            Defaults to ``True``.
        save_figure (bool, optional):
            Whether to save the figure to disk.  Defaults to ``False``.
        output_dir (str | Path, optional):
            Output directory.  Created if it does not exist.
        filename (str, optional):
            Output filename.  Defaults to
            ``"collective_calibration_curve.png"``.
        dpi (int, optional):
            Resolution of the saved figure.  Defaults to ``300``.
        transparent (bool, optional):
            Whether to use a transparent background when saving.
            Defaults to ``False``.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        ``None``.

    Raises:
        ValueError:
            If ``model_curves`` is empty or any probability array has an
            incompatible length.

    Example:
        Compare OOF calibration across several models::

            models = [
                {
                    "probas": lr_probas_oof,
                    "name": "Logistic_Regression_Lasso_TSFEL",
                    "intercept_oof": 0.12,
                    "slope_oof": 0.95,
                    "ici_oof": 0.034,
                },
                {
                    "probas": xgb_probas_oof,
                    "name": "XGBoost_TSFEL",
                    "intercept_oof": -0.05,
                    "slope_oof": 1.10,
                    "ici_oof": 0.028,
                },
            ]

            plot_collective_calibration_curves(
                models,
                y_true=y_true_oof,
                save_figure=True,
                output_dir="outputs/comparison",
            )
    """
    if not model_curves:
        raise ValueError("model_curves must contain at least one model.")

    y_true = np.asarray(y_true, dtype=int).ravel()

    n_models = len(model_curves)
    default_colors = sns.color_palette("tab10", max(n_models, 10))
    color_idx = 0

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Perfect-calibration reference
    if show_perfect:
        ax.plot(
            [0, 1],
            [0, 1],
            "k:",
            alpha=0.7,
            label="Perfect calibration",
        )

    for idx, spec in enumerate(model_curves):
        probas = np.asarray(spec.get("probas"), dtype=float).ravel()

        if probas.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Model '{spec.get('name', idx)}': probas length "
                f"({probas.shape[0]}) does not match y_true length "
                f"({y_true.shape[0]})."
            )

        name = spec.get("name", f"Model {idx + 1}")
        color = spec.get("color")
        if color is None:
            color = default_colors[color_idx % len(default_colors)]
            color_idx += 1

        intercept_oof = spec.get("intercept_oof")
        slope_oof = spec.get("slope_oof")
        ici_oof = spec.get("ici_oof")

        # Build legend label
        if intercept_oof is not None and slope_oof is not None and ici_oof is not None:
            calibration_label = (
                f"{name} "
                f"(Int={intercept_oof:.2f}, "
                f"Slope={slope_oof:.2f}, "
                f"ICI={ici_oof:.3f})"
            )
        else:
            calibration_label = name

        # Detect constant predictions (degenerate case)
        constant_predictions = np.all(probas == probas[0])

        if constant_predictions:
            prevalence = float(np.mean(y_true))
            fop = np.array([prevalence])
            mpv = np.array([float(probas[0])])
        else:
            fop, mpv = calibration_curve(
                y_true,
                probas,
                n_bins=n_bins,
                strategy=bin_strategy,
            )

        ax.plot(
            mpv,
            fop,
            "s-",
            label=calibration_label,
            color=color,
            linewidth=2,
        )

    # Axes configuration
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("True Fraction of Positives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    if grid:
        ax.grid(True, linestyle=":", alpha=grid_alpha)

    fig.tight_layout()

    # Save logic
    if save_figure and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_figure_file(fig, output_path / filename, output_format, bbox_inches="tight", transparent=transparent)

    plt.show()
    plt.close(fig)


# ============================================================================
# CALIBRATION STATISTICS
# ============================================================================

def _validate_calibration_inputs(
    probas,
    y_test,
):
    """Validate and normalize inputs used for calibration analysis.
    
    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
    
    Returns:
        A tuple containing flattened NumPy arrays for probabilities and labels.
    
    Raises:
        ValueError:
            If the arrays are empty, have different lengths, contain non-finite
            values, contain probabilities outside ``[0, 1]``, or do not include
            both binary classes.
    """
    probas = np.asarray(
        probas,
        dtype=float,
    ).ravel()

    y_test = np.asarray(
        y_test,
        dtype=int,
    ).ravel()

    if probas.size == 0:
        raise ValueError(
            "probas and y_test must not be empty."
        )

    if probas.shape[0] != y_test.shape[0]:
        raise ValueError(
            "probas and y_test must have the same length."
        )

    if not np.all(np.isfinite(probas)):
        raise ValueError(
            "probas contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_test)):
        raise ValueError(
            "y_test contains NaN or infinite values."
        )

    if np.any((probas < 0) | (probas > 1)):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    unique_classes = np.unique(y_test)

    if not np.array_equal(
        unique_classes,
        np.array([0, 1]),
    ):
        raise ValueError(
            "y_test must contain both binary classes 0 and 1."
        )

    return probas, y_test

def get_calibration_stats(probas, y_test):
    """Compute calibration statistics and binned coordinates using 10% fixed risk brackets.
    
    Calibration intercept and slope are estimated via logistic regression on logit-transformed
    predicted probabilities. Brier score, Integrated Calibration Index (ICI), E90, and EMax
    are calculated directly from the fixed 10% probability bins.
    
    Args:
        probas (array-like): Predicted probabilities for the positive class.
        y_test (array-like): Binary ground-truth labels encoded as 0 and 1.
    
    Returns:
        dict: A dictionary containing intercept, slope, brier, ici, e90
              the Polars binned DataFrame ('binned_df'), and bin coordinates ('x', 'obs_rate', 'pred_mean').
    """
    probas, y_test = _validate_calibration_inputs(
        probas,
        y_test,
    )

    # ---------------------------------------------------------
    # 1. Compute 10% fixed bins with Polars
    # ---------------------------------------------------------
    df_brier = pl.DataFrame({"y": y_test, "pred": probas})

    fixed_brier_df = (
        df_brier.with_columns(
            (
                pl.col("pred")
                .clip(0, 0.999999)
                .mul(10)
                .floor()
                .cast(pl.Int64)
            ).alias("bin_fixed")
        )
        .group_by("bin_fixed")
        .agg([
            pl.len().alias("n"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        ])
        .sort("bin_fixed")
        .with_columns(
            ((pl.col("bin_fixed") + 0.5) / 10).alias("x")
        )
    )

    fixed_brier_pd = fixed_brier_df.to_pandas()

    # ---------------------------------------------------------
    # 2. Calibration Intercept & Slope (Logistic Regression)
    # ---------------------------------------------------------
    eps = 1e-7
    probas_clipped = np.clip(probas, eps, 1 - eps)
    logits = np.log(probas_clipped / (1 - probas_clipped)).reshape(-1, 1)

    calib_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    calib_model.fit(logits, y_test)

    # ---------------------------------------------------------
    # 3. Bin-based ICI, E90, EMax
    # ---------------------------------------------------------
    bin_errors = np.abs(fixed_brier_pd["obs_rate"] - fixed_brier_pd["pred_mean"])
    bin_weights = fixed_brier_pd["n"] / fixed_brier_pd["n"].sum()

    return {
        "intercept": float(calib_model.intercept_[0]),
        "slope": float(calib_model.coef_[0, 0]),
        "brier": float(brier_score_loss(y_test, probas)),
        "ici": float(np.sum(bin_errors * bin_weights)),
        "e90": float(np.percentile(bin_errors, 90)),
        # "eMax": float(np.max(bin_errors)),
        "binned_df": fixed_brier_df,
        "x": fixed_brier_pd["x"].to_numpy(),
        "obs_rate": fixed_brier_pd["obs_rate"].to_numpy(),
        "pred_mean": fixed_brier_pd["pred_mean"].to_numpy(),
        "n": fixed_brier_pd["n"].to_numpy(),
    }



# ============================================================================
# FOLD METRICS
# ============================================================================

def compute_binary_metrics(
    probas,
    y_true,
):
    """Compute binary prediction metrics without generating figures.

    Args:
        probas:
            Positive-class predicted probabilities.
        y_true:
            Binary ground-truth labels.

    Returns:
        A dictionary containing ROC AUC, AUPRC, F1 score, MCC,
        Brier score, calibration intercept, calibration slope,
        ICI, E90, EMax, and the fold-specific F1 threshold.
    """
    probas = np.asarray(
        probas,
        dtype=float,
    ).reshape(-1)

    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    if len(probas) != len(y_true):
        raise ValueError(
            "probas and y_true must have the same length: "
            f"{len(probas)} != {len(y_true)}."
        )

    if np.unique(y_true).size < 2:
        raise ValueError(
            "Both classes are required to compute fold metrics."
        )

    precision, recall, _ = precision_recall_curve(
        y_true,
        probas,
    )

    calibration_stats = get_calibration_stats(
        probas,
        y_true,
    )

    thresholds = np.linspace(
        0.1,
        0.9,
        50,
    )

    fold_f1_scores = np.asarray(
        [
            f1_score(
                y_true,
                (probas >= threshold).astype(int),
                zero_division=0,
            )
            for threshold in thresholds
        ],
        dtype=float,
    )

    best_threshold_index = int(
        np.argmax(fold_f1_scores)
    )
    best_threshold = float(
        thresholds[best_threshold_index]
    )
    y_pred = (
        probas >= best_threshold
    ).astype(int)

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                probas,
            )
        ),
        "auprc": float(
            auc(
                recall,
                precision,
            )
        ),
        "f1_score": float(
            fold_f1_scores[
                best_threshold_index
            ]
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "best_threshold": best_threshold,
        "brier": float(
            calibration_stats["brier"]
        ),
        "calibration_intercept": float(
            calibration_stats["intercept"]
        ),
        "calibration_slope": float(
            calibration_stats["slope"]
        ),
        "ici": float(
            calibration_stats["ici"]
        ),
        "e90": float(
            calibration_stats["e90"]
        ),
        # "eMax": float(
        #     calibration_stats["eMax"]
        # ),
    }

def summarize_fold_metrics(
    fold_metrics,
    ddof=1,
):
    """Summarize metrics computed independently on multiple folds.

    Args:
        fold_metrics:
            Sequence of dictionaries returned by
            ``compute_binary_metrics``.
        ddof:
            Delta degrees of freedom used for the standard deviation.
            ``ddof=1`` computes the sample standard deviation.

    Returns:
        A dictionary containing per-fold values, mean, and standard
        deviation for every metric.
    """
    if not fold_metrics:
        raise ValueError(
            "fold_metrics must contain at least one fold."
        )

    metric_names = fold_metrics[0].keys()
    summary = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                fold_result[metric_name]
                for fold_result in fold_metrics
            ],
            dtype=float,
        )

        summary[metric_name] = {
            "fold_values": values,
            "mean": float(
                np.nanmean(values)
            ),
            "std": float(
                np.nanstd(
                    values,
                    ddof=ddof,
                )
            ),
        }

    return summary

# ============================================================================
# SINGLE-MODEL DIAGNOSTIC FIGURES
# ============================================================================

def roc_curve_homemade(probas, y_test, model_name, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Plot a receiver operating characteristic curve and compute its AUC.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        model_name:
            Model name displayed in the figure title and legend.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the ROC AUC, false-positive rates, true-positive
        rates, and decision thresholds.
    """
    (fpr, tpr, thresholds) = roc_curve(y_test, probas)
    # The displayed AUC is computed from pooled out-of-fold predictions.
    auc_final = roc_auc_score(y_test, probas)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC {model_name} (OOF AUC = {auc_final:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance', color = "green")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    _show_title(title, f'ROC Curve for model {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_figure :
        save_figure_file(plt, output_dir / Path("roc_curve.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()
    return auc_final, fpr, tpr, thresholds

def prc_curve_homemade(probas, y_test, model_name, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Plot a precision-recall curve and compute its area.

    The positive-class prevalence is displayed as the no-skill baseline, which
    makes the figure especially useful for imbalanced binary datasets.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        model_name:
            Model name displayed in the figure title and legend.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure files should be written.
        transparent:
            Whether the saved figures should use a transparent background.
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the area under the precision-recall curve,
        precision values, recall values, and decision thresholds.
    """
    # 1. Compute curve coordinates
    # Note: scikit-learn returns thresholds in ascending order
    # and appends precision=1.0 and recall=0.0 without a matching threshold.
    precision, recall, thresholds = precision_recall_curve(y_test, probas)
    # 2. Compute AUPRC with trapezoidal integration

    auprc_final = auc(recall, precision)

    # 3. Compute the baseline (chance level)
    # Unlike ROC, whose chance level is always 0.5, the PRC baseline
    # depends on the proportion of positive samples in the dataset.

    baseline = sum(y_test) / len(y_test)
    # 4. Build the figure

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PRC {model_name} (OOF AUPRC = {auprc_final:.3f})', color='blue', linewidth=2)
    # Horizontal baseline representing chance
    plt.axhline(y=baseline, linestyle='--', color='green', label=f'Chance (Ratio Positifs = {baseline:.3f})')

    plt.xlabel('Recall (True Positive Rate / Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    _show_title(title, f'Precision-Recall Curve for model {model_name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.grid(True)

    if save_figure:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        save_figure_file(plt, path / "prc_curve", output_format, bbox_inches="tight", transparent=transparent)

    plt.show()

    return auprc_final, precision, recall, thresholds

def kde_plot_homemade(probas, y_test, model_name, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Plot class-conditional probability densities and quantify separation.

    Kernel density estimates are created for survivors and deaths. The
    function computes their non-overlapping area, compares class-specific
    uncertainty, measures the difference between mean predicted risks, and
    estimates a decision threshold from the first density intersection.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels used to split the predictions.
        model_name:
            Model name displayed in the figure title and console output.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the non-overlapping density area, the absolute
        difference between class standard deviations, the absolute difference
        between class mean risks, and the mean risk for the positive class.
    """
    plt.figure()

    p0 = probas[y_test == 0]
    p1 = probas[y_test == 1]

    sns.kdeplot(p0, label="Survivors", fill=True)
    sns.kdeplot(p1, label="Deaths", fill=True)

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    _show_title(title, f"Score distribution (KDE) for {model_name}")
    plt.legend()
    plt.grid()

    # Compute the overlap area
    kde0 = gaussian_kde(p0)
    kde1 = gaussian_kde(p1)

    # Create an x-axis on which to evaluate the densities
    x = np.linspace(-0.1, 1.1, 2000)

    y0 = kde0(x)
    y1 = kde1(x)

    intersection = np.minimum(y0, y1)

    overlap_area = np.trapezoid(intersection, x)
    non_overlap_area = 1.0 - overlap_area

    plt.text(0.5, 0.9, f"Non-overlapping area: {non_overlap_area:.1%}",
             transform=plt.gca().transAxes, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))

    if save_figure :
        save_figure_file(plt, output_dir / Path("kde_plot.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()

    # --- SHAPE IMBALANCE ANALYSIS ---
    # Locate the exact intersection points between both density curves
    # An intersection occurs where the density difference changes sign
    intersection_indices = np.argwhere(np.diff(np.sign(y0 - y1))).flatten()

    # Compute the mean predicted risk for each group
    mean_p0 = np.mean(p0)
    mean_p1 = np.mean(p1)
    mean_risk_diff = abs(mean_p0 - mean_p1)

    # Compute the standard deviation as a measure of model uncertainty
    std_p0 = np.std(p0)
    std_p1 = np.std(p1)
    print(f"[{model_name}] --- Shape & Fairness Analysis ---")
    print(f"[{model_name}] Survivors -> Mean Risk: {mean_p0:.2f} (±{std_p0:.2f})")
    print(f"[{model_name}] Deaths    -> Mean Risk: {mean_p1:.2f} (±{std_p1:.2f})")
    # A substantially larger standard deviation for deaths indicates
    # that the model is less certain for deaths than for survivors.
    asymmetric_uncertainty = abs(std_p0 - std_p1)
    if asymmetric_uncertainty > 0.05:
        print(f"Warning: Unbalanced uncertainty! The model is less confident on one of the classes.")

    if len(intersection_indices) > 0:
        # Use the first detected intersection, usually the main one in [0, 1]
        crossing_threshold = x[intersection_indices[0]]
        print(f"[{model_name}] Decision Threshold (KDE Cross): {crossing_threshold:.2f}")
    else:
        crossing_threshold = -1
        print(f"[{model_name}] No decision threshold found (curves do not cross).")
    return non_overlap_area, asymmetric_uncertainty, mean_risk_diff, mean_p1

def brier_evolution(probas, y_test, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Analyze Brier scores across fixed risk brackets and patient deciles.

    The function computes the global Brier score and generates four figures:
    Brier score by fixed risk bracket, Brier score by patient decile,
    calibration by patient decile, and calibration by fixed risk bracket.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        save_figure:
            Whether the generated figures should be saved.
        output_dir:
            Directory in which the figures should be written.
        transparent:
            Whether the saved figures should use a transparent background.
        title:
            Optional figure title prefix. Omit (default ``...``) to keep the
            automatic titles, pass ``None`` to remove them, or provide a custom
            string prepended to each subtitle.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        The global Brier score.
    """
    df_brier = pl.DataFrame({"y" : y_test, "pred" : probas})

    df_brier = df_brier.with_columns(
        ((pl.col("pred") - pl.col("y")) ** 2).alias("brier")
    )

    # Global Brier score
    global_brier = brier_score_loss(df_brier["y"], df_brier["pred"])
    print("Brier score: ",global_brier)

    # Fixed risk bins
    fixed_brier_df = (
        df_brier.with_columns(
            (
                pl.col("pred")
                .clip(0, 0.999999)
                .mul(10)
                .floor()
                .cast(pl.Int64)
            ).alias("bin_fixed")
        )
        .group_by("bin_fixed")
        .agg([
            pl.col("brier").mean().alias("brier_mean"),
            pl.len().alias("n"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        ])
        .sort("bin_fixed")
        .with_columns(
            ((pl.col("bin_fixed") + 0.5) / 10).alias("x")
        )
    )

    # Patient deciles
    n_total = df_brier.height

    decile_brier_df = (
        df_brier.sort("pred")
        .with_row_count("row_idx")
        .with_columns(
            (
                (pl.col("row_idx") * 10 / n_total)
                .floor()
                .clip(upper_bound=9)
                .cast(pl.Int64)
            ).alias("decile")
        )
        .group_by("decile")
        .agg([
            pl.col("brier").mean().alias("brier_mean"),
            pl.len().alias("n"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        ])
        .sort("decile")
        .with_columns(
            (pl.col("decile") + 1).alias("x")
        )
    )

    fixed_brier_pd = fixed_brier_df.to_pandas()
    decile_brier_pd = decile_brier_df.to_pandas()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(fixed_brier_pd["x"], fixed_brier_pd["n"], width=0.08, alpha=0.3)
    ax1.set_xlabel("Predicted risk")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(fixed_brier_pd["x"], fixed_brier_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Mean Brier score")

    _show_title(title, "Brier score by fixed risk brackets", ax=ax1)
    plt.tight_layout()
    if save_figure:
        save_figure_file(plt, output_dir / Path("brier_per_risk_bracket.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(decile_brier_pd["x"], decile_brier_pd["n"], alpha=0.3)
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Mean Brier score")

    _show_title(title, "Brier score by patient deciles", ax=ax1)
    plt.tight_layout()
    if save_figure:
        save_figure_file(plt, output_dir / Path("brier_per_decile.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(decile_brier_pd["x"], decile_brier_pd["n"], alpha=0.3, color="grey", edgecolor="black")
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["obs_rate"], marker="o", label="Predicted risk", color="black")
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")

    _show_title(title, "Calibration Curve by patient deciles", ax=ax1)
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure:
        save_figure_file(plt, output_dir / Path("calib_per_decile.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(fixed_brier_pd["x"], fixed_brier_pd["n"], width=0.08, alpha=0.3, color="grey", edgecolor="black")
    ax1.set_xlabel("Predicted risk (10% brackets)")
    ax1.set_ylabel("Number of patients")
    ax1.set_xlim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(fixed_brier_pd["x"], fixed_brier_pd["obs_rate"], marker="o", label="Mean predicted risk", color="black", linestyle="-")
    ax2.plot(fixed_brier_pd["x"], fixed_brier_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")
    ax2.set_ylim(0, 1)

    _show_title(title, "Calibration Curve by fixed risk brackets", ax=ax1)
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure:
        save_figure_file(plt, output_dir / Path("calib_per_risk_bracket.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()
    return global_brier


def calibration_per_risk_brackets(
    probas,
    y_test,
    save_figure=False,
    output_dir=None,
    transparent=False,
    model_name="Model",
    title=...,

    output_format: str = "pdf",
):
    """Plot a calibration curve aggregated by fixed 10% risk brackets using get_calibration_stats.

    Args:
        probas (array-like): Predicted probabilities for the positive class.
        y_test (array-like): Binary ground-truth labels (0 or 1).
        save_figure (bool, optional): Whether to save the figure to disk. Defaults to False.
        output_dir (str or Path, optional): Path where the PNG will be saved. Defaults to None.
        transparent (bool, optional): Whether the background should be transparent when saved. Defaults to False.
        model_name (str, optional): Model name displayed in file name. Defaults to "Model".
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        dict: The dictionary containing computed calibration statistics and data.
    """
    # 1. Refactored Call: Get all stats and binned data
    stats = get_calibration_stats(probas, y_test)

    # 2. Setup Figure & Axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Patient distribution histogram (Left Axis)
    ax1.bar(
        stats["x"], 
        stats["n"], 
        width=0.08, 
        alpha=0.3, 
        color="grey", 
        edgecolor="black"
    )
    ax1.set_xlabel("Predicted Risk / Probability")
    ax1.set_ylabel("Number of Patients")
    ax1.set_xlim(0, 1)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))

    # Calibration curves (Right Axis)
    ax2 = ax1.twinx()
    ax2.plot(
        stats["x"], 
        stats["pred_mean"], 
        marker="o", 
        label="Mean predicted risk", 
        color="black", 
        linestyle="-"
    )
    ax2.plot(
        stats["x"], 
        stats["obs_rate"], 
        marker="s", 
        label="Observed mortality", 
        color="black", 
        linestyle="--"
    )
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")
    ax2.set_ylim(0, 1)

    # 3. Display Stats Box
    text_str = (
        f"Intercept: {stats['intercept']:.2f}\n"
        f"Slope: {stats['slope']:.2f}\n"
        f"ICI: {stats['ici']:.3f}\n"
        f"E90: {stats['e90']:.3f}\n"
        # f"EMax: {stats['eMax']:.3f}\n"
        f"Brier: {stats['brier']:.3f}"
    )

    text_box_properties = {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85,
        "edgecolor": "gray"
    }

    ax2.text(
        0.05,
        0.95,
        text_str,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=text_box_properties,
    )

    # Legend & Layout
    fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))

    _show_title(title, f"Calibration Curve - {model_name}", ax=ax1)

    fig.tight_layout()

    # 4. Save & Close
    if save_figure and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_figure_file(fig, output_path / f"{model_name}_calib_per_risk_bracket.png", output_format, bbox_inches="tight", transparent=transparent)

    plt.show()
    plt.close(fig)

    return stats

def f1_score_evolution(probas, y_test, model_name, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Evaluate the F1 score over a grid of decision thresholds.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        model_name:
            Model name displayed in the figure title.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the highest F1 score and its associated threshold.
    """
    _thresholds = np.linspace(0.1, 0.9, 50)
    f1s = []
    best_f1 = 0
    for t in _thresholds:
        _y_pred = (probas >= t).astype(int)
        f1s.append(f1_score(y_test, _y_pred))
        f1 = f1_score(y_test, _y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    plt.figure(figsize=(6, 6))
    plt.plot(_thresholds, f1s)
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    _show_title(title, f"F1 score evolution vs Threshold for model {model_name}")
    plt.grid()
    if save_figure:
        save_figure_file(plt, output_dir / Path("threshold_evolution.png"), output_format, bbox_inches="tight", transparent=transparent)
    plt.show()
    print(f'The best F1 score of{best_f1: .2f} is reached at threshold{best_t: .2f}')
    return best_f1, best_t


def confusion_matrix_homemade(probas, y_test, best_t, model_name, save_figure, output_dir, transparent, title=...,
    output_format: str = "pdf",
):
    """Plot a confusion matrix at a selected probability threshold.

    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        best_t:
            Probability threshold used to convert probabilities into classes.
        model_name:
            Model name displayed in the figure title.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        title:
            Optional figure title. Omit (default ``...``) to keep the automatic
            title, pass ``None`` to remove it, or provide a custom string.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing the predicted binary labels and the Matthews
        correlation coefficient.
    """
    y_pred = (probas >= best_t).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _show_title(title, f"Confusion matrix of model {model_name}  (threshold={ best_t: .2f}, MCC = {mcc})")
    if save_figure:
        save_figure_file(plt, output_dir / Path("confusion_matrix.png"), output_format, bbox_inches="tight", transparent=transparent, facecolor="white")
    plt.show()
    return y_pred, mcc

# ============================================================================
# SINGLE-MODEL REPORT ORCHESTRATION
# ============================================================================

def plot_all_figs(
    probas_uncalib,
    y_test,
    config_models,
    calibration,
    save_figure,
    output_dir,
    transparent,
    probas_calib=None,
    calibration_mode="Platt",

    output_format: str = "pdf",
):
    """Generate standard evaluation figures and collect their metrics.

    Args:
        probas_uncalib:
            Uncalibrated positive-class probabilities.
        y_test:
            Binary ground-truth labels.
        config_models:
            Model configuration exposing ``models_name``.
        calibration:
            Whether calibrated predictions should be displayed.
        save_figure:
            Whether generated figures should be saved.
        output_dir:
            Directory in which figures should be written.
        transparent:
            Whether saved figures should use a transparent background.
        probas_calib:
            Optional calibrated positive-class probabilities.
        calibration_mode:
            Name of the calibration method displayed in the legend.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing ROC metrics, Brier score, optimal F1 information,
        predicted labels, MCC, and KDE-based separation statistics.
    """
    probas_uncalib, y_test = _validate_calibration_inputs(
        probas_uncalib,
        y_test,
    )

    if calibration:
        if probas_calib is None:
            raise ValueError(
                "probas_calib must be provided when calibration=True."
            )

        probas_calib, y_test_calib = _validate_calibration_inputs(
            probas_calib,
            y_test,
        )

        if not np.array_equal(y_test, y_test_calib):
            raise ValueError(
                "The calibrated and uncalibrated predictions must share "
                "the same ground-truth labels."
            )

    probabilities = (
        probas_calib
        if calibration and probas_calib is not None
        else probas_uncalib
    )

    cfg = {
        "save_figure": save_figure,
        "output_dir": output_dir,
        "transparent": transparent,
    }

    calibration_curve_homemade(
        probas_uncalib=probas_uncalib,
        probas_calib=probas_calib,
        y_test_global=y_test,
        model_name=config_models.models_name,
        calibration=calibration,
        calibration_mode=calibration_mode,
        **cfg,
    output_format=output_format)

    auc_final, fpr, tpr, thresholds_roc = roc_curve_homemade(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    output_format=output_format)

    auprc_final, precision, recall, thresholds_prc = prc_curve_homemade(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    output_format=output_format)

    (
        non_overlap_area,
        asymmetric_uncertainty,
        mean_risk_diff,
        mean_p1,
    ) = kde_plot_homemade(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    output_format=output_format)

    brier_score = brier_evolution(
        probabilities,
        y_test,
        **cfg,
    output_format=output_format)

    best_f1, best_t = f1_score_evolution(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    output_format=output_format)

    y_pred, mcc = confusion_matrix_homemade(
        probabilities,
        y_test,
        best_t,
        config_models.models_name,
        **cfg,
    output_format=output_format)

    return (
        auc_final,
        fpr,
        tpr,
        thresholds_roc,
        auprc_final,
        precision,
        recall,
        thresholds_prc,
        brier_score,
        best_f1,
        best_t,
        y_pred,
        mcc,
        non_overlap_area,
        asymmetric_uncertainty,
        mean_risk_diff,
        mean_p1,
    )

# ============================================================================
# SAVED-FIGURE COMPARISON
# ============================================================================

