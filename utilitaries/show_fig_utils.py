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

from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

import utilitaries.features_extraction_utils as feu
from pathlib import Path

import utilitaries.evaluate_utils as evaluate
from utilitaries.postprocessing_utils import bootstrap_holdout_metrics


def plot_collected_learning_curve(
    sample_sizes,
    train_matrix,
    val_matrix,
    model_name="Model",
    folder="",
    savefig=True,
    transparent=True,
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

    plt.title(
        f"Learning Curve — {model_name} (Embedded Fold Splitting)",
        fontsize=13,
        fontweight="bold",
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
        plt.savefig(
            output_path / "learning_curve.png",
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )

    plt.show()


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

        fig.savefig(
            output_path / filename,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

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
    )


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

        fig.savefig(
            output_path / filename,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

    plt.show()
    plt.close(fig)


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
        dict: A dictionary containing intercept, slope, brier, ici, e90, eMax,
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
        "eMax": float(np.max(bin_errors)),
        "binned_df": fixed_brier_df,
        "x": fixed_brier_pd["x"].to_numpy(),
        "obs_rate": fixed_brier_pd["obs_rate"].to_numpy(),
        "pred_mean": fixed_brier_pd["pred_mean"].to_numpy(),
        "n": fixed_brier_pd["n"].to_numpy(),
    }



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
        "eMax": float(
            calibration_stats["eMax"]
        ),
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

def roc_curve_homemade(probas, y_test, model_name, save_figure, output_dir, transparent):
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
    plt.title(f'ROC Curve for model {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_figure :
        plt.savefig(output_dir / Path("roc_curve"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    return auc_final, fpr, tpr, thresholds

def prc_curve_homemade(probas, y_test, model_name, save_figure, output_dir, transparent):
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
    plt.title(f'Precision-Recall Curve for model {model_name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) # Add a small upper margin
    plt.legend(loc='lower left') # PRC curves commonly decrease toward the right
    plt.grid(True)
    
    if save_figure:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / "prc_curve.png", dpi=300, bbox_inches="tight", transparent=transparent)
        plt.savefig(path / "prc_curve.pdf", bbox_inches="tight", transparent=transparent)
        
    plt.show()
    
    return auprc_final, precision, recall, thresholds

def kde_plot_homemade(probas, y_test, model_name, save_figure, output_dir, transparent):
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
    
    Returns:
        A tuple containing the non-overlapping density area, the absolute
        difference between class standard deviations, the absolute difference
        between class mean risks, and the mean risk for the positive class.
    """
    plt.figure()

    # Split probabilities by observed class
    p0 = probas[y_test == 0]
    p1 = probas[y_test == 1]

    sns.kdeplot(p0, label="Survivors", fill=True)
    sns.kdeplot(p1, label="Deaths", fill=True)

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(f"Score distribution (KDE) for {model_name}")
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
        plt.savefig(output_dir / Path("kde_plot"), dpi = 300, bbox_inches="tight", transparent=transparent)
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

def brier_evolution(probas, y_test, save_figure, output_dir, transparent):
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

    plt.title(f"Brier score by fixed risk brackets")
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("brier_per_risk_bracket"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(decile_brier_pd["x"], decile_brier_pd["n"], alpha=0.3)
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Mean Brier score")

    plt.title(f"Brier score by patient deciles")
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("brier_per_decile"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Patient histogram
    ax1.bar(decile_brier_pd["x"], decile_brier_pd["n"], alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    # Calibration curve
    ax2 = ax1.twinx()
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["obs_rate"], marker="o", label="Predicted risk", color="black")
    ax2.plot(decile_brier_pd["x"], decile_brier_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")

    plt.title(f"Calibration Curve by patient deciles")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("calib_per_decile"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Patient histogram (fixed-risk brackets)
    ax1.bar(fixed_brier_pd["x"], fixed_brier_pd["n"], width=0.08, alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Predicted risk (10% brackets)")
    ax1.set_ylabel("Number of patients")
    ax1.set_xlim(0, 1)

    # Calibration curve (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(fixed_brier_pd["x"], fixed_brier_pd["obs_rate"], marker="o", label="Mean predicted risk", color="black", linestyle="-")
    ax2.plot(fixed_brier_pd["x"], fixed_brier_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")
    ax2.set_ylim(0, 1) 

    plt.title(f"Calibration Curve by fixed risk brackets")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("calib_per_risk_bracket"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    return global_brier

def calibration_per_risk_brackets(
    probas,
    y_test,
    save_figure=False,
    output_dir=None,
    transparent=False,
    model_name="Model"
):
    """Plot a calibration curve aggregated by fixed 10% risk brackets using get_calibration_stats.

    Args:
        probas (array-like): Predicted probabilities for the positive class.
        y_test (array-like): Binary ground-truth labels (0 or 1).
        save_figure (bool, optional): Whether to save the figure to disk. Defaults to False.
        output_dir (str or Path, optional): Path where the PNG will be saved. Defaults to None.
        transparent (bool, optional): Whether the background should be transparent when saved. Defaults to False.
        model_name (str, optional): Model name displayed in file name. Defaults to "Model".

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
        f"EMax: {stats['eMax']:.3f}\n"
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
    fig.tight_layout()

    # 4. Save & Close
    if save_figure and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path / f"{model_name}_calib_per_risk_bracket.png", 
            dpi=300, 
            bbox_inches="tight", 
            transparent=transparent
        )

    plt.show()
    plt.close(fig)

    return stats

def f1_score_evolution(probas, y_test, model_name, save_figure, output_dir, transparent):
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
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title(f"F1 score evolution vs Threshold for model {model_name}")
    plt.grid()
    if save_figure:
        plt.savefig(output_dir / Path("threshold_evolution"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()
    print(f'The best F1 score of{best_f1: .2f} is reached at threshold{best_t: .2f}')
    return best_f1, best_t


def confusion_matrix_homemade(probas, y_test, best_t, model_name, save_figure, output_dir, transparent):
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
    
    Returns:
        A tuple containing the predicted binary labels and the Matthews
        correlation coefficient.
    """
    y_pred = (probas >= best_t).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion matrix of model {model_name}  (threshold={ best_t: .2f}, MCC = {mcc})')
    if save_figure :
        plt.savefig(output_dir / Path("confusion_matrix"), dpi = 300, bbox_inches="tight", transparent=transparent, facecolor = "white")
    plt.show()
    return y_pred, mcc

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
    )

    auc_final, fpr, tpr, thresholds_roc = roc_curve_homemade(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    )

    auprc_final, precision, recall, thresholds_prc = prc_curve_homemade(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    )

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
    )

    brier_score = brier_evolution(
        probabilities,
        y_test,
        **cfg,
    )

    best_f1, best_t = f1_score_evolution(
        probabilities,
        y_test,
        config_models.models_name,
        **cfg,
    )

    y_pred, mcc = confusion_matrix_homemade(
        probabilities,
        y_test,
        best_t,
        config_models.models_name,
        **cfg,
    )

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

def mesureImportance_tsfel(model, X_train, varnames, top_n=20, class_labels=None, folder="", savefig=True, transparent=True, seed=42):
    """Analyze and plot TSFEL feature importance with MDI and SHAP.
    
    The function unwraps fitted estimators when necessary, computes individual
    and globally aggregated feature importances, creates SHAP summary plots,
    and groups observations by predicted class. Feature names sharing a TSFEL
    suffix are aggregated under their common root feature.
    
    Args:
        model:
            Fitted estimator or fitted estimator wrapper.
        X_train:
            Training features as a Polars DataFrame, pandas DataFrame, or
            array-like object.
        varnames:
            Feature names corresponding to the columns of ``X_train``.
        top_n:
            Maximum number of individual or aggregated features displayed.
        class_labels:
            Optional human-readable labels for output classes.
        folder:
            Directory in which generated figures should be written.
        savefig:
            Whether generated figures should be saved.
        transparent:
            Whether PDF figures should use a transparent background.
        seed:
            NumPy random seed used for reproducibility.
    
    Returns:
        A dictionary containing samples grouped by predicted class, the top
        individual SHAP features, the top aggregated MDI features, and the top
        aggregated SHAP features.
    
    Raises:
        ValueError:
            If SHAP returns an unsupported output format.
    """
    np.random.seed(seed)
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()
    
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values
    else:
        X_arr = np.asarray(X_train) 
        
    varnames = list(varnames)
    k = int(min(top_n, len(varnames)))
    
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    shap_model = evaluate.get_root_estimator(model)
    # ----- 1) Tree-based feature importance -----
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        importances = np.mean([
            clf.estimator.feature_importances_ 
            for clf in model.calibrated_classifiers_
        ], axis=0)
    elif hasattr(shap_model, 'feature_importances_'): # Additional safeguard when a wrapper hides the attribute
        importances = shap_model.feature_importances_
    else:
        print("⚠️ This model does not expose 'feature_importances_'. Falling back to SHAP.")

    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        sorted_feature_names = np.array(varnames)[sorted_idx]
        sorted_importances = importances[sorted_idx]

        # Aggregate remaining individual MDI values
        if len(sorted_importances) > k:
            other_importance = np.sum(sorted_importances[k:])
            display_importances = np.append(sorted_importances[:k], other_importance)
            display_feature_names = np.append(sorted_feature_names[:k], f"Others Features (N={len(sorted_feature_names[:k])})")
        else:
            display_importances = sorted_importances
            display_feature_names = sorted_feature_names

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(display_importances)), display_importances)
        plt.xticks(range(len(display_feature_names)), display_feature_names, rotation=90)
        plt.ylabel("Importance (forest)")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}/feature_importance.pdf", bbox_inches="tight", transparent=transparent)
            plt.savefig(f"{folder}/feature_importance.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Aggregate feature importance by root feature name
        suffixes = feu.generer_suffixes_tsfel()
        root_feature_names = [feu.extraire_racine(name, suffixes) for name in varnames]
        mdi_df = pd.DataFrame({
            'Global_Feature': root_feature_names,
            'Importance': importances
        })
        
        aggregated_mdi_df = mdi_df.groupby('Global_Feature').sum().sort_values(by='Importance', ascending=False)
        aggregated_top_k = int(min(top_n, len(aggregated_mdi_df)))

        # Aggregate remaining global MDI values
        if len(aggregated_mdi_df) > aggregated_top_k:
            other_aggregated_mdi = aggregated_mdi_df.iloc[aggregated_top_k:].sum()
            mdi_plot_df = pd.concat([
                aggregated_mdi_df.head(aggregated_top_k),
                pd.DataFrame([other_aggregated_mdi], index=["Other Global Features"])
            ])
        else:
            mdi_plot_df = aggregated_mdi_df
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(mdi_plot_df)), mdi_plot_df['Importance'])
        plt.xticks(range(len(mdi_plot_df)), mdi_plot_df.index, rotation=90)
        plt.ylabel("Cumulative Global Importance (forest)")
        plt.title("Top Global Feature Importance (MDI)")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}/global_feature_importance_mdi.pdf", bbox_inches="tight", transparent=transparent)
            plt.savefig(f"{folder}/global_feature_importance_mdi.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        suffixes = feu.generer_suffixes_tsfel()
        root_feature_names = [feu.extraire_racine(name, suffixes) for name in varnames]

    # ----- 2) SHAP values -----
    x_numpy = np.array(X_arr, dtype=np.float32)

    if hasattr(shap_model, 'n_jobs'):
        shap_model.n_jobs = -1
        
    if isinstance(shap_model, XGBClassifier):
        explainer = shap.TreeExplainer(shap_model)
    else:
        try:
            explainer = shap.Explainer(shap_model)
        except Exception:
            explainer = shap.TreeExplainer(shap_model)
            
    shap_values = explainer.shap_values(x_numpy)

    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=-1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_arr = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_arr = shap_values[:, :, None]
    else:
        raise ValueError(f"Unexpected SHAP format: type={type(shap_values)}")

    n_samples, n_features, n_classes = shap_arr.shape

    # ----- 3) Individual SHAP aggregations -----
    mean_abs_by_class = np.abs(shap_arr).mean(axis=0)  # (n_features, n_classes)
    feature_sum = mean_abs_by_class.sum(axis=1)
    
    top_idx = np.argsort(feature_sum)[::-1][:min(k, n_features)]
    
    # Aggregate remaining individual SHAP values
    if n_features > k:
        other_idx = np.argsort(feature_sum)[::-1][k:]
        others_shap = mean_abs_by_class[other_idx].sum(axis=0) # Sum by class
        plot_data = np.vstack([mean_abs_by_class[top_idx], others_shap])
        plot_labels = [varnames[i] for i in top_idx] + ["Other Features"]
    else:
        plot_data = mean_abs_by_class[top_idx]
        plot_labels = [varnames[i] for i in top_idx]

    if class_labels is not None:
        class_names = list(class_labels)
    else:
        class_names = ["0", "1"] if (shap_arr.ndim == 3 or (shap_arr.ndim == 2 and "XGB" in str(type(shap_model)))) else [str(i) for i in range(n_classes)]

    if n_classes == 1 and len(class_names) > 1:
        class_names = [class_names[-1]]

    # ----- 4) Stacked SHAP bars -----
    fig, ax = plt.subplots(figsize=(16, 8))
    left = np.zeros(len(plot_labels))
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(len(plot_labels)),
            width=plot_data[:, c_id],
            left=left,
            label=class_names[c_id]
        )
        left += plot_data[:, c_id]
        
    ax.set_yticks(np.arange(len(plot_labels)))
    ax.set_yticklabels(plot_labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|) (average impact per class)")
    ax.legend(title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}/shap_importance.pdf", bbox_inches="tight", transparent=transparent)
        plt.savefig(f"{folder}/shap_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ----- 5) Class-specific summary plots -----
    for class_id, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_arr[..., class_id],
            X_arr,
            feature_names=varnames,
            show=False
        )
        plt.title(f"SHAP Value Impact for {class_name}")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}/shap_values_{class_name}.pdf", bbox_inches="tight", transparent=transparent)
            plt.savefig(f"{folder}/shap_values_{class_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    # ----- 6) Split samples by predicted class -----
    y_pred = model.predict(X_arr)
    unique_classes = np.unique(y_pred)
    X_by_class = {cls: X_arr[y_pred == cls] for cls in unique_classes}
    
    # ----- 7) Globally aggregated SHAP importance -----
    shap_data = {'Global_Feature': root_feature_names}
    for c_id in range(n_classes):
        shap_data[f'SHAP_class_{c_id}'] = mean_abs_by_class[:, c_id]
    
    aggregated_shap_df = pd.DataFrame(shap_data).groupby('Global_Feature').sum()
    aggregated_shap_df['Total_Impact'] = aggregated_shap_df.sum(axis=1)
    aggregated_shap_df = aggregated_shap_df.sort_values(by='Total_Impact', ascending=False).drop(columns=['Total_Impact'])
    
    aggregated_shap_top_k = int(min(top_n, len(aggregated_shap_df)))
    
    # Aggregate remaining global SHAP values
    if len(aggregated_shap_df) > aggregated_shap_top_k:
        other_aggregated_shap = aggregated_shap_df.iloc[aggregated_shap_top_k:].sum()
        shap_plot_df = pd.concat([
            aggregated_shap_df.head(aggregated_shap_top_k),
            pd.DataFrame([other_aggregated_shap], index=["Other Global Features"])
        ])
    else:
        shap_plot_df = aggregated_shap_df
    
    # Globally aggregated SHAP figure
    fig, ax = plt.subplots(figsize=(16, 8))
    aggregated_left = np.zeros(len(shap_plot_df))
    
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(len(shap_plot_df)),
            width=shap_plot_df[f'SHAP_class_{c_id}'].values,
            left=aggregated_left,
            label=class_names[c_id]
        )
        aggregated_left += shap_plot_df[f'SHAP_class_{c_id}'].values
        
    ax.set_yticks(np.arange(len(shap_plot_df)))
    ax.set_yticklabels(shap_plot_df.index)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative mean(|SHAP value|) (average impact per class)")
    ax.set_title("Top Global Feature Importance (SHAP)")
    ax.legend(title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}/global_shap_importance.pdf", bbox_inches="tight", transparent=transparent)
        plt.savefig(f"{folder}/global_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Return only original top features, excluding the aggregated "Others" category
    return {
        "X_by_class": X_by_class,
        "top_feat": [varnames[i] for i in top_idx],
        "top_global_feat_mdi" : list(aggregated_mdi_df.index[:aggregated_top_k]) if importances is not None else [],
        "top_global_feat_shap" : list(aggregated_shap_df.index[:aggregated_shap_top_k])
    }

def compare_models_figure(figname, max_cols=3, savefig = False, folder = "", **paths):
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
        plt.savefig(f"{folder}/comparison_{figname}", dpi=300, bbox_inches="tight")
    plt.show()

def _build_holdout_metric_functions():
    """Return metric functions compatible with bootstrap_holdout_metrics.

    Returns:
        A dictionary mapping metric names to callables accepting
        ``(y_true, probabilities)``.
    """
    from sklearn.metrics import (
        brier_score_loss,
        roc_auc_score,
    )

    def _auprc(y_true, probas):
        p, r, _ = precision_recall_curve(y_true, probas)
        return float(auc(r, p))

    return {
        "auc": roc_auc_score,
        "auprc": _auprc,
        "brier": brier_score_loss,
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

    metric_functions = _build_holdout_metric_functions()
    unique_labels = np.unique(subgroup_labels)
    results = []

    for label in unique_labels:
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

        # Point estimates via compute_binary_metrics
        try:
            metrics = compute_binary_metrics(sub_probas, sub_y_true)
        except ValueError as exc:
            print(
                f"[compute_subgroup_holdout_metrics] Skipping subgroup "
                f"'{display_name}': {exc}"
            )
            continue

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
):
    """Generate a comparative holdout report across patient subgroups.

    Convenience wrapper that chains
    :func:`compute_subgroup_holdout_metrics` and
    :func:`generate_comparative_report` in **"holdout"** mode.  Holdout
    predictions are split by subgroup, bootstrap confidence intervals are
    computed per subgroup, and a collective report (tables + figures) is
    generated.

    Args:
        probas:
            Array-like of predicted probabilities for the positive class.
        y_true:
            Array-like of binary ground-truth labels.
        subgroup_labels:
            Array-like of categorical subgroup identifiers (same length as
            ``probas``).
        subgroup_names:
            Optional mapping from raw subgroup label to a human-readable name.
        n_bootstrap:
            Number of bootstrap resamples per subgroup.
        confidence_level:
            Confidence level for percentile intervals.
        seed:
            Random seed for reproducibility.
        save_dir:
            Optional output directory for figures and tables.
        table_format:
            Console table format passed to :func:`tabulate`.

    Returns:
        A pandas DataFrame containing formatted metric values (with
        estimate [CI95%]) per subgroup, as returned by
        :func:`generate_comparative_report`.
    """
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
    )


def generate_comparative_report(
    configurations,
    y_true_base=None,
    save_dir=None,
    table_format="fancy_grid",
    evaluation_mode="oof",
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

    Returns:
        A pandas DataFrame containing formatted metric values.
    """
    if evaluation_mode not in ("oof", "holdout"):
        raise ValueError(
            f"evaluation_mode must be 'oof' or 'holdout', got {evaluation_mode!r}."
        )

    configurations = list(configurations)

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
        model_name = item[0]
        if model_name in predefined_order:
            return 0, predefined_order.index(model_name)
        return 1, model_name

    def require_key(config, key, model_name):
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
    reference_y = None

    for idx, (name, config) in enumerate(configurations):
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

        if reference_y is None:
            reference_y = y_true

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
            # For holdout calibration stats, compute from probas/y_true directly
            cal_stats = get_calibration_stats(probas, y_true)
            intercept_val = cal_stats["intercept"]
            slope_val = cal_stats["slope"]
            ici_val = cal_stats["ici"]

        # ---- Results table ----
        if evaluation_mode == "oof":
            results[name] = {
                "AUC ROC": format_metric(config, "auc", name),
                "AUPRC": format_metric(config, "auprc", name),
                "F1-Score": format_metric(config, "f1_score", name),
                "MCC": format_metric(config, "mcc", name),
                "Brier": format_metric(config, "brier", name),
                "Intercept": format_metric(config, "calibration_intercept", name),
                "Slope": format_metric(config, "calibration_slope", name),
                "ICI": format_metric(config, "ici", name),
                "E90": format_metric(config, "e90", name),
                "EMax": format_metric(config, "eMax", name),
            }
        else:
            results[name] = {
                "AUC ROC": format_metric(config, "auc", name),
                "AUPRC": format_metric(config, "auprc", name),
                "F1-Score": format_metric(config, "f1", name),
                "MCC": format_metric(config, "mcc", name),
                "Brier": format_metric(config, "brier", name),
            }

        label_prefix = "OOF" if evaluation_mode == "oof" else "Holdout"

        plot_data_list.append({
            "name": name,
            "color": color,
            "fpr": fpr,
            "tpr": tpr,
            "auc_val": auc_val,
            "recall": recall,
            "precision": precision,
            "auprc_val": auprc_val,
            "fop": fop,
            "mpv": mpv,
            "intercept_val": intercept_val,
            "slope_val": slope_val,
            "ici_val": ici_val,
            "label_prefix": label_prefix,
        })

    # ---- Figures ----
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8), layout="constrained")
    fig_prc, ax_prc = plt.subplots(figsize=(8, 8), layout="constrained")
    fig_cal, ax_cal = plt.subplots(figsize=(8, 8), layout="constrained")

    for item in plot_data_list:
        prefix = item["label_prefix"]
        ax_roc.plot(
            item["fpr"], item["tpr"],
            label=f"{item['name']} ({prefix} AUC = {item['auc_val']:.3f})",
            color=item["color"], linewidth=2,
        )
        ax_prc.plot(
            item["recall"], item["precision"],
            label=f"{item['name']} ({prefix} AUPRC = {item['auprc_val']:.3f})",
            color=item["color"], linewidth=2,
        )
        cal_label = (
            f"{item['name']} "
            f"(Int={item['intercept_val']:.2f}, "
            f"Slope={item['slope_val']:.2f}, "
            f"ICI={item['ici_val']:.3f})"
        )
        ax_cal.plot(
            item["mpv"], item["fop"], "s-",
            label=cal_label, color=item["color"], linewidth=2,
        )

    # ROC styling
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Chance", color="gray")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.05)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=9)

    # PRC styling
    baseline = float(np.mean(reference_y)) if reference_y is not None else 0.5
    ax_prc.axhline(y=baseline, linestyle="--", color="green", alpha=0.7,
                    label=f"Chance (Pos Ratio = {baseline:.3f})")
    ax_prc.set_xlabel("Recall (Sensitivity)")
    ax_prc.set_ylabel("Precision (PPV)")
    ax_prc.set_xlim(0.0, 1.0)
    ax_prc.set_ylim(0.0, 1.05)
    ax_prc.grid(True, linestyle=":", alpha=0.6)
    ax_prc.legend(loc="upper right", fontsize=9)

    # Calibration styling
    ax_cal.plot([0, 1], [0, 1], "k:", alpha=0.7, label="Perfect calibration")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("True Fraction of Positives")
    ax_cal.set_xlim(0.0, 1.0)
    ax_cal.set_ylim(0.0, 1.05)
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    ax_cal.legend(loc="upper left", fontsize=9)

    # ---- Table ----
    results_df = pd.DataFrame(results).T
    mode_label = "OOF" if evaluation_mode == "oof" else "Holdout"
    print(f"\n=== {mode_label} PERFORMANCE COMPARISON TABLE ===")
    print(tabulate(results_df, headers="keys", tablefmt=table_format, showindex=True))

    # ---- Save ----
    if save_dir is not None:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig_roc.savefig(output_path / f"collective_roc_curve_{mode_label.lower()}.png", dpi=300, bbox_inches="tight")
        fig_prc.savefig(output_path / f"collective_prc_curve_{mode_label.lower()}.png", dpi=300, bbox_inches="tight")
        fig_cal.savefig(output_path / f"collective_calibration_curve_{mode_label.lower()}.png", dpi=300, bbox_inches="tight")
        with open(output_path / f"results_table_{mode_label.lower()}.tex", "w", encoding="utf-8") as output_file:
            output_file.write(tabulate(results_df, headers="keys", tablefmt="latex_booktabs", showindex=True))

    plt.show()
    plt.close(fig_roc)
    plt.close(fig_prc)
    plt.close(fig_cal)

    return results_df


