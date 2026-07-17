import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import polars as pl
import math
import matplotlib.pyplot as plt
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
from sklearn.metrics import brier_score_loss 
from statsmodels.nonparametric.smoothers_lowess import lowess

import utilitaries.features_extraction_utils as feu
from pathlib import Path

import utilitaries.evaluate_utils as evaluate

def calibration_curve_homemade(
    probas_uncalib,
    probas_calib,
    y_test_global,
    model_name,
    extraction_type,
    calibration,
    save_figure,
    output_dir,
    transparent,
    calibration_mode="Platt",
):
    """Plot a binned calibration curve for raw and calibrated predictions.
    
    The function compares the observed positive-class frequency with the mean
    predicted probability in ten uniform probability bins. When applicable, a
    second curve is added for calibrated TSFEL predictions.
    
    Args:
        probas_uncalib:
            Uncalibrated probabilities predicted for the positive class.
        probas_calib:
            Calibrated positive-class probabilities. May be ``None`` when no
            calibrated curve is required.
        y_test_global:
            Binary ground-truth labels associated with the predictions.
        model_name:
            Model name displayed in the figure title and legend.
        extraction_type:
            Feature-extraction type used by the model.
        calibration:
            Whether calibrated predictions should be displayed.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        calibration_mode:
            Name of the calibration strategy displayed in the legend.
    
    Returns:
        ``None``.
    """
    probas_uncalib = np.asarray(
        probas_uncalib,
        dtype=float,
    ).ravel()

    y_test_global = np.asarray(
        y_test_global,
        dtype=int,
    ).ravel()

    if probas_calib is not None:
        probas_calib = np.asarray(
            probas_calib,
            dtype=float,
        ).ravel()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        "k:",
        label="Perfect calibration",
    )

    # Raw-model curve
    uncalibrated_fraction_positive, uncalibrated_mean_prediction = calibration_curve(
        y_test_global,
        probas_uncalib,
        n_bins=10,
        strategy="uniform",
    )

    ax.plot(
        uncalibrated_mean_prediction,
        uncalibrated_fraction_positive,
        "s-",
        color="red",
        label=f"Curve ({model_name})",
    )

    # Calibrated-model curve
    if (
        extraction_type == "TSFEL"
        and calibration
        and probas_calib is not None
    ):
        calibrated_fraction_positive, calibrated_mean_prediction = calibration_curve(
            y_test_global,
            probas_calib,
            n_bins=10,
            strategy="uniform",
        )

        ax.plot(
            calibrated_mean_prediction,
            calibrated_fraction_positive,
            "s-",
            color="blue",
            label=f"After calibration ({calibration_mode})",
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("True fraction of positives")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Global Calibration Curve "
        "(5-Fold Cross-Validation)\n"
        f"Model: {model_name}"
    )

    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()

    if save_figure:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "calibration_curve.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
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


def _prepare_lowess_for_interpolation(
    lowess_result,
):
    """Prepare LOWESS coordinates for numerical interpolation.
    
    Duplicate x-coordinates returned by LOWESS are grouped and their y-values
    are averaged so that the resulting curve can safely be passed to
    ``numpy.interp``.
    
    Args:
        lowess_result:
            Two-column array containing LOWESS x- and y-coordinates.
    
    Returns:
        A tuple containing unique sorted x-coordinates and their averaged
        y-coordinates.
    """
    lowess_x = lowess_result[:, 0]
    lowess_y = lowess_result[:, 1]

    unique_x, inverse_indices = np.unique(
        lowess_x,
        return_inverse=True,
    )

    unique_y = np.zeros(
        unique_x.shape[0],
        dtype=float,
    )

    counts = np.zeros(
        unique_x.shape[0],
        dtype=int,
    )

    np.add.at(
        unique_y,
        inverse_indices,
        lowess_y,
    )

    np.add.at(
        counts,
        inverse_indices,
        1,
    )

    unique_y = unique_y / counts

    return unique_x, unique_y


def get_calibration_stats(
    probas,
    y_test,
    lowess_frac=0.30,
    lowess_it=0,
):
    """Compute calibration statistics and a LOWESS calibration curve.
    
    Calibration intercept and slope are estimated with logistic regression on
    the logit-transformed predicted probabilities. The function also computes
    the Brier score, the Integrated Calibration Index, and the 90th percentile
    of the absolute calibration error.
    
    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels encoded as 0 and 1.
        lowess_frac:
            Fraction of observations used in each LOWESS neighborhood.
        lowess_it:
            Number of additional robust LOWESS iterations.
    
    Returns:
        A dictionary containing calibration intercept, calibration slope,
        Brier score, ICI, E90, and the LOWESS curve coordinates.
    
    Raises:
        ValueError:
            If the inputs are invalid, ``lowess_frac`` is outside ``(0, 1]``,
            ``lowess_it`` is negative, or too few distinct probabilities are
            available to estimate the LOWESS curve.
    """
    probas, y_test = _validate_calibration_inputs(
        probas,
        y_test,
    )

    if not 0 < lowess_frac <= 1:
        raise ValueError(
            "lowess_frac must be strictly greater than 0 "
            "and lower than or equal to 1."
        )

    if lowess_it < 0:
        raise ValueError(
            "lowess_it must be greater than or equal to 0."
        )

    # ---------------------------------------------------------
    # Calibration intercept and slope
    # ---------------------------------------------------------
    eps = 1e-7

    probas_clipped = np.clip(
        probas,
        eps,
        1 - eps,
    )

    logits = np.log(
        probas_clipped
        / (1 - probas_clipped)
    ).reshape(-1, 1)

    calibration_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )

    calibration_model.fit(
        logits,
        y_test,
    )

    # ---------------------------------------------------------
    # LOWESS curve
    # ---------------------------------------------------------
    lowess_result = lowess(
        endog=y_test,
        exog=probas,
        frac=lowess_frac,
        it=lowess_it,
        is_sorted=False,
        return_sorted=True,
    )

    lowess_x, lowess_y = (
        _prepare_lowess_for_interpolation(
            lowess_result
        )
    )

    if lowess_x.size < 2:
        raise ValueError(
            "Not enough distinct predicted probabilities "
            "to estimate a LOWESS calibration curve."
        )

    # LOWESS is not constrained to the [0, 1] interval.
    lowess_y = np.clip(
        lowess_y,
        0,
        1,
    )

    # ---------------------------------------------------------
    # ICI and E90 computation
    # ---------------------------------------------------------
    smooth_observed_at_predictions = np.interp(
        probas,
        lowess_x,
        lowess_y,
        left=lowess_y[0],
        right=lowess_y[-1],
    )

    smooth_observed_at_predictions = np.clip(
        smooth_observed_at_predictions,
        0,
        1,
    )

    absolute_errors = np.abs(
        smooth_observed_at_predictions
        - probas
    )

    return {
        "intercept": float(
            calibration_model.intercept_[0]
        ),
        "slope": float(
            calibration_model.coef_[0, 0]
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                probas,
            )
        ),
        "ici": float(
            np.mean(absolute_errors)
        ),
        "e90": float(
            np.percentile(
                absolute_errors,
                90,
            )
        ),
        "x": lowess_x,
        "y": lowess_y,
    }


def calibration_curve_advanced(
    probas_uncalib,
    probas_calib,
    y_test_global,
    model_name,
    extraction_type,
    calibration,
    save_figure,
    output_dir,
    transparent,
    calibration_mode="Platt",
    lowess_frac=0.30,
    lowess_it=0,
):
    """Plot an advanced calibration assessment with summary metrics.
    
    The figure combines a prediction histogram, the perfect-calibration
    diagonal, LOWESS curves before and after calibration, and a text box with
    calibration intercept, slope, Brier score, ICI, and E90.
    
    Args:
        probas_uncalib:
            Uncalibrated probabilities predicted for the positive class.
        probas_calib:
            Calibrated positive-class probabilities. May be ``None`` when
            calibration is not applied.
        y_test_global:
            Binary ground-truth labels associated with the predictions.
        model_name:
            Model name displayed in the figure title.
        extraction_type:
            Feature-extraction type used by the model.
        calibration:
            Whether a calibrated curve should be computed and displayed.
        save_figure:
            Whether the generated figure should be saved.
        output_dir:
            Directory in which the figure should be written.
        transparent:
            Whether the saved figure should use a transparent background.
        calibration_mode:
            Name of the calibration strategy displayed in the legend.
        lowess_frac:
            Fraction of observations used in each LOWESS neighborhood.
        lowess_it:
            Number of additional robust LOWESS iterations.
    
    Returns:
        ``None``.
    
    Raises:
        ValueError:
            If calibration is enabled for TSFEL but ``probas_calib`` is not
            provided, or if any calibration input is invalid.
    """
    probas_uncalib, y_test_global = (
        _validate_calibration_inputs(
            probas_uncalib,
            y_test_global,
        )
    )

    if probas_calib is not None:
        probas_calib = np.asarray(
            probas_calib,
            dtype=float,
        ).ravel()

    fig, ax1 = plt.subplots(
        figsize=(9, 7)
    )

    # ---------------------------------------------------------
    # Background histogram
    # ---------------------------------------------------------
    ax1.hist(
        probas_uncalib,
        bins=40,
        range=(0, 1),
        alpha=0.10,
        color="grey",
        density=False,
    )

    ax1.set_xlabel(
        "Predicted Probability / Risk"
    )

    ax1.set_ylabel(
        "Number of Patients"
    )

    ax1.set_xlim(0, 1)

    # ---------------------------------------------------------
    # Calibration axis
    # ---------------------------------------------------------
    ax2 = ax1.twinx()

    ax2.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Perfect calibration",
    )

    # ---------------------------------------------------------
    # Raw model
    # ---------------------------------------------------------
    stats_raw = get_calibration_stats(
        probas=probas_uncalib,
        y_test=y_test_global,
        lowess_frac=lowess_frac,
        lowess_it=lowess_it,
    )

    ax2.plot(
        stats_raw["x"],
        stats_raw["y"],
        color="red",
        linewidth=2,
        label=(
            "Before Calibration "
            f"(Brier: {stats_raw['brier']:.3f})"
        ),
    )

    text_str = (
        "[Raw Model]\n"
        f"Intercept: {stats_raw['intercept']:.2f}\n"
        f"Slope: {stats_raw['slope']:.2f}\n"
        f"ICI: {stats_raw['ici']:.3f}\n"
        f"E90: {stats_raw['e90']:.3f}\n"
        f"Brier: {stats_raw['brier']:.3f}"
    )

    # ---------------------------------------------------------
    # Calibrated model
    # ---------------------------------------------------------
    if extraction_type == "TSFEL" and calibration:
        if probas_calib is None:
            raise ValueError(
                "probas_calib must be provided when calibration "
                "is enabled for TSFEL."
            )

        probas_calib, calibrated_y_test = (
            _validate_calibration_inputs(
                probas_calib,
                y_test_global,
            )
        )

        stats_calib = get_calibration_stats(
            probas=probas_calib,
            y_test=calibrated_y_test,
            lowess_frac=lowess_frac,
            lowess_it=lowess_it,
        )

        ax2.plot(
            stats_calib["x"],
            stats_calib["y"],
            color="blue",
            linewidth=2,
            label=(
                f"After {calibration_mode} "
                f"(Brier: {stats_calib['brier']:.3f})"
            ),
        )

        text_str += (
            "\n\n"
            "[Calibrated Model]\n"
            f"Intercept: {stats_calib['intercept']:.2f}\n"
            f"Slope: {stats_calib['slope']:.2f}\n"
            f"ICI: {stats_calib['ici']:.3f}\n"
            f"E90: {stats_calib['e90']:.3f}\n"
            f"Brier: {stats_calib['brier']:.3f}"
        )

    # ---------------------------------------------------------
    # Metrics box
    # ---------------------------------------------------------
    text_box_properties = {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.8,
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

    ax2.set_ylabel(
        "Observed Proportion / Mortality Rate"
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.legend(
        loc="lower right"
    )

    ax2.grid(
        True,
        alpha=0.25,
    )

    ax2.set_title(
        "Advanced Calibration Assessment\n"
        f"Model: {model_name}"
    )

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    if save_figure:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "advanced_calibration_curve.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )

    plt.show()
    plt.close(fig)

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
    # If the AUC differs from the LSTM model metric, it is because the model AUC
    # is computed on the validation split, whereas this value is computed on the test set.
    auc_final = roc_auc_score(y_test, probas)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC {model_name} (AUC = {auc_final:.3f})')

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
    plt.plot(recall, precision, label=f'PRC {model_name} (AUPRC = {auprc_final:.3f})', color='blue', linewidth=2)
    
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

def plot_all_figs(probas, y_test, config_models, calibration, save_figure, output_dir, transparent):
    """Generate the standard evaluation figures and collect their metrics.
    
    Args:
        probas:
            Predicted probabilities for the positive class.
        y_test:
            Binary ground-truth labels.
        config_models:
            Model configuration exposing ``models_name`` and
            ``extraction_type`` attributes.
        calibration:
            Whether calibration-related output should be enabled.
        save_figure:
            Whether generated figures should be saved.
        output_dir:
            Directory in which figures should be written.
        transparent:
            Whether saved figures should use a transparent background.
    
    Returns:
        A tuple containing ROC metrics, Brier score, optimal F1 information,
        predicted labels, MCC, and KDE-based separation statistics.
    """
    cfg = {
        "save_figure": save_figure,
        "output_dir": output_dir,
        "transparent": transparent
    }
    calibration_curve_homemade(probas, [], y_test, config_models.models_name, 
                               config_models.extraction_type, calibration,
                                 **cfg)
    auc_final, fpr, tpr, th = roc_curve_homemade(probas, y_test, config_models.models_name, **cfg)
    non_overlap_area, asymmetric_uncertainty, mean_risk_diff, mean_p1 = kde_plot_homemade(probas, y_test, config_models.models_name, **cfg)
    brier_score = brier_evolution(probas, y_test, **cfg)
    best_f1, best_t = f1_score_evolution(probas, y_test, config_models.models_name, **cfg)
    y_pred, mcc = confusion_matrix_homemade(probas, y_test, best_t, config_models.models_name, **cfg)
    return auc_final, fpr, tpr, th, brier_score, best_f1, best_t, y_pred, mcc, non_overlap_area, asymmetric_uncertainty, mean_risk_diff, mean_p1

def plot_decision_curve_analysis(probas, y_test, model_name, save_figure, output_dir, transparent):
    """Plot a decision curve analysis for a binary prediction model.
    
    The model's net benefit is compared with the strategies of treating all
    patients and treating no patients over probability thresholds from 0.01 to
    0.99.
    
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
        A tuple containing the evaluated thresholds, model net benefits, and
        treat-all net benefits.
    """
    y_test = np.asarray(y_test)
    probas = np.asarray(probas)
    n = len(y_test)
    
    # Total number of positives (deaths) and negatives (survivors)
    total_pos = np.sum(y_test == 1)
    total_neg = np.sum(y_test == 0)
    
    # Probability-threshold grid from 1% to 99%
    thresholds = np.linspace(0.01, 0.99, 100)
    
    net_benefit_model = []
    net_benefit_all = []
    
    for p in thresholds:
        # Harm-to-benefit weight p / (1 - p)
        weight = p / (1 - p)
        
        # 1. Model-based strategy
        y_pred = (probas >= p).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        nb_model = (tp / n) - (fp / n) * weight
        net_benefit_model.append(nb_model)
        
        # 2. Treat-all strategy: every patient is considered positive
        nb_all = (total_pos / n) - (total_neg / n) * weight
        net_benefit_all.append(nb_all)
        
    # 3. Treat-none strategy: net benefit is always zero
    net_benefit_none = np.zeros_like(thresholds)
    
    # --- Figure construction ---
    plt.figure(figsize=(8, 6))
    
    # Model curve
    plt.plot(thresholds, net_benefit_model, color="blue", linewidth=2.5, 
             label=f"Modèle : {model_name}")
    
    # Treat-all curve
    plt.plot(thresholds, net_benefit_all, color="red", linestyle="--", linewidth=1.5, 
             label="Stratégie : Considérer tout le monde Positif")
    
    # Treat-none curve
    plt.plot(thresholds, net_benefit_none, color="black", linestyle="-", alpha=0.6, linewidth=1.5, 
             label="Stratégie : Considérer tout le monde Négatif")
    
    # Adjust axes for clinical interpretability
    plt.xlim(0.0, 1.0)
    
    # Limit the y-axis so large negative benefits do not flatten the useful region
    max_visible_nb = max(max(net_benefit_model), total_pos / n)
    plt.ylim(-0.05, max_visible_nb + 0.05)
    
    plt.xlabel("Seuil de probabilité critique (p)", fontsize=10)
    plt.ylabel("Bénéfice Net (Net Benefit)", fontsize=10)
    plt.title(f"Decision Curve Analysis (DCA)\nModel: {model_name}", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if save_figure:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / "decision_curve_analysis.png", dpi=300, bbox_inches="tight", transparent=transparent)
        plt.savefig(path / "decision_curve_analysis.pdf", bbox_inches="tight", transparent=transparent)
        
    plt.show()
    
    return thresholds, net_benefit_model, net_benefit_all

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

def générer_rapport_comparatif(configurations, y_true_base = None, save_dir=None, table_format='fancy_grid', saps2_pred = None, saps2_true = None):
    """Generate a comparative performance table and collective ROC curve.
    
    Args:
        configurations:
            Iterable of ``(name, configuration)`` pairs. Each configuration
            must provide predicted probabilities and the precomputed F1, MCC,
            AUC, and Brier metrics. It must also provide ``y_true`` when
            ``y_true_base`` is not supplied.
        y_true_base:
            Optional common ground-truth label array shared by all models.
        save_dir:
            Optional directory in which the ROC figure and LaTeX table should
            be saved.
        table_format:
            Output format passed to ``tabulate`` for console display.
        saps2_pred:
            Optional SAPS II or IGS II predicted risks added to the ROC curve.
        saps2_true:
            Optional labels associated with ``saps2_pred``.
    
    Returns:
        A pandas DataFrame containing the comparative model metrics.
    """
    results = {}

    # Configure the collective ROC figure
    plt.figure(figsize=(8, 8))
    for name, config in configurations:
        # Extract precomputed vectors
        probas = config['probas']
        if y_true_base is None:
            y_true = config['y_true']
        else:
            y_true = y_true_base
            
        # Retrieve metrics
        f1 = config['f1_score']
        mcc = config['mcc']
        auc = config['auc']
        brier = config ['brier']

        # Store metrics for the comparison table
        results[name] = {
            'F1-Score': f1,
            'MCC': mcc,
            'AUC': auc,
            "brier" : brier
        }

        # Add the model to the collective ROC curve
        fpr, tpr, _ = roc_curve(y_true, probas)
        color = config.get('color', None)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

    # 1. Generate the comparison table with tabulate
    results_df = pd.DataFrame(results).T
    print("\n=== PERFORMANCE COMPARISON TABLE ===")
    print(tabulate(results_df, headers='keys', tablefmt=table_format, floatfmt=".3f"))
    # Compute the SAPS II AUC
    if saps2_pred is not None and saps2_true is not None:
        saps2_fpr, saps2_tpr, _ = roc_curve(saps2_true, saps2_pred)
        saps2_auc = roc_auc_score(saps2_true, saps2_pred)
        saps2_name = "IGS II Score"
        plt.plot(saps2_fpr, saps2_tpr, label=f'{saps2_name} (AUC = {saps2_auc:.3f})', lw=2, linestyle='-.')
    # 2. Finalize the ROC figure
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance', color='gray')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Save generated artifacts
    if save_dir:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the figure
        plt.savefig(output_path / "collective_roc_curve.png", dpi=300, bbox_inches="tight")

        # Save the table as LaTeX booktabs for Overleaf
        with open(output_path / "results_table.tex", "w") as f:
            f.write(tabulate(results_df, headers='keys', tablefmt='latex_booktabs', floatfmt=".3f"))

    plt.show()

    return results_df



def plot_collected_learning_curve(sample_sizes, train_matrix, val_matrix, model_name="Model",  folder = "", savefig = True, transparent = True):
    """Plot learning curves collected during cross-validation training.
    
    Mean training and out-of-fold validation scores are plotted for each sample
    size. The validation curve includes a one-standard-deviation band across
    folds.
    
    Args:
        sample_sizes:
            Training-set sizes associated with the score columns.
        train_matrix:
            Training-score matrix shaped as ``(n_folds, n_sample_sizes)``.
        val_matrix:
            Validation-score matrix shaped as ``(n_folds, n_sample_sizes)``.
        model_name:
            Model name displayed in the figure title.
        folder:
            Directory in which the figure should be written.
        savefig:
            Whether the generated figure should be saved.
        transparent:
            Whether the saved figure should use a transparent background.
    
    Returns:
        ``None``.
    """

    # Compute means and standard deviations across folds
    train_mean = np.mean(train_matrix, axis=0)
    val_mean = np.mean(val_matrix, axis=0)
    val_std = np.std(val_matrix, axis=0)
    
    plt.figure(figsize=(10, 5))
    
    # Training curve
    plt.plot(sample_sizes, train_mean, "o-", color="crimson", label="Training Score (Mean)", linewidth=2)
    
    # Out-of-fold validation curve with its variability band
    plt.plot(sample_sizes, val_mean, "o-", color="royalblue", label="Validation Score (Mean OOF)", linewidth=2)
    plt.fill_between(sample_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="royalblue", label="OOF Volatility (± 1 STD)")
    
    plt.title(f"Learning Curve — {model_name} (Embedded Fold Splitting)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Training Samples (Aggregated)")
    plt.ylabel("AUC-ROC Score")
    plt.ylim(0.6, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}/learning_curve.png", dpi=300, bbox_inches="tight", transparent=transparent)
    plt.show()