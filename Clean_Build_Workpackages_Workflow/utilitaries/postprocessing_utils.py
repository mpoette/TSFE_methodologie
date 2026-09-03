"""Post-processing metrics for calibrated holdout model evaluation.

Historical explainability plotting imports remain available from this module
through the compatibility exports at the end of the file.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, matthews_corrcoef

# ============================================================================
# CALIBRATION AND CLASSIFICATION METRICS
# ============================================================================

def calibration_intercept(
    y_true,
    probabilities,
):
    """Compute the calibration intercept from predicted probabilities.
    
    The intercept is estimated by fitting logistic regression on the logits of
    the predicted probabilities. A well-calibrated model has an intercept close
    to zero.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
    
    Returns:
        float:
            Calibration intercept.
    
    Raises:
        ValueError:
            If the inputs have different lengths or do not contain both classes.
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
    """Compute the calibration slope from predicted probabilities.
    
    The slope is estimated by fitting logistic regression on the logits of the
    predicted probabilities. A well-calibrated model has a slope close to one.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
    
    Returns:
        float:
            Calibration slope.
    
    Raises:
        ValueError:
            If the inputs have different lengths or do not contain both classes.
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
    """Compute absolute calibration errors for fixed 10% probability bins.
    
    Args:
        probabilities:
            Predicted probabilities for the positive class.
        y_true:
            Binary ground-truth labels.
    
    Returns:
        numpy.ndarray:
            Absolute calibration errors for non-empty bins.
    
    Raises:
        ValueError:
            If the inputs have different lengths or if ``probabilities`` is empty.
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
    """Compute the Integrated Calibration Index using fixed 10% bins.
    
    The score is the weighted mean of the absolute calibration errors across
    fixed 10% probability bins. Empty bins are ignored.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
    
    Returns:
        float:
            Integrated Calibration Index.
    
    Raises:
        ValueError:
            If the inputs have different lengths or if ``probabilities`` is empty.
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
    """Compute the 90th percentile of absolute calibration errors.
    
    Errors are computed over fixed 10% probability bins. Empty bins are excluded.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
    
    Returns:
        float:
            E90 calibration error.
    """
    bin_errs = _bin_errors_fixed_10pct(probabilities, y_true)

    if bin_errs.size == 0:
        return 0.0

    return float(np.percentile(bin_errs, 90))


def eMax_score(
    y_true,
    probabilities,
):
    """Compute the maximum absolute calibration error.
    
    Errors are computed over fixed 10% probability bins. Empty bins are excluded.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
    
    Returns:
        float:
            Maximum absolute calibration error.
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
    """Compute the F1 score at a fixed decision threshold.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
        threshold:
            Fixed decision threshold selected before holdout evaluation.
    
    Returns:
        float:
            F1 score at the specified threshold.
    
    Raises:
        ValueError:
            If the inputs have different lengths.
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
    """Compute the Matthews correlation coefficient at a fixed threshold.
    
    Args:
        y_true:
            Binary ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
        threshold:
            Fixed decision threshold selected before holdout evaluation.
    
    Returns:
        float:
            Matthews correlation coefficient at the specified threshold.
    
    Raises:
        ValueError:
            If the inputs have different lengths.
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


# ============================================================================
# HOLDOUT BOOTSTRAP
# ============================================================================

def bootstrap_holdout_metrics(
    y_true,
    probabilities,
    metric_functions,
    n_bootstrap=2000,
    confidence_level=0.95,
    seed=42,
):
    """Bootstrap metrics on an independent holdout set.
    
    Models are not retrained. Each bootstrap iteration resamples holdout patients
    with replacement and evaluates the already-computed probabilities.
    
    Args:
        y_true:
            Binary holdout labels.
        probabilities:
            Final holdout probabilities, typically averaged across fold models.
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
        dict:
            Point estimates, bootstrap distributions, confidence intervals, and
            bootstrap metadata.
    
    Raises:
        ValueError:
            If the inputs are invalid or no metric is provided.
        RuntimeError:
            If the requested number of valid bootstrap resamples cannot be
            generated.
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


# ============================================================================
# FEATURE-NAME NORMALIZATION
# ============================================================================


# Backward-compatible exports for historical call sites.
from utilitaries.figures.explainability import *
