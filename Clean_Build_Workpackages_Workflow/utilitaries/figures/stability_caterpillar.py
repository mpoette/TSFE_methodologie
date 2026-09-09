"""Descriptive odds-ratio stability and paired bootstrap performance evaluation.

This module provides tools for extracting standardized logistic regression
coefficients, generating bootstrap confidence intervals, and plotting
caterpillar charts without computing asymptotic p-values.
"""

from pathlib import Path
import textwrap
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def aligned_betas(
    betas: pd.Series,
    feature_names: Sequence[str]
) -> np.ndarray:
    """Align a named coefficient Series with the full feature universe.

    Args:
        betas: Named pandas Series of coefficients.
        feature_names: Expected list or index of feature names.

    Returns:
        One-dimensional array aligned to feature_names, with missing variables
        filled with zero.

    Raises:
        ValueError: If feature_names are empty, duplicated, if non-finite
            values are present, or if unexpected features exist in betas.
        TypeError: If betas is not a pandas Series.
    """
    names = pd.Index(feature_names)
    if names.empty or not names.is_unique:
        raise ValueError("Feature names must be non-empty and unique.")
    if not isinstance(betas, pd.Series) or not betas.index.is_unique:
        raise TypeError("betas must be a pd.Series with unique feature names.")
    if len(betas.index.difference(names)):
        raise ValueError("Coefficients contain names absent from feature_names.")
    if not np.isfinite(betas.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite coefficient detected: cannot fill with zero.")
    return betas.reindex(names, fill_value=0.0).to_numpy(dtype=float)


def extract_linear_betas(
    fitted: Union[Pipeline, LogisticRegression],
    input_names: Sequence[str]
) -> pd.Series:
    """Extract standardized coefficients in input feature units.

    Supports pipelines composed of feature selectors (exposing `get_support` or
    `support_`), `StandardScaler`, and a final binary `LogisticRegression`.

    Args:
        fitted: Fitted pipeline or LogisticRegression instance.
        input_names: Sequence of original feature names.

    Returns:
        Series of coefficients scaled back to the original feature units.

    Raises:
        ValueError: If selectors or masks are misaligned, or target classes
            are not binary [0, 1].
        TypeError: If an unsupported pipeline step is encountered.
    """
    names = np.asarray(input_names, dtype=object)
    scales = np.ones(len(names))
    steps = fitted.steps if isinstance(fitted, Pipeline) else [("lr", fitted)]

    for _, step in steps[:-1]:
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if isinstance(step, StandardScaler):
            if step.with_std:
                scales *= step.scale_
        elif hasattr(step, "get_support") or hasattr(step, "support_"):
            mask = np.asarray(
                step.get_support() if hasattr(step, "get_support") else step.support_
            )
            if mask.dtype != bool or mask.shape != names.shape:
                raise ValueError("Feature selector must provide a aligned boolean mask.")
            names, scales = names[mask], scales[mask]
        else:
            raise TypeError(f"Extraction not implemented for step: {type(step).__name__}.")

    lr = steps[-1][1]
    if not isinstance(lr, LogisticRegression) or lr.coef_.shape != (1, len(names)):
        raise ValueError("A binary LogisticRegression with matching shape is required.")
    if not np.array_equal(lr.classes_, [0, 1]):
        raise ValueError("Binary target must be encoded as 0 and 1.")

    return pd.Series(lr.coef_[0] / scales, index=names, dtype=float)


def coefficient_summary(
    draws: np.ndarray,
    feature_names: Sequence[str],
    *,
    mode: str = "bootstrap",
    level: float = 0.95,
    effect_units: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    outlier_k: float = 3.0,
    zero_tol: float = 1e-10
) -> pd.DataFrame:
    """Compute descriptive log-OR summary statistics across bootstrap draws or models.

    Args:
        draws: 2D array of coefficients with shape (n_draws, n_features).
        feature_names: Sequence of feature names corresponding to draws columns.
        mode: Either 'bootstrap' (percentile intervals) or 'intermodel'
            (min-max intervals).
        level: Confidence level for percentile intervals (default: 0.95).
        effect_units: Multiplicative positive scaling factor per feature.
        outlier_k: Tukey fence multiplier for outlier detection. Defaults to
            3.0 (extreme outliers) to avoid over-flagging sparse variables.
        zero_tol: Tolerance below which a coefficient is considered zero.

    Returns:
        DataFrame summarizing median betas, bounds, frequencies, and outlier status.

    Raises:
        ValueError: If shapes, intervals, or arguments are invalid.
    """
    a = np.asarray(draws, dtype=float)
    names = pd.Index(feature_names)
    if (
        a.ndim != 2
        or a.shape[0] < 2
        or a.shape[1] != len(names)
        or not names.is_unique
        or len(names) == 0
        or not np.isfinite(a).all()
    ):
        raise ValueError("draws must be a finite 2D array with matching unique names.")
    if not 0 < level < 1 or outlier_k <= 0:
        raise ValueError("Invalid level or outlier_k parameter.")

    units = (
        pd.Series(1.0, index=names)
        if effect_units is None
        else pd.Series(effect_units).reindex(names)
    )
    if not np.isfinite(units).all() or (units <= 0).any():
        raise ValueError("Each feature requires a positive, finite effect increment.")

    a = a * units.to_numpy()[None, :]
    center = np.median(a, axis=0)

    if mode == "intermodel":
        low, high = a.min(axis=0), a.max(axis=0)
        label = f"Min-max range across {len(a)} models (not a calibrated CI)"
    elif mode == "bootstrap":
        low, high = np.quantile(a, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
        label = f"Central {level:.0%} bootstrap percentile interval"
    else:
        raise ValueError("mode must be either 'intermodel' or 'bootstrap'.")

    active = np.any(np.abs(a) > zero_tol, axis=0)
    outlier = np.zeros(len(names), dtype=bool)

    # Tukey fences on active variables using outlier_k=3.0 (extreme outliers)
    if active.sum() >= 6:
        active_centers = center[active]
        q1, q3 = np.quantile(active_centers, [0.25, 0.75])
        iqr = q3 - q1
        # Prevent division collapse when medians are strongly peaked
        robust_spread = max(iqr, float(np.std(active_centers)))
        if robust_spread > zero_tol:
            outlier = active & (
                (center < q1 - outlier_k * robust_spread)
                | (center > q3 + outlier_k * robust_spread)
            )

    return pd.DataFrame({
        "feature": names,
        "beta": center,
        "beta_lower": low,
        "beta_upper": high,
        "importance": np.mean(np.abs(a), axis=0),
        "nonzero_frequency": np.mean(np.abs(a) > zero_tol, axis=0),
        "positive_frequency": np.mean(a > zero_tol, axis=0),
        "negative_frequency": np.mean(a < -zero_tol, axis=0),
        "effect_unit": units.to_numpy(),
        "is_outlier": outlier,
        "interval_label": label,
        "n_draws": len(a),
    })


def compare_fixed_models(
    beta_series: Sequence[pd.Series],
    feature_names: Sequence[str],
    **summary_options
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Summarize coefficient variability across pre-fitted fixed models.

    Args:
        beta_series: Sequence of pd.Series containing fitted coefficients.
        feature_names: Master list of feature names.
        **summary_options: Optional keyword arguments passed to `coefficient_summary`.

    Returns:
        A tuple containing:
            - A summary DataFrame of the coefficients.
            - A 2D array of aligned raw beta weights across models.
    """
    a = np.stack([aligned_betas(b, feature_names) for b in beta_series])
    return coefficient_summary(a, feature_names, mode="intermodel", **summary_options), a


def _sample_indices(
    rng: np.random.Generator,
    n_samples: int,
    patient_ids: Optional[Sequence] = None
) -> np.ndarray:
    """Generate bootstrap resample indices, clustered by patient ID if provided."""
    if patient_ids is None:
        return rng.integers(n_samples, size=n_samples)
    ids = np.asarray(patient_ids)
    if ids.shape != (n_samples,) or pd.isna(ids).any():
        raise ValueError("patient_ids must be complete and row-aligned.")
    codes, unique = pd.factorize(ids)
    picked_clusters = rng.integers(len(unique), size=len(unique))
    return np.concatenate([np.flatnonzero(codes == k) for k in picked_clusters])


def bootstrap_coefficients(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    fit_extract: Callable[[pd.DataFrame, np.ndarray, int], pd.Series],
    *,
    n_bootstrap: int = 400,
    seed: int = 42,
    patient_ids: Optional[Sequence] = None,
    progress_every: int = 50
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Execute cluster-aware bootstrap resampling to extract model coefficients.

    Args:
        X_train: Training feature DataFrame.
        y_train: Binary training labels.
        feature_names: Full feature universe for coefficient alignment.
        fit_extract: Callback function `(X_b, y_b, seed) -> pd.Series`.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random generator seed.
        patient_ids: Group identifiers for clustered patient resampling.
        progress_every: Step frequency for console progress output.

    Returns:
        A tuple containing:
            - A 2D array of valid bootstrap coefficient draws.
            - A DataFrame logging execution status and errors per draw.

    Raises:
        TypeError: If X_train is not a DataFrame.
        ValueError: If input labels or dimensions are invalid.
        RuntimeError: If fewer than two iterations succeed.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    y = np.asarray(y_train).reshape(-1)
    if len(y) != len(X_train) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("y_train must be binary (0/1) and aligned with X_train.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    rng = np.random.default_rng(seed)
    valid, log = [], []

    for b in range(n_bootstrap):
        idx = _sample_indices(rng, len(y), patient_ids)
        status, reason = "ok", ""
        try:
            if np.unique(y[idx]).size != 2:
                raise ValueError("Resampled draw contains only a single class.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                result = fit_extract(X_train.iloc[idx].copy(), y[idx], seed)
            valid.append(aligned_betas(result, feature_names))
        except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
            status, reason = "failed", str(exc)

        log.append({"draw": b, "status": status, "reason": reason})
        if progress_every and (b + 1) % progress_every == 0:
            print(f"{b+1}/{n_bootstrap} draws processed ({len(valid)} valid)", flush=True)

    if len(valid) < 2:
        raise RuntimeError(f"Fewer than two draws succeeded. Initial failures: {log[:5]}")
    if len(valid) < n_bootstrap:
        warnings.warn(
            f"{n_bootstrap - len(valid)}/{n_bootstrap} bootstrap fits failed. "
            "Inspect log file for details."
        )

    return np.stack(valid), pd.DataFrame(log)


def fixed_hyperparameter_callback(
    template: Union[Pipeline, LogisticRegression]
) -> Callable[[pd.DataFrame, np.ndarray, int], pd.Series]:
    """Generate a refit callback keeping hyperparameter settings invariant.

    Args:
        template: Base estimator or pipeline to clone.

    Returns:
        Callable suitable for `bootstrap_coefficients`.
    """
    def fit_extract(X: pd.DataFrame, y: np.ndarray, seed: int) -> pd.Series:
        estimator = clone(template)
        params = estimator.get_params(deep=True)
        estimator.set_params(
            **{k: seed for k in params if k.split("__")[-1] == "random_state"}
        )
        estimator.fit(X, y)
        return extract_linear_betas(estimator, X.columns)

    return fit_extract


def bootstrap_predictions(
    y: np.ndarray,
    predictions: Union[pd.DataFrame, np.ndarray],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    level: float = 0.95,
    patient_ids: Optional[Sequence] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Evaluate paired bootstrap metrics on pre-computed holdout probabilities.

    Args:
        y: True binary outcomes.
        predictions: Matrix or DataFrame of predicted probabilities.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random generator seed.
        level: Confidence interval level.
        patient_ids: Group identifiers for patient-level clustering.

    Returns:
        A tuple containing:
            - Summary DataFrame reporting estimate, lower, and upper bounds.
            - 3D array of bootstrap performance evaluations.
    """
    p = pd.DataFrame(predictions)
    y = np.asarray(y).reshape(-1)
    if (
        len(p) != len(y)
        or not p.columns.is_unique
        or p.shape[1] == 0
        or not np.array_equal(np.unique(y), [0, 1])
        or not np.isfinite(p.to_numpy()).all()
        or ((p < 0) | (p > 1)).any().any()
    ):
        raise ValueError("Aligned binary targets and valid probabilities [0, 1] required.")
    if n_bootstrap < 2 or not 0 < level < 1:
        raise ValueError("Invalid bootstrap parameters.")

    scores = {"AUC": roc_auc_score, "Brier": brier_score_loss}
    a = np.full((n_bootstrap, p.shape[1], 2), np.nan)
    rng = np.random.default_rng(seed)
    values = p.to_numpy()

    for b in range(n_bootstrap):
        idx = _sample_indices(rng, len(y), patient_ids)
        for j in range(p.shape[1]):
            a[b, j, 1] = brier_score_loss(y[idx], values[idx, j])
            if np.unique(y[idx]).size == 2:
                a[b, j, 0] = roc_auc_score(y[idx], values[idx, j])

    rows = []
    for j, name in enumerate(p.columns):
        for k, (metric, score) in enumerate(scores.items()):
            v = a[:, j, k]
            v = v[np.isfinite(v)]
            lo, hi = (
                np.quantile(v, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
                if len(v)
                else [np.nan, np.nan]
            )
            rows.append({
                "model": name,
                "metric": metric,
                "estimate": score(y, values[:, j]),
                "lower": lo,
                "upper": hi,
                "n_valid": len(v),
            })

    return pd.DataFrame(rows), a


def plot_performance(
    summary: pd.DataFrame,
    folder: Union[str, Path],
    *,
    level: float = 0.95
) -> plt.Figure:
    """Plot paired model performance comparisons and export to PDF, PNG, and SVG.

    Args:
        summary: Performance summary DataFrame from `bootstrap_predictions`.
        folder: Target directory for saving artifacts.
        level: Confidence level label for bootstrap intervals.

    Returns:
        Generated matplotlib Figure instance.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    for ax, metric in zip(axes, ["AUC", "Brier"]):
        d = summary[summary.metric == metric].sort_values("estimate")
        y = np.arange(len(d))
        ax.hlines(y, d.lower, d.upper, color="#8aa2b2", lw=2)
        ax.scatter(d.estimate, y, c="#193e52", zorder=3)
        ax.set_yticks(y, d.model.astype(str))
        direction = "higher is better" if metric == "AUC" else "lower is better"
        ax.set_title(f"{metric} ({direction})")
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Fixed Models · {level:.0%} Percentile Bootstrap Performance")

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / f"performances.{ext}", dpi=300)
    return fig


def caterpillar_pair(
    summary: pd.DataFrame,
    folder: Union[str, Path],
    *,
    prefix: str = "coefficients",
    top_n: int = 20,
    include_outliers_in_top: bool = True,
    title: str = "Odds Ratio Stability",
    label_transform: Optional[Callable[[str], str]] = None
) -> Dict[str, plt.Figure]:
    """Generate global and top-N caterpillar plots with significance coloring.

    Features whose confidence interval crosses OR=1 (log-OR=0) are rendered in
    gray. Statistically separated features are rendered in coral (OR > 1) or
    slate-teal (OR < 1). Plots are exported to PDF, PNG, and SVG formats.

    Args:
        summary: Coefficient summary DataFrame.
        folder: Output folder path.
        prefix: Prefix for saved plot file names.
        top_n: Number of top features to display in the detailed view.
        include_outliers_in_top: Whether to append outlier features to top view.
        title: Plot super-title.
        label_transform: Optional callable function to reformat feature names.

    Returns:
        Dictionary mapping view keys ('all', 'top') to their Figure objects.

    Raises:
        ValueError: If bounds are inconsistent or data is empty.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    d = summary.copy()

    if d.empty or top_n < 1:
        raise ValueError("Empty summary or invalid top_n parameter.")
    finite = np.isfinite(d[["beta", "beta_lower", "beta_upper", "importance"]]).all().all()
    if not finite or (d.beta_lower > d.beta_upper).any():
        raise ValueError("Invalid lower and upper interval bounds.")

    nonout = d.loc[~d.is_outlier].nlargest(top_n, "importance")
    detail = pd.concat([nonout, d.loc[d.is_outlier]]) if include_outliers_in_top else nonout
    views = {"all": d, "top": detail}
    figures = {}

    for view, rows in views.items():
        rows = rows.sort_values("beta").reset_index(drop=True)
        if rows.empty:
            warnings.warn("No features available for detail plot.")
            continue

        n = len(rows)
        height = 7 if view == "all" else max(5, 2.2 + 0.31 * n)
        fig, ax = plt.subplots(figsize=(11, height), layout="constrained")
        y = np.arange(n)

        # Confidence interval crosses OR = 1 (log-OR = 0) -> gray out as non-significant
        crosses_zero = (rows.beta_lower <= 0.0) & (rows.beta_upper >= 0.0)
        colors = np.where(
            crosses_zero,
            "#94a3b8",  # Non-significant: neutral slate gray
            np.where(rows.beta > 0.0, "#b64c38", "#26738b")  # Significant: coral vs teal
        )

        ax.axvline(0, color="#475569", ls="--", lw=1.1, zorder=0)
        ax.hlines(
            y,
            rows.beta_lower,
            rows.beta_upper,
            colors=colors,
            alpha=0.6,
            lw=1.0 if view == "all" else 2.0,
        )

        for flag, marker in [(False, "o"), (True, "D")]:
            mask = rows.is_outlier.to_numpy() == flag
            ax.scatter(
                rows.beta[mask],
                y[mask],
                c=colors[mask],
                marker=marker,
                s=(19 if view == "all" else 36),
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

        if view == "all":
            ax.set_yticks([])
            ax.set_ylabel(f"{n} features sorted by median OR")
            subtitle = f"All features · {n} variables"
        else:
            labels = [
                label_transform(x) if label_transform else str(x)
                for x in rows.feature
            ]
            ax.set_yticks(y, ["\n".join(textwrap.wrap(x, 44)) for x in labels], fontsize=9)
            subtitle = f"Top {len(nonout)} features (excluding outliers)"
            if include_outliers_in_top:
                subtitle += f" + {int(d.is_outlier.sum())} extreme outliers"

        lo = min(float(rows.beta_lower.min()), float(rows.beta.min()), 0.0)
        hi = max(float(rows.beta_upper.max()), float(rows.beta.max()), 0.0)
        span = max(hi - lo, 0.2)
        ax.set_xlim(lo - 0.06 * span, hi + 0.06 * span)

        candidates = np.linspace(lo, hi, 5)
        ticks = np.unique(np.r_[candidates[np.abs(candidates) > 0.12 * span], 0.0])

        def tick_label(x: float) -> str:
            return f"{np.exp(x):.3g}" if abs(x) < 12 else f"$e^{{{x:.1f}}}$"

        ax.set_xticks(ticks, [tick_label(x) for x in ticks])
        ax.set_xlabel("Odds Ratio for standard reference increment (logarithmic scale)")
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, pad=14)
        ax.grid(axis="x", color="#e2e8f0", alpha=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        legend_elements = [
            Line2D([], [], marker="o", color="#475569", ls="", label="Median estimate"),
            Line2D([], [], marker="D", color="#475569", ls="", label="Outlier feature"),
            Line2D([], [], color="#b64c38", lw=2, label="Significant risk increase (OR > 1)"),
            Line2D([], [], color="#26738b", lw=2, label="Significant protective (OR < 1)"),
            Line2D([], [], color="#94a3b8", lw=2, label="Non-significant (CI crosses 1)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

        for ext in ["pdf", "png", "svg"]:
            fig.savefig(folder / f"{prefix}_{view}.{ext}", dpi=300)

        rows.to_csv(folder / f"{prefix}_{view}.csv", index=False)
        figures[view] = fig

    return figures