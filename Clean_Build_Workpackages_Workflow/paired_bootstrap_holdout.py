#!/usr/bin/env python3
"""Paired bootstrap comparison of models on the independent holdout.

Reads the ``all_res.joblib`` files produced by the pipeline, realigns every
model on the same holdout patients, and resamples those patients once per
bootstrap replicate so that all models are evaluated on exactly the same
draws. Reports per-model metrics with percentile CIs and pairwise deltas
against a reference model.

Usage (from Clean_Build_Workpackages_Workflow, with the venv activated):

    python paired_bootstrap_holdout.py
    python paired_bootstrap_holdout.py --reference "XGBoost TSFEL" --n-boot 2000
    python paired_bootstrap_holdout.py --outputs-root outputs --filter Mode_Commonly_Used_Without_pmsi

Outputs (printed and written next to the compared models):
    paired_bootstrap/model_metrics.csv
    paired_bootstrap/pairwise_deltas.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

EPSILON = 1e-12


def _logit(probabilities):
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def calibration_slope_intercept(y_true, probabilities):
    """Calibration slope and intercept (logistic recalibration)."""
    from sklearn.linear_model import LogisticRegression

    x = _logit(probabilities).reshape(-1, 1)
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    model.fit(x, y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def integrated_calibration_index(y_true, probabilities):
    """ICI: mean absolute difference between predicted and smoothed observed risk.

    Uses the pipeline's own estimator when it can be imported, so the values
    match those reported by the pipeline; otherwise falls back to a LOWESS
    smoother, or to isotonic regression when statsmodels is unavailable.
    """
    if integrated_calibration_index.backend is None:
        integrated_calibration_index.backend = _select_ici_backend()
    return integrated_calibration_index.backend(y_true, probabilities)


integrated_calibration_index.backend = None


def _select_ici_backend():
    try:
        from utilitaries.figures.performance import get_calibration_stats

        def pipeline_backend(y_true, probabilities):
            stats = get_calibration_stats(probabilities, y_true)
            for key in ("ici", "ICI", "ici_holdout"):
                if isinstance(stats, dict) and key in stats:
                    return float(stats[key])
            raise KeyError("ICI not found in get_calibration_stats output")

        pipeline_backend(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7]))
        print("[ICI] using the pipeline's get_calibration_stats", flush=True)
        return pipeline_backend
    except Exception:
        pass

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        def lowess_backend(y_true, probabilities):
            smoothed = lowess(
                y_true,
                probabilities,
                frac=0.75,
                it=0,
                return_sorted=False,
            )
            return float(np.mean(np.abs(smoothed - probabilities)))

        print("[ICI] using a LOWESS smoother (statsmodels)", flush=True)
        return lowess_backend
    except Exception:
        pass

    from sklearn.isotonic import IsotonicRegression

    def isotonic_backend(y_true, probabilities):
        fitted = IsotonicRegression(out_of_bounds="clip").fit_transform(
            probabilities, y_true
        )
        return float(np.mean(np.abs(fitted - probabilities)))

    print("[ICI] using isotonic regression (approximation)", flush=True)
    return isotonic_backend


def compute_metrics(y_true, probabilities, with_calibration=True):
    metrics = {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(np.mean((probabilities - y_true) ** 2)),
    }
    if with_calibration:
        try:
            slope, intercept = calibration_slope_intercept(y_true, probabilities)
        except Exception:
            slope, intercept = np.nan, np.nan
        metrics["calibration_slope"] = slope
        metrics["calibration_intercept"] = intercept
        try:
            metrics["ici"] = integrated_calibration_index(y_true, probabilities)
        except Exception:
            metrics["ici"] = np.nan
    return metrics


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def model_label(path, outputs_root):
    relative = os.path.relpath(path, outputs_root)
    parts = relative.split(os.sep)
    # .../<feature mode>/<model>/<variant>/all_res.joblib
    model = parts[-3] if len(parts) >= 3 else parts[0]
    variant = parts[-2] if len(parts) >= 2 else ""
    return model.replace("_", " "), variant


def load_models(outputs_root, name_filter):
    paths = sorted(glob.glob(os.path.join(outputs_root, "**", "all_res.joblib"), recursive=True))
    if name_filter:
        paths = [p for p in paths if name_filter in p]
    models = {}
    for path in paths:
        try:
            data = joblib.load(path)
        except Exception as error:
            print(f"[skip] {path}: {error}", flush=True)
            continue
        if not isinstance(data, dict) or "probas_holdout" not in data:
            print(f"[skip] {path}: no holdout predictions", flush=True)
            continue
        name, variant = model_label(path, outputs_root)
        models[name] = {
            "path": path,
            "variant": variant,
            "y": np.asarray(data["y_true_holdout"]).ravel(),
            "p": np.asarray(data["probas_holdout"]).ravel(),
            "ids": np.asarray(data.get("holdout_patient_ids", [])).ravel(),
        }
        print(f"[load] {name:35s} ({variant}) n={models[name]['y'].size}", flush=True)
    return models


def align_models(models):
    """Reorder every model on a common, identically ordered set of patients."""
    reference_ids = None
    for name, entry in models.items():
        if entry["ids"].size != entry["y"].size:
            raise SystemExit(f"{name}: patient ids missing, alignment impossible")
        if reference_ids is None:
            reference_ids = np.sort(entry["ids"])
        elif not np.array_equal(np.sort(entry["ids"]), reference_ids):
            raise SystemExit(f"{name}: holdout patients differ from the other models")

    for name, entry in models.items():
        order = np.argsort(entry["ids"])
        entry["y"] = entry["y"][order]
        entry["p"] = entry["p"][order]
        entry["ids"] = entry["ids"][order]

    return reference_ids.size


def harmonize_labels(models, reference_labels):
    """Check label agreement between models; optionally impose one model's labels."""
    names = list(models)
    base = models[names[0]]["y"]
    mismatches = {
        name: int((models[name]["y"] != base).sum())
        for name in names[1:]
    }
    if any(mismatches.values()):
        print("\n[labels] outcome labels differ between models "
              f"(vs {names[0]}): " +
              ", ".join(f"{k}={v}" for k, v in mismatches.items()), flush=True)
        if reference_labels is None:
            raise SystemExit(
                "Labels differ. Re-run with --reference-labels <model> to evaluate "
                "every model against the same outcome vector, e.g.\n"
                f"  --reference-labels \"{names[0]}\""
            )
        if reference_labels not in models:
            raise SystemExit(
                f"Unknown model '{reference_labels}'. Available: {names}"
            )
        shared = np.asarray(models[reference_labels]["y"]).astype(int)
        for name in names:
            models[name]["y"] = shared
        print(f"[labels] all models evaluated against the labels of "
              f"{reference_labels} ({int(shared.sum())} events)", flush=True)
    else:
        for name in names:
            models[name]["y"] = np.asarray(models[name]["y"]).astype(int)


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------

def paired_bootstrap(models, n_boot, seed, metrics_with_calibration):
    names = list(models)
    y = models[names[0]]["y"]
    n = y.size
    rng = np.random.default_rng(seed)

    distributions = {name: {} for name in names}
    for replicate in range(n_boot):
        index = rng.integers(0, n, n)
        y_boot = y[index]
        if y_boot.sum() == 0 or y_boot.sum() == y_boot.size:
            continue
        for name in names:
            values = compute_metrics(
                y_boot,
                models[name]["p"][index],
                with_calibration=metrics_with_calibration,
            )
            for metric, value in values.items():
                distributions[name].setdefault(metric, []).append(value)
        if (replicate + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {replicate + 1}/{n_boot}", flush=True)

    return {
        name: {metric: np.asarray(values) for metric, values in per_metric.items()}
        for name, per_metric in distributions.items()
    }


def summarize(models, distributions, metrics_with_calibration):
    rows = []
    for name, entry in models.items():
        point = compute_metrics(entry["y"], entry["p"], with_calibration=metrics_with_calibration)
        for metric, value in point.items():
            draws = distributions[name][metric]
            draws = draws[np.isfinite(draws)]
            rows.append({
                "model": name,
                "variant": entry["variant"],
                "metric": metric,
                "value": value,
                "ci_low": float(np.percentile(draws, 2.5)) if draws.size else np.nan,
                "ci_high": float(np.percentile(draws, 97.5)) if draws.size else np.nan,
            })
    return pd.DataFrame(rows)


def pairwise_deltas(models, distributions, reference, metrics_with_calibration):
    rows = []
    for name, entry in models.items():
        if name == reference:
            continue
        for metric in distributions[name]:
            delta_point = (
                compute_metrics(entry["y"], entry["p"], metrics_with_calibration)[metric]
                - compute_metrics(
                    models[reference]["y"], models[reference]["p"], metrics_with_calibration
                )[metric]
            )
            draws = distributions[name][metric] - distributions[reference][metric]
            draws = draws[np.isfinite(draws)]
            if draws.size == 0:
                continue
            p_value = float(min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
            rows.append({
                "model": name,
                "reference": reference,
                "metric": metric,
                "delta": delta_point,
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "p_value": p_value,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--filter", default="Mode_Commonly_Used_Without_pmsi",
                        help="substring the model path must contain")
    parser.add_argument("--reference", default=None,
                        help="reference model (default: highest AUC)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-calibration", action="store_true",
                        help="AUC, AUPRC and Brier only (much faster)")
    parser.add_argument("--reference-labels", default=None,
                        help="model whose outcome labels are imposed on all models "
                             "when labels disagree")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    models = load_models(args.outputs_root, args.filter)
    if len(models) < 2:
        sys.exit("At least two models with holdout predictions are required.")

    n_patients = align_models(models)
    harmonize_labels(models, args.reference_labels)
    events = int(models[next(iter(models))]["y"].sum())
    print(f"\n{len(models)} models | {n_patients} holdout patients | {events} events "
          f"({100 * events / n_patients:.1f}%)\n", flush=True)

    with_calibration = not args.no_calibration
    distributions = paired_bootstrap(models, args.n_boot, args.seed, with_calibration)

    metrics_table = summarize(models, distributions, with_calibration)
    reference = args.reference
    if reference is None:
        auc_rows = metrics_table[metrics_table["metric"] == "auc"]
        reference = auc_rows.loc[auc_rows["value"].idxmax(), "model"]
    if reference not in models:
        sys.exit(f"Reference model '{reference}' not found. Available: {list(models)}")
    deltas_table = pairwise_deltas(models, distributions, reference, with_calibration)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(models[reference]["path"]))),
        "paired_bootstrap",
    )
    os.makedirs(out_dir, exist_ok=True)
    metrics_table.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False)
    deltas_table.to_csv(os.path.join(out_dir, "pairwise_deltas.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n=== Holdout metrics (paired bootstrap CIs) ===")
    wide = metrics_table.assign(
        formatted=lambda d: d.apply(
            lambda r: f"{r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1
        )
    ).pivot(index="model", columns="metric", values="formatted")
    print(wide.to_string())

    print(f"\n=== Paired deltas vs {reference} (model - reference) ===")
    wide_delta = deltas_table.assign(
        formatted=lambda d: d.apply(
            lambda r: f"{r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p_value']:.3f}",
            axis=1,
        )
    ).pivot(index="model", columns="metric", values="formatted")
    print(wide_delta.to_string())
    print(f"\nCSV written to {out_dir}")


if __name__ == "__main__":
    main()