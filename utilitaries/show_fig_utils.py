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
    """
    Affiche une courbe de calibration classique basée sur des groupes
    de probabilités prédictes.
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

    # Courbe du modèle brut
    fraction_pos_uncalib, mean_pred_uncalib = calibration_curve(
        y_test_global,
        probas_uncalib,
        n_bins=10,
        strategy="uniform",
    )

    ax.plot(
        mean_pred_uncalib,
        fraction_pos_uncalib,
        "s-",
        color="red",
        label=f"Curve ({model_name})",
    )

    # Courbe du modèle calibré
    if (
        extraction_type == "TSFEL"
        and calibration
        and probas_calib is not None
    ):
        fraction_pos_calib, mean_pred_calib = calibration_curve(
            y_test_global,
            probas_calib,
            n_bins=10,
            strategy="uniform",
        )

        ax.plot(
            mean_pred_calib,
            fraction_pos_calib,
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
    """
    Valide et convertit les données utilisées pour l'analyse
    de calibration.
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
    """
    Prépare une courbe LOWESS pour l'interpolation.

    LOWESS peut renvoyer plusieurs lignes ayant la même valeur de x,
    notamment lorsque le modèle produit de nombreuses probabilités
    identiques. Ces valeurs sont regroupées avant interpolation.
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
    """
    Calcule les principales statistiques de calibration.

    La fonction de calibration est estimée directement à partir
    des observations individuelles avec une régression LOWESS.

    Paramètres
    ----------
    probas : array-like
        Probabilités prédites pour la classe positive.

    y_test : array-like
        Valeurs observées binaires, codées 0 et 1.

    lowess_frac : float, default=0.30
        Fraction des observations utilisée dans chaque voisinage
        local LOWESS. Une valeur plus faible produit une courbe plus
        flexible ; une valeur plus élevée produit une courbe plus
        lissée.

    lowess_it : int, default=0
        Nombre d'itérations robustes supplémentaires de LOWESS.
        Pour une réponse binaire, 0 est généralement un choix simple
        et reproductible.

    Retours
    -------
    dict
        Calibration intercept, calibration slope, Brier score,
        ICI, E90 et coordonnées de la courbe LOWESS.
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
    # Calibration intercept et calibration slope
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
    # Courbe LOWESS
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

    # La LOWESS n'est pas contrainte dans [0, 1].
    lowess_y = np.clip(
        lowess_y,
        0,
        1,
    )

    # ---------------------------------------------------------
    # Calcul de l'ICI et de l'E90
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
    """
    Affiche :

    - l'histogramme des probabilités prédites en arrière-plan ;
    - la diagonale de calibration parfaite ;
    - la courbe LOWESS avant calibration ;
    - la courbe LOWESS après calibration si applicable ;
    - le calibration intercept ;
    - la calibration slope ;
    - le Brier score ;
    - l'ICI ;
    - l'E90.
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
    # Histogramme en arrière-plan
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
    # Axe de calibration
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
    # Modèle brut
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
    # Modèle calibré
    # ---------------------------------------------------------
    if extraction_type == "TSFEL" and calibration:
        if probas_calib is None:
            raise ValueError(
                "probas_calib must be provided when calibration "
                "is enabled for TSFEL."
            )

        probas_calib, y_test_calib = (
            _validate_calibration_inputs(
                probas_calib,
                y_test_global,
            )
        )

        stats_calib = get_calibration_stats(
            probas=probas_calib,
            y_test=y_test_calib,
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
    # Boîte de métriques
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
    # Sauvegarde
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
    (fpr, tpr, thresholds) = roc_curve(y_test, probas)
    # là si l'AUC est différente entre le modèle LSTM et ici c'est parce que pour le modèle 
    # elle est calculée par rapport à 20% des données de train (validation) alors que là c'est par rapport à test.
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
    """
    Génère la courbe Précision-Rappel (PRC) et calcule l'AUPRC.
    Idéal pour les jeux de données déséquilibrés.
    """
    # 1. Calcul des points de la courbe
    # Note : scikit-learn retourne les thresholds par ordre croissant, 
    # et ajoute une valeur de précision à 1.0 et de rappel à 0.0 à la fin sans threshold associé.
    precision, recall, thresholds = precision_recall_curve(y_test, probas)
    
    # 2. Calcul de l'AUPRC (Aire sous la courbe Precision-Recall) via la méthode des trapèzes (auc)
    auprc_final = auc(recall, precision)
    
    # 3. Calcul de la ligne de base (Baseline / Chance)
    # Contrairement à la ROC où la chance vaut toujours 0.5, pour la PRC,
    # la chance dépend uniquement de la proportion de la classe positive dans le dataset.
    baseline = sum(y_test) / len(y_test)
    
    # 4. Construction graphique
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PRC {model_name} (AUPRC = {auprc_final:.3f})', color='blue', linewidth=2)
    
    # Ligne de base horizontale (Hasard)
    plt.axhline(y=baseline, linestyle='--', color='green', label=f'Chance (Ratio Positifs = {baseline:.3f})')

    plt.xlabel('Recall (True Positive Rate / Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title(f'Precision-Recall Curve for model {model_name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) # 1.05 pour respirer un peu en haut
    plt.legend(loc='lower left') # Souvent en bas à gauche pour les PRC car la courbe chute vers la droite
    plt.grid(True)
    
    if save_figure:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / "prc_curve.png", dpi=300, bbox_inches="tight", transparent=transparent)
        plt.savefig(path / "prc_curve.pdf", bbox_inches="tight", transparent=transparent)
        
    plt.show()
    
    return auprc_final, precision, recall, thresholds

def kde_plot_homemade(probas, y_test, model_name, save_figure, output_dir, transparent):
    plt.figure()

    # Séparation des probabilités
    p0 = probas[y_test == 0]
    p1 = probas[y_test == 1]

    sns.kdeplot(p0, label="Survivors", fill=True)
    sns.kdeplot(p1, label="Deaths", fill=True)

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(f"Score distribution (KDE) for {model_name}")
    plt.legend()
    plt.grid()

    # Calcul de l'aire de superposition
    kde0 = gaussian_kde(p0)
    kde1 = gaussian_kde(p1)

    # On créé un axe x pour évaluer l'aire
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

    # --- CALCUL DU DÉSÉQUILIBRE DE FORME ---
    # On cherche le point d'intersection précis (où les deux courbes se croisent)
    # C'est le point où la différence change de signe proche du milieu
    idx_intersection = np.argwhere(np.diff(np.sign(y0 - y1))).flatten()

    # On calcule la probabilité moyenne pour chaque groupe pour voir le "biais"
    mean_p0 = np.mean(p0)
    mean_p1 = np.mean(p1)
    mean_risk_diff = abs(mean_p0 - mean_p1)

    # On calcule l'écart-type (l'étalement / l'incertitude du modèle)
    std_p0 = np.std(p0)
    std_p1 = np.std(p1)
    print(f"[{model_name}] --- Shape & Fairness Analysis ---")
    print(f"[{model_name}] Survivors -> Mean Risk: {mean_p0:.2f} (±{std_p0:.2f})")
    print(f"[{model_name}] Deaths    -> Mean Risk: {mean_p1:.2f} (±{std_p1:.2f})")
    # Si l'écart-type des Deaths est beaucoup plus grand que celui des Survivors, 
    # le modèle est beaucoup plus "indécis" sur les décès que sur les survivants.
    asymetric_incertitude = abs(std_p0 - std_p1)
    if asymetric_incertitude > 0.05:
        print(f"Warning: Unbalanced uncertainty! The model is less confident on one of the classes.")

    if len(idx_intersection) > 0:
        # On prend le premier point d'intersection trouvé (souvent le principal entre 0 et 1)
        x_cross = x[idx_intersection[0]]
        print(f"[{model_name}] Decision Threshold (KDE Cross): {x_cross:.2f}")
    else:
        x_cross = -1
        print(f"[{model_name}] No decision threshold found (curves do not cross).")
    return non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1

def brier_evolution(probas, y_test, save_figure, output_dir, transparent):
    df_brier = pl.DataFrame({"y" : y_test, "pred" : probas})

    df_brier = df_brier.with_columns(
        ((pl.col("pred") - pl.col("y")) ** 2).alias("brier")
    )

    # score global de brier
    global_brier = brier_score_loss(df_brier["y"], df_brier["pred"])
    print("Brier score: ",global_brier)

    # bins fixes de risque
    df_brier_fixed = (
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

    # déciles de patients 
    n_total = df_brier.height

    df_brier_dec = (
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

    fixed_pd = df_brier_fixed.to_pandas()
    dec_pd = df_brier_dec.to_pandas()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(fixed_pd["x"], fixed_pd["n"], width=0.08, alpha=0.3)
    ax1.set_xlabel("Predicted risk")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(fixed_pd["x"], fixed_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Mean Brier score")

    plt.title(f"Brier score by fixed risk brackets")
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("brier_per_risk_bracket"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3)
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["brier_mean"], marker="o")
    ax2.set_ylabel("Mean Brier score")

    plt.title(f"Brier score by patient deciles")
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("brier_per_decile"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients
    ax1.bar(dec_pd["x"], dec_pd["n"], alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Patient decile")
    ax1.set_ylabel("Number of patients")

    # Courbe de calibration
    ax2 = ax1.twinx()
    ax2.plot(dec_pd["x"], dec_pd["obs_rate"], marker="o", label="Predicted risk", color="black")
    ax2.plot(dec_pd["x"], dec_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
    ax2.set_ylabel("Mortality (Observed rate vs Predicted risk)")

    plt.title(f"Calibration Curve by patient deciles")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
    plt.tight_layout()
    if save_figure:
        plt.savefig(output_dir / Path("calib_per_decile"), dpi = 300, bbox_inches="tight", transparent=transparent)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Histogramme des patients (Tranches fixes)
    ax1.bar(fixed_pd["x"], fixed_pd["n"], width=0.08, alpha=0.3, color='grey', edgecolor='black')
    ax1.set_xlabel("Predicted risk (10% brackets)")
    ax1.set_ylabel("Number of patients")
    ax1.set_xlim(0, 1)

    # Courbe de calibration (axe Y droit)
    ax2 = ax1.twinx()
    ax2.plot(fixed_pd["x"], fixed_pd["obs_rate"], marker="o", label="Mean predicted risk", color="black", linestyle="-")
    ax2.plot(fixed_pd["x"], fixed_pd["pred_mean"], marker="s", label="Observed mortality", color="black", linestyle="--")
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
    cfg = {
        "save_figure": save_figure,
        "output_dir": output_dir,
        "transparent": transparent
    }
    calibration_curve_homemade(probas, [], y_test, config_models.models_name, 
                               config_models.extraction_type, calibration,
                                 **cfg)
    auc_final, fpr, tpr, th = roc_curve_homemade(probas, y_test, config_models.models_name, **cfg)
    non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1 = kde_plot_homemade(probas, y_test, config_models.models_name, **cfg)
    brier_score = brier_evolution(probas, y_test, **cfg)
    best_f1, best_t = f1_score_evolution(probas, y_test, config_models.models_name, **cfg)
    y_pred, mcc = confusion_matrix_homemade(probas, y_test, best_t, config_models.models_name, **cfg)
    return auc_final, fpr, tpr, th, brier_score, best_f1, best_t, y_pred, mcc, non_overlap_area, asymetric_incertitude, mean_risk_diff, mean_p1

def plot_decision_curve_analysis(probas, y_test, model_name, save_figure, output_dir, transparent):
    """
    Calcule et affiche la courbe de Bénéfice Net (Decision Curve Analysis).
    Compare le modèle aux stratégies 'Traiter tout le monde' et 'Ne traiter personne'.
    """
    y_test = np.asarray(y_test)
    probas = np.asarray(probas)
    n = len(y_test)
    
    # Nombre total de vrais positifs (décès) et vrais négatifs (survivants)
    total_pos = np.sum(y_test == 1)
    total_neg = np.sum(y_test == 0)
    
    # Grille de seuils de probabilité p (de 1% à 99%)
    thresholds = np.linspace(0.01, 0.99, 100)
    
    net_benefit_model = []
    net_benefit_all = []
    
    for p in thresholds:
        # Poids mathématique du dommage (ratio p / (1 - p))
        weight = p / (1 - p)
        
        # 1. Stratégie basée sur le modèle
        y_pred = (probas >= p).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        nb_model = (tp / n) - (fp / n) * weight
        net_benefit_model.append(nb_model)
        
        # 2. Stratégie "Traiter tout le monde" (Tout le monde est considéré positif)
        nb_all = (total_pos / n) - (total_neg / n) * weight
        net_benefit_all.append(nb_all)
        
    # 3. Stratégie "Ne traiter personne" -> Le bénéfice net est structurellement égal à 0
    net_benefit_none = np.zeros_like(thresholds)
    
    # --- Construction Graphique ---
    plt.figure(figsize=(8, 6))
    
    # Courbe du modèle
    plt.plot(thresholds, net_benefit_model, color="blue", linewidth=2.5, 
             label=f"Modèle : {model_name}")
    
    # Courbe "Traiter tout le monde"
    plt.plot(thresholds, net_benefit_all, color="red", linestyle="--", linewidth=1.5, 
             label="Stratégie : Considérer tout le monde Positif")
    
    # Courbe "Ne traiter personne"
    plt.plot(thresholds, net_benefit_none, color="black", linestyle="-", alpha=0.6, linewidth=1.5, 
             label="Stratégie : Considérer tout le monde Négatif")
    
    # Ajustement des axes pour la pertinence clinique
    plt.xlim(0.0, 1.0)
    
    # On cadre l'axe Y pour éviter que l'effondrement du bénéfice dans les négatifs n'écrase le graphique
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
    """
    Analyse et visualise l'importance des features (MDI, SHAP) pour n'importe quel 
    ensemble de features TSFEL.
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
    # ----- 1) Importances "forêt" -----
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        importances = np.mean([
            clf.estimator.feature_importances_ 
            for clf in model.calibrated_classifiers_
        ], axis=0)
    elif hasattr(shap_model, 'feature_importances_'): # Sécurité supplémentaire si le wrapper masquait l'attribut
        importances = shap_model.feature_importances_
    else:
        print("⚠️ Ce modèle ne supporte pas 'feature_importances_'. Passage au SHAP.")

    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        sorted_varnames = np.array(varnames)[sorted_idx]
        sorted_importances = importances[sorted_idx]

        # === AJOUT : Agrégation des MDI individuelles ===
        if len(sorted_importances) > k:
            autres_importance = np.sum(sorted_importances[k:])
            plot_importances = np.append(sorted_importances[:k], autres_importance)
            plot_varnames = np.append(sorted_varnames[:k], f"Others Features (N={len(sorted_varnames[:k])})")
        else:
            plot_importances = sorted_importances
            plot_varnames = sorted_varnames

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(plot_importances)), plot_importances)
        plt.xticks(range(len(plot_varnames)), plot_varnames, rotation=90)
        plt.ylabel("Importance (forest)")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}/feature_importance.pdf", bbox_inches="tight", transparent=transparent)
            plt.savefig(f"{folder}/feature_importance.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Pareil mais cumulé : 
        suffixes = feu.generer_suffixes_tsfel()
        racines_varnames = [feu.extraire_racine(name, suffixes) for name in varnames]
        df_mdi = pd.DataFrame({
            'Feature_Globale': racines_varnames,
            'Importance': importances
        })
        
        df_mdi_agg = df_mdi.groupby('Feature_Globale').sum().sort_values(by='Importance', ascending=False)
        k_agg = int(min(top_n, len(df_mdi_agg)))

        # === AJOUT : Agrégation des MDI globales ===
        if len(df_mdi_agg) > k_agg:
            autres_mdi_agg = df_mdi_agg.iloc[k_agg:].sum()
            df_plot_mdi_agg = pd.concat([
                df_mdi_agg.head(k_agg),
                pd.DataFrame([autres_mdi_agg], index=["Autres Features Globales"])
            ])
        else:
            df_plot_mdi_agg = df_mdi_agg
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(df_plot_mdi_agg)), df_plot_mdi_agg['Importance'])
        plt.xticks(range(len(df_plot_mdi_agg)), df_plot_mdi_agg.index, rotation=90)
        plt.ylabel("Cumulative Global Importance (forest)")
        plt.title("Top Global Feature Importance (MDI)")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}/global_feature_importance_mdi.pdf", bbox_inches="tight", transparent=transparent)
            plt.savefig(f"{folder}/global_feature_importance_mdi.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        suffixes = feu.generer_suffixes_tsfel()
        racines_varnames = [feu.extraire_racine(name, suffixes) for name in varnames]

    # ----- 2) SHAP -----
    X_pure_numpy = np.array(X_arr, dtype=np.float32)

    if hasattr(shap_model, 'n_jobs'):
        shap_model.n_jobs = -1
        
    if isinstance(shap_model, XGBClassifier):
        explainer = shap.TreeExplainer(shap_model)
    else:
        try:
            explainer = shap.Explainer(shap_model)
        except Exception:
            explainer = shap.TreeExplainer(shap_model)
            
    shap_values = explainer.shap_values(X_pure_numpy)

    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=-1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_arr = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_arr = shap_values[:, :, None]
    else:
        raise ValueError(f"Format SHAP inattendu: type={type(shap_values)}")

    n_samples, n_features, n_classes = shap_arr.shape

    # ----- 3) Agrégations SHAP individuelles -----
    mean_abs_by_class = np.abs(shap_arr).mean(axis=0)  # (n_features, n_classes)
    feature_sum = mean_abs_by_class.sum(axis=1)
    
    top_idx = np.argsort(feature_sum)[::-1][:min(k, n_features)]
    
    # === AJOUT : Agrégation des SHAP individuels ===
    if n_features > k:
        other_idx = np.argsort(feature_sum)[::-1][k:]
        others_shap = mean_abs_by_class[other_idx].sum(axis=0) # Somme par classe
        plot_data = np.vstack([mean_abs_by_class[top_idx], others_shap])
        plot_labels = [varnames[i] for i in top_idx] + ["Autres Features"]
    else:
        plot_data = mean_abs_by_class[top_idx]
        plot_labels = [varnames[i] for i in top_idx]

    if class_labels is not None:
        class_names = list(class_labels)
    else:
        class_names = ["0", "1"] if (shap_arr.ndim == 3 or (shap_arr.ndim == 2 and "XGB" in str(type(shap_model)))) else [str(i) for i in range(n_classes)]

    if n_classes == 1 and len(class_names) > 1:
        class_names = [class_names[-1]]

    # ----- 4) Barres empilées SHAP -----
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

    # ----- 5) Summary plot par classe -----
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

    # ----- 6) Découpage par classe prédite -----
    y_pred = model.predict(X_arr)
    unique_classes = np.unique(y_pred)
    X_by_class = {cls: X_arr[y_pred == cls] for cls in unique_classes}
    
    # ----- 7) Importance globale agrégée SHAP -----
    df_shap_dict = {'Feature_Globale': racines_varnames}
    for c_id in range(n_classes):
        df_shap_dict[f'SHAP_class_{c_id}'] = mean_abs_by_class[:, c_id]
    
    df_shap_agg = pd.DataFrame(df_shap_dict).groupby('Feature_Globale').sum()
    df_shap_agg['Total_Impact'] = df_shap_agg.sum(axis=1)
    df_shap_agg = df_shap_agg.sort_values(by='Total_Impact', ascending=False).drop(columns=['Total_Impact'])
    
    k_shap_agg = int(min(top_n, len(df_shap_agg)))
    
    # === AJOUT : Agrégation des SHAP globaux ===
    if len(df_shap_agg) > k_shap_agg:
        autres_shap_agg = df_shap_agg.iloc[k_shap_agg:].sum()
        df_shap_plot = pd.concat([
            df_shap_agg.head(k_shap_agg),
            pd.DataFrame([autres_shap_agg], index=["Autres Features Globales"])
        ])
    else:
        df_shap_plot = df_shap_agg
    
    # Graphique SHAP agrégé global
    fig, ax = plt.subplots(figsize=(16, 8))
    left_agg = np.zeros(len(df_shap_plot))
    
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(len(df_shap_plot)),
            width=df_shap_plot[f'SHAP_class_{c_id}'].values,
            left=left_agg,
            label=class_names[c_id]
        )
        left_agg += df_shap_plot[f'SHAP_class_{c_id}'].values
        
    ax.set_yticks(np.arange(len(df_shap_plot)))
    ax.set_yticklabels(df_shap_plot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative mean(|SHAP value|) (average impact per class)")
    ax.set_title("Top Global Feature Importance (SHAP)")
    ax.legend(title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}/global_shap_importance.pdf", bbox_inches="tight", transparent=transparent)
        plt.savefig(f"{folder}/global_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    # On retourne les tops originaux purs (sans la catégorie "Autres")
    return {
        "X_by_class": X_by_class,
        "top_feat": [varnames[i] for i in top_idx],
        "top_global_feat_mdi" : list(df_mdi_agg.index[:k_agg]) if importances is not None else [],
        "top_global_feat_shap" : list(df_shap_agg.index[:k_shap_agg])
    }

def compare_models_figure(figname, max_cols=3, savefig = False, folder = "", **paths):
    """
    Affiche une figure avec une grille dynamique (lignes x colonnes)
    à partir de plusieurs dossiers.
    """
    num_paths = len(paths)
    
    if num_paths == 0:
        print("Error: You must provide at least one path.")
        return
    
    n_cols = min(num_paths, max_cols)
    n_rows = math.ceil(num_paths / n_cols)
    # Création du subplot dynamique (1 ligne, X colonnes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    # Si num_paths == 1, matplotlib ne retourne pas un tableau d'axes mais un seul axe.
    # On le transforme en liste pour que la boucle for fonctionne dans tous les cas.
    if num_paths == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # On itère sur les dossiers fournis
    for i, (model_name, dir_path) in enumerate(paths.items()):
        ax = axes[i]
        full_path = Path(dir_path) / figname        
        
        if full_path.exists():
            img = plt.imread(full_path)
            ax.imshow(img)
            ax.set_title(model_name)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"Image not found\n{model_name}", 
                    ha='center', va='center', color='red')
            ax.set_title(model_name)
            ax.axis('off')
    
    for j in range(num_paths, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}/comparison_{figname}", dpi=300, bbox_inches="tight")
    plt.show()

def générer_rapport_comparatif(configurations, y_true_base = None, save_dir=None, table_format='fancy_grid', saps2_pred = None, saps2_true = None):
    """
    Génère un tableau comparatif et une courbe ROC unique à partir de scores et 
    de prédictions déjà calculés.

    y_true_base : tableau des vraies étiquettes 
    configurations : dictionnaire contenant les scores, prédictions et couleurs pour chaque modèle
    """
    results = {}

    # Configuration de la figure ROC
    plt.figure(figsize=(8, 8))
    for name, config in configurations:
        # Extraction des vecteurs précalculés
        probas = config['probas']
        if y_true_base is None:
            y_true = config['y_true']
        else:
            y_true = y_true_base
            
        # Calcul des métriques
        f1 = config['f1_score']
        mcc = config['mcc']
        auc = config['auc']
        brier = config ['brier']

        # Stockage pour le tableau
        results[name] = {
            'F1-Score': f1,
            'MCC': mcc,
            'AUC': auc,
            "brier" : brier
        }

        # Ajout à la courbe ROC collective
        fpr, tpr, _ = roc_curve(y_true, probas)
        color = config.get('color', None)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

    # 1. Génération du tableau avec tabulate
    df_results = pd.DataFrame(results).T
    print("\n=== PERFORMANCE COMPARISON TABLE ===")
    print(tabulate(df_results, headers='keys', tablefmt=table_format, floatfmt=".3f"))
    # Calcul de l'AUC de IGS2 : 
    if saps2_pred is not None and saps2_true is not None:
        fpr_saps2, tpr_saps2, _ = roc_curve(saps2_true, saps2_pred)
        auc_saps2 = roc_auc_score(saps2_true, saps2_pred)
        name_saps2 = "IGS II Score"
        plt.plot(fpr_saps2, tpr_saps2, label=f'{name_saps2} (AUC = {auc_saps2:.3f})', lw=2, linestyle='-.')
    # 2. Finalisation de la courbe ROC
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance', color='gray')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Sauvegarde automatique des artefacts
    if save_dir:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarde de l'image
        plt.savefig(output_path / "collective_roc_curve.png", dpi=300, bbox_inches="tight")

        # Sauvegarde du tableau au format LaTeX booktabs pour Overleaf
        with open(output_path / "results_table.tex", "w") as f:
            f.write(tabulate(df_results, headers='keys', tablefmt='latex_booktabs', floatfmt=".3f"))

    plt.show()

    return df_results



def plot_collected_learning_curve(sample_sizes, train_matrix, val_matrix, model_name="Model",  folder = "", savefig = True, transparent = True):
    """
    Trace la courbe d'apprentissage récoltée en direct pendant le training des folds.
    
    train_matrix : np.array de shape [5_folds, n_paliers]
    val_matrix   : np.array de shape [5_folds, n_paliers]
    """

    # Calcul des moyennes et std sur l'axe des folds (axis=0)
    train_mean = np.mean(train_matrix, axis=0)
    val_mean = np.mean(val_matrix, axis=0)
    val_std = np.std(val_matrix, axis=0)
    
    plt.figure(figsize=(10, 5))
    
    # Courbe de Train
    plt.plot(sample_sizes, train_mean, "o-", color="crimson", label="Training Score (Mean)", linewidth=2)
    
    # Courbe de Validation Out-Of-Fold avec son ombre de volatilité
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