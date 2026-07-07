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
    )
import utilitaries.features_extraction_utils as feu
from pathlib import Path

def calibration_curve_homemade(probas_uncalib, probas_calib, y_test_global, 
                             model_name, extraction_type, calibration, save_figure, 
                             output_dir, transparent, calibration_mode = "Platt"):
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")

    # Courbe de base (Modèle brut ou modèle DL déjà calibré en température)
    fraction_pos_uncalib, mean_pred_uncalib = calibration_curve(y_test_global, probas_uncalib, n_bins=10)
    plt.plot(mean_pred_uncalib, fraction_pos_uncalib, "s-", color="red", label=f"Curve ({model_name})")

    # Courbe calibrée (Uniquement affichée pour TSFEL si demandée)
    if extraction_type == "TSFEL" and calibration:
        fraction_pos_calib, mean_pred_calib = calibration_curve(y_test_global, probas_calib, n_bins=10)
        plt.plot(mean_pred_calib, fraction_pos_calib, "s-", color="blue", label=f"After calibration ({calibration_mode[1:]})")

    plt.ylabel("True fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title(f"Global Calibration Curve (5-Fold Cross-Validation)\nModel: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Sauvegarde propre de l'image
    if save_figure:
        plt.savefig(output_dir / Path("calibration_curve"), dpi=300, bbox_inches="tight", transparent=transparent)

    plt.show()
    
def roc_curve_homemade(probas, y_test, model_name, save_figure, output_dir, transparent):
    (fpr, tpr, thresholds) = roc_curve(y_test, probas)
    # là si l'AUC est différente entre le modèle LSTM et ici c'est parce que pour le modèle elle est calculée par rapport à 20% des données de train (validation) alors que là c'est par rapport à test.
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

    # Pour avoir 2 plots au même endroit, on utilise twinx
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


def mesureImportance_tsfel(model, X_train, varnames, top_n=20, class_labels=None, folder = "", savefig = True, transparent = True, seed = 42):
    """
    Analyse et visualise l'importance des features (MDI, SHAP) pour n'importe quel 
    ensemble de features TSFEL (ex: après filtrage Boruta).
    
    X_train : numpy.ndarray ou pandas.DataFrame (les features déjà filtrées)
    varnames : liste ou array des noms de ces features
    """
    np.random.seed(seed)
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()
    
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values
    else:
        X_arr = np.asarray(X_train) # Conserve le format 3D pour InceptionTime
        
    varnames = list(varnames)
    
    # On ajuste top_n si on a moins de features que prévu
    k = int(min(top_n, len(varnames)))
    # Si c'est un GridSearchCV / RandomizedSearchCV, on prend le meilleur modèle
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    # ----- 1) Importances "forêt" -----
    importances = None
    # Gestion des modèles calibrés pour récupérer l'importance des features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        # On fait la moyenne des importances de tous les sous-modèles de la calibration
        importances = np.mean([
            clf.estimator.feature_importances_ 
            for clf in model.calibrated_classifiers_
        ], axis=0)
    else:
        print("⚠️ Ce modèle ne supporte pas 'feature_importances_'.")
        print("   -> Saut de l'étape MDI, passage direct à l'analyse SHAP.")
    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        sorted_varnames = np.array(varnames)[sorted_idx]
        sorted_importances = importances[sorted_idx]
        plt.figure(figsize=(12, 4))
        plt.bar(range(k), sorted_importances[:k])
        plt.xticks(range(k), sorted_varnames[:k], rotation=90)
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
        # On additionne les importances des sous-features appartenant à la même feature globale
        df_mdi_agg = df_mdi.groupby('Feature_Globale').sum().sort_values(by='Importance', ascending=False)
        k_agg = int(min(top_n, len(df_mdi_agg)))
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(k_agg), df_mdi_agg['Importance'].head(k_agg))
        plt.xticks(range(k_agg), df_mdi_agg.index[:k_agg], rotation=90)
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
    X_pure_numpy = np.array(X_arr, dtype = np.float32)
    shap_model = model
    # On autorise le multicoeur
    if hasattr(model, 'n_jobs'):
        model.n_jobs = -1

    if hasattr(model, 'calibrated_classifiers_'):
        # On prend le premier estimateur de la calibration (ils partagent la même structure)
        shap_model = model.calibrated_classifiers_[0].estimator
    
    if shap_model.__class__.__name__ == "FrozenEstimator" and hasattr(shap_model, "estimator"):
        shap_model = shap_model.estimator
    if isinstance(shap_model, XGBClassifier):
        explainer = shap.TreeExplainer(shap_model)
    else:
        try:
            explainer = shap.Explainer(shap_model)
        except Exception:
            explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_pure_numpy)

    # Gestion de la structure des shap_values selon la version de SHAP / type de modèle
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=-1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_arr = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_arr = shap_values[:, :, None]
    else:
        raise ValueError(f"Format SHAP inattendu: type={type(shap_values)}, shape={getattr(shap_values, 'shape', None)}")

    n_samples, n_features, n_classes = shap_arr.shape

    # ----- 3) Agrégations SHAP -----
    mean_abs_by_class = np.abs(shap_arr).mean(axis=0)  # (n_features, n_classes)
    feature_sum = mean_abs_by_class.sum(axis=1)
    
    # On ajuste le top index au nombre réel de features disponibles
    top_idx = np.argsort(feature_sum)[::-1][:min(k, n_features)]
    plot_data = mean_abs_by_class[top_idx]
    plot_labels = [varnames[i] for i in top_idx]

    # Noms de classes dynamiques
    if class_labels is not None:
        class_names = list(class_labels)
    else:
        # Si pas de labels, on génère par défaut ["0", "1"] basés sur la vraie logique binaire
        # et non pas sur la dimension technique de shap_arr (n_classes)
        class_names = ["0", "1"] if (shap_arr.ndim == 3 or (shap_arr.ndim == 2 and "XGB" in str(type(shap_model)))) else [str(i) for i in range(n_classes)]

    # Si XGBoost nous donne une seule matrice, elle correspond TOUJOURS à la classe positive (la dernière)
    if n_classes == 1 and len(class_names) > 1:
        # On force la liste des noms à ne contenir que la classe d'intérêt (Deaths)
        class_names = [class_names[-1]]

    # ----- 4) Barres empilées -----
    fig, ax = plt.subplots(figsize=(16, 8))
    left = np.zeros(len(top_idx))
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(len(top_idx)),
            width=plot_data[:, c_id],
            left=left,
            label=class_names[c_id]
        )
        left += plot_data[:, c_id]
        
    ax.set_yticks(np.arange(len(top_idx)))
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
    
    # ----- 7) Importance globale agrégée
    df_shap_dict = {'Feature_Globale': racines_varnames}
    for c_id in range(n_classes):
        df_shap_dict[f'SHAP_class_{c_id}'] = mean_abs_by_class[:, c_id]
    
    # Fusionner et grouper par feature globale en sommant
    df_shap_agg = pd.DataFrame(df_shap_dict).groupby('Feature_Globale').sum()
    
    # Calcul de la somme toutes classes confondues pour trier le Top N global
    df_shap_agg['Total_Impact'] = df_shap_agg.sum(axis=1)
    df_shap_agg = df_shap_agg.sort_values(by='Total_Impact', ascending=False).drop(columns=['Total_Impact'])
    
    k_shap_agg = int(min(top_n, len(df_shap_agg)))
    df_shap_plot = df_shap_agg.head(k_shap_agg)
    
    # Graphique SHAP agrégé
    fig, ax = plt.subplots(figsize=(16, 8))
    left_agg = np.zeros(k_shap_agg)
    
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(k_shap_agg),
            width=df_shap_plot[f'SHAP_class_{c_id}'].values,
            left=left_agg,
            label=class_names[c_id]
        )
        left_agg += df_shap_plot[f'SHAP_class_{c_id}'].values
        
    ax.set_yticks(np.arange(k_shap_agg))
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

    # =====================================================================
    # BLOC : GENERATION DU GRAPHIQUE SHAP SUMMARY PLOT (ROSE/BLEU) CUMULÉ
    # =====================================================================

    # 1. Extraction des racines uniques via les suffixes TSFEL
    suffixes = feu.generer_suffixes_tsfel()
    racines_varnames = [feu.extraire_racine(name, suffixes) for name in varnames]

    # 2. Agrégation de la matrice X par racine (en transposant pour éviter les warnings Pandas)
    # On prend la moyenne des sous-features pour conserver une notion de valeur "Basse" ou "Haute"
    df_X = pd.DataFrame(X_arr, columns=varnames)
    X_global_df = df_X.T.groupby(racines_varnames).mean().T

    # Liste finale des variables fusionnées (triées automatiquement par le groupby)
    liste_racines = list(X_global_df.columns)
    X_global_arr = X_global_df.values

    # 3. Agrégation des Shapley Values pour la classe de ton choix (ex: classe 0)
    target_class_idx = n_classes - 1
    actual_class_name = class_names[-1]
    shap_classe_pure = shap_arr[..., target_class_idx] # Forme (n_samples, n_features)

    # Initialisation de la matrice SHAP globale : (n_samples, n_racines)
    shap_global_arr = np.zeros((n_samples, len(liste_racines)), dtype=np.float32)

    # Remplissage par la somme des contributions SHAP de chaque sous-feature
    for idx, racine in enumerate(liste_racines):
        indices_sous_features = [i for i, r in enumerate(racines_varnames) if r == racine]
        shap_global_arr[:, idx] = shap_classe_pure[:, indices_sous_features].sum(axis=1)

    # 4. Affichage du graphique SHAP Summary Plot (Bleu / Rose) fusionné
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_global_arr, 
        X_global_arr, 
        feature_names=liste_racines, 
        show=False
    )
    plt.title(f"SHAP Value Impact (Global Fused) - {actual_class_name}")
    plt.tight_layout()

    if savefig:
        plt.savefig(f"{folder}/global_fused_shap_summary_{actual_class_name}.pdf", bbox_inches="tight", transparent=transparent)
        plt.savefig(f"{folder}/global_fused_shap_summary_{actual_class_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return {
        "X_by_class": X_by_class,
        "top_feat": plot_labels,
        "top_global_feat_mdi" : list(df_mdi_agg.index[:k_agg]),
        "top_global_feat_shap" : list(df_shap_plot.index)
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