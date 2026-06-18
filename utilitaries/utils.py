from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import os


class Experiment:
    def __init__(self, config_mode, config_cleaning, config_y, str_balance_method, modex, class_weight_choice, str_pop, seed):
        self.clean = config_cleaning.clean
        self.target_name = config_y.target_name
        self.str_balance_method = str_balance_method
        self.modex = modex.value
        self.class_weight = class_weight_choice.value
        self.str_pop = str_pop
        self.seed = seed
        self.config_mode = config_mode.name

    @property
    def dirname(self):
        """Génère le nom du dossier dynamiquement via self."""
        return (
            f"Clean_{self.clean}_{self.target_name}_"
            f"{self.str_balance_method}{self.modex}"
            f"{self.class_weight}{self.str_pop}_seed_{self.seed}"
        )
    
    def shortdirname(self, class_weight = ""):
        return (
            f"Clean_{self.clean}_{self.target_name}_"
            f"{self.str_balance_method}{self.modex}"
            f"{class_weight}{self.str_pop}_seed_{self.seed}"
        )

    def get_model_path(self, model_name, fold_idx, extension=".joblib", class_weight = "",):
        """Retourne le chemin d'un fold spécifique et crée le dossier parent s'il manque."""
        path = Path("models") / model_name / self.shortdirname(class_weight)
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"fold_{fold_idx}{extension}"
    
    def get_output_path(self, model_name, class_weight = ""):
        path = Path("outputs") / model_name / self.shortdirname(class_weight)
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path
    
    def load_model(self, model_name, class_weight = "", name_file = "all_res.joblib"):
        file_path = self.get_output_path(model_name, class_weight) / name_file
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        return model_name, joblib.load(file_path)

    def get_tsfel_parquet_path(self):
        path = Path("inputs") 
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"tsfel_global_brut_{self.config_mode}_{self.clean}_{self.target_name}_{self.modex}.parquet"
    
    def get_tsfel_boruta(self, mode, fold_idx):
        path = Path("inputs") / f"tsfel_{mode}_{self.shortdirname()}"
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"fold_{fold_idx}.parquet"



def compare_models_figure(figname, **paths):
    """
    Affiche une figure côte à côte à partir de plusieurs dossiers.
    """
    num_paths = len(paths)
    
    if num_paths == 0:
        print("Erreur : Vous devez fournir au moins un chemin (path).")
        return

    # Création du subplot dynamique (1 ligne, X colonnes)
    fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 5))
    
    # Si num_paths == 1, matplotlib ne retourne pas un tableau d'axes mais un seul axe.
    # On le transforme en liste pour que la boucle for fonctionne dans tous les cas.
    if num_paths == 1:
        axes = [axes]
        
    # On itère sur les dossiers fournis
    for ax, (model_name, dir_path) in zip(axes, paths.items()):
        full_path = Path(dir_path) / figname        
        if full_path.exists():
            img = plt.imread(full_path)
            ax.imshow(img)
            ax.set_title(model_name) # Utilise le nom de l'argument comme titre
            ax.axis('off') # Masque les axes pour une image plus propre
        else:
            ax.text(0.5, 0.5, f"Image introuvable\n{model_name}", 
                    ha='center', va='center', color='red')
            ax.set_title(model_name)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def générer_rapport_comparatif(y_true, configurations, save_dir=None, table_format='fancy_grid', saps2_pred = None, saps2_true = None):
    """
    Génère un tableau comparatif et une courbe ROC unique à partir de scores et 
    de prédictions déjà calculés.

    y_true : tableau des vraies étiquettes (ex: y_test ou y_news)
    configurations : dictionnaire contenant les scores, prédictions et couleurs pour chaque modèle
    """
    results = {}

    # Configuration de la figure ROC
    plt.figure(figsize=(8, 8))
    for name, config in configurations:
        # Extraction des vecteurs précalculés
        probas = config['probas']

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
    print("\n=== TABLEAU COMPARATIF DES PERFORMANCES ===")
    print(tabulate(df_results, headers='keys', tablefmt=table_format, floatfmt=".3f"))
    # Calcul de l'AUC de IGS2 : 
    if saps2_pred is not None and saps2_true is not None:
        fpr_saps2, tpr_saps2, _ = roc_curve(saps2_true, saps2_pred)
        auc_saps2 = roc_auc_score(saps2_true, saps2_pred)
        name_saps2 = "Score IGS2"
        plt.plot(fpr_saps2, tpr_saps2, label=f'{name_saps2} (AUC = {auc_saps2:.3f})', lw=2, linestyle='-.')
    # 2. Finalisation de la courbe ROC
    plt.plot([0, 1], [0, 1], linestyle='--', label='Hasard', color='gray')
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Comparaison des Courbes ROC')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Sauvegarde automatique des artefacts pour votre publi
    if save_dir:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarde de l'image
        plt.savefig(output_path / "courbe_roc_collective.png", dpi=300, bbox_inches="tight")

        # Sauvegarde du tableau au format LaTeX booktabs pour Overleaf
        with open(output_path / "tableau_resultats.tex", "w") as f:
            f.write(tabulate(df_results, headers='keys', tablefmt='latex_booktabs', floatfmt=".3f"))

    plt.show()

    return df_results