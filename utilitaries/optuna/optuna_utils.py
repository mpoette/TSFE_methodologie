import optuna
import numpy as np

# ─── LA CONSTANTE COMMUNE À TOUT LE PROJET ───
OPTUNA_STORAGE = "sqlite:///optuna.db"


def get_params(study_name, default_params, storage=OPTUNA_STORAGE):
    # On met OPTUNA_STORAGE par défaut ici aussi !
    params = default_params.copy()
    use_defaults = True

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        if study.best_trial is not None:
            params.update(study.best_params)
            use_defaults = False

    except KeyError:
        pass
    except Exception as e:
        print("Erreur Optuna:", e)

    return params, use_defaults


def extract_best_val_loss(history, metric_name):
    """
    Fonction qui extrait la meilleure valeur de loss/auc sur validation (val_loss/val_auc)
    """
    values = [
        v for v in history.get(metric_name, [])
        if v is not None and np.isfinite(v)
    ]
    if not values:
        raise ValueError("Aucune val_loss/val_auc valide trouvée.")
 
    if metric_name == "val_loss":
        return float(min(values))
    elif metric_name == "val_auc":
        return float(max(values))