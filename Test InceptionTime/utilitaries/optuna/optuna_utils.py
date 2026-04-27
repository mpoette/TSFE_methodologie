import optuna
import numpy as np
def get_params(study_name, default_params, storage):
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


def extract_best_val_loss(history):
    """
    Fonction qui extrait la meilleure valeur de loss sur validation (val_loss)
    """
    if history is None:
        raise ValueError("history est None.")
    if "val_loss" not in history:
        raise ValueError("La clé 'val_loss' est absente de history.")
 
    values = [v for v in history["val_loss"] if v is not None and np.isfinite(v)]
    if not values:
        raise ValueError("Aucune val_loss valide trouvée.")
 
    return float(min(values))
