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



def neighbors_from_choices(best_value, choices):
    """
    Pour un hyperparam catégoriel ordinal, on prend le voisin du dessous,
    lui-même, et le voisin du dessus.
    """
    choices = list(choices)
    idx = choices.index(best_value)
    low_idx = max(0, idx - 1)
    high_idx = min(len(choices) - 1, idx + 1)
    return choices[low_idx:high_idx + 1]
