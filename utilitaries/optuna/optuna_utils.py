import optuna
import numpy as np

OPTUNA_STORAGE = "sqlite:///optuna.db"


def get_params(study_name, default_params, storage=OPTUNA_STORAGE):
    """Load the best Optuna parameters or fall back to defaults.

    Args:
        study_name:
            Name of the Optuna study to load.
        default_params:
            Default parameter dictionary.
        storage:
            Optuna storage URL.

    Returns:
        A tuple containing the merged parameter dictionary and a Boolean
        indicating whether only default parameters were used.
    """
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
    """Extract the best finite validation metric from a training history.

    The minimum value is returned for ``"val_loss"``, while the maximum value
    is returned for ``"val_auc"``.

    Args:
        history:
            Mapping containing metric histories.
        metric_name:
            Validation metric to extract.

    Returns:
        The best finite validation metric value.

    Raises:
        ValueError:
            If no valid finite metric value is available.
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
