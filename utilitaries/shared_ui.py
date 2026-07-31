import sys

import marimo as mo

import utilitaries.config_utils as mo_utils


def create_pipeline_widgets():
    """Create and return the main pipeline widgets as a dictionary."""
    return {
        "transparent": mo.ui.dropdown(
            options={
                "Oui": True,
                "Non": False,
            },
            value="Non",
            label="Figures transparentes",
        ),
        "cleaning": mo.ui.dropdown(
            options=mo_utils.CLEAN,
            value="Enlever Surveillance Continue",
            label="Nettoyage SC",
        ),
        "mode": mo.ui.dropdown(
            options=mo_utils.MODES,
            value="24h début réanimation sans remplissage",
            label="Fenêtrage",
        ),
        "type_donnees": mo.ui.dropdown(
            options={
                "modèle": "modèle",
                "score": "score",
            },
            value="modèle",
            label="Type de données",
        ),
        "modex": mo.ui.dropdown(
            options=list(mo_utils.FEAT.keys()),
            value="Mode IGS2",
            label="Features gardées",
        ),
        "balance": mo.ui.dropdown(
            options=mo_utils.BALANCE,
            value="Aucune Méthode",
            label="Équilibrage",
        ),
        "y_dd": mo.ui.dropdown(
            options=mo_utils.Y,
            value="Survie à 28 jours",
            label="Cible (y)",
        ),
        "save_figure": mo.ui.dropdown(
            options={
                "Oui": True,
                "Non": False,
            },
            value="Oui",
            label="Sauver les figures",
        ),
        "boruta_filter": mo.ui.dropdown(
            options={
                "Oui": True,
                "Non": False,
            },
            value="Oui",
            label="Filtre Boruta",
        ),
        "keep_pop": mo.ui.dropdown(
            options=mo_utils.POPULATION,
            value="Tout",
            label="Filtre Population ICU_DP",
        ),
        "use_optuna": mo.ui.dropdown(
            options={
                "Oui": True,
                "Non": False,
            },
            value="Oui",
            label="Utiliser Optuna (si disponible)",
        ),
    }


def get_model_dropdown(type_donnees_value):
    """Create the model or clinical-score dropdown.

    The returned widget depends on the selected data type.

    Args:
        type_donnees_value:
            Selected data type, expected to be either ``"modèle"`` or
            ``"score"``.

    Returns:
        A Marimo dropdown containing either model configurations or clinical
        score configurations.
    """
    if type_donnees_value == "modèle":
        return mo.ui.dropdown(
            options=mo_utils.MODELS,
            value="XGBoost TSFEL",
            label="Modèle utilisé",
        )

    else:
        return mo.ui.dropdown(
            options=mo_utils.SCORE,
            value="IGS2",
            label="Score utilisé",
        )


def get_calibration_widgets(
    models_value,
    balance_value,
):
    """Create calibration widgets based on model and balancing settings.

    Prior correction is forced when class balancing is enabled for a
    non-time-series model. Otherwise, the available calibration choices
    depend on the selected model type and feature extraction strategy.

    Args:
        models_value:
            Selected model configuration.
        balance_value:
            Selected balancing method.

    Returns:
        A tuple containing the calibration activation widget and the
        calibration-method widget.
    """
    # Retrieve model properties.
    # Fallback values are used if the selected object does not expose the
    # expected attributes.
    extraction_type = getattr(
        models_value,
        "extraction_type",
        "TSFEL",
    )

    models_type = getattr(
        models_value,
        "models_type",
        "standard",
    )

    # Determine whether prior correction must be forced.
    # An empty balancing value means that no balancing method is active.
    has_balancing = balance_value != ""

    is_prior_forced = (
        has_balancing
        and extraction_type != "time"
    )

    # Build widget options and default values dynamically.
    if is_prior_forced:
        # Prior correction is mandatory, so the user cannot change either
        # the calibration status or the calibration method.
        options_calib = {
            "Oui (Forcé par le resampling)": True,
        }

        bool_calib = "Oui (Forcé par le resampling)"

        options_mode = {
            "Prior": "prior",
        }

        calib_mode = "Prior"

    else:
        # Prior correction is excluded from the available options.
        options_calib = {
            "Oui": True,
            "Non": False,
        }

        options_mode = {
            "Platt": "platt",
            "Temperature Scaling": "temperature_scaling",
        }

        # Define default values according to the model configuration.
        if models_type == "calibrated":
            bool_calib = "Non"
            calib_mode = "Platt"

        else:
            bool_calib = "Oui"

            # Time-series models default to temperature scaling, while other
            # models default to Platt scaling.
            calib_mode = (
                "Temperature Scaling"
                if extraction_type == "time"
                else "Platt"
            )

    # Create the Marimo calibration widgets.
    calibration = mo.ui.dropdown(
        options=options_calib,
        value=bool_calib,
        label="Activer calibration",
    )

    calibration_mode = mo.ui.dropdown(
        options=options_mode,
        value=calib_mode,
        label="Méthode calibration",
    )

    return calibration, calibration_mode


def get_tsfel_ui_components(
    extraction_type,
    models_type,
    boruta_filter,
    calibration,
    calibration_mode,
):
    """Create and assemble TSFEL-specific interface components.

    Args:
        extraction_type:
            Feature extraction strategy used by the selected model.
        models_type:
            Selected model type or calibration status.
        boruta_filter:
            Boruta filtering widget.
        calibration:
            Calibration activation widget.
        calibration_mode:
            Calibration-method widget.

    Returns:
        A tuple containing:

        - The assembled TSFEL interface container.
        - The TSFEL extraction widget.
        - The class-weight widget.
    """
    if extraction_type == "TSFEL":
        extract_tsfel = mo.ui.dropdown(
            options={
                "Oui": True,
                "Non": False,
            },
            value="Non",
            label="Extraire les données TSFEL",
        )

        class_weight_choice = mo.ui.dropdown(
            options={
                "balanced": "balanced",
                "balanced_subsample": "balanced_subsample",
            },
            value="balanced",
            label="class_weight",
        )

        # Hide calibration widgets when the selected model is already
        # calibrated.
        if models_type == "calibrated":
            ui_tsfel = mo.vstack(
                [
                    extract_tsfel,
                    boruta_filter,
                    class_weight_choice,
                ]
            )

        else:
            ui_tsfel = mo.vstack(
                [
                    extract_tsfel,
                    boruta_filter,
                    calibration,
                    calibration_mode,
                    class_weight_choice,
                ]
            )

    else:
        # Provide neutral fallback widgets for non-TSFEL models so that the
        # rest of the pipeline can use a consistent interface.
        extract_tsfel = None

        class_weight_choice = mo.ui.dropdown(
            options={
                "": "",
            },
            value="",
        )

        ui_tsfel = mo.md("")

    return (
        ui_tsfel,
        extract_tsfel,
        class_weight_choice,
    )


def render_sidebar(
    uid,
    models,
    ui_tsfel,
    keep_feats,
    str_keep_feats,
    run,
    run_optuna,
    seed,
    custom_features,
):
    """Build the list of components displayed in the notebook sidebar.

    The displayed widgets depend on whether the user selected a machine-
    learning model or a clinical score.

    Args:
        uid:
            Dictionary containing the main pipeline widgets.
        models:
            Dynamic model or score selection widget.
        ui_tsfel:
            TSFEL-specific widget container.
        keep_feats:
            Selected feature names.
        str_keep_feats:
            Human-readable feature description.
        run:
            Main execution button.
        run_optuna:
            Optuna execution button.
        seed:
            Global random seed.
        custom_features:
            Widget used to select custom features.

    Returns:
        A list of Marimo components to display in the sidebar.
    """
    # Add the common components displayed at the top of the sidebar.
    sidebar_items = [
        mo.md(mo_utils.config_sidebar),
        mo.md(
            f"<U>Seed utilisée pour l'ensemble du code : "
            f"**{seed}**</U>"
        ),
        mo.md(
            f"Version de python : {sys.version}"
        ),
        mo.md("-----"),
        uid["type_donnees"],
        mo.md("-----"),
        uid["mode"],
        mo.md("-----"),
        models,
    ]

    # Add model-specific or score-specific components.
    if uid["type_donnees"].value == "modèle":
        modex = uid["modex"]

        sidebar_items.extend(
            [
                uid["balance"],
                ui_tsfel,
                uid["cleaning"],
                uid["y_dd"],
                uid["keep_pop"],
                uid["use_optuna"],
                modex,
                (
                    custom_features
                    if modex.value == "Mode Custom"
                    else mo.md("*(features fixe)*")
                ),
                mo.md(
                    f"**Features gardées :** `{keep_feats}`"
                ),
                mo.md(
                    "**Soit en Français "
                    "(dynamic feature only):** "
                    f"\n{str_keep_feats}"
                ),
            ]
        )

    else:
        sidebar_items.extend(
            [
                uid["y_dd"],
                uid["keep_pop"],
            ]
        )

    # Add components shared by both models and clinical scores.
    sidebar_items.extend(
        [
            uid["save_figure"],
            uid["transparent"],
            mo.md("-----"),
        ]
    )

    # Execution buttons are only displayed for machine-learning models.
    if uid["type_donnees"].value == "modèle":
        sidebar_items.extend(
            [
                run_optuna,
                mo.md("-----"),
                run,
            ]
        )

    sidebar_items.extend(
        [
            mo.md(mo_utils.config_end),
        ]
    )

    return sidebar_items