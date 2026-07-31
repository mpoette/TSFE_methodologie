from dataclasses import dataclass

import utilitaries.config_utils as mo_utils


@dataclass(frozen=True)
class ConfigValue:
    """Wrap a configuration value in an immutable object.

    Attributes:
        value:
            Wrapped configuration value.
    """

    value: object


@dataclass(frozen=True)
class PipelineConfig:
    """Store the complete configuration of a preprocessing and model pipeline.

    Attributes:
        config_transparent:
            Whether generated figures should use a transparent background.
        config_cleaning:
            Dataset-cleaning configuration.
        config_mode:
            Windowing or resampling configuration.
        type_donnees:
            Selected data or evaluation type.
        modex:
            Selected feature mode.
        config_balance:
            Class-balancing configuration.
        config_y:
            Prediction-target configuration.
        save_figure:
            Whether generated figures should be saved.
        boruta_filter:
            Whether Boruta feature selection should be enabled.
        config_boruta:
            Boolean Boruta configuration used directly by the pipeline.
        config_keep_pop:
            Population-filtering configuration.
        use_optuna:
            Whether Optuna hyperparameter optimization should be enabled.
        config_models:
            Selected model or clinical-score configuration.
        calibration:
            Whether calibration should be applied.
        calibration_mode:
            Selected calibration strategy.
        extract_tsfel:
            Whether TSFEL features should be extracted.
        class_weight_choice:
            Selected class-weight strategy.
    """

    config_transparent: bool
    config_cleaning: object
    config_mode: object
    type_donnees: ConfigValue
    modex: ConfigValue
    config_balance: object
    config_y: object
    save_figure: ConfigValue
    boruta_filter: ConfigValue
    config_boruta: bool
    config_keep_pop: object
    use_optuna: ConfigValue
    config_models: object
    calibration: ConfigValue
    calibration_mode: ConfigValue
    extract_tsfel: ConfigValue
    class_weight_choice: ConfigValue


def get_model_config(type_donnees, model_name, score_name):
    """Return the selected model or clinical-score configuration.

    Args:
        type_donnees:
            Selected data type. Supported values are ``"modèle"`` and
            ``"score"``.
        model_name:
            Key used to retrieve a model configuration from
            ``mo_utils.MODELS``.
        score_name:
            Key used to retrieve a score configuration from
            ``mo_utils.SCORE``.

    Returns:
        The selected model or score configuration object.

    Raises:
        ValueError:
            If ``type_donnees`` is not supported.
    """
    if type_donnees == "modèle":
        return mo_utils.MODELS[model_name]

    if type_donnees == "score":
        return mo_utils.SCORE[score_name]

    raise ValueError(
        f"Type de données inconnu : {type_donnees}"
    )


def get_calibration_config(model_config, balance_method):
    """Determine whether calibration is required and select its strategy.

    Prior correction is enforced when class balancing is applied to a
    non-time-series model. Models already marked as calibrated do not require
    an additional calibration step. Time-series models use temperature
    scaling, while other models default to Platt scaling.

    Args:
        model_config:
            Model configuration exposing ``extraction_type`` and
            ``models_type`` attributes.
        balance_method:
            Selected class-balancing method.

    Returns:
        A tuple containing:

        - A wrapped Boolean indicating whether calibration is enabled.
        - A wrapped string identifying the calibration strategy.
    """
    extraction_type = model_config.extraction_type
    models_type = model_config.models_type

    has_balancing = balance_method != ""

    # Balancing changes the class prior. Non-time-series models therefore
    # require prior correction.
    is_prior_forced = (
        has_balancing
        and extraction_type != "time"
    )

    if is_prior_forced:
        return ConfigValue(True), ConfigValue("prior")

    # Models already stored as calibrated do not require a new calibration
    # fitting step.
    if models_type == "calibrated":
        return ConfigValue(False), ConfigValue("platt")

    # Neural time-series models use temperature scaling.
    if extraction_type == "time":
        return (
            ConfigValue(True),
            ConfigValue("temperature_scaling"),
        )

    # Other models use Platt scaling by default.
    return ConfigValue(True), ConfigValue("platt")


def get_extraction_config(
    model_config,
    extract_tsfel=False,
    class_weight="balanced",
):
    """Build TSFEL extraction and class-weight configuration values.

    TSFEL extraction and class weighting are only relevant for models whose
    extraction type is ``"TSFEL"``.

    Args:
        model_config:
            Model configuration exposing an ``extraction_type`` attribute.
        extract_tsfel:
            Whether TSFEL features should be extracted.
        class_weight:
            Class-weight strategy used by TSFEL-based models.

    Returns:
        A tuple containing:

        - A wrapped Boolean controlling TSFEL extraction.
        - A wrapped class-weight value.
    """
    if model_config.extraction_type == "TSFEL":
        return (
            ConfigValue(extract_tsfel),
            ConfigValue(class_weight),
        )

    return ConfigValue(False), ConfigValue("")


def create_pipeline_config(
    *,
    type_donnees,
    mode_fenetrage,
    model_name,
    score_name,
    nettoyage,
    cible,
    mode_features,
    equilibrage,
    population,
    save_figure,
    transparent,
    boruta_filter,
    use_optuna,
    extract_tsfel,
    class_weight,
):
    """Create the complete pipeline configuration from interface selections.

    The function retrieves the corresponding immutable configuration objects
    from ``config_utils``, determines the required calibration strategy, and
    configures TSFEL extraction and class weighting according to the selected
    model.

    Args:
        type_donnees:
            Selected data type, such as model or clinical score.
        mode_fenetrage:
            Key identifying the windowing or resampling configuration.
        model_name:
            Key identifying the selected machine-learning model.
        score_name:
            Key identifying the selected clinical score.
        nettoyage:
            Key identifying the cleaning configuration.
        cible:
            Key identifying the prediction target.
        mode_features:
            Selected feature-set mode.
        equilibrage:
            Key identifying the class-balancing configuration.
        population:
            Key identifying the population configuration.
        save_figure:
            Whether generated figures should be saved.
        transparent:
            Whether generated figures should use transparent backgrounds.
        boruta_filter:
            Whether Boruta feature selection should be enabled.
        use_optuna:
            Whether Optuna hyperparameter optimization should be enabled.
        extract_tsfel:
            Whether TSFEL features should be extracted.
        class_weight:
            Selected class-weight strategy.

    Returns:
        A fully initialized ``PipelineConfig`` instance.
    """
    # Retrieve the selected configuration objects from Marimo utilities.
    config_mode = mo_utils.MODES[mode_fenetrage]
    config_cleaning = mo_utils.CLEAN[nettoyage]
    config_y = mo_utils.Y[cible]
    config_balance = mo_utils.BALANCE[equilibrage]
    config_keep_pop = mo_utils.POPULATION[population]

    # Retrieve either the selected machine-learning model or clinical score.
    config_models = get_model_config(
        type_donnees=type_donnees,
        model_name=model_name,
        score_name=score_name,
    )

    # Determine whether calibration is required and which strategy to use.
    calibration, calibration_mode = get_calibration_config(
        model_config=config_models,
        balance_method=config_balance.balance_method,
    )

    # Configure TSFEL extraction and class weighting when applicable.
    extract_tsfel_config, class_weight_choice = (
        get_extraction_config(
            model_config=config_models,
            extract_tsfel=extract_tsfel,
            class_weight=class_weight,
        )
    )

    # Assemble all configuration components into one immutable object.
    return PipelineConfig(
        config_transparent=transparent,
        config_cleaning=config_cleaning,
        config_mode=config_mode,
        type_donnees=ConfigValue(type_donnees),
        modex=ConfigValue(mode_features),
        config_balance=config_balance,
        config_y=config_y,
        save_figure=ConfigValue(save_figure),
        boruta_filter=ConfigValue(boruta_filter),
        config_boruta=boruta_filter,
        config_keep_pop=config_keep_pop,
        use_optuna=ConfigValue(use_optuna),
        config_models=config_models,
        calibration=calibration,
        calibration_mode=calibration_mode,
        extract_tsfel=extract_tsfel_config,
        class_weight_choice=class_weight_choice,
    )