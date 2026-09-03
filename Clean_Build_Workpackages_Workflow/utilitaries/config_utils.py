"""Global configuration dataclasses, type aliases, and widget styles.

Provides the shared configuration objects used across the preprocessing
and machine-learning pipelines (windowing, population, clinical scores,
models, feature sets, class balancing, and prediction targets).
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias


# ---------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------

FeatureSelection: TypeAlias = list[str] | Literal[
    "custom",
    "all_wp",
    "all",
    "mcuwp",
    "mcu",
]


# ---------------------------------------------------------------------
# Global Marimo widget styles
# ---------------------------------------------------------------------

config_dropdown_color = (
    "<div style='background:#F0FFD4;"
    "padding:8px;"
    "border-radius:6px'>"
)

config_run_button = (
    "<div style='background:#FFFA7A;"
    "padding:8px;"
    "border-radius:6px'>"
)

config_msg = (
    "<div style='background:#FFABAB;"
    "padding:8px;"
    "border-radius:6px'>"
)

config_sidebar = (
    "<div style='background:#F9F9F9;"
    "padding:8px;"
    "border-radius:6px'>"
)

config_end = "</div>"


# ---------------------------------------------------------------------
# Marimo windowing and resampling configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigFenetrage:
    """Configuration for time-window extraction or ICU stay resampling.

    Attributes:
        name:
            Internal identifier of the preprocessing configuration.
        hour_offset:
            Offset used when extracting a fixed temporal window.
        random:
            Whether random window selection is enabled.
        max_hour:
            Number of hours preserved before the prediction time.
        strict_mode:
            Whether incomplete windows must be discarded.
        used_distribution:
            Distribution used for random window selection.
        mode:
            Main preprocessing mode, either ``"windows"`` or
            ``"resampling"``.
    """

    name: str
    hour_offset: int
    random: bool
    max_hour: int
    strict_mode: bool
    used_distribution: str
    mode: str


MODES = {
    "24h début réanimation sans remplissage": ConfigFenetrage(
        name="24h_debut_rea_no-fill",
        hour_offset=0,
        random=False,
        max_hour=6,
        strict_mode=True,
        used_distribution="uniform",  # Unused in this configuration.
        mode="windows",
    ),
    "24h fin réanimation sans remplissage": ConfigFenetrage(
        name="24h_fin_rea_no-fill",
        hour_offset=-1,
        max_hour=6,
        strict_mode=True,
        random=False,
        used_distribution="uniform",  # Unused in this configuration.
        mode="windows",
    ),
    "resampling aléatoire 'lomax' prio 24h sans remplissage": ConfigFenetrage(
        name="resampling_x_points_alea_lomax_prio_24h_no-fill",
        hour_offset=0,  # Unused in this configuration.
        max_hour=6,
        strict_mode=True,
        random=True,
        used_distribution="lomax",  # Currently unused by this mode.
        mode="resampling",
    ),
    "resampling aléatoire 'lomax' prio 50-50 sans remplissage": ConfigFenetrage(
        name="resampling_x_points_alea_lomax_prio_50-50_no-fill",
        hour_offset=0,  # Unused in this configuration.
        max_hour=6,
        strict_mode=True,
        random=True,
        used_distribution="lomax",  # Currently unused by this mode.
        mode="resampling",
    ),
    "24h aléatoire 'lomax' prio 24h sans remplissage": ConfigFenetrage(
        name="24h_alea_lomax_prio_24h_no-fill",
        hour_offset=0,  # Unused in this configuration.
        max_hour=6,
        strict_mode=True,
        random=False,
        used_distribution="lomax",  # Currently unused by this mode.
        mode="windows",
    ),
    "24h aléatoire 'lomax' prio 50-50 sans remplissage": ConfigFenetrage(
        name="24h_alea_lomax_prio_50-50_no-fill",
        hour_offset=0,  # Unused in this configuration.
        max_hour=6,
        strict_mode=True,
        random=False,
        used_distribution="lomax",  # Currently unused by this mode.
        mode="windows",
    ),
    "resampling X points": ConfigFenetrage(
        name="resampling_x_points",
        hour_offset=0,  # Unused in this configuration.
        max_hour=6,
        strict_mode=True,  # Currently unused by this mode.
        random=False,  # Currently unused by this mode.
        used_distribution="lomax",  # Currently unused by this mode.
        mode="resampling",
    ),
}


# ---------------------------------------------------------------------
# Population configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigPopulation:
    """Configuration defining the patient population to retain.

    Attributes:
        keep_population:
            Internal identifier of the population filter.
    """

    keep_population: str


POPULATION = {
    "Tout": ConfigPopulation(
        keep_population="all_diseases",
    ),
    "Sepsis": ConfigPopulation(
        keep_population="sepsis",
    ),
}


# ---------------------------------------------------------------------
# Clinical score configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigScores:
    """Configuration describing a clinical severity score.

    Attributes:
        models_name:
            Internal score or model name.
        extraction_type:
            Feature extraction strategy, if applicable.
        models_type:
            Type of model or evaluation workflow.
    """

    models_name: str
    extraction_type: str | None
    models_type: str


SCORE = {
    "IGS2": ConfigScores(
        models_name="IGS2",
        extraction_type=None,
        models_type="Normal",
    ),
    "NEWS2": ConfigScores(
        models_name="NEWS2",
        extraction_type=None,
        models_type="Normal",
    ),
}


# ---------------------------------------------------------------------
# Machine-learning model configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigModels:
    """Configuration describing a machine-learning model.

    Attributes:
        models_name:
            Internal model name.
        extraction_type:
            Type of input representation expected by the model.
        models_type:
            Model family or calibration status.
    """

    models_name: str
    extraction_type: str
    models_type: str


MODELS = {
    "InceptionTimeModified": ConfigModels(
        models_name="InceptionTimeModified",
        extraction_type="time",
        models_type="Normal",
    ),
    "LstmTimeModified": ConfigModels(
        models_name="LstmTimeModified",
        extraction_type="time",
        models_type="Normal",
    ),
    "VanillaTransformerModified": ConfigModels(
        models_name="VanillaTransformerModified",
        extraction_type="time",
        models_type="Normal",
    ),
    "RandomForest TSFEL": ConfigModels(
        models_name="RandomForest TSFEL",
        extraction_type="TSFEL",
        models_type="Normal",
    ),
    "XGBoost TSFEL": ConfigModels(
        models_name="XGBoost TSFEL",
        extraction_type="TSFEL",
        models_type="Normal",
    ),
    "SVC TSFEL": ConfigModels(
        models_name="SVC TSFEL",
        extraction_type="TSFEL",
        models_type="calibrated",
    ),
    "Logistic Regression Lasso TSFEL": ConfigModels(
        models_name="Logistic Regression Lasso TSFEL",
        extraction_type="TSFEL",
        models_type="calibrated",
    ),
}


# ---------------------------------------------------------------------
# Dataset-cleaning configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigCleaning:
    """Configuration controlling dataset population filtering.

    Attributes:
        clean:
            Whether continuous-monitoring units must be removed.
    """

    clean: bool


CLEAN = {
    "Enlever Surveillance Continue": ConfigCleaning(
        clean=True,
    ),
    "Garder le dataset intact": ConfigCleaning(
        clean=False,
    ),
}


# ---------------------------------------------------------------------
# Prediction-target configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigY:
    """Configuration defining the prediction target.

    Attributes:
        target_name:
            Name of the target column in the dataset.
    """

    target_name: str


Y = {
    "Survie à 24 heures": ConfigY(
        target_name="isDeceased_lt_24h",
    ),
    "Survie à 7 jours": ConfigY(
        target_name="isDeceased_lt_7d",
    ),
    "Survie à 28 jours": ConfigY(
        target_name="isDeceased_lt_28d",
    ),
    "Survie à 3 mois": ConfigY(
        target_name="isDeceased_lt_3m",
    ),
    "Survie (isDeceased)": ConfigY(
        target_name="isDeceased",
    ),
}


# ---------------------------------------------------------------------
# Feature-set configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigFeatures:
    """Configuration defining the feature subset used by the pipeline.

    Attributes:
        keep_feats:
            Explicit list of feature names or a predefined feature-selection
            mode identifier.
    """

    keep_feats: FeatureSelection


FEAT = {
    "Mode IGS2": ConfigFeatures(
        keep_feats=[
            "admission_type",
            "score_glasgow",
            "age",
            "pas",
            "heart_rate",
            "temp",
            "pao2",
            "fio2_corr",
            "urine_rate",
            "blood_urea",
            "leucocytes",
            "potassium",
            "sodium",
            "hco3",
            "bili_tot",
        ],
    ),
    "Mode NEWS": ConfigFeatures(
        keep_feats=[
            "fio2_corr",
            "fr",
            "spo2",
            "temp",
            "is_conscious",
            "pas",
            "heart_rate",
        ],
    ),
    "Mode NEWS2": ConfigFeatures(
        keep_feats=[
            "fio2_corr",
            "fr",
            "spo2",
            "temp",
            "is_conscious",
            "pas",
            "heart_rate",
            "hx_respi_chronique",
        ],
    ),
    "Mode Commonly Used Without pmsi": ConfigFeatures(
        keep_feats="mcuwp",
    ),
    "Mode Commonly Used": ConfigFeatures(
        keep_feats="mcu",
    ),
}


# ---------------------------------------------------------------------
# Class-balancing configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigEquilibre:
    """Configuration defining the class-balancing strategy.

    Attributes:
        balance_method:
            Internal identifier of the balancing method.
    """

    balance_method: str


BALANCE = {
    "Aucune Méthode": ConfigEquilibre(
        balance_method="",
    ),
    "DownSampling HomeMade": ConfigEquilibre(
        balance_method="downsampling_homemade",
    ),
    "DownSampling 50-50": ConfigEquilibre(
        balance_method="downsampling_50-50",
    ),
    "UpSampling 50-50": ConfigEquilibre(
        balance_method="upsampling_50-50",
    ),
    "Up/DownSampling 50-50": ConfigEquilibre(
        balance_method="updownsampling_50-50",
    ),
}