import marimo as mo
from dataclasses import dataclass
# Configurations globales des widgets marimo
config_dropdown_color = "<div style='background:#F0FFD4;padding:8px;border-radius:6px'>"

config_run_button = "<div style='background:#FFFA7A;padding:8px;border-radius:6px'>"

config_msg = "<div style='background:#FFABAB;padding:8px;border-radius:6px>"

config_sidebar = "<div style='background:#F9F9F9;padding:8px;border-radius:6px'>"

config_end = "</div>"
# Configuration des dropdowns marimo

@dataclass(frozen=True)
class ConfigFenetrage:
    name : str
    hour_offset: int
    random: bool
    max_hour: int
    strict_mode: bool
    used_distribution : str

MODES = {
    "24h début réanimation avec remplissage": ConfigFenetrage(
        name = "24h_debut_rea_fill",
        hour_offset = 0,
        random = False,
        max_hour = 12,
        strict_mode = False,
        used_distribution = "uniform",
    ),
    "24h début réanimation sans remplissage": ConfigFenetrage(
        name = "24h_debut_rea_no-fill",
        hour_offset = 0,
        random = False,
        max_hour = 12,
        strict_mode = True,
        used_distribution = "uniform", # pas utilisé
    ),
    "24h fin réanimation avec remplissage" : ConfigFenetrage(
        name = "24h_fin_rea-fill",
        hour_offset = -1,
        max_hour = 12,
        strict_mode = False,
        random = False,
        used_distribution = "uniform", # pas utilisé
    ),
     "24h fin réanimation sans remplissage" : ConfigFenetrage(
        name = "24h_fin_rea_no-fill",
        hour_offset = -1,
        max_hour = 12,
        strict_mode = True,
        random = False,
        used_distribution = "uniform", # pas utilisé
     ),
    "24h aléatoire 'real' avec remplissage": ConfigFenetrage(
        name = "24h_alea_real_fill",
        hour_offset = 0, # pas utilisé en pratique
        random = True,
        max_hour = 12,
        strict_mode = False,
        used_distribution = "real",
    ),
    "24h aléatoire 'real' sans remplissage": ConfigFenetrage(
        name = "24h_alea_real_no-fill",
        hour_offset = 0,
        random = True,
        max_hour = 12,
        strict_mode = True,
        used_distribution = "real",
    ),
}

@dataclass(frozen=True)
class ConfigPopulation:
    keep_population : str
POPULATION = {
    "Tout" : ConfigPopulation(
        keep_population = "all_diseases"
    ),
    "Sepsis" : ConfigPopulation(
        keep_population = "sepsis"
    ),
}

@dataclass(frozen=True)
class ConfigModels:
    models_name : str
MODELS = {
    "InceptionTimeModified" : ConfigModels(
        models_name = "InceptionTimeModified"
    )
}

@dataclass(frozen=True)
class ConfigCleaning:
    clean : bool
CLEAN = {
    "Enlever Surveillance Continue" : ConfigCleaning(
        clean = True
    ),
    "Garder le dataset intact" : ConfigCleaning(
        clean = False
    ),
}

@dataclass(frozen=True)
class ConfigY:
    target_name : str
Y = {
    "Survie à 24 heures" : ConfigY(
        target_name = "isDeceased_lt_24h"
    ),
    "Survie à 7 jours" : ConfigY(
        target_name = "isDeceased_lt_7d"
    ),
    "Survie à 28 jours" : ConfigY(
        target_name = "isDeceased_lt_28d"
    ),
    "Survie à 3 mois" : ConfigY(
        target_name = "isDeceased_lt_3m"
    ),
    "Survie (isDeceased)" : ConfigY(
        target_name = "isDeceased")

}

@dataclass(frozen=True)
class ConfigFeatures:
    keep_feats : list
FEAT = {
    "Mode classique" : ConfigFeatures(
        keep_feats = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf']
    ),
    "Mode NEWS" : ConfigFeatures(
        keep_feats = ["fio2_corr", "fr", "spo2", "temp", "is_conscious", "pas", "heart_rate"]
    ),
    "Mode Custom" : ConfigFeatures(
        keep_feats= ['heure_calibree', 'pam', 'pad', 'heart_rate']
    )
}