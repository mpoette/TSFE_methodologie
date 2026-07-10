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
    mode : str

MODES = {

    "24h début réanimation sans remplissage": ConfigFenetrage(
        name = "24h_debut_rea_no-fill",
        hour_offset = 0,
        random = False,
        max_hour = 6,
        strict_mode = True,
        used_distribution = "uniform", # pas utilisé
        mode = "windows"
    ),
     "24h fin réanimation sans remplissage" : ConfigFenetrage(
        name = "24h_fin_rea_no-fill",
        hour_offset = -1,
        max_hour = 6,
        strict_mode = True,
        random = False,
        used_distribution = "uniform", # pas utilisé
        mode = "windows"
     ),
     "resampling aléatoire 'lomax' prio 24h sans remplissage" : ConfigFenetrage(
         name = "resampling_x_points_alea_lomax_prio_24h_no-fill",
         hour_offset = 0, # pas utilsé
         max_hour = 6,
         strict_mode = True,
         random = True, 
         used_distribution = "lomax", # pas utilsé
         mode = "resampling"
     ),
     "resampling aléatoire 'lomax' prio 50-50 sans remplissage" : ConfigFenetrage(
         name = "resampling_x_points_alea_lomax_prio50-50_no-fill",
         hour_offset = 0, # pas utilsé
         max_hour = 6,
         strict_mode = True,
         random = True, 
         used_distribution = "lomax", # pas utilsé
         mode = "resampling"
     ),
     "24h aléatoire 'lomax' prio 24h sans remplissage" : ConfigFenetrage(
        name = "24h_alea_lomax_prio24h_no-fill",
        hour_offset = 0, # pas utilisé
        max_hour = 6,
        strict_mode = True,
        random = False,
        used_distribution = "lomax", # pas utilisé
        mode = "windows"
     ),
     "24h aléatoire 'lomax' prio 50-50 sans remplissage" : ConfigFenetrage(
        name = "24h_alea_lomax_prio50-50_no-fill",
        hour_offset = 0, # pas utilisé
        max_hour = 6,
        strict_mode = True,
        random = False,
        used_distribution = "lomax", # pas utilisé
        mode = "windows"
     ),
     "resampling X points" : ConfigFenetrage(
        name = "resampling_x_points",
        hour_offset = 0, # pas utilisé
        max_hour = 6,
        strict_mode = True, # pas utilisé
        random = False, # pas utilisé
        used_distribution = "lomax", # pas utilisé
        mode = "resampling"
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
class ConfigScores:
    models_name : str
    extraction_type : str
    models_type : str
SCORE = {
    "IGS2" : ConfigScores(
        models_name = "IGS2",
        extraction_type = None,
        models_type = "Normal"
    ),
    "NEWS2" : ConfigScores(
        models_name = "NEWS2",
        extraction_type = None,
        models_type = "Normal"
    ),
}

@dataclass(frozen=True)
class ConfigModels:
    models_name : str
    extraction_type : str
    models_type : str
MODELS = {
    # "InceptionTimeModified" : ConfigModels(
    #     models_name = "InceptionTimeModified",
    #     extraction_type = "time",
    #     models_type = "Normal"
    # ),
    "LstmTimeModified" : ConfigModels(
        models_name = "LstmTimeModified",
        extraction_type = "time",
        models_type = "Normal"
    ),
    "RandomForest TSFEL" : ConfigModels(
        models_name = "RandomForest TSFEL",
        extraction_type = "TSFEL",
        models_type = "Normal"
    ),
    "XGBoost TSFEL" : ConfigModels(
        models_name = "XGBoost TSFEL",
        extraction_type = "TSFEL",
        models_type = "Normal"
    ),
    "SVC TSFEL" : ConfigModels(
        models_name = "SVC TSFEL",
        extraction_type = "TSFEL",
        models_type = "calibrated"
    ),
    "Logistic Regression Lasso TSFEL" : ConfigModels(
        models_name = "Logistic Regression Lasso TSFEL",
        extraction_type = "TSFEL",
        models_type = "calibrated"
    ),
    "RandomForest Imbalanced TSFEL":ConfigModels(
        models_name = "RandomForest Imbalanced TSFEL",
        extraction_type = "TSFEL",
        models_type = "imbalanced"
    ),
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
    "Mode gabrielle" : ConfigFeatures(
        keep_feats = ['heure_calibree', 'pam', 'pad', 'heart_rate', 'spo2', 'temp', 'fio2_corr', 'glyc_cap', 'nad_dose_poids', 'is_ventilated', 'is_conscious', 'is_sedated', 'is_not_alert', 'age', 'creat', 'num_plq', 'bili_tot', 'tp', 'abs_dialyse', 'dialyse_hdi', 'dialyse_cvvhf']
    ),
    "Mode IGS2" : ConfigFeatures(
        keep_feats = ["admission_type", "score_glasgow", "age", "pas", "heart_rate", "temp", "pao2", "fio2_corr", "urine_rate", "blood_urea", "leucocytes", "potassium","sodium",  "hco3", "bili_tot"]
    ),
    "Mode NEWS" : ConfigFeatures(
        keep_feats = ["fio2_corr", "fr", "spo2", "temp", "is_conscious", "pas", "heart_rate"]
    ),
    "Mode NEWS2" : ConfigFeatures(
        keep_feats = ["fio2_corr", "fr", "spo2", "temp", "is_conscious", "pas", "heart_rate", "hx_respi_chronique"]
    ),
    "Mode Custom" : ConfigFeatures(
        keep_feats= "custom"
    ),
    "Mode All Without pmsi" : ConfigFeatures(
        keep_feats= "all_wp"
    ),
    "Mode All" : ConfigFeatures(
        keep_feats = "all"
    ),
    "Mode Commonly Used Without pmsi" : ConfigFeatures(
        keep_feats = "mcuwp"
    ),
    "Mode Commonly Used": ConfigFeatures(
        keep_feats = "mcu"
    )
}


@dataclass(frozen=True)
class ConfigEquilibre:
    balance_method : str
BALANCE = {
    "Aucune Méthode" : ConfigEquilibre(
        balance_method = ""
    ),
    "DownSampling HomeMade" : ConfigEquilibre(
        balance_method = "downsampling_homemade"
    ),
    "DownSampling 50-50": ConfigEquilibre(
        balance_method = "downsampling_50-50"
    ),
    "UpSampling 50-50": ConfigEquilibre(
        balance_method = "upsampling_50-50"
    ),
    "Up/DownSampling 50-50" : ConfigEquilibre(
        balance_method = "updownsampling_50-50"
    )
}