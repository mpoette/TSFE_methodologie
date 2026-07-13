import sys
import marimo as mo
import utilitaries.marimo_utils as mo_utils

def create_pipeline_widgets():
    """Crée et retourne tous les widgets de base sous forme de dictionnaire."""
    return {
        "transparent": mo.ui.dropdown(options={"Oui": True, "Non": False}, value="Non", label="Figures transparentes"),
        "cleaning": mo.ui.dropdown(options=mo_utils.CLEAN, value="Enlever Surveillance Continue", label="Nettoyage SC"),
        "mode": mo.ui.dropdown(options=mo_utils.MODES, value="24h début réanimation sans remplissage", label="Fenêtrage"),
        "type_donnees": mo.ui.dropdown(options={"modèle": "modèle", "score": "score"}, value="modèle", label="Type de données"),
        "modex": mo.ui.dropdown(options=list(mo_utils.FEAT.keys()), value="Mode IGS2", label="Features gardées"),
        "balance": mo.ui.dropdown(options=mo_utils.BALANCE, value="Aucune Méthode", label="Équilibrage"),
        "y_dd": mo.ui.dropdown(options=mo_utils.Y, value="Survie à 28 jours", label="Cible (y)"),
        "save_figure": mo.ui.dropdown(options={"Oui": True, "Non": False}, value="Oui", label="Sauver les figures"),
        "boruta_filter": mo.ui.dropdown(options={"Oui": True, "Non": False}, value="Oui", label="Filtre Boruta"),
        "keep_pop": mo.ui.dropdown(options=mo_utils.POPULATION, value="Tout", label="Filtre Population ICU_DP"),
        "use_optuna" : mo.ui.dropdown(options={"Oui" : True, "Non" : False}, value="Oui", label="Utiliser Optuna (si disponible)"),
    }

def get_model_dropdown(type_donnees_value):
    """Widget dynamique qui dépend du type de données choisi."""
    if type_donnees_value == "modèle":
        return mo.ui.dropdown(options=mo_utils.MODELS, value="XGBoost TSFEL", label="Modèle utilisé")
    else:
        return mo.ui.dropdown(options=mo_utils.SCORE, value="IGS2", label="Score utilisé")

def get_calibration_widgets(models_value, balance_value):
    """Widgets de calibration dépendants du modèle et de l'équilibrage."""
    
    # 1. Extraction des propriétés du modèle
    # On ajoute des fallbacks au cas où l'objet n'a pas ces attributs
    extraction_type = getattr(models_value, "extraction_type", "TSFEL")
    models_type = getattr(models_value, "models_type", "standard")
    
    # 2. Définition de la condition stricte pour le mode Prior
    # On gère le cas où balance vaut "" ou "Aucune Méthode"
    has_balancing = balance_value != ""
    is_prior_forced = has_balancing and extraction_type != "time"

    # 3. Construction dynamique des options et des valeurs par défaut
    if is_prior_forced:
        # CAS PRIOR : L'utilisateur n'a pas le choix, on verrouille tout
        options_calib = {"Oui (Forcé par le resampling)": True}
        bool_calib = "Oui (Forcé par le resampling)"
        
        options_mode = {"Prior": "prior"}
        calib_mode = "Prior"
        
    else:
        # AUTRES CAS : L'option Prior est totalement EXCLUE du dictionnaire
        options_calib = {"Oui": True, "Non": False}
        options_mode = {
            "Platt": "platt", 
            "Temperature Scaling": "temperature_scaling"
        }
        
        # Logique des valeurs par défaut classiques
        if models_type == "calibrated":
            bool_calib = "Non"
            calib_mode = "Platt"
        else:
            bool_calib = "Oui"
            # Sécurité : "Temperature Scaling" pour le temporel, "Platt" pour le reste
            calib_mode = "Temperature Scaling" if extraction_type == "time" else "Platt"
        
    # 4. Création des widgets Marimo avec les configurations injectées
    calibration = mo.ui.dropdown(
        options=options_calib, 
        value=bool_calib, 
        label="Activer calibration"
    )
    calibration_mode = mo.ui.dropdown(
        options=options_mode, 
        value=calib_mode, 
        label="Méthode calibration"
    )
    
    return calibration, calibration_mode

def get_tsfel_ui_components(extraction_type, models_type, boruta_filter, calibration, calibration_mode):
    """
    Génère et assemble les widgets spécifiques à l'extraction TSFEL.
    Retourne le conteneur UI, le widget d'extraction et le widget de poids des classes.
    """
    if extraction_type == "TSFEL":
        extract_tsfel = mo.ui.dropdown(
            options={"Oui": True, "Non": False},
            value="Non",
            label="Extraire les données TSFEL"
        )
        class_weight_choice = mo.ui.dropdown(
            options={"balanced": "balanced", "balanced_subsample": "balanced_subsample"},
            value="balanced",
            label="class_weight"
        )

        # Si le modèle est déjà calibré, on n'affiche pas les widgets de calibration
        if models_type == "calibrated":
            ui_tsfel = mo.vstack([extract_tsfel, boruta_filter, class_weight_choice])
        else:
            ui_tsfel = mo.vstack([
                extract_tsfel, 
                boruta_filter, 
                calibration, 
                calibration_mode, 
                class_weight_choice,
            ])
            
    else:
        # Fallback pour ne pas casser la pipeline si ce n'est pas du TSFEL
        extract_tsfel = None
        class_weight_choice = mo.ui.dropdown(options={"": ""}, value="")
        ui_tsfel = mo.md("")

    return ui_tsfel, extract_tsfel, class_weight_choice

def render_sidebar(uid, models, ui_tsfel, keep_feats, str_keep_feats, run, run_optuna, seed, custom_features):
    """S'occupe de l'affichage esthétique dans le notebook."""

    # 1. On définit les éléments communs du haut
    sidebar_items = [
        mo.md(mo_utils.config_sidebar),
        mo.md(f"<U>Seed utilisée pour l'ensemble du code : **{seed}**</U>"),
        mo.md(f"Version de python : {sys.version}"),
        mo.md("-----"),
        uid["type_donnees"],
        mo.md("-----"),
        uid["mode"],
        mo.md("-----"),
        models,
    ]

    # 2. On ajoute les éléments spécifiques selon la condition
    if uid["type_donnees"].value == "modèle":
        modex = uid["modex"]
        sidebar_items.extend([
            uid["balance"],
            ui_tsfel,
            uid["cleaning"],
            uid["y_dd"],
            uid["keep_pop"],
            uid["use_optuna"],
            modex,
            custom_features if modex.value == "Mode Custom" else mo.md("*(features fixe)*"),
            mo.md(f"**Features gardées :** `{keep_feats}`"),
            mo.md(f"**Soit en Français (dynamic feature only):** \n{str_keep_feats}"),
        ])
    else:
        sidebar_items.extend([
           uid["y_dd"],
           uid["keep_pop"],
        ])

    # 3. On ajoute les éléments communs du bas
    sidebar_items.extend([
        uid["save_figure"],
        uid["transparent"],
        mo.md("-----"),
    ])

    if  uid["type_donnees"].value == "modèle":
        sidebar_items.extend([
            run_optuna,
            mo.md("-----"),
            run,
        ])

    sidebar_items.extend([
        mo.md(mo_utils.config_end)
    ])
    return sidebar_items