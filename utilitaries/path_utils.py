from pathlib import Path
import joblib
import numpy as np

class Experiment:
    def __init__(self, config_mode, config_cleaning, config_y, str_balance_method, 
                 modex, class_weight_choice, str_pop, seed, 
                 stratify_mode="", calibrated_mode=""):
        
        self.config_mode = config_mode
        self.clean = config_cleaning.clean
        self.target_name = config_y.target_name
        self.str_balance_method = str_balance_method
        self.modex = modex.value
        self.class_weight = class_weight_choice.value
        self.str_pop = str_pop
        self.seed = seed
        self.stratify_mode = stratify_mode
        self.calibrated_mode = calibrated_mode

    # ─── COMPOSANTS DE NOMMAGE MODULAIRES ───

    def _get_data_slug(self, config_mode=None):
        """Identifiant unique lié uniquement aux données de base et au nettoyage."""
        cfg_mode = config_mode if config_mode is not None else self.config_mode
        return f"{cfg_mode}_clean_{self.clean}_{self.target_name}"

    def _get_feature_slug(self, config_mode=None):
        """Identifiant lié à la stratégie de features (Modex, Population, Split)."""
        return f"{self._get_data_slug(config_mode)}_{self.modex}{self.str_pop}_seed_{self.seed}{self.stratify_mode}"

    def _get_model_slug(self, class_weight=None, config_mode=None):
        """Identifiant complet incluant la modélisation (équilibrage, poids, calibration)."""
        w = class_weight if class_weight is not None else self.class_weight
        return (
            f"{self._get_data_slug(config_mode)}_"
            f"{self.str_balance_method}{self.modex}"
            f"{w}{self.str_pop}_seed_{self.seed}"
            f"{self.stratify_mode}{self.calibrated_mode}"
        )

    # ─── ANCIENS NOMS (Pour rétrocompatibilité si besoin) ───

    @property
    def dirname(self):
        return self._get_model_slug()
    
    def shortdirname(self, class_weight="", config_mode=""):
        # Si une chaîne vide est passée, on force à None pour utiliser la valeur par défaut de self
        w = class_weight if class_weight != "" else None
        cfg = config_mode if config_mode != "" else None
        return self._get_model_slug(class_weight=w, config_mode=cfg)

    # ─── ACCÈS AUX CHEMINS (INPUTS / OUTPUTS / MODELS) ───

    def get_model_path(self, model_name, fold_idx, extension=".joblib", class_weight="", config_mode=""):
        path = Path("models") / model_name / self.shortdirname(class_weight, config_mode)
        path.mkdir(parents=True, exist_ok=True)
        return path / f"fold_{fold_idx}{extension}"
    
    def get_output_path(self, model_name, class_weight="", config_mode=""):
        path = Path("outputs") / model_name / self.shortdirname(class_weight, config_mode)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def load_model(self, model_name, class_weight="", name_file = "all_res.joblib", config_mode=""):
        file_path = self.get_output_path(model_name, class_weight, config_mode) / name_file
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        return model_name, joblib.load(file_path)

    # ─── DOSSIER INPUTS : Découplé des paramètres de modélisation ! ───

    def get_tsfel_parquet_path(self):
        path = Path("inputs") 
        path.mkdir(parents=True, exist_ok=True)
        return path / f"tsfel_global_brut_{self.config_mode}_{self.clean}_{self.target_name}_{self.modex}.parquet"
    
    def get_tsfel_boruta(self, mode, fold_idx, config_mode=""):
        cfg = config_mode if config_mode != "" else None
        # Boruta ne dépend que des features et de la pop, pas de la calibration ni du rééquilibrage !
        path = Path("inputs") / f"tsfel_{mode}_{self._get_feature_slug(config_mode=cfg)}"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"fold_{fold_idx}.parquet"
    
    def get_time_path(self, mode, fold_idx, config_mode=""):
        cfg = config_mode if config_mode != "" else None
        path = Path("inputs") / f"time_{mode}_{self._get_feature_slug(config_mode=cfg)}"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"fold_{fold_idx}.npy"
    
    def get_var_path(self, config_mode=""):
        cfg = config_mode if config_mode != "" else None
        path = Path("inputs")
        path.mkdir(parents=True, exist_ok=True)
        return path / f"keepVarTime_{self._get_feature_slug(config_mode=cfg)}.npy"    

    def get_lasso_path(self, mode, fold_idx, extension, config_mode=""):
        cfg = config_mode if config_mode != "" else None
        path = Path("inputs") / f"lasso_{mode}_{self._get_feature_slug(config_mode=cfg)}"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"fold_{fold_idx}.{extension}"

    def get_resampling_path(self, target_length, class_weight=""):
        path = Path("inputs")
        path.mkdir(parents=True, exist_ok=True)
        # Utilise le slug minimal de la donnée de base
        return path / f"resampling_{target_length}_{self._get_data_slug()}.parquet"