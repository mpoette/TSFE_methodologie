from pathlib import Path
import joblib

class Experiment:
    def __init__(self, config_mode, config_cleaning, config_y, str_balance_method, modex, class_weight_choice, str_pop, seed):
        self.clean = config_cleaning.clean
        self.target_name = config_y.target_name
        self.str_balance_method = str_balance_method
        self.modex = modex.value
        self.class_weight = class_weight_choice.value
        self.str_pop = str_pop
        self.seed = seed
        self.config_mode = config_mode.name

    @property
    def dirname(self):
        """Génère le nom du dossier dynamiquement via self."""
        return (
            f"Clean_{self.clean}_{self.target_name}_"
            f"{self.str_balance_method}{self.modex}"
            f"{self.class_weight}{self.str_pop}_seed_{self.seed}"
        )
    
    def shortdirname(self, class_weight = ""):
        return (
            f"Clean_{self.clean}_{self.target_name}_"
            f"{self.str_balance_method}{self.modex}"
            f"{class_weight}{self.str_pop}_seed_{self.seed}"
        )

    def get_model_path(self, model_name, fold_idx, extension=".joblib", class_weight = "",):
        """Retourne le chemin d'un fold spécifique et crée le dossier parent s'il manque."""
        path = Path("models") / model_name / self.shortdirname(class_weight)
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"fold_{fold_idx}{extension}"
    
    def get_output_path(self, model_name, class_weight = ""):
        path = Path("outputs") / model_name / self.shortdirname(class_weight)
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path
    
    def load_model(self, model_name, class_weight = "", name_file = "all_res.joblib"):
        file_path = self.get_output_path(model_name, class_weight) / name_file
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        return model_name, joblib.load(file_path)

    def get_tsfel_parquet_path(self):
        path = Path("inputs") 
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"tsfel_global_brut_{self.config_mode}_{self.clean}_{self.target_name}_{self.modex}.parquet"
    
    def get_tsfel_boruta(self, mode, fold_idx):
        path = Path("inputs") / f"tsfel_{mode}_{self.shortdirname()}"
        path.mkdir(parents=True, exist_ok=True) # Sécurité création de dossier
        return path / f"fold_{fold_idx}.parquet"
    
    def get_lasso_path(self, mode, fold_idx, extension):
        path = Path("inputs") / f"lasso_{mode}_{self.shortdirname()}"
        path.mkdir(parents = True, exist_ok = True)
        return path / f"fold_{fold_idx}.{extension}"