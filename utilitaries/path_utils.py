from pathlib import Path

import joblib


class Experiment:
    def __init__(
        self,
        config_mode,
        config_cleaning,
        config_y,
        str_balance_method,
        modex,
        class_weight_choice,
        str_pop,
        seed,
        stratify_mode="",
        calibrated_mode="",
    ):
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
    

    def shortdirname(self, class_weight=None, config_optuna=False):
        return "_".join([
            self._normalize_component(self.target_name),
            self._get_stratify_slug(),
            self._normalize_component(self.config_mode),
            self._get_feature_slug(),
            self._get_run_slug(
                class_weight=class_weight,
                config_optuna=config_optuna,
            ),
        ])

    def _normalize_component(self, value, default="default"):
        if value is None:
            return default

        value = str(value).strip().strip("_")

        if not value:
            return default

        return value.replace(" ", "_").replace("/", "-")

    def _get_stratify_slug(self):
        return self._normalize_component(self.stratify_mode)

    def _get_feature_slug(self):
        return self._normalize_component(self.modex)

    def _get_cleaning_slug(self):
        return f"clean_{self._normalize_component(self.clean)}"

    def _get_base_path(self, root, config_mode=""):
        mode = config_mode if config_mode != "" else self.config_mode

        return (
            Path(root)
            / self._normalize_component(self.target_name)
            / self._get_stratify_slug()
            / self._normalize_component(mode)
            / self._get_cleaning_slug()
            / self._get_feature_slug()
        )

    def _get_run_slug(
        self,
        class_weight=None,
        config_optuna=False,
        include_calibration=True,
    ):
        weight = self.class_weight if class_weight is None else class_weight
        components = []

        if (
            self.str_balance_method
            and self.str_balance_method != "Aucune Méthode"
        ):
            components.append(
                self._normalize_component(self.str_balance_method)
            )

        if weight:
            components.append(self._normalize_component(weight))

        if self.str_pop and self.str_pop != "Tout":
            components.append(self._normalize_component(self.str_pop))

        components.append(f"seed_{self.seed}")

        if include_calibration and self.calibrated_mode:
            components.append(
                self._normalize_component(self.calibrated_mode)
            )

        if config_optuna:
            components.append("optuna")

        return "_".join(components)

    def _get_model_slug(
        self,
        class_weight=None,
        config_mode=None,
        config_optuna=False,
    ):
        mode = config_mode if config_mode is not None else self.config_mode

        return (
            f"{self._normalize_component(mode)}_"
            f"{self._get_feature_slug()}_"
            f"{self._get_run_slug(class_weight, config_optuna)}"
        )

    def get_model_path(
        self,
        model_name,
        fold_idx,
        extension=".joblib",
        class_weight="",
        config_mode="",
        config_optuna=False,
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("models", config_mode)
            / self._normalize_component(model_name)
            / self._get_run_slug(
                class_weight=weight,
                config_optuna=config_optuna,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"fold_{fold_idx}{extension}"

    def get_output_path(
        self,
        model_name,
        class_weight="",
        config_mode="",
        config_optuna=False,
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("outputs", config_mode)
            / self._normalize_component(model_name)
            / self._get_run_slug(
                class_weight=weight,
                config_optuna=config_optuna,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    def load_model(
        self,
        model_name,
        class_weight="",
        name_file="all_res.joblib",
        config_mode="",
        config_optuna=False,
    ):
        file_path = (
            self.get_output_path(
                model_name=model_name,
                class_weight=class_weight,
                config_mode=config_mode,
                config_optuna=config_optuna,
            )
            / name_file
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        return model_name, joblib.load(file_path)

    def get_tsfel_parquet_path(self, config_mode=""):
        path = self._get_base_path("inputs", config_mode) / "tsfel_global"
        path.mkdir(parents=True, exist_ok=True)

        return path / "global_brut.parquet"

    def get_tsfel_boruta(
        self,
        mode,
        fold_idx,
        config_mode="",
        class_weight="",
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("inputs", config_mode)
            / "tsfel_boruta"
            / self._normalize_component(mode)
            / self._get_run_slug(
                class_weight=weight,
                include_calibration=False,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"fold_{fold_idx}.parquet"

    def get_time_path(
        self,
        mode,
        fold_idx,
        config_mode="",
        class_weight="",
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("inputs", config_mode)
            / "time"
            / self._normalize_component(mode)
            / self._get_run_slug(
                class_weight=weight,
                include_calibration=False,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"fold_{fold_idx}.npy"

    def get_var_path(
        self,
        config_mode="",
        class_weight="",
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("inputs", config_mode)
            / "variables"
            / self._get_run_slug(
                class_weight=weight,
                include_calibration=False,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / "keep_variables.npy"

    def get_lasso_path(
        self,
        mode,
        fold_idx,
        extension,
        config_mode="",
        class_weight="",
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("inputs", config_mode)
            / "lasso"
            / self._normalize_component(mode)
            / self._get_run_slug(
                class_weight=weight,
                include_calibration=False,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"fold_{fold_idx}.{extension}"

    def get_resampling_path(
        self,
        target_length,
        class_weight="",
        config_mode="",
    ):
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("inputs", config_mode)
            / "resampling"
            / self._get_run_slug(
                class_weight=weight,
                include_calibration=False,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"length_{target_length}.parquet"