from pathlib import Path

import joblib


class Experiment:
    """Store an experiment configuration and generate its artifact paths.

    The class centralizes the naming and directory structure used for models,
    outputs, preprocessed inputs, selected variables, and resampled datasets.

    Directory and file names are derived from the experiment configuration to
    keep results from different targets, preprocessing modes, feature sets,
    populations, seeds, and calibration settings separated.
    """

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
        keep_duplicates = False,
        mode_duplicates = "prio_first"
    ):
        """Initialize an experiment configuration.

        Args:
            config_mode:
                Windowing or resampling configuration identifier.
            config_cleaning:
                Cleaning configuration object exposing a ``clean`` attribute.
            config_y:
                Target configuration object exposing a ``target_name``
                attribute.
            str_balance_method:
                Class-balancing method identifier.
            modex:
                Feature configuration object exposing a ``value`` attribute.
            class_weight_choice:
                Class-weight configuration object exposing a ``value``
                attribute.
            str_pop:
                Population selection identifier.
            seed:
                Random seed associated with the experiment.
            stratify_mode:
                Stratification strategy identifier.
            calibrated_mode:
                Calibration strategy identifier.
            keep_duplicates:
                Whether to retain patient duplicates across encounters.
            mode_duplicates:
                Strategy used when dropping duplicates (e.g., "prio_first" or "prio_last").
                Ignored if ``keep_duplicates`` is True.
        """
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
        self.mode_duplicates = mode_duplicates
        self.duplicates_folder = "duplicates" if keep_duplicates else f"no_duplicates_{mode_duplicates}"

    def shortdirname(self, class_weight=None, uses_optuna_config=False):
        """Build a compact directory name for the experiment.

        Args:
            class_weight:
                Optional class-weight value overriding the experiment default.
            uses_optuna_config:
                Whether the directory name must indicate Optuna optimization.

        Returns:
            A normalized experiment directory name.
        """
        return "_".join([
            self._normalize_component(self.target_name),
            self._get_stratify_slug(),
            self._normalize_component(self.config_mode),
            self._get_feature_slug(),
            self._get_run_slug(
                class_weight=class_weight,
                uses_optuna_config=uses_optuna_config,
            ),
        ])

    def _normalize_component(self, value, default="default"):
        """Normalize a value for safe use in a path component.

        Args:
            value:
                Value to normalize.
            default:
                Fallback value used when the input is empty or ``None``.

        Returns:
            A normalized string with spaces replaced by underscores and
            slashes replaced by hyphens.
        """
        if value is None:
            return default

        value = str(value).strip().strip("_")

        if not value:
            return default

        return value.replace(" ", "_").replace("/", "-")

    def _get_stratify_slug(self):
        """Return the normalized stratification identifier.

        Returns:
            The normalized stratification path component.
        """
        return self._normalize_component(self.stratify_mode)

    def _get_feature_slug(self):
        """Return the normalized feature-set identifier.

        Returns:
            The normalized feature-set path component.
        """
        return self._normalize_component(self.modex)

    def _get_cleaning_slug(self):
        """Return the normalized cleaning configuration identifier.

        Returns:
            A cleaning path component based on the ``clean`` attribute.

        Notes:
            This component is no longer included in generated paths because
            cleaning is always enabled in the current pipeline.
        """
        return f"clean_{self._normalize_component(self.clean)}"

    def _get_base_path(self, root, config_mode=""):
        """Build the common base directory for experiment artifacts.

        Args:
            root:
                Top-level artifact directory, such as ``"models"``,
                ``"outputs"``, or ``"inputs"``.
            config_mode:
                Optional preprocessing mode overriding the experiment default.

        Returns:
            The common experiment base path.

        Notes:
            The cleaning configuration is intentionally omitted because
            cleaning is always enabled.
        """
        mode = config_mode if config_mode != "" else self.config_mode

        return (
            Path(root)
            / self.duplicates_folder
            / self._normalize_component(self.target_name)
            / self._get_stratify_slug()
            / self._normalize_component(mode)
            / self._get_feature_slug()
        )

    def _get_run_slug(
        self,
        class_weight=None,
        uses_optuna_config=False,
        include_calibration=True,
    ):
        """Build the run-specific path component.

        Args:
            class_weight:
                Optional class-weight value overriding the experiment default.
            uses_optuna_config:
                Whether to include an Optuna marker.
            include_calibration:
                Whether to include the calibration mode.

        Returns:
            A normalized run identifier containing the applicable balancing,
            weighting, population, seed, calibration, and optimization
            components.
        """
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

        if uses_optuna_config:
            components.append("optuna")

        return "_".join(components)

    def _get_model_slug(
        self,
        class_weight=None,
        config_mode=None,
        uses_optuna_config=False,
    ):
        """Build a normalized model identifier.

        Args:
            class_weight:
                Optional class-weight value overriding the experiment default.
            config_mode:
                Optional preprocessing mode overriding the experiment default.
            uses_optuna_config:
                Whether to include an Optuna marker.

        Returns:
            A normalized identifier combining preprocessing mode, feature set,
            and run configuration.
        """
        mode = config_mode if config_mode is not None else self.config_mode

        return (
            f"{self._normalize_component(mode)}_"
            f"{self._get_feature_slug()}_"
            f"{self._get_run_slug(class_weight, uses_optuna_config)}"
        )

    def get_model_path(
        self,
        model_name,
        fold_idx,
        extension=".joblib",
        class_weight="",
        config_mode="",
        uses_optuna_config=False,
    ):
        """Return the path used to save a fold-specific model.

        The parent directory is created automatically when needed.

        Args:
            model_name:
                Model identifier.
            fold_idx:
                Fold index included in the filename.
            extension:
                Model file extension.
            class_weight:
                Optional class-weight override.
            config_mode:
                Optional preprocessing mode override.
            uses_optuna_config:
                Whether the model was optimized with Optuna.

        Returns:
            Path to the fold-specific model file.
        """
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("models", config_mode)
            / self._normalize_component(model_name)
            / self._get_run_slug(
                class_weight=weight,
                uses_optuna_config=uses_optuna_config,
            )
        )

        path.mkdir(parents=True, exist_ok=True)

        return path / f"fold_{fold_idx}{extension}"

    def get_output_path(
        self,
        model_name,
        class_weight="",
        config_mode="",
        uses_optuna_config=False,
    ):
        """Return the output directory for a model experiment.

        The directory is created automatically when needed.

        Args:
            model_name:
                Model identifier.
            class_weight:
                Optional class-weight override.
            config_mode:
                Optional preprocessing mode override.
            uses_optuna_config:
                Whether the model was optimized with Optuna.

        Returns:
            Path to the model output directory.
        """
        weight = class_weight if class_weight != "" else None

        path = (
            self._get_base_path("outputs", config_mode)
            / self._normalize_component(model_name)
            / self._get_run_slug(
                class_weight=weight,
                uses_optuna_config=uses_optuna_config,
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
        uses_optuna_config=False,
    ):
        """Load a serialized model result from the experiment output folder.

        Args:
            model_name:
                Model identifier.
            class_weight:
                Optional class-weight override.
            name_file:
                Name of the serialized file to load.
            config_mode:
                Optional preprocessing mode override.
            uses_optuna_config:
                Whether the model was optimized with Optuna.

        Returns:
            A tuple containing the model name and the deserialized object.

        Raises:
            FileNotFoundError:
                If the requested serialized file does not exist.
        """
        file_path = (
            self.get_output_path(
                model_name=model_name,
                class_weight=class_weight,
                config_mode=config_mode,
                uses_optuna_config=uses_optuna_config,
            )
            / name_file
        )

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        return model_name, joblib.load(file_path)

    def get_tsfel_parquet_path(self, config_mode=""):
        """Return the path of the global raw TSFEL feature dataset.

        The parent directory is created automatically when needed.

        Args:
            config_mode:
                Optional preprocessing mode override.

        Returns:
            Path to the global TSFEL Parquet file.
        """
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
        """Return the path of a fold-specific Boruta-filtered TSFEL dataset.

        The parent directory is created automatically when needed.

        Args:
            mode:
                Boruta or feature-selection mode identifier.
            fold_idx:
                Fold index included in the filename.
            config_mode:
                Optional preprocessing mode override.
            class_weight:
                Optional class-weight override.

        Returns:
            Path to the fold-specific Parquet file.
        """
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
        """Return the path of a fold-specific time-series input file.

        The parent directory is created automatically when needed.

        Args:
            mode:
                Time-series preprocessing mode identifier.
            fold_idx:
                Fold index included in the filename.
            config_mode:
                Optional preprocessing mode override.
            class_weight:
                Optional class-weight override.

        Returns:
            Path to the fold-specific NumPy file.
        """
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
        """Return the path of the retained-variable file.

        The parent directory is created automatically when needed.

        Args:
            config_mode:
                Optional preprocessing mode override.
            class_weight:
                Optional class-weight override.

        Returns:
            Path to the NumPy file containing retained variable names.
        """
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
        """Return the path of a fold-specific Lasso artifact.

        The parent directory is created automatically when needed.

        Args:
            mode:
                Lasso preprocessing or selection mode identifier.
            fold_idx:
                Fold index included in the filename.
            extension:
                File extension without a leading period.
            config_mode:
                Optional preprocessing mode override.
            class_weight:
                Optional class-weight override.

        Returns:
            Path to the fold-specific Lasso artifact.
        """
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
        """Return the path of a resampled patient dataset.

        The parent directory is created automatically when needed.

        Args:
            target_length:
                Number of time points used during resampling.
            class_weight:
                Optional class-weight override.
            config_mode:
                Optional preprocessing mode override.

        Returns:
            Path to the resampled Parquet file.
        """
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
    

    def get_comparison_path(
        self,
        comparison_name="Comparaison_All",
        config_mode="",
    ):
        """Return the directory path for multi-model comparisons within the experiment.

        The parent directory is created automatically when needed. This method centralizes 
        the storage of comparative reports to prevent cross-experiment data overwrites.

        Args:
            comparison_name (str): Identifier for the comparison group or report.
                Defaults to "Comparaison_All".
            config_mode (str): Optional preprocessing mode overriding the experiment 
                default configuration. Defaults to "".

        Returns:
            pathlib.Path: Path to the experiment-specific comparison directory.
        """
        path = self._get_base_path("outputs", config_mode) / self._normalize_component(comparison_name)
        path.mkdir(parents=True, exist_ok=True)
        return path