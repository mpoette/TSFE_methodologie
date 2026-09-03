"""Validated JSON and terminal configuration for pipeline runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "icu_model_comparison"
    / "configs"
    / "run.default.json"
)


@dataclass(frozen=True)
class ModeConfig:
    """Define the experiment matrix for one run mode.

    Args:
        windowing_mode: Windowing strategy label consumed by the pipeline.
        target_labels: Target labels evaluated by this mode.
        model_names: Models evaluated by this mode.
        stratify_modes: Cross-validation stratification strategies.
        optuna_run_options: Whether each experiment uses Optuna.
        feature_modes: Feature configurations evaluated by this mode.
        balancing_methods: Dataset-balancing configurations.
        mode_duplicates: Duplicate-resolution strategy.
    """

    windowing_mode: str
    target_labels: tuple[str, ...]
    model_names: tuple[str, ...]
    stratify_modes: tuple[str, ...]
    optuna_run_options: tuple[bool, ...]
    feature_modes: tuple[str, ...]
    balancing_methods: tuple[str, ...]
    mode_duplicates: str = "prio_first"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModeConfig":
        """Build a mode configuration from decoded JSON data.

        Args:
            data: Mapping containing one mode definition.

        Returns:
            The validated immutable mode configuration.
        """
        list_fields = (
            "target_labels",
            "model_names",
            "stratify_modes",
            "optuna_run_options",
            "feature_modes",
            "balancing_methods",
        )
        missing = [name for name in ("windowing_mode", *list_fields) if name not in data]
        if missing:
            raise ValueError(f"Mode configuration is missing: {', '.join(missing)}")
        if any(
            not isinstance(data[name], (list, tuple)) or not data[name]
            for name in list_fields
        ):
            raise ValueError("Every mode matrix field must be a non-empty sequence.")
        return cls(
            windowing_mode=str(data["windowing_mode"]),
            target_labels=tuple(str(value) for value in data["target_labels"]),
            model_names=tuple(str(value) for value in data["model_names"]),
            stratify_modes=tuple(str(value) for value in data["stratify_modes"]),
            optuna_run_options=tuple(bool(value) for value in data["optuna_run_options"]),
            feature_modes=tuple(str(value) for value in data["feature_modes"]),
            balancing_methods=tuple(str(value) for value in data["balancing_methods"]),
            mode_duplicates=str(data.get("mode_duplicates", "prio_first")),
        )


@dataclass(frozen=True)
class RunConfig:
    """Contain all user-selectable pipeline launch parameters."""

    seed: int
    modes: tuple[str, ...]
    run_comparison: bool
    run_training: bool
    run_lasso: bool
    run_interpretability: bool
    run_test: bool
    figure_output_format: str
    population: str
    save_figure: bool
    transparent: bool
    dataset_path: str
    target_columns: dict[str, str]
    keep_duplicates: bool
    profiles: dict[str, ModeConfig]
    reference : bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Build and validate a run configuration.

        Args:
            data: Decoded JSON configuration.

        Returns:
            The validated immutable run configuration.

        Raises:
            ValueError: If a required field or referenced mode is invalid.
        """
        required = {
            "seed", "modes", "run_comparison", "run_training", "run_lasso",
            "run_interpretability", "run_test", "figure_output_format",
            "population", "save_figure", "transparent", "dataset_path",
            "target_columns", "keep_duplicates", "profiles",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Run configuration is missing: {', '.join(missing)}")

        profiles = {
            str(name): ModeConfig.from_dict(profile)
            for name, profile in data["profiles"].items()
        }
        modes = tuple(str(mode) for mode in data["modes"])
        unknown_modes = sorted(set(modes).difference(profiles))
        if not modes:
            raise ValueError("At least one run mode must be selected.")
        if unknown_modes:
            raise ValueError(f"Unknown run mode(s): {', '.join(unknown_modes)}")

        figure_format = str(data["figure_output_format"]).lower().lstrip(".")
        if figure_format not in {"pdf", "png"}:
            raise ValueError("figure_output_format must be either 'pdf' or 'png'.")

        return cls(
            seed=int(data["seed"]),
            modes=modes,
            run_comparison=bool(data["run_comparison"]),
            run_training=bool(data["run_training"]),
            run_lasso=bool(data["run_lasso"]),
            run_interpretability=bool(data["run_interpretability"]),
            run_test=bool(data["run_test"]),
            figure_output_format=figure_format,
            population=str(data["population"]),
            save_figure=bool(data["save_figure"]),
            transparent=bool(data["transparent"]),
            dataset_path=str(data["dataset_path"]),
            target_columns={str(key): str(value) for key, value in data["target_columns"].items()},
            keep_duplicates=bool(data["keep_duplicates"]),
            profiles=profiles,
            reference=bool(data.get("reference", False)),
        )

    def experiment_count(self) -> int:
        """Return the number of experiments represented by the matrix.

        Returns:
            Number of experiments scheduled for the selected modes.
        """
        total = 0
        for mode in self.modes:
            profile = self.profiles[mode]
            fixed_dimensions = (
                len(profile.target_labels)
                * len(profile.stratify_modes)
                * len(profile.feature_modes)
                * len(profile.balancing_methods)
            )
            optuna_variants = sum(
                1
                if self.reference or model == "Logistic Regression Lasso TSFEL"
                else len(profile.optuna_run_options)
                for model in profile.model_names
            )
            total += fixed_dimensions * optuna_variants
        return total

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Configuration mapping suitable for logging or serialization.
        """
        return asdict(self)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run configuration from JSON.

    Args:
        path: JSON configuration path.

    Returns:
        The validated run configuration.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return RunConfig.from_dict(json.load(handle))


def _prompt_boolean(label: str, current: bool) -> bool:
    """Prompt for a Boolean value while preserving an empty default.

    Args:
        label: Human-readable setting name.
        current: Current Boolean value.

    Returns:
        The selected Boolean value.
    """
    suffix = "Y/n" if current else "y/N"
    answer = input(f"{label} [{suffix}]: ").strip().lower()
    if not answer:
        return current
    if answer in {"y", "yes", "1", "true"}:
        return True
    if answer in {"n", "no", "0", "false"}:
        return False
    print("Invalid answer; keeping the current value.")
    return current


def _apply_interactive_overrides(config: RunConfig) -> RunConfig:
    """Collect common run overrides from the terminal.

    Args:
        config: Configuration loaded from JSON and CLI overrides.

    Returns:
        A new configuration containing the interactive selections.
    """
    data = config.to_dict()
    available = list(config.profiles)
    print("Available modes: " + ", ".join(available))
    selected = input(
        "Modes separated by commas "
        f"[{', '.join(config.modes)}]: "
    ).strip()
    if selected:
        data["modes"] = [value.strip() for value in selected.split(",") if value.strip()]

    for field, label in (
        ("run_training", "Run training"),
        ("run_test", "Run holdout testing"),
        ("run_lasso", "Generate Lasso paths"),
        ("run_interpretability", "Run explainability"),
        ("run_comparison", "Generate comparison reports"),
        ("save_figure", "Save figures"),
    ):
        data[field] = _prompt_boolean(label, bool(data[field]))
    return RunConfig.from_dict(data)


def load_run_config_from_cli(argv: Sequence[str] | None = None) -> RunConfig:
    """Load JSON configuration and apply command-line or terminal overrides.

    Args:
        argv: Optional argument sequence. Defaults to ``sys.argv``.

    Returns:
        The final validated run configuration.
    """
    parser = argparse.ArgumentParser(description="Run the model-comparison pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--figure-format", choices=("pdf", "png"))
    parser.add_argument(
        "--reference",
        action="store_true",
        help=(
            "Load ref_hyperparameters.json from each experiment directory "
            "without running Optuna."
        ),
    )
    for flag in (
        "run-training", "run-test", "run-lasso", "run-interpretability",
        "run-comparison", "save-figure",
    ):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    data = config.to_dict()
    scalar_overrides = {
        "modes": args.modes,
        "seed": args.seed,
        "dataset_path": args.dataset_path,
        "figure_output_format": args.figure_format,
        "run_training": args.run_training,
        "run_test": args.run_test,
        "run_lasso": args.run_lasso,
        "run_interpretability": args.run_interpretability,
        "run_comparison": args.run_comparison,
        "save_figure": args.save_figure,
        "reference" : args.reference,
    }
    data.update({key: value for key, value in scalar_overrides.items() if value is not None})
    config = RunConfig.from_dict(data)
    if args.interactive:
        config = _apply_interactive_overrides(config)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    return config
