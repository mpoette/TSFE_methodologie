"""Tests for the lightweight public package API."""

from icu_model_comparison import DEFAULT_CONFIG_PATH, __version__, load_run_config
from icu_model_comparison.figures.feature_names import short_feature_name


def test_default_configuration_is_packaged_and_valid() -> None:
    """Load the packaged default configuration without scientific imports."""
    config = load_run_config(DEFAULT_CONFIG_PATH)

    assert __version__ == "0.1.0"
    assert config.modes == ("LR",)
    assert config.experiment_count() == 1


def test_short_feature_names_preserve_internal_suffix_meaning() -> None:
    """Map exact and TSFEL-derived names to publication labels."""
    assert short_feature_name("heart_rate") == "HR"
    assert short_feature_name("heart_rate_Mean") == "HR - Mean"
    assert short_feature_name("unknown_feature") == "unknown_feature"
