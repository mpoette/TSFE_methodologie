"""Tests for the per-run artifact directory scoping (RunContext)."""

from types import SimpleNamespace

from utilitaries.path_utils import Experiment, RunContext


def _make_exp(run_context=None):
    """Build a minimal experiment configuration for path checks."""
    config_cleaning = SimpleNamespace(clean="Enlever Surveillance Continue")
    config_y = SimpleNamespace(target_name="isDeceased_lt_28d")
    modex = SimpleNamespace(value="Mode Commonly Used")
    class_weight_choice = SimpleNamespace(value="")
    return Experiment(
        "24h debut",
        config_cleaning,
        config_y,
        "Aucune Methode",
        modex,
        class_weight_choice,
        "Tout",
        42,
        stratify_mode="target_col",
        calibrated_mode="",
        keep_duplicates=True,
        mode_duplicates="prio_first",
        run_context=run_context,
    )


def test_run_context_creates_unique_directories(tmp_path):
    """Two consecutive runs get distinct, predictably named folders."""
    run_a = RunContext.new(base_dir=tmp_path, suffix="run_mimic")
    run_b = RunContext.new(base_dir=tmp_path, suffix="run_mimic")

    assert run_a.run_dir.parent == tmp_path
    assert run_b.run_dir.parent == tmp_path
    assert run_a.run_dir.name != run_b.run_dir.name
    assert run_a.run_dir.name.startswith("run_")
    assert run_a.run_dir.name.endswith("_run_mimic")
    assert run_a.run_name == run_a.run_dir.name
    assert (run_a.run_dir / "optuna.db").is_relative_to(run_a.run_dir)


def test_run_context_properties(tmp_path):
    """Artifact roots are resolved inside the run directory."""
    run = RunContext.new(base_dir=tmp_path)

    assert run.models_dir == run.run_dir / "models"
    assert run.outputs_dir == run.run_dir / "outputs"
    assert run.inputs_dir == run.run_dir / "inputs"
    assert run.log_dir == run.run_dir / "pipeline_logs"
    assert run.optuna_db == run.run_dir / "optuna.db"


def test_experiment_paths_scoped_to_run_context(tmp_path):
    """Every artifact path lands inside the run directory."""
    run = RunContext.new(base_dir=tmp_path)
    exp = _make_exp(run_context=run)

    model_path = exp.get_model_path("XGBoost TSFEL", 0)
    output_path = exp.get_output_path("XGBoost TSFEL")
    var_path = exp.get_var_path()
    parquet_path = exp.get_tsfel_parquet_path()
    comparison_path = exp.get_comparison_path("Comparisons All")

    for path in (model_path, output_path, var_path, parquet_path, comparison_path):
        assert path.is_relative_to(run.run_dir), path

    model_rel = model_path.relative_to(run.run_dir)
    assert model_rel.parts[0] == "models"
    assert model_rel.parts[1] == "duplicates"
    assert model_rel.parts[2] == "isDeceased_lt_28d"
    assert parquet_path.relative_to(run.run_dir).parts[0] == "inputs"


def test_experiment_without_run_context_stays_cwd_relative(tmp_path, monkeypatch):
    """Legacy behavior: without a run context, paths stay CWD-relative."""
    monkeypatch.chdir(tmp_path)
    exp = _make_exp(run_context=None)

    output_path = exp.get_output_path("XGBoost TSFEL")

    assert output_path.is_absolute() is False
    assert output_path.parts[0] == "outputs"
