# ICU Model Comparison

`icu-model-comparison` is a reproducible pipeline for training, evaluating,
interpreting, and comparing ICU mortality prediction models. It supports
classical estimators, TSFEL feature extraction, deep-learning time-series
models, calibration, holdout evaluation, and publication-ready figures.

## Installation

From the project directory:

```powershell
python -m pip install -e .
```

For development and publication checks:

```powershell
python -m pip install -e ".[dev]"
```

## Running the pipeline

```powershell
# Use the packaged default configuration
icu-model-comparison

# Select options interactively
icu-model-comparison --interactive

# Use a version-controlled experiment configuration
icu-model-comparison --config configs/run.default.json

# Inspect the effective configuration without starting the pipeline
icu-model-comparison --print-config
```

The original script command remains supported:

```powershell
python ICU_Models_Comparison.py
```

## Python API

The top-level import is deliberately lightweight and does not import PyTorch,
Matplotlib, or start the pipeline:

```python
from icu_model_comparison import RunConfig, load_run_config
from icu_model_comparison.figures.feature_names import short_feature_name

config = load_run_config("configs/run.default.json")
print(config.experiment_count())
print(short_feature_name("heart_rate_Mean"))  # HR — Mean
```

Domain modules are available under `icu_model_comparison`, including
`training`, `evaluation`, `features`, `preprocessing`, `postprocessing`,
`sampling`, and `figures`.

## Reproducibility

Run settings live in JSON files and are validated before scientific
dependencies are imported. Keep one configuration file per reported
experiment and archive the generated pipeline log with the results.

## Publication checklist

Before publishing a release, run the full pipeline on the target environment,
execute the test suite, build the wheel and source distribution with
`python -m build`, then validate both artifacts with `python -m twine check
dist/*`.
