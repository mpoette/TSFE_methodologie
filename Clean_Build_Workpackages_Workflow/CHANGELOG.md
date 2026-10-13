# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- **Restrictable TSFEL feature domains.** A per-profile `tsfel_domains` config
  field (e.g. `["statistical"]` or `["statistical", "temporal"]`) limits which
  TSFEL domains are extracted, enabling feature-set ablations (statistical-only,
  statistical+temporal) without touching the model or preprocessing code. The
  extraction is filtered by a new `utilitaries.features_extraction_utils.
  build_tsfel_config` helper (standing exclusions — fractal, Histogram mode,
  MFCC — still apply). Each domain set gets its own artifact subtree
  (`.../tsfel_<domains>/`) so distinct variants never share the TSFEL global
  cache, model, or output trees; an empty/omitted value preserves the previous
  behavior (statistical + temporal + spectral).
- **Phase 3 — per-run artifact directories.** Every pipeline launch now writes
  its artifacts into a dedicated `run_YYYYMMDD_HHMMSS[_<config>]/` folder (created
  in the launch directory) instead of shared `inputs/`, `models/`, and
  `outputs/` trees. Successive runs on the same dataset no longer overwrite or
  mix results; each run is attributable by folder alone.
- Add `utilitaries.path_utils.RunContext`: identifies a run, owns its
  `models/`, `outputs/`, `inputs/`, `pipeline_logs/`, and `optuna.db`.
- `path_utils.Experiment` accepts an optional `run_context=`; when set, every
  generated artifact path (models, outputs, inputs, boruta, resampling,
  lasso, comparison) is resolved inside the run directory. Without it, the
  legacy CWD-relative layout is preserved.
- The pipeline chdirs into the run directory at startup (restored on exit),
  resolves `dataset_path` to an absolute path, writes a `run_metadata.json`
  (name, start time, full config), and logs to the run-local
  `pipeline_logs/`.
- Optuna studies are stored in the run-local `optuna.db` (storage URL passed
  explicitly to every search); a `utilitaries.optuna.optuna_utils.set_storage`
  helper is provided.
- CWD-independent feature-metadata resolution: the `dynamic_features.json` /
  `calculated_features_to_original_features.json` paths are resolved from the
  module location rather than a hardcoded `../../` relative path, so the
  per-run chdir cannot break them.
- Add `tests/test_run_context.py` covering run-directory creation, artifact
  scoping, and the legacy no-context behavior.

## 0.1.0

- Add the installable `icu_model_comparison` public package.
- Add a console entry point and JSON-based run configuration.
- Organize figures by performance, comparison, explainability, feature
  analysis, and data distribution.
- Add PDF output by default and 300-DPI PNG output on request.
- Add literature-friendly feature labels for publication figures.
- Add feature-level removal provenance for zero-variance, correlation, and
  Boruta filtering, with stage-specific gray shades in SHAP heatmaps.
