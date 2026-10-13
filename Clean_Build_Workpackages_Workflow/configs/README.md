# Run configuration

`run.default.json` reproduces the launch parameters that were previously
hard-coded in the main pipeline script.

Run with the default configuration:

```powershell
python ICU_Models_Comparison.py
```

Load another configuration:

```powershell
python ICU_Models_Comparison.py --config configs/my_run.json
```

Open the interactive terminal selector:

```powershell
python ICU_Models_Comparison.py --interactive
```

Command-line values override the JSON without modifying it:

```powershell
python ICU_Models_Comparison.py `
  --modes wp3_test `
  --no-run-training `
  --figure-format png
```

Use `--help` for the complete option list and `--print-config` to inspect the
effective configuration without importing the machine-learning dependencies or
starting the pipeline.

## Configuration files

| File | Dataset | Scope |
|---|---|---|
| `run.default.json` | CHU (full) | reference grid + quick-test modes |
| `run.mimic.json` / `run.mimic.smoke.json` | MIMIC | wp1 full / smoke |
| `run.smoke_phase3.json`, `run.smoke_chu100.json` | smoke | Phase 3 validation (1 XGBoost exp) |
| `run.chu.wp1_2017.json` | CHU 2017+ subset | wp1: 7 models, `commonly_used_without_pmsi`, optuna on |
| `run.chu.tsfeldomains.json` | CHU 2017+ subset | TSFEL-domain ablation: 2 profiles × 7 models |

CHU 2017+ subset: `/data2/poette.m/chu_2017plus/` (built by
`chu/scripts/build_chu_2017plus.py` at the workspace root — encounters with
`adm_year >= 2017`, 12 906 patients after the pipeline's `prio_first` dedup).
See `../RUNS_CHU_2017.md` for the full runbook.

## Per-profile fields

Beyond the standard matrix fields (`windowing_mode`, `target_labels`,
`model_names`, `stratify_modes`, `optuna_run_options`, `feature_modes`,
`balancing_methods`, `mode_duplicates`), a profile may define:

- `tsfel_domains` (optional list): TSFEL feature domains to extract.
  Valid values: `statistical`, `temporal`, `spectral`.
  - omitted / `[]` → historical behavior (statistical + temporal + spectral,
    59 features/variable after the standing exclusions)
  - `["statistical"]` → 19 features/variable
  - `["statistical", "temporal"]` → 34 features/variable

  Each distinct domain set gets its own artifact subtree
  (`.../tsfel_<domain>[_<domain>]/` inside the run directory) so ablation
  variants never share the TSFEL global cache or model/output trees.
