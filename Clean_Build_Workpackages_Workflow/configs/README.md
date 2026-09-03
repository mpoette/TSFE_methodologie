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
