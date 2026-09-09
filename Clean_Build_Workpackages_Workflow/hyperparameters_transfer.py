"""Copy hyperparameter files between workflows, preserving relative paths."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys


SOURCE = "/home/paquie.d/Projet Stage M2/WorkPackages_WorkFlow/outputs"
DESTINATION = "/home/paquie.d/Projet Stage M2/Clean_Build_Workpackages_Workflow/outputs"

MODEL_DISPLAY_NAMES = {
    "InceptionTimeModified": "ResNet-1D",
    "LstmTimeModified": "LSTM",
    "VanillaTransformerModified": "Transformer encoder",
    "RandomForest TSFEL": "Random Forest",
    "SVC TSFEL": "SVC",
    "XGBoost TSFEL": "XGBoost",
    "Logistic Regression Lasso TSFEL" : "L1-LR",
}

MODE_DISPLAY_NAMES = {
    "Mode_Commonly_Used_Without_pmsi": "Structured Predictor Set",
    "Mode_IGS2": "SAPS II Predictor Set",
    "Mode_Commonly_Used": "Full Predictor Set",
}

# Search spaces taken from the supplied Optuna objective functions. Exponent
# parameters are displayed as the effective values passed to the models.
SEARCH_SPACES = {
    "InceptionTimeModified": {
        "out_channels": r"$\{8, 16, 32, 64\}$",
        "bottleneck_channels": r"$\{4, 8, 16, 32\}$",
        "batch_size": r"$\{16, 32, 64, 128\}$",
        "num_blocks": r"$\{2, \dots, 5\}$",
        "kernel_sizes": r"$\text{odd } \in \{3, \dots, \min(T-1, 15)\}$",
        "lr": r"$\log [10^{-4},\, 3 \times 10^{-3}]$",
        "weight_decay": r"$\log [10^{-6},\, 10^{-2}]$",
        "clip_grad": r"$\{0{,}5,\, 1{,}0,\, 1{,}5,\, 2{,}0\}$",
        "use_scheduler": r"$\{\text{True}, \text{False}\}$",
    },
    "LstmTimeModified": {
        "fc_units": r"$\{\emptyset, 64, 128, 256, [128, 64], [256, 128]\}$",
        "hidden_size": r"$\{32, 64, 128, 256\}$",
        "batch_size": r"$\{16, 32, 64, 128\}$",
        "clip_grad": r"$\{\emptyset, 0{,}5,\, 1{,}0,\, 1{,}5,\, 2{,}0\}$",
        "num_layers": r"$\{1, \dots, 4\}$",
        "bidirectional": r"$\{\text{True}, \text{False}\}$",
        "dropout": r"$[0,\, 0{,}5]$",
        "layernorm": r"$\{\text{True}, \text{False}\}$",
        "lr": r"$\log [10^{-4},\, 3 \times 10^{-3}]$",
        "weight_decay": r"$\log [10^{-6},\, 10^{-2}]$",
        "use_scheduler": r"$\{\text{True}, \text{False}\}$",
    },
    "VanillaTransformerModified": {
        "d_model": r"$\{16, 32, 64\}$",
        "dim_feedforward": r"$\{32, 64, 128\}$",
        "batch_size": r"$\{16, 32, 64, 128\}$",
        "num_layers": r"$\{1, 2, 3\}$",
        "nhead": r"$\{2, 4\} \text{ (divisors of } d_{\text{model}}\text{)}$",
        "dropout": r"$\{0{,}05,\, 0{,}10,\, \dots,\, 0{,}30\}$",
        "lr": r"$\log [10^{-4},\, 3 \times 10^{-3}]$",
        "weight_decay": r"$\log [10^{-6},\, 10^{-2}]$",
        "clip_grad": r"$\{0{,}5,\, 1{,}0,\, 1{,}5,\, 2{,}0\}$",
        "use_scheduler": r"$\{\text{True}, \text{False}\}$",
    },
    "RandomForest TSFEL": {
        "n_estimators": r"$\{100, 200, \dots, 1000\}$",
        "max_depth": r"$\{3, \dots, 12\}$",
        "min_samples_split": r"$\{2, \dots, 40\}$",
        "min_samples_leaf": r"$\{1, \dots, 20\}$",
        "max_features": r"$\{\sqrt{p},\, \log_2(p)\}$",
        "bootstrap": r"$\{\text{True}, \text{False}\}$",
        "class_weight": r"$\{\text{balanced}, \text{subsample}, \emptyset\}$",
    },
    "SVC TSFEL": {
        "C": r"$\log [0{,}1,\, 100]$",
        "kernel": r"$\{\text{rbf}, \text{linear}, \text{poly}\}$",
        "class_weight": r"$\{\text{balanced}, \emptyset\}$",
    },
    "XGBoost TSFEL": {
        "n_estimators": r"$\{100, 200, \dots, 800\}$",
        "learning_rate": r"$\log [10^{-3},\, 5 \times 10^{-2}]$",
        "max_depth": r"$\{2, \dots, 5\}$",
        "min_child_weight": r"$\{10, \dots, 80\}$",
        "gamma": r"$\log [10^{-3},\, 5]$",
        "subsample": r"$\{0{,}4,\, 0{,}5,\, 0{,}6,\, 0{,}7\}$",
        "colsample_bytree": r"$\{0{,}4,\, 0{,}5,\, 0{,}6,\, 0{,}7\}$",
        "reg_alpha": r"$\log [10^{-3},\, 10]$",
        "reg_lambda": r"$\log [0{,}1,\, 20]$",
    },
    "Logistic Regression Lasso TSFEL" : {
    "C": r"$\log\text{-grid } 10 \text{ pts } \in [10^{-4},\, 10^{4}]$",
    "penalty": r"$\{\text{L1}\}$",
    "solver": r"$\{\text{saga}\}$",
    },
}

DEFAULTS = {
    "InceptionTimeModified": {"lr": 0.001, "num_blocks": 4, "out_channels": 32,
        "bottleneck_channels": 16, "kernel_sizes": 11, "batch_size": 64},
    "LstmTimeModified": {"lr": 0.001},
    "VanillaTransformerModified": {"lr": 0.001, "d_model": 64, "nhead": 4,
        "num_layers": 2, "dim_feedforward": 128, "batch_size": 64},
    "RandomForest TSFEL": {"n_estimators": 200, "max_depth": 12,
        "min_samples_split": 5, "min_samples_leaf": 2, "class_weight": "balanced"},
    "XGBoost TSFEL": {"n_estimators": 300, "max_depth": 5,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "SVC TSFEL": {"C": 1.0, "kernel": "rbf", "class_weight": "balanced"},
    "Logistic Regression Lasso TSFEL" : {
        "C": 1.0,
    "penalty": "l1",
    "solver": "saga",
    }
}


def _latex(value):
    replacements = {
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%",
        "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _display(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_display(item) for item in value) + ")"
    return str(value)


def write_latex_table_for_set(path: Path, reference_configs: list[dict], set_title: str = "") -> str:
    """Write one LaTeX table for a specific feature/variable set."""
    selected = {}
    for config in reference_configs:
        for model, params in config.items():
            if isinstance(params, dict):
                model_values = selected.setdefault(model, {})
                for parameter, value in params.items():
                    if isinstance(value, (list, tuple, set)):
                        for item in value:
                            model_values.setdefault(parameter, set()).add(_display(item))
                    else:
                        model_values.setdefault(parameter, set()).add(_display(value))

    def multiline(items, escape=True):
        if escape:
            rendered = [r"\strut " + _latex(item) for item in items]
        else:
            rendered = [r"\strut " + str(item) for item in items]
        return (
            r"\begin{minipage}[t]{\linewidth}" + "\n"
            + (r"\\" + "\n").join(rendered)
            + "\n" + r"\end{minipage}"
        )

    display_title = MODE_DISPLAY_NAMES.get(set_title, set_title)
    caption_title = f"Hyperparameters -- {display_title}" if display_title else "Hyperparameters"
    # Clean label key: alphanumeric + underscores only (no backslashes or math artifacts)
    safe_label = "".join(c if c.isalnum() else "_" for c in set_title).strip("_")

    lines = [
        r"% Requires: \usepackage{booktabs,longtable,array}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.15\linewidth} @{\hspace{8pt}} >{\raggedright\arraybackslash}p{0.16\linewidth} >{\raggedright\arraybackslash}p{0.12\linewidth} >{\raggedright\arraybackslash}p{0.27\linewidth} >{\raggedright\arraybackslash}p{0.20\linewidth}}",
        rf"\caption{{{caption_title}}}\label{{tab:hyperparameters_{safe_label}}}\\ ",
        r"\toprule",
        r"Model & Hyperparameters & Defaults & Optuna search spaces & Selected reference value(s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Hyperparameters & Defaults & Optuna search spaces & Selected reference value(s) \\",
        r"\midrule",
        r"\endhead",
    ]
    model_items = list(SEARCH_SPACES.items())
    for idx, (model, spaces) in enumerate(model_items):
        parameters = []
        defaults = []
        ranges = []
        references = []
        for parameter, space in spaces.items():
            parameters.append(parameter)
            defaults.append(_display(DEFAULTS.get(model, {}).get(parameter, "not specified")))
            ranges.append(space)
            values = sorted(selected.get(model, {}).get(parameter, set()))
            references.append(", ".join(values) if values else "not found")

        display_name = MODEL_DISPLAY_NAMES.get(model, model)

        lines.append(
            f"{_latex(display_name)} & {multiline(parameters)} & {multiline(defaults)} & "
            f"{multiline(ranges, escape=False)} & {multiline(references)} \\\\"
        )
        
        # Add a midrule between models for clear separation, skip after the last one
        if idx < len(model_items) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{longtable}", r"\endgroup", ""))

    path.parent.mkdir(parents=True, exist_ok=True)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE))
    parser.add_argument("--destination", type=Path, default=Path(DESTINATION))
    parser.add_argument("--apply", action="store_true", help="Perform the copies.")
    parser.add_argument("--overwrite", action="store_true", help="Also replace differing existing destination files.")
    parser.add_argument("--latex-output", type=Path, help="Generate the LaTeX hyperparameter table at this path.")
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Source directory not found: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Source and destination must be separate, non-nested directories.")

    sources = sorted(source.rglob("best_hyperparameters.json"))
    if not sources:
        print(f"No best_hyperparameters.json files found in {source}")
        return 1

    # Validate the entire batch before making any directories or copying files.
    copies = []
    reference_configs = []
    conflicts = []
    unchanged = 0
    for src in sources:
        relative = src.relative_to(source)
        if src.resolve() != source / relative:
            raise ValueError(f"Symbolic links are not supported: {src}")
        payload = src.read_bytes()
        decoded = json.loads(payload)
        if not isinstance(decoded, dict) or not decoded:
            raise ValueError(f"Expected a non-empty JSON object: {src}")
        reference_configs.append(decoded)

        for filename in ("best_hyperparameters.json", "ref_hyperparameters.json"):
            dst = destination / relative.parent / filename
            if dst.resolve() != dst:
                raise ValueError(f"Symbolic links are not supported: {dst}")
            if dst.exists():
                if not dst.is_file():
                    raise ValueError(f"Destination is not a regular file: {dst}")
                if dst.read_bytes() == payload:
                    unchanged += 1
                    continue
                if not args.overwrite:
                    conflicts.append(dst)
                    continue
            copies.append((src, dst, payload))

    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print(f"Source files: {len(sources)}; identical destinations: {unchanged}")
    for src, dst, _ in copies:
        action = "REPLACE" if dst.exists() else "CREATE"
        print(f"[{action}] {src.relative_to(source)} -> {dst.relative_to(destination)}")

    if conflicts:
        for dst in conflicts:
            print(f"[CONFLICT] {dst}", file=sys.stderr)
        print("Nothing copied. Review conflicts, then use --overwrite if intentional.", file=sys.stderr)
        return 1

    if args.latex_output is not None:
        latex_path = args.latex_output.expanduser().resolve()

        # 1. Group configuration files by variable set (subfolder in relative path)
        grouped_configs = defaultdict(list)
        for src in sources:
            parts = src.relative_to(source).parts
            # Extract the folder segment identifying the variable set (e.g., 'Mode_...')
            set_name = [p for p in parts if "Mode_" in p or "set" in p.lower()]
            group_key = set_name[0] if set_name else "default"

            payload = src.read_bytes()
            grouped_configs[group_key].append(json.loads(payload))

        print(f"Detected variable sets: {list(grouped_configs.keys())}")

        # 2. Render each distinct set table
        all_tables_text = []
        for set_key, configs in grouped_configs.items():
            table_latex = write_latex_table_for_set(latex_path, configs, set_title=set_key)
            all_tables_text.append(table_latex)

            # Export standalone .tex file per variable set
            set_file = latex_path.with_name(f"{latex_path.stem}_{set_key}.tex")
            set_file.write_text(table_latex, encoding="utf-8")
            print(f"Wrote table for set '{set_key}' to: {set_file}")

        # Write combined multi-table document separated by page breaks
        latex_path.write_text("\n\n\\clearpage\n\n".join(all_tables_text), encoding="utf-8")
        print(f"Combined LaTeX table written: {latex_path}")

    if not args.apply:
        print(f"Preview only: {len(copies)} copies planned. Use --apply to execute.")
        return 0

    completed = 0
    for src, dst, payload in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental overwrites unless authorized.
        with dst.open("wb" if args.overwrite else "xb") as handle:
            handle.write(payload)
        shutil.copystat(src, dst)
        completed += 1
    print(f"Done: {completed} destination files written; source files unchanged.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
