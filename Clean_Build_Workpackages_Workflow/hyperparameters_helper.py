"""Copy hyperparameter files between workflows and generate LaTeX summary tables."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

DEFAULT_CONFIG_PATH = Path("config_hyperparameters.json")


def load_config(config_path: Path) -> dict:
    """Load settings from JSON config file if present."""
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a JSON object: {config_path}")
        return data
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON syntax in config {config_path}: {err}") from err


def _latex(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(c, c) for c in str(value))


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


def write_latex_table_for_set(
    reference_configs: list[dict],
    set_title: str,
    search_spaces: dict,
    defaults: dict,
    model_display: dict,
    mode_display: dict,
) -> str:
    """Generate a single LaTeX table for a specific feature set."""
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
        rendered = [r"\strut " + (_latex(item) if escape else str(item)) for item in items]
        return (
            r"\begin{minipage}[t]{\linewidth}" + "\n"
            + (r"\\" + "\n").join(rendered)
            + "\n" + r"\end{minipage}"
        )

    display_title = mode_display.get(set_title, set_title)
    caption_title = f"Hyperparameters -- {display_title}" if display_title else "Hyperparameters"
    safe_label = "".join(c if c.isalnum() else "_" for c in set_title).strip("_")

    lines = [
        r"% Requires: \usepackage{booktabs,longtable,array}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{longtable}{@{} "
        r">{\raggedright\arraybackslash}p{0.15\linewidth} "
        r">{\raggedright\arraybackslash}p{0.18\linewidth} "
        r">{\raggedright\arraybackslash}p{0.12\linewidth} "
        r">{\raggedright\arraybackslash}p{0.36\linewidth} "
        r">{\raggedright\arraybackslash}p{0.16\linewidth} @{}}",
        rf"\caption{{{caption_title}}}\label{{tab:hyperparameters_{safe_label}}}\\ ",
        r"\toprule",
        r"Model & Hyperparameters & Defaults & Optuna search spaces & Selected value(s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Hyperparameters & Defaults & Optuna search spaces & Selected value(s) \\",
        r"\midrule",
        r"\endhead",
]

    model_items = list(search_spaces.items())
    for idx, (model, spaces) in enumerate(model_items):
        parameters = []
        model_defaults = []
        ranges = []
        references = []

        for parameter, space in spaces.items():
            parameters.append(parameter)
            model_defaults.append(_display(defaults.get(model, {}).get(parameter, "not specified")))
            ranges.append(space)
            values = sorted(selected.get(model, {}).get(parameter, set()))
            references.append(", ".join(values) if values else "not found")

        display_name = model_display.get(model, model)
        lines.append(
            f"{_latex(display_name)} & {multiline(parameters)} & {multiline(model_defaults)} & "
            f"{multiline(ranges, escape=False)} & {multiline(references)} \\\\"
        )
        if idx < len(model_items) - 1:
            lines.append(r"\midrule")

    lines.extend((r"\bottomrule", r"\end{longtable}", r"\endgroup", ""))
    return "\n".join(lines)


def main(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)

    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path(cfg["source"]) if "source" in cfg else None,
        required="source" not in cfg,
        help="Source directory tree containing hyperparameter files.",
    )
    parser.add_argument(
        "--destination",
        "-d",
        type=Path,
        default=Path(cfg["destination"]) if "destination" in cfg else None,
        required="destination" not in cfg,
        help="Destination directory tree.",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default=cfg.get("pattern", "best_hyperparameters.json"),
        help="Glob pattern to search in source.",
    )
    parser.add_argument(
        "--target-names",
        nargs="+",
        default=cfg.get("target_names", ["best_hyperparameters.json", "ref_hyperparameters.json"]),
        help="Destination filename(s) to write per matched file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute file copies (dry-run preview by default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=cfg.get("overwrite", False),
        help="Replace differing destination files.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        help="Generate the LaTeX hyperparameter tables at this path.",
    )
    args = parser.parse_args(remaining_argv)

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not source.is_dir():
        parser.error(f"Source directory not found: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Source and destination must be separate, non-nested directories.")

    sources = sorted(source.rglob(args.pattern))
    if not sources:
        print(f"No files matching '{args.pattern}' found in {source}")
        return 1

    copies = []
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

        for filename in args.target_names:
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

    print(f"Config file  : {pre_args.config.resolve() if pre_args.config.exists() else 'None'}")
    print(f"Source       : {source}")
    print(f"Destination  : {destination}")
    print(f"Source files : {len(sources)}; identical destinations: {unchanged}")

    for src, dst, _ in copies:
        action = "REPLACE" if dst.exists() else "CREATE"
        print(f"[{action}] {src.relative_to(source)} -> {dst.relative_to(destination)}")

    if conflicts:
        for dst in conflicts:
            print(f"[CONFLICT] {dst}", file=sys.stderr)
        print("Nothing copied. Review conflicts, then use --overwrite if intentional.", file=sys.stderr)
        return 1

    # LaTeX tables generation
    if args.latex_output is not None:
        search_spaces = cfg.get("search_spaces", {})
        defaults = cfg.get("defaults", {})
        model_display = cfg.get("model_display_names", {})
        mode_display = cfg.get("mode_display_names", {})

        if not search_spaces:
            print("WARNING: 'search_spaces' is empty or missing in config. LaTeX output will be empty.", file=sys.stderr)

        latex_path = args.latex_output.expanduser().resolve()
        grouped_configs = defaultdict(list)

        for src in sources:
            parts = src.relative_to(source).parts
            set_name = [p for p in parts if "Mode_" in p or "set" in p.lower()]
            group_key = set_name[0] if set_name else "default"
            grouped_configs[group_key].append(json.loads(src.read_bytes()))

        print(f"Detected variable sets: {list(grouped_configs.keys())}")

        all_tables_text = []
        for set_key, configs in grouped_configs.items():
            table_latex = write_latex_table_for_set(
                configs,
                set_title=set_key,
                search_spaces=search_spaces,
                defaults=defaults,
                model_display=model_display,
                mode_display=mode_display,
            )
            all_tables_text.append(table_latex)

            set_file = latex_path.with_name(f"{latex_path.stem}_{set_key}.tex")
            set_file.parent.mkdir(parents=True, exist_ok=True)
            set_file.write_text(table_latex, encoding="utf-8")
            print(f"Wrote table for set '{set_key}' to: {set_file}")

        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text("\n\n\\clearpage\n\n".join(all_tables_text), encoding="utf-8")
        print(f"Combined LaTeX table written: {latex_path}")

    if not args.apply:
        print(f"Preview only: {len(copies)} copies planned. Use --apply to execute.")
        return 0

    completed = 0
    for src, dst, payload in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
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