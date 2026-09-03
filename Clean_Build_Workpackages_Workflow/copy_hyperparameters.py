"""Copy hyperparameter files between workflows, preserving relative paths."""

import argparse
import json
from pathlib import Path
import shutil
import sys


SOURCE = "/home/paquie.d/Projet Stage M2/WorkPackages_WorkFlow/outputs"
DESTINATION = "/home/paquie.d/Projet Stage M2/Clean_Build_Workpackages_Workflow/outputs"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE))
    parser.add_argument("--destination", type=Path, default=Path(DESTINATION))
    parser.add_argument("--apply", action="store_true", help="Perform the copies.")
    parser.add_argument("--overwrite", action="store_true", help="Also replace differing existing destination files.")
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