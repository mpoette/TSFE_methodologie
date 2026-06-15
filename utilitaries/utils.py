import glob
import re
import os
from pathlib import Path
def get_unique_path(path):
    path = Path(path)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    new_path = parent / f"{stem}_{i}{suffix}"

    while new_path.exists():
        i += 1
        new_path = parent / f"{stem}_{i}{suffix}"

    return new_path


def get_latest_model_path(base_pattern, extension):
    files = glob.glob(base_pattern)

    if not files:
        return None

    def extract_number(f, extension = extension):
        match = re.search(rf"_(\d+)\{extension}$", f)
        return int(match.group(1)) if match else -1

    return max(files, key=extract_number)