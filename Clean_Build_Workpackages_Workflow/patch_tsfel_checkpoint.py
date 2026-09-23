#!/usr/bin/env python3
"""Patch: chunked, resumable TSFEL extraction (bounded memory).

Run from Clean_Build_Workpackages_Workflow:
    python patch_tsfel_checkpoint.py

Edits (with .bak_tsfel backups):
  - utilitaries/features_extraction_utils.py : extract_tsfel_per_patient
  - ICU_Models_Comparison.py : checkpoint_dir argument + atomic write of the
    global TSFEL parquet + cleanup of chunk checkpoints
"""
import os
import py_compile
import shutil
import sys

NEW_FUNCTION = 'def extract_tsfel_per_patient(\n    df: PolarsFrame,\n    patient_col: str,\n    time_col: str,\n    feature_cols: list[str],\n    target_col: str,\n    n_jobs: int = 8,\n    sampling_interval_hours: float = 1.0,\n    checkpoint_dir: str | os.PathLike | None = None,\n    chunk_size: int = 1000,\n) -> pl.DataFrame:\n    """Extract TSFEL features independently for each patient or ICU stay.\n\n    Patients are processed in chunks. Each chunk is dispatched to a fresh pool\n    of Joblib workers and its per-patient results are concatenated into a\n    single DataFrame, which bounds the memory held by the parent process (tens\n    of thousands of one-row, very wide DataFrames otherwise exhaust RAM). When\n    ``checkpoint_dir`` is given, each finished chunk is written atomically to\n    disk so that an interrupted extraction resumes at the first missing chunk.\n    Fractal features are excluded, along with Histogram mode and MFCC spectral\n    features. All other spectral features are retained. Row order (patients\n    sorted by ``patient_col``) and column values are unchanged compared with a\n    single-pass extraction.\n\n    Args:\n        df: Patient time-series dataset.\n        patient_col: Patient or stay identifier column.\n        time_col: Temporal ordering column.\n        feature_cols: Dynamic variables used for feature extraction.\n        target_col: Prediction target column.\n        n_jobs: Number of parallel Joblib workers. Defaults to 8 to avoid OOM\n            spikes on systems with large core counts.\n        sampling_interval_hours: Time between two consecutive observations, in\n            hours. Defaults to 1.0 (one observation per hour). It is converted\n            to hertz before being passed to TSFEL: ``fs = 1 / (hours * 3600)``.\n        checkpoint_dir: Optional directory used to store one parquet file per\n            finished chunk. Checkpoints are discarded automatically when the\n            extraction settings or the patient list differ from the stored\n            metadata.\n        chunk_size: Number of patients per chunk.\n\n    Returns:\n        A DataFrame containing one row per patient and one column per\n        extracted TSFEL feature.\n\n    Raises:\n        ValueError: If the input is empty, required columns are missing, no\n            feature column is provided, or no patient group is processed.\n    """\n    import gc\n    import json\n    import shutil\n    from pathlib import Path\n\n    df = _ensure_dataframe(df)\n\n    if df.is_empty():\n        raise ValueError(\n            "Cannot run TSFEL extraction on an empty DataFrame."\n        )\n\n    if not feature_cols:\n        raise ValueError(\n            "At least one dynamic feature column is required."\n        )\n\n    if not np.isfinite(sampling_interval_hours) or sampling_interval_hours <= 0:\n        raise ValueError(\n            "sampling_interval_hours must be a finite, strictly positive "\n            f"number, got {sampling_interval_hours!r}."\n        )\n\n    sampling_interval_seconds = sampling_interval_hours * 3600.0\n    sampling_frequency_hz = 1.0 / sampling_interval_seconds\n    sampling_frequency_per_hour = sampling_frequency_hz * 3600.0\n\n    logger.debug(\n        "TSFEL sampling: interval=%.12g hour(s), fs=%.15g Hz, rate=%.12g sample/hour",\n        sampling_interval_hours,\n        sampling_frequency_hz,\n        sampling_frequency_per_hour,\n    )\n\n    _validate_columns(\n        df,\n        [\n            patient_col,\n            time_col,\n            target_col,\n            *feature_cols,\n        ],\n        context="TSFEL extraction",\n    )\n\n    _validate_regular_sampling(\n        df=df,\n        patient_col=patient_col,\n        time_col=time_col,\n        expected_interval_hours=sampling_interval_hours,\n    )\n\n    df = df.sort(\n        [\n            patient_col,\n            time_col,\n        ]\n    )\n\n    config = tsfel.get_features_by_domain()\n\n    # Fractal features are intentionally excluded from this pipeline.\n    config.pop("fractal", None)\n\n    # Histogram mode is excluded because it may crash on constant signals.\n    config["statistical"].pop("Histogram mode", None)\n\n    # MFCC is excluded because it reduces explainability.\n    config["spectral"].pop("MFCC", None)\n\n    # Log feature columns passed to TSFEL to detect static features leaking in.\n    logger.info(\n        "TSFEL feature_cols (%d columns): %s",\n        len(feature_cols),\n        feature_cols,\n    )\n\n    patient_ids = (\n        df.get_column(patient_col)\n        .unique(maintain_order=True)\n        .to_list()\n    )\n    n_patients = len(patient_ids)\n    if n_patients == 0:\n        raise ValueError(\n            "No patient group was processed during TSFEL extraction."\n        )\n\n    chunk_size = max(1, int(chunk_size))\n    chunks = [\n        patient_ids[start:start + chunk_size]\n        for start in range(0, n_patients, chunk_size)\n    ]\n\n    logger.debug(\n        "Starting parallel TSFEL extraction for %d patients in %d chunk(s) (n_jobs=%d)",\n        n_patients,\n        len(chunks),\n        n_jobs,\n    )\n\n    # ------------------------------------------------------------------\n    # Checkpoint directory and metadata\n    # ------------------------------------------------------------------\n    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None\n\n    def _chunk_file(chunk_index):\n        return checkpoint_path / f"chunk_{chunk_index:05d}.parquet"\n\n    if checkpoint_path is not None:\n        metadata = {\n            "feature_cols": [str(c) for c in feature_cols],\n            "patient_col": str(patient_col),\n            "time_col": str(time_col),\n            "target_col": str(target_col),\n            "sampling_interval_hours": float(sampling_interval_hours),\n            "n_patients": n_patients,\n            "chunk_size": chunk_size,\n            "first_ids": [str(x) for x in patient_ids[:5]],\n            "last_ids": [str(x) for x in patient_ids[-5:]],\n        }\n        metadata_file = checkpoint_path / "meta.json"\n        if checkpoint_path.exists():\n            try:\n                previous_metadata = json.loads(metadata_file.read_text())\n            except Exception:\n                previous_metadata = None\n            if previous_metadata != metadata:\n                logger.warning(\n                    "TSFEL checkpoints in %s do not match the current extraction "\n                    "settings and will be discarded.",\n                    checkpoint_path,\n                )\n                shutil.rmtree(checkpoint_path, ignore_errors=True)\n        checkpoint_path.mkdir(parents=True, exist_ok=True)\n        metadata_file.write_text(json.dumps(metadata))\n\n    # ------------------------------------------------------------------\n    # Identify chunks that still need to be computed\n    # ------------------------------------------------------------------\n    already_done = 0\n    pending_chunks = []\n    for chunk_index, chunk_ids in enumerate(chunks):\n        if checkpoint_path is not None:\n            chunk_file = _chunk_file(chunk_index)\n            if chunk_file.exists():\n                try:\n                    stored_ids = (\n                        pl.read_parquet(chunk_file, columns=[patient_col])\n                        .get_column(patient_col)\n                        .to_list()\n                    )\n                    if stored_ids and {str(x) for x in stored_ids} <= {str(x) for x in chunk_ids}:\n                        already_done += len(chunk_ids)\n                        continue\n                except Exception:\n                    pass\n                chunk_file.unlink(missing_ok=True)\n        pending_chunks.append(chunk_index)\n\n    if already_done:\n        print(\n            f"[TSFEL] Resuming from checkpoints: {already_done}/{n_patients} "\n            f"patients already extracted ({len(chunks) - len(pending_chunks)}/{len(chunks)} chunks).",\n            flush=True,\n        )\n\n    def _restart_workers():\n        # Fresh worker processes for each chunk release memory accumulated by\n        # long-lived Loky workers.\n        try:\n            from joblib.externals.loky import get_reusable_executor\n\n            get_reusable_executor().shutdown(wait=True)\n        except Exception:\n            pass\n\n    # ------------------------------------------------------------------\n    # Chunked extraction\n    # ------------------------------------------------------------------\n    in_memory_chunks = []\n    progress_bar = tqdm(\n        total=n_patients,\n        initial=already_done,\n        desc="Parallel TSFEL extraction",\n        unit="patient",\n    )\n    try:\n        for chunk_index in pending_chunks:\n            chunk_ids = chunks[chunk_index]\n            patient_groups = (\n                df.filter(pl.col(patient_col).is_in(chunk_ids))\n                .partition_by(patient_col, maintain_order=True)\n            )\n\n            results_generator = Parallel(n_jobs=n_jobs, return_generator=True)(\n                delayed(_process_single_patient)(\n                    patient_df=patient_group,\n                    config=config,\n                    feature_cols=feature_cols,\n                    patient_col=patient_col,\n                    target_col=target_col,\n                    sampling_frequency_hz=sampling_frequency_hz,\n                )\n                for patient_group in patient_groups\n            )\n\n            chunk_results = []\n            for patient_result in results_generator:\n                chunk_results.append(patient_result)\n                progress_bar.update(1)\n            del patient_groups\n\n            if chunk_results:\n                # One vertical concatenation per chunk keeps memory bounded.\n                chunk_df = pl.concat(chunk_results, how="vertical")\n                del chunk_results\n\n                if checkpoint_path is not None:\n                    chunk_file = _chunk_file(chunk_index)\n                    temporary_file = chunk_file.with_name(chunk_file.name + ".tmp")\n                    chunk_df.write_parquet(temporary_file)\n                    os.replace(temporary_file, chunk_file)\n                    del chunk_df\n                else:\n                    in_memory_chunks.append(chunk_df)\n\n            _restart_workers()\n            gc.collect()\n    finally:\n        progress_bar.close()\n\n    # ------------------------------------------------------------------\n    # Final assembly (same patient order as a single-pass extraction)\n    # ------------------------------------------------------------------\n    if checkpoint_path is not None:\n        chunk_frames = [\n            pl.read_parquet(_chunk_file(chunk_index))\n            for chunk_index in range(len(chunks))\n            if _chunk_file(chunk_index).exists()\n        ]\n    else:\n        chunk_frames = in_memory_chunks\n\n    if not chunk_frames:\n        raise ValueError(\n            "No patient group was processed during TSFEL extraction."\n        )\n\n    return pl.concat(\n        chunk_frames,\n        how="vertical_relaxed",\n    )\n'

FEU = os.path.join("utilitaries", "features_extraction_utils.py")
ICU = "ICU_Models_Comparison.py"


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def backup(path):
    dst = path + ".bak_tsfel"
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    print(f"  backup: {dst}")


for p in (FEU, ICU):
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run this script from Clean_Build_Workpackages_Workflow.")

# ---------------------------------------------------------------- features_extraction_utils.py
src = read(FEU)
nl = "\r\n" if "\r\n" in src else "\n"
if "checkpoint_dir: str | os.PathLike | None = None" in src:
    print(f"[skip] {FEU} already patched")
else:
    start = src.index("def extract_tsfel_per_patient(")
    marker = src.index("FOLD-LEVEL FEATURE SELECTION", start)
    # go back to the '# ====' line preceding the marker
    block_start = src.rindex(nl + "# =", start, marker) + len(nl)
    old_block = src[start:block_start]
    assert "results = list(results_generator)" in old_block, "unexpected version of extract_tsfel_per_patient"
    assert old_block.count("def ") == 1, "unexpected content between function and marker"
    new_block = NEW_FUNCTION.replace("\n", nl).rstrip() + nl + nl
    backup(FEU)
    write(FEU, src[:start] + new_block + src[block_start:])
    print(f"[ok] {FEU} patched")

# ---------------------------------------------------------------- ICU_Models_Comparison.py
src = read(ICU)
nl = "\r\n" if "\r\n" in src else "\n"
old_call = "tsfel_global_df = extract_feat.extract_tsfel_per_patient(df_clean_3, extract.ID_COL, extract.TIME_COL, tsfel_features, target_col, n_jobs = n_jobs)"
new_call = old_call[:-1] + ", checkpoint_dir=str(raw_global_tsfel_path) + '.chunks')"
old_write = "complete_tsfel_df.write_parquet(raw_global_tsfel_path)"
if "checkpoint_dir=str(raw_global_tsfel_path)" in src:
    print(f"[skip] {ICU} already patched")
else:
    assert src.count(old_call) == 1, "extract_tsfel_per_patient call not found exactly once"
    assert src.count(old_write) == 1, "write_parquet(raw_global_tsfel_path) not found exactly once"
    src = src.replace(old_call, new_call)
    pos = src.index(old_write)
    line_start = src.rindex(nl, 0, pos) + len(nl)
    indent = src[line_start:pos]
    assert indent.strip() == ""
    replacement = nl.join([
        "# Atomic write: a crash during the write never leaves a truncated cache.",
        "import shutil as _shutil",
        "_tmp_tsfel_path = str(raw_global_tsfel_path) + '.tmp'",
        "complete_tsfel_df.write_parquet(_tmp_tsfel_path)",
        "os.replace(_tmp_tsfel_path, raw_global_tsfel_path)",
        "_shutil.rmtree(str(raw_global_tsfel_path) + '.chunks', ignore_errors=True)",
    ])
    replacement = nl.join(indent + l if i else l for i, l in enumerate(replacement.split(nl)))
    src = src[:pos] + replacement + src[pos + len(old_write):]
    backup(ICU)
    write(ICU, src)
    print(f"[ok] {ICU} patched")

# ---------------------------------------------------------------- checks
for p in (FEU, ICU):
    py_compile.compile(p, doraise=True)
print("[ok] both files compile")
