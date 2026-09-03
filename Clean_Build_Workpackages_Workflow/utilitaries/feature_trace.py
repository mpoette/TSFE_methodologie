"""Persistent feature-removal trace for explainability figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


VALID_REMOVAL_STAGES = frozenset({"zero_variance", "correlation", "boruta"})


def load_feature_trace(path: str | Path | None) -> dict[str, str]:
    """Load feature removal stages from a JSON trace.

    Args:
        path: Trace path, or ``None`` when tracing is disabled.

    Returns:
        Mapping from internal feature name to removal stage. Missing files
        produce an empty mapping.

    Raises:
        ValueError: If the trace contains an unsupported removal stage.
    """
    if path is None:
        return {}
    trace_path = Path(path)
    if not trace_path.exists():
        return {}
    with trace_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features", payload)
    trace = {str(name): str(stage) for name, stage in features.items()}
    invalid = sorted(set(trace.values()).difference(VALID_REMOVAL_STAGES))
    if invalid:
        raise ValueError(f"Unsupported feature-removal stage(s): {invalid}")
    return trace


def save_feature_trace(
    path: str | Path,
    feature_stages: Mapping[str, str],
) -> Path:
    """Persist a complete feature-removal trace atomically.

    Args:
        path: Destination JSON path.
        feature_stages: Mapping from feature names to removal stages.

    Returns:
        Resolved trace path.

    Raises:
        ValueError: If a removal stage is unsupported.
    """
    trace_path = Path(path)
    normalized = {str(name): str(stage) for name, stage in feature_stages.items()}
    invalid = sorted(set(normalized.values()).difference(VALID_REMOVAL_STAGES))
    if invalid:
        raise ValueError(f"Unsupported feature-removal stage(s): {invalid}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = trace_path.with_suffix(f"{trace_path.suffix}.tmp")
    payload = {"version": 1, "features": dict(sorted(normalized.items()))}
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(trace_path)
    return trace_path


def record_removed_features(
    path: str | Path | None,
    stage: str,
    features: Iterable[str],
    *,
    reset: bool = False,
) -> dict[str, str]:
    """Record features removed at one pipeline stage.

    Args:
        path: Destination trace path, or ``None`` to disable persistence.
        stage: One of ``zero_variance``, ``correlation``, or ``boruta``.
        features: Internal feature names removed at this stage.
        reset: Whether to replace any existing trace before recording.

    Returns:
        Updated in-memory trace mapping.
    """
    if stage not in VALID_REMOVAL_STAGES:
        raise ValueError(f"Unsupported feature-removal stage: {stage!r}")
    trace = {} if reset else load_feature_trace(path)
    for feature in features:
        trace.setdefault(str(feature), stage)
    if path is not None:
        save_feature_trace(path, trace)
    return trace
