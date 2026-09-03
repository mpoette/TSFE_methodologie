"""Shared helpers for saving Matplotlib figures in PDF or PNG format."""

from pathlib import Path
from typing import Any


SUPPORTED_OUTPUT_FORMATS = frozenset({"pdf", "png"})


def normalize_output_format(output_format: str) -> str:
    """Normalize and validate a figure output format.

    Args:
        output_format: Requested format, with or without a leading dot.

    Returns:
        The normalized lowercase format name.

    Raises:
        ValueError: If the requested format is not PDF or PNG.
    """
    normalized = str(output_format).lower().lstrip(".")
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            "output_format must be either 'pdf' or 'png'; "
            f"received {output_format!r}."
        )
    return normalized


def resolve_figure_path(path: str | Path, output_format: str = "pdf") -> Path:
    """Return a figure path carrying the requested extension.

    Args:
        path: Desired output path, with or without an extension.
        output_format: Requested output format.

    Returns:
        A path whose suffix matches the normalized output format.
    """
    normalized = normalize_output_format(output_format)
    return Path(path).with_suffix(f".{normalized}")


def save_figure(
    figure: Any,
    path: str | Path,
    output_format: str = "pdf",
    **savefig_kwargs: Any,
) -> Path:
    """Save a Matplotlib figure as PDF or as a 300-DPI PNG.

    Args:
        figure: Matplotlib figure or pyplot-compatible object exposing
            ``savefig``.
        path: Desired output path, with or without an extension.
        output_format: Requested output format.
        **savefig_kwargs: Additional keyword arguments passed to ``savefig``.

    Returns:
        The resolved output path.
    """
    normalized = normalize_output_format(output_format)
    resolved_path = resolve_figure_path(path, normalized)
    savefig_kwargs.pop("dpi", None)
    if normalized == "png":
        savefig_kwargs["dpi"] = 300
    figure.savefig(resolved_path, **savefig_kwargs)
    return resolved_path
