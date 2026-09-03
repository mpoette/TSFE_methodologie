"""Correlation and feature-selection analysis visualizations."""

from collections.abc import Sequence
from pathlib import Path
import gc
import os
from tqdm.auto import tqdm
import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from utilitaries.figures.output import save_figure as save_figure_file
from utilitaries.figures.feature_names import short_feature_name

from utilitaries.features_extraction_utils import (
    PolarsFrame,
    _compute_numeric_correlation,
    _ensure_dataframe,
    _log_source_feature_groups,
    _log_memory,
    grouper_colonnes_par_variable_source,
)

logger = logging.getLogger(__name__)


def _safe_save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 120,
    output_format: str = "pdf",
) -> None:
    """Save a Matplotlib figure without the expensive tight-bbox pass.

    Args:
        fig: The Matplotlib figure to save.
        path: Output file path.
        dpi: Image resolution in dots per inch.
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at
            300 DPI.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_figure_file(fig, path, output_format)
    logger.info("Saved: %s (dpi=%d)", path, dpi)

def calculer_matrice_correlation(
    df: PolarsFrame,
    exclude_cols: Sequence[str] | None = None,
    threshold: float = 0.9,
) -> tuple[pl.DataFrame, list[tuple[str, str, float]]]:
    """Compute the correlation matrix and identify highly correlated pairs.

    Correlations are computed using Pearson's method on numeric columns.
    Missing values are excluded pairwise.

    Args:
        df:
            DataFrame containing numerical features.
        exclude_cols:
            Columns to exclude from the correlation computation (typically
            metadata columns such as patient ID or target).
        threshold:
            Absolute correlation threshold above which a pair is considered
            highly correlated. Must be in ``(0, 1]``.

    Returns:
        A tuple containing:

        - The full correlation matrix as a Polars DataFrame.
        - A list of ``(col_a, col_b, corr)`` tuples where the absolute
          correlation exceeds the threshold, sorted by descending absolute
          correlation.

    Raises:
        ValueError:
            If the input is empty, required columns are missing, or the
            threshold is out of range.
    """
    df = _ensure_dataframe(df)

    if df.is_empty():
        raise ValueError(
            "Cannot compute correlation on an empty DataFrame."
        )

    if not (0 < threshold <= 1):
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
        )
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            "At least two numeric columns are required for correlation."
        )

    # Convert to Pandas for correlation computation.
    numeric_df = (
        df.select(numeric_cols)
        .to_pandas()
        .apply(pd.to_numeric, errors="coerce")
    )

    corr_matrix = numeric_df.corr(method="pearson")

    # Identify highly correlated pairs (upper triangle only).
    correlated_pairs: list[tuple[str, str, float]] = []

    for i, col_a in enumerate(numeric_cols):
        for j, col_b in enumerate(numeric_cols):
            if j <= i:
                continue

            corr_value = corr_matrix.loc[col_a, col_b]

            if pd.isna(corr_value):
                continue

            if abs(corr_value) >= threshold:
                correlated_pairs.append(
                    (col_a, col_b, float(corr_value))
                )

    # Sort by descending absolute correlation.
    correlated_pairs.sort(
        key=lambda pair: abs(pair[2]),
        reverse=True,
    )

    corr_pl = pl.from_pandas(corr_matrix, include_index=False)

    if correlated_pairs:
        logger.warning(
            "Found %d feature pair(s) with |correlation| >= %s:",
            len(correlated_pairs),
            threshold,
        )
        for col_a, col_b, corr_value in correlated_pairs:
            logger.info("  %s  <->  %s:  %.4f", col_a, col_b, corr_value)
    else:
        logger.info(
            "No feature pair with |correlation| >= %s.", threshold
        )

    return corr_pl, correlated_pairs


def _compute_block_figure_size(
    n_rows: int,
    n_cols: int,
    min_size: float = 12.0,
    max_size: float = 26.0,
    inches_per_feature: float = 0.18,
) -> tuple[float, float]:
    """Return a large but bounded figure size for a correlation block.

    Args:
        n_rows: Number of rows in the correlation matrix.
        n_cols: Number of columns in the correlation matrix.
        min_size: Minimum figure dimension in inches.
        max_size: Maximum figure dimension in inches.
        inches_per_feature: Inches allocated per feature dimension.

    Returns:
        A ``(width, height)`` tuple in inches.
    """
    width = min(max_size, max(min_size, n_cols * inches_per_feature))
    height = min(max_size, max(min_size, n_rows * inches_per_feature))
    return width, height


def _short_feature_label(
    feature_name: str,
    source_name: str,
) -> str:
    """Remove the repeated source prefix from a derived feature label.

    Args:
        feature_name: Full TSFEL feature name.
        source_name: Original source variable name.

    Returns:
        The feature name with the source prefix removed.
    """
    prefix = f"{source_name}_"
    return (
        feature_name[len(prefix):]
        if feature_name.startswith(prefix)
        else feature_name
    )


def _render_cross_block(
    sub_corr: pd.DataFrame,
    path: str,
    cmap: str,
    dpi: int,
    source_a: str,
    source_b: str,
    source_a_is_static: bool,
    source_b_is_static: bool,
    threshold: float | None = None,
    show_colorbar: bool = True,

    output_format: str = "pdf",
) -> None:
    """Render one cross-source correlation block.

    Dynamic x dynamic blocks remain label-free for readability. For a
    static x dynamic block, the singleton static axis displays its source
    name and the dynamic axis displays the associated TSFEL feature names.

    Args:
        sub_corr: Sub-matrix of correlations between two source groups.
        path: Output image file path.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.
        source_a: Name of the row source variable.
        source_b: Name of the column source variable.
        source_a_is_static: Whether ``source_a`` is a static feature.
        source_b_is_static: Whether ``source_b`` is a static feature.
        threshold: Optional absolute correlation threshold for masking.
        show_colorbar: Whether to include a colorbar.
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.
    """
    matrix = sub_corr.to_numpy(dtype=np.float32, copy=True)

    if threshold is not None:
        matrix = np.ma.masked_where(
            ~np.isfinite(matrix) | (np.abs(matrix) < threshold),
            matrix,
        )
    else:
        matrix = np.ma.masked_invalid(matrix)

    if source_a_is_static or source_b_is_static:
        # A singleton static row/column benefits from a wide, shallow figure.
        width = min(34.0, max(16.0, sub_corr.shape[1] * 0.24))
        height = min(18.0, max(5.0, sub_corr.shape[0] * 0.24))
    else:
        width, height = _compute_block_figure_size(
            n_rows=sub_corr.shape[0],
            n_cols=sub_corr.shape[1],
        )

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )

    if source_a_is_static:
        ax.set_yticks(np.arange(sub_corr.shape[0]))
        ax.set_yticklabels(
            [short_feature_name(source_a)] * sub_corr.shape[0],
            fontsize=11,
        )
    else:
        ax.set_yticks([])

    if source_b_is_static:
        ax.set_xticks(np.arange(sub_corr.shape[1]))
        ax.set_xticklabels(
            [short_feature_name(source_b)] * sub_corr.shape[1],
            fontsize=11,
        )
    elif source_a_is_static:
        ax.set_xticks(np.arange(sub_corr.shape[1]))
        ax.set_xticklabels(
            [
                _short_feature_label(column, source_b)
                for column in sub_corr.columns
            ],
            rotation=90,
            fontsize=7,
        )
    else:
        ax.set_xticks([])

    if source_b_is_static and not source_a_is_static:
        ax.set_yticks(np.arange(sub_corr.shape[0]))
        ax.set_yticklabels(
            [
                _short_feature_label(index, source_a)
                for index in sub_corr.index
            ],
            fontsize=7,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        colorbar.set_label("Pearson correlation", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

    # Avoid tight_layout on dense labels; reserve explicit margins instead.
    if source_a_is_static:
        fig.subplots_adjust(
            left=0.08, right=0.93, bottom=0.38, top=0.98
        )
    elif source_b_is_static:
        fig.subplots_adjust(
            left=0.38, right=0.93, bottom=0.08, top=0.98
        )
    else:
        fig.subplots_adjust(
            left=0.01, right=0.93, bottom=0.01, top=0.99
        )

    _safe_save_figure(fig, path, dpi=dpi, output_format=output_format)
    plt.close(fig)
    del fig, ax, image, matrix
    gc.collect()



def _build_static_temporal_summary(
    corr_matrix: pd.DataFrame,
    root_map: dict[str, list[str]],
    static_roots: Sequence[str],
    temporal_roots: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate static-to-temporal correlations by source variable.

    Each cell contains the signed correlation belonging to the TSFEL feature
    pair with the largest absolute correlation for the corresponding
    ``static source x temporal source`` pair. A detail table records the exact
    columns responsible for every displayed cell.

    Args:
        corr_matrix: Full Pearson correlation matrix.
        root_map: Mapping from source variable to TSFEL columns.
        static_roots: Static source variable names.
        temporal_roots: Temporal source variable names.

    Returns:
        A tuple of the summary matrix and the detail DataFrame.
    """
    summary = pd.DataFrame(
        index=list(static_roots),
        columns=list(temporal_roots),
        dtype=float,
    )
    details: list[dict[str, object]] = []

    for static_source in static_roots:
        for temporal_source in temporal_roots:
            sub_corr = corr_matrix.loc[
                root_map[static_source],
                root_map[temporal_source],
            ]
            values = sub_corr.to_numpy(dtype=float, copy=False)

            if values.size == 0 or np.isnan(values).all():
                summary.loc[static_source, temporal_source] = np.nan
                details.append({
                    "static_feature": static_source,
                    "temporal_feature": temporal_source,
                    "static_column": None,
                    "temporal_tsfel_column": None,
                    "correlation": np.nan,
                    "absolute_correlation": np.nan,
                })
                continue

            flat_index = int(np.nanargmax(np.abs(values)))
            row_index, column_index = np.unravel_index(
                flat_index, values.shape
            )
            correlation = float(values[row_index, column_index])
            static_column = str(sub_corr.index[row_index])
            temporal_column = str(sub_corr.columns[column_index])

            summary.loc[static_source, temporal_source] = correlation
            details.append({
                "static_feature": static_source,
                "temporal_feature": temporal_source,
                "static_column": static_column,
                "temporal_tsfel_column": temporal_column,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            })

    return summary, pd.DataFrame(details)


def _render_static_temporal_summary(
    summary: pd.DataFrame,
    path: str,
    cmap: str,
    dpi: int,
    threshold: float | None = None,
    show_colorbar: bool = True,
    annotate_values: bool = True,

    output_format: str = "pdf",
) -> None:
    """Render one compact static-features x temporal-sources matrix.

    Args:
        summary: Aggregated static x temporal correlation matrix.
        path: Output image file path.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.
        threshold: Optional absolute correlation threshold for masking.
        show_colorbar: Whether to include a colorbar.
        annotate_values: Whether to overlay numeric values on cells.
        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.
    """
    matrix = summary.to_numpy(dtype=np.float32, copy=True)
    invalid_mask = ~np.isfinite(matrix)

    if threshold is not None:
        matrix = np.ma.masked_where(
            invalid_mask | (np.abs(matrix) < threshold),
            matrix,
        )
    else:
        matrix = np.ma.masked_where(invalid_mask, matrix)

    n_rows, n_cols = summary.shape
    width = min(30.0, max(10.0, 1.35 * n_cols + 3.0))
    height = min(20.0, max(5.0, 0.72 * n_rows + 3.0))

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [short_feature_name(name) for name in summary.columns],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [short_feature_name(name) for name in summary.index],
        fontsize=10,
    )
    ax.set_xlabel("Temporal source features", fontsize=11)
    ax.set_ylabel("Static features", fontsize=11)

    if annotate_values:
        raw_values = summary.to_numpy(dtype=float, copy=False)
        for row in range(n_rows):
            for column in range(n_cols):
                value = raw_values[row, column]
                if not np.isfinite(value):
                    continue
                if threshold is not None and abs(value) < threshold:
                    continue
                ax.text(
                    column, row, f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.035,
            pad=0.03,
            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        colorbar.set_label("Pearson correlation", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(
        left=0.18, right=0.93, bottom=0.25, top=0.97
    )
    _safe_save_figure(fig, path, dpi=dpi, output_format=output_format)
    plt.close(fig)
    del fig, ax, image, matrix
    gc.collect()

def afficher_correlation_par_blocs(
    df: PolarsFrame,
    source_features: Sequence[str],
    static_features: Sequence[str] | None = None,
    exclude_cols: Sequence[str] | None = None,
    threshold: float = 0.5,
    cmap: str = "RdBu_r",
    output_dir: str | None = "comparison_figs",
    corr_matrix: pd.DataFrame | None = None,
    max_cross_heatmaps: int | None = None,
    block_heatmap_dpi: int = 160,
    show_colorbar: bool = True,
    annotate_static_summary: bool = True,

    output_format: str = "pdf",
) -> None:
    """Generate temporal cross-blocks and one static-temporal summary.

    Dynamic x dynamic source pairs are rendered as detailed TSFEL blocks.
    Static x static pairs are omitted. All static x temporal relationships are
    condensed into one matrix whose rows are the original static features and
    whose columns are the original temporal source features. Each summary cell
    is the signed correlation of the strongest underlying TSFEL pair.

    Args:
        df: DataFrame containing TSFEL feature columns.
        source_features: Original dynamic and static variable names.
        static_features: Optional list of static feature names.
        exclude_cols: Columns to exclude from correlation computation.
        threshold: Absolute correlation threshold for threshold-mode output.
        cmap: Matplotlib colormap name.
        output_dir: Base directory for output images.
        corr_matrix: Optional precomputed Pearson correlation matrix.
        max_cross_heatmaps: Maximum number of temporal cross-blocks to render.
        block_heatmap_dpi: Image resolution for block heatmaps.
        show_colorbar: Whether to include colorbars on heatmaps.
        annotate_static_summary: Whether to annotate the static summary cells.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Raises:
        ValueError:
            If the input is empty, the threshold is invalid, or static
            features are not present in source features.
    """
    df = _ensure_dataframe(df)
    if df.is_empty():
        raise ValueError(
            "Cannot display correlation blocks for an empty DataFrame."
        )
    if not (0 < threshold <= 1):
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}."
        )

    static_set = set(static_features or [])
    unknown_static = static_set - set(source_features)
    if unknown_static:
        raise ValueError(
            "static_features contains values absent from source_features: "
            f"{sorted(unknown_static)}"
        )

    exclude_set = set(exclude_cols) if exclude_cols else set()
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (
            pl.Float32, pl.Float64, pl.Int32, pl.Int64
        )
    ]
    if len(numeric_cols) < 2:
        raise ValueError("At least two numeric columns are required.")

    root_map = grouper_colonnes_par_variable_source(
        columns=numeric_cols,
        source_features=source_features,
        strict=True,
    )
    sorted_roots = _log_source_feature_groups(
        root_map=root_map,
        source_features=source_features,
        total_columns=len(numeric_cols),
    )
    ordered_cols = [
        column
        for source in sorted_roots
        for column in root_map[source]
    ]

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(
            df.select(ordered_cols).to_pandas()
        )
    else:
        missing = set(ordered_cols) - set(corr_matrix.columns)
        if missing:
            raise ValueError(
                "Precomputed correlation matrix is missing columns: "
                f"{sorted(missing)}"
            )
        corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
        logger.info(
            "[STEP 2.1] Reusing precomputed correlation matrix.",
        )

    normal_dir = (
        os.path.join(output_dir, "Normal_Mode")
        if output_dir is not None else None
    )
    threshold_dir = (
        os.path.join(output_dir, "Threshold_Mode")
        if output_dir is not None else None
    )
    for directory in (normal_dir, threshold_dir):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

    static_roots = [
        source for source in sorted_roots if source in static_set
    ]
    temporal_roots = [
        source for source in sorted_roots if source not in static_set
    ]

    if static_roots and temporal_roots:
        static_summary, static_details = _build_static_temporal_summary(
            corr_matrix=corr_matrix,
            root_map=root_map,
            static_roots=static_roots,
            temporal_roots=temporal_roots,
        )
        if normal_dir is not None:
            _render_static_temporal_summary(
                summary=static_summary,
                path=os.path.join(
                    normal_dir, "static_vs_temporal_summary.png"
                ),
                cmap=cmap,
                dpi=block_heatmap_dpi,
                threshold=None,
                show_colorbar=show_colorbar,
                annotate_values=annotate_static_summary,
            output_format=output_format)
            static_details.to_csv(
                os.path.join(
                    normal_dir, "static_vs_temporal_details.csv"
                ),
                index=False,
            )

        if threshold_dir is not None:
            _render_static_temporal_summary(
                summary=static_summary,
                path=os.path.join(
                    threshold_dir, "static_vs_temporal_summary.png"
                ),
                cmap=cmap,
                dpi=block_heatmap_dpi,
                threshold=threshold,
                show_colorbar=show_colorbar,
                annotate_values=annotate_static_summary,
            output_format=output_format)
            static_details.loc[
                static_details["absolute_correlation"] >= threshold
            ].to_csv(
                os.path.join(
                    threshold_dir,
                    "static_vs_temporal_details.csv",
                ),
                index=False,
            )

        for detail in static_details.itertuples(index=False):
            logger.info(
                "[STATIC CORR] %s x %s: corr=%+.6f, via %r",
                detail.static_feature,
                detail.temporal_feature,
                detail.correlation,
                detail.temporal_tsfel_column,
            )

    # Individual images are now reserved for temporal x temporal blocks.
    candidates: list[tuple[float, str, str]] = []
    for index_a, source_a in enumerate(temporal_roots):
        for source_b in temporal_roots[index_a + 1:]:
            sub_corr = corr_matrix.loc[
                root_map[source_a],
                root_map[source_b],
            ]
            values = sub_corr.to_numpy(dtype=float, copy=False)
            max_abs = (
                float(np.nanmax(np.abs(values)))
                if values.size and not np.isnan(values).all()
                else float("nan")
            )
            candidates.append((max_abs, source_a, source_b))

    candidates.sort(
        key=lambda item: -np.nan_to_num(item[0], nan=-1.0)
    )
    if max_cross_heatmaps is not None:
        candidates = candidates[:max_cross_heatmaps]

    logger.info(
        "[STEP 2.2] Rendering %d temporal x temporal cross-block pair(s).",
        len(candidates),
    )

    normal_count = 0
    threshold_count = 0
    for rank, (max_abs, source_a, source_b) in enumerate(
        candidates, start=1
    ):
        sub_corr = corr_matrix.loc[
            root_map[source_a], root_map[source_b]
        ]
        safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_a)
        safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_b)
        filename = (
            f"cross_block_{rank:03d}_{safe_a}_x_{safe_b}.png"
        )

        render_kwargs = {
            "sub_corr": sub_corr,
            "cmap": cmap,
            "dpi": block_heatmap_dpi,
            "source_a": source_a,
            "source_b": source_b,
            "source_a_is_static": False,
            "source_b_is_static": False,
            "show_colorbar": show_colorbar,
        }

        if normal_dir is not None:
            _render_cross_block(
                path=os.path.join(normal_dir, filename),
                threshold=None,
                **render_kwargs,
            output_format=output_format)
        normal_count += 1

        if np.isfinite(max_abs) and max_abs >= threshold:
            if threshold_dir is not None:
                _render_cross_block(
                    path=os.path.join(threshold_dir, filename),
                    threshold=threshold,
                    **render_kwargs,
                output_format=output_format)
            threshold_count += 1

        logger.info(
            "[BLOCK %d/%d] %s x %s: %d x %d, max |corr|=%.4f",
            rank,
            len(candidates),
            source_a,
            source_b,
            len(root_map[source_a]),
            len(root_map[source_b]),
            max_abs,
        )
        del sub_corr

    logger.info(
        "[STEP 2] Correlation figures complete: "
        "Normal temporal blocks=%d, Threshold temporal blocks=%d, "
        "static summary=%s.",
        normal_count,
        threshold_count,
        "generated" if static_roots and temporal_roots else "not generated",
    )

def generer_map_correlation_complete(
    df: PolarsFrame,
    source_features: Sequence[str],
    exclude_cols: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    output_path: str | None = "comparison_figs/corr_complete_map.png",
    corr_matrix: pd.DataFrame | None = None,
    max_full_heatmap_features: int = 300,
    max_block_heatmaps: int = 100,
    full_heatmap_dpi: int = 120,
    block_heatmap_dpi: int = 150,

    output_format: str = "pdf",
) -> pl.DataFrame:
    """Generate a bounded-memory complete map or a capped set of blocks.

    Args:
        df: DataFrame containing TSFEL feature columns.
        source_features: Original dynamic and static variable names.
        exclude_cols: Columns to exclude from correlation computation.
        cmap: Matplotlib colormap name.
        output_path: Output image file path for the complete map.
        corr_matrix: Optional precomputed Pearson correlation matrix.
        max_full_heatmap_features: Maximum feature count before switching to
            block mode.
        max_block_heatmaps: Maximum number of block maps to generate.
        full_heatmap_dpi: Image resolution for the complete map.
        block_heatmap_dpi: Image resolution for individual blocks.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        The correlation matrix as a Polars DataFrame.

    Raises:
        ValueError:
            If the input is empty or fewer than two numeric columns remain.
    """
    df = _ensure_dataframe(df)
    if df.is_empty():
        raise ValueError("Cannot generate correlation map for an empty DataFrame.")

    exclude_set = set(exclude_cols) if exclude_cols else set()
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_set
        and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    if len(numeric_cols) < 2:
        raise ValueError("At least two numeric columns are required.")

    root_map = grouper_colonnes_par_variable_source(
        columns=numeric_cols,
        source_features=source_features,
        strict=True,
    )
    sorted_roots = _log_source_feature_groups(
        root_map=root_map,
        source_features=source_features,
        total_columns=len(numeric_cols),
    )
    ordered_cols = [
        column
        for source in sorted_roots
        for column in root_map[source]
    ]

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(
            df.select(ordered_cols).to_pandas()
        )
    else:
        missing = set(ordered_cols) - set(corr_matrix.columns)
        if missing:
            raise ValueError(
                "Precomputed correlation matrix is missing columns: "
                f"{sorted(missing)}"
            )
        corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]
        logger.info("[STEP 3.1] Reusing precomputed correlation matrix.")

    n_features = len(ordered_cols)
    if n_features <= max_full_heatmap_features:
        logger.info(
            "[STEP 3.2] Rendering complete map (%d features)...",
            n_features,
        )
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            cbar_kws={"shrink": 0.6},
            ax=ax,
            linewidths=0,
            xticklabels=False,
            yticklabels=False,
        )
        cumulative = 0
        for root in sorted_roots[:-1]:
            cumulative += len(root_map[root])
            ax.axhline(y=cumulative, color="black", linewidth=1.2)
            ax.axvline(x=cumulative, color="black", linewidth=1.2)
        ax.set_title(
            f"Complete Correlation Map\n"
            f"{n_features} features across {len(sorted_roots)} groups"
        )
        fig.tight_layout()
        if output_path is not None:
            _safe_save_figure(fig, output_path, dpi=full_heatmap_dpi, output_format=output_format)
        plt.close(fig)
        del fig, ax
        gc.collect()
    else:
        blocks_dir = os.path.join(
            os.path.dirname(output_path) if output_path else "comparison_figs",
            "all_correlations",
        )
        os.makedirs(blocks_dir, exist_ok=True)
        logger.info(
            "[STEP 3.2] Complete map skipped: %d features exceed "
            "the safe limit of %d. "
            "Generating at most %d block map(s).",
            n_features,
            max_full_heatmap_features,
            max_block_heatmaps,
        )

        candidates: list[tuple[float, str, str]] = []
        for i, root_a in enumerate(sorted_roots):
            for root_b in sorted_roots[i:]:
                sub_corr = corr_matrix.loc[root_map[root_a], root_map[root_b]]
                values = sub_corr.abs().to_numpy()
                max_abs = 1.0 if root_a == root_b else (
                    float(np.nanmax(values))
                    if values.size and not np.isnan(values).all()
                    else float("nan")
                )
                candidates.append((max_abs, root_a, root_b))

        candidates.sort(
            key=lambda item: -np.nan_to_num(item[0], nan=-1.0)
        )
        for rank, (max_abs, root_a, root_b) in enumerate(
            candidates[:max_block_heatmaps], start=1
        ):
            sub_corr = corr_matrix.loc[root_map[root_a], root_map[root_b]]
            width = min(16, max(6, 0.35 * len(root_map[root_b])))
            height = min(14, max(5, 0.35 * len(root_map[root_a])))
            fig_b, ax_b = plt.subplots(figsize=(width, height))
            sns.heatmap(
                sub_corr,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=False,
                cbar_kws={"shrink": 0.8},
                ax=ax_b,
                linewidths=0,
                xticklabels=False,
                yticklabels=False,
            )
            title_suffix = "self-block" if root_a == root_b else f"max |corr|={max_abs:.3f}"
            ax_b.set_title(f"{root_a} x {root_b}\n{title_suffix}")
            fig_b.tight_layout()
            safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_a)
            safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_b)
            _safe_save_figure(
                fig_b,
                os.path.join(
                    blocks_dir,
                    f"block_{rank:03d}_{safe_a}_x_{safe_b}.png",
                ),
                dpi=block_heatmap_dpi,
            output_format=output_format)
            plt.close(fig_b)
            del fig_b, ax_b, sub_corr
            gc.collect()

    return pl.from_pandas(corr_matrix, include_index=False)


# ============================================================================
# AUTOMATIC THRESHOLD ESTIMATION
# ============================================================================

def _distance_to_chord_2d(
    point: np.ndarray,
    chord_start: np.ndarray,
    chord_end: np.ndarray,
) -> float:
    """Return the perpendicular distance from a 2D point to a chord."""
    chord = chord_end - chord_start
    chord_norm = np.linalg.norm(chord)

    if chord_norm == 0:
        return 0.0

    relative_point = point - chord_start
    cross_product = (
        chord[0] * relative_point[1]
        - chord[1] * relative_point[0]
    )

    return float(abs(cross_product) / chord_norm)

def _manual_elbow(
    thresholds: np.ndarray,
    values: list[int],
) -> float:
    """Compute the elbow point using the maximum distance to the chord."""
    p1 = np.array([thresholds[0], values[0]], dtype=float)
    p2 = np.array([thresholds[-1], values[-1]], dtype=float)

    max_dist = -1.0
    elbow_idx = 0

    for i, threshold in enumerate(thresholds):
        point = np.array([threshold, values[i]], dtype=float)

        distance = _distance_to_chord_2d(
            point=point,
            chord_start=p1,
            chord_end=p2,
        )

        if distance > max_dist:
            max_dist = distance
            elbow_idx = i

    return float(thresholds[elbow_idx])


def estimate_correlation_threshold(
    df: pd.DataFrame,
    threshold_step: float = 0.05,
    use_kneed: bool = True,
    save_path: str | None = None,
    corr_matrix: pd.DataFrame | None = None,

    output_format: str = "pdf",
) -> tuple[float, pd.DataFrame]:
    """Estimate a correlation threshold from one reusable correlation matrix.

    Args:
        df: DataFrame containing numerical features.
        threshold_step: Step size for the threshold grid search.
        use_kneed: Whether to use the kneed library for elbow detection when
            available.
        save_path: Optional output path for the elbow curve figure.
        corr_matrix: Optional precomputed Pearson correlation matrix.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple of the estimated threshold and the elbow curve DataFrame.

    Raises:
        ValueError: If ``threshold_step`` is not in (0, 1).
    """
    if not (0 < threshold_step < 1):
        raise ValueError("threshold_step must be in (0, 1).")

    if corr_matrix is None:
        _, corr_matrix = _compute_numeric_correlation(df)
    else:
        logger.info("[STEP 1.1] Reusing precomputed correlation matrix.")

    thresholds = np.arange(0.05, 1.0, threshold_step)
    vars_remaining: list[int] = []
    abs_corr = np.abs(corr_matrix.to_numpy(dtype=float))
    np.fill_diagonal(abs_corr, np.nan)

    for thresh in tqdm(
    thresholds,
    desc="Testing correlation thresholds",
    unit="threshold",):
        correlated_vars = np.nan_to_num(
            abs_corr >= thresh,
            nan=False,
        ).any(axis=1)
        vars_remaining.append(int((~correlated_vars).sum()))

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "vars_remaining": vars_remaining,
    })

    if use_kneed:
        try:
            from kneed import KneeLocator
            kl = KneeLocator(
                thresholds,
                vars_remaining,
                curve="convex",
                direction="increasing",
            )
            elbow_threshold = float(
                kl.elbow if kl.elbow is not None else thresholds[len(thresholds) // 2]
            )
        except ImportError:
            elbow_threshold = float(_manual_elbow(thresholds, vars_remaining))
    else:
        elbow_threshold = float(_manual_elbow(thresholds, vars_remaining))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, vars_remaining, "o-", linewidth=2, markersize=6, label="Curve")
    ax.axvline(
        x=elbow_threshold,
        linestyle="--",
        alpha=0.7,
        label=f"Estimated elbow (t={elbow_threshold:.2f})",
    )
    elbow_idx = int(np.argmin(np.abs(thresholds - elbow_threshold)))
    ax.plot(elbow_threshold, vars_remaining[elbow_idx], "*", markersize=16)
    ax.set_xlabel("Correlation threshold")
    ax.set_ylabel("Number of remaining variables")
    ax.set_title("Elbow curve — Correlation threshold selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _safe_save_figure(fig, save_path, dpi=150, output_format=output_format)
    plt.close(fig)
    del fig, ax
    gc.collect()

    logger.info("Estimated optimal threshold: %.2f", elbow_threshold)
    logger.info(
        "Remaining variables at this threshold: %d",
        vars_remaining[elbow_idx],
    )
    return elbow_threshold, results_df


def generate_correlation_analysis(
    df: pd.DataFrame,
    source_features: Sequence[str],
    static_features: Sequence[str] | None = None,
    output_folder: str = "comparison_figs",
    threshold_step: float = 0.05,
    use_kneed: bool = False,
    corr_threshold: float | None = None,
    max_cross_heatmaps: int | None = None,
    block_heatmap_dpi: int = 160,
    show_colorbar: bool = True,
    annotate_static_summary: bool = True,
    max_full_heatmap_features: int = 300,
    full_heatmap_dpi: int = 140,

    output_format: str = "pdf",
) -> dict[str, object]:
    """Run dual-mode correlation analysis with static-aware block output.

    Args:
        df: DataFrame containing numerical features.
        source_features: Original dynamic and static variable names.
        static_features: Optional list of static feature names.
        output_folder: Directory for output figures.
        threshold_step: Step size for the elbow threshold estimation.
        use_kneed: Whether to use the kneed library for elbow detection.
        corr_threshold: Optional fixed correlation threshold. Estimated from
            data when None.
        max_cross_heatmaps: Maximum number of temporal cross-blocks.
        block_heatmap_dpi: Image resolution for block heatmaps.
        show_colorbar: Whether to include colorbars.
        annotate_static_summary: Whether to annotate the static summary.
        max_full_heatmap_features: Maximum feature count for the full map.
        full_heatmap_dpi: Image resolution for the complete map.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        Dictionary containing ``optimal_threshold``, ``elbow_results``, and
        ``corr_matrix``.

    Raises:
        ValueError: If the final correlation threshold is not in (0, 1].
    """
    os.makedirs(output_folder, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Preparing reusable correlation matrix...")
    logger.info("=" * 60)

    static_set = set(static_features) if static_features else set()
    dynamic_cols = [col for col in df.columns if col not in static_set]

    logger.info(
        "[CORRELATION] Computing matrix for %d columns...",
        len(dynamic_cols),
    )

    numeric_df, corr_matrix_pd = _compute_numeric_correlation(df[dynamic_cols])

    logger.info(
        "[CORRELATION] Matrix computed: %d × %d",
        corr_matrix_pd.shape[0],
        corr_matrix_pd.shape[1],
    )

    logger.info("=" * 60)
    logger.info("Step 1: Estimating optimal correlation threshold...")
    logger.info("=" * 60)
    estimated_threshold, elbow_results = estimate_correlation_threshold(
        df=numeric_df,
        threshold_step=threshold_step,
        use_kneed=use_kneed,
        save_path=os.path.join(output_folder, "elbow_curve.png"),
        corr_matrix=corr_matrix_pd,
    output_format=output_format)

    final_threshold = (
        float(corr_threshold)
        if corr_threshold is not None
        else estimated_threshold
    )
    if not (0 < final_threshold <= 1):
        raise ValueError(
            "Correlation threshold must be in (0, 1], "
            f"got {final_threshold}."
        )

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Generating Normal_Mode and Threshold_Mode blocks...")
    logger.info("=" * 60)

    df_pl = pl.from_pandas(numeric_df, include_index=False)
    afficher_correlation_par_blocs(
        df=df_pl,
        source_features=source_features,
        static_features=static_features,
        threshold=final_threshold,
        output_dir=output_folder,
        corr_matrix=corr_matrix_pd,
        max_cross_heatmaps=max_cross_heatmaps,
        block_heatmap_dpi=block_heatmap_dpi,
        show_colorbar=show_colorbar,
        annotate_static_summary=annotate_static_summary,
    output_format=output_format)

    # Keep the global map only when its size remains safe. Produce both an
    # unmasked and a threshold-masked version in the corresponding folders.
    n_features = corr_matrix_pd.shape[0]
    if n_features <= max_full_heatmap_features:
        logger.info(
            "[GLOBAL MAP] %d features <= %d: rendering both modes.",
            n_features,
            max_full_heatmap_features,
        )
        for mode, threshold_value in (
            ("Normal_Mode", None),
            ("Threshold_Mode", final_threshold),
        ):
            matrix = corr_matrix_pd.to_numpy(
                dtype=np.float32, copy=True
            )
            if threshold_value is not None:
                matrix = np.ma.masked_where(
                    ~np.isfinite(matrix)
                    | (np.abs(matrix) < threshold_value),
                    matrix,
                )
            else:
                matrix = np.ma.masked_invalid(matrix)

            fig, ax = plt.subplots(figsize=(18, 16))
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.03,
                pad=0.02,
                ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            )
            colorbar.set_label("Pearson correlation")
            fig.subplots_adjust(
                left=0.01, right=0.93, bottom=0.01, top=0.99
            )
            _safe_save_figure(
                fig,
                os.path.join(
                    output_folder, mode, "corr_complete_map.png"
                ),
                dpi=full_heatmap_dpi,
            output_format=output_format)
            plt.close(fig)
            del fig, ax, image, matrix
            gc.collect()
    else:
        logger.info(
            "[GLOBAL MAP] Skipped: %d > %d.",
            n_features,
            max_full_heatmap_features,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("  Threshold used: %.2f", final_threshold)
    logger.info("  Normal maps: %s/Normal_Mode/", output_folder)
    logger.info("  Threshold maps: %s/Threshold_Mode/", output_folder)
    _log_memory("analysis complete")
    logger.info("=" * 60)

    return {
        "optimal_threshold": final_threshold,
        "elbow_results": elbow_results,
        "corr_matrix": pl.from_pandas(
            corr_matrix_pd, include_index=False
        ),
    }


# ============================================================================
# CROSS-FOLD BORUTA RESOLUTION
# ============================================================================

def estimate_boruta_frequency_threshold(
    fold_feature_lists: list[list[str]],
    n_folds: int,
    output_folder: str | None = None,

    output_format: str = "pdf",
) -> tuple[float, pd.DataFrame]:
    """Estimate the optimal Boruta frequency threshold by elbow analysis.

    Given the list of Boruta-selected features for each fold, this function
    computes how often each feature appears across all folds, then finds the
    optimal frequency threshold using the elbow method (maximum distance to
    the chord between the first and last points).

    Args:
        fold_feature_lists:
            A list of length ``n_folds``, where each element is the list of
            feature names selected by Boruta for that fold.
        n_folds:
            Total number of folds used in cross-validation.
        output_folder:
            Directory in which to save the elbow curve figure. If None, no
            figure is saved.

        output_format: Figure format: ``"pdf"`` (default) or ``"png"`` at 300 DPI.

    Returns:
        A tuple containing:

        - The optimal frequency threshold (value in [0, 1]).
        - A DataFrame with columns ``frequency``, ``vars_remaining``, and
          ``n_folds`` for each threshold step.
    """
    if len(fold_feature_lists) != n_folds:
        raise ValueError(
            f"Expected {n_folds} fold feature lists, "
            f"got {len(fold_feature_lists)}."
        )

    # Count how many folds each feature was selected in.
    feature_counts: dict[str, int] = {}
    for fold_features in fold_feature_lists:
        for feature in fold_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    if not feature_counts:
        raise ValueError(
            "No feature was selected by Boruta in any fold."
        )

    # Build frequency bins: thresholds from 1/n_folds to 1.0.
    threshold_step = 1.0 / n_folds
    thresholds = np.arange(threshold_step, 1.0 + threshold_step, threshold_step)

    # For each threshold, count how many features have frequency >= threshold.
    vars_remaining: list[int] = []
    for thresh in thresholds:
        count = sum(
            1 for freq in feature_counts.values() if freq / n_folds >= thresh
        )
        vars_remaining.append(count)

    results_df = pd.DataFrame({
        "frequency": thresholds,
        "vars_remaining": vars_remaining,
    })

    # Elbow detection: maximum perpendicular distance to the chord.
    p1 = np.array([thresholds[0], vars_remaining[0]])
    p2 = np.array([thresholds[-1], vars_remaining[-1]])
    max_dist, elbow_idx = 0.0, 0

    for i in range(len(thresholds)):
        p = np.array([thresholds[i], vars_remaining[i]])
        dist = _distance_to_chord_2d(
            point=p,
            chord_start=p1,
            chord_end=p2,
        )
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    elbow_threshold = float(thresholds[elbow_idx])

    # Save the elbow curve figure.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            thresholds, vars_remaining, "o-",
            linewidth=2, markersize=6, label="Curve",
        )
        ax.axvline(
            x=elbow_threshold,
            linestyle="--",
            alpha=0.7,
            label=f"Elbow (freq={elbow_threshold:.2f})",
        )
        ax.plot(
            elbow_threshold, vars_remaining[elbow_idx],
            "*", markersize=16,
        )
        ax.set_xlabel("Boruta frequency threshold")
        ax.set_ylabel("Number of remaining features")
        ax.set_title(
            "Elbow curve — Boruta cross-fold frequency selection"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_path = os.path.join(output_folder, "boruta_frequency_elbow.png")
        _safe_save_figure(fig, save_path, dpi=150, output_format=output_format)
        plt.close(fig)

    logger.debug(
        "BORUTA FREQ: Elbow threshold=%.2f (%.0f/%d folds), %d features retained.",
        elbow_threshold,
        elbow_threshold * n_folds,
        n_folds,
        vars_remaining[elbow_idx],
    )

    return elbow_threshold, results_df


