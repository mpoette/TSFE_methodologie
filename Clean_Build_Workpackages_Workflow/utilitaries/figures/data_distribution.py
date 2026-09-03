"""Data-sampling distribution visualizations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_offset_distribution(
    offsets: np.ndarray,
    max_hour: int,
    distribution_name: str,
) -> None:
    """Plot the central 90% of sampled relative offsets.

    Args:
        offsets: Sampled relative offsets.
        max_hour: Sanctuary period in hours.
        distribution_name: Name of the sampling distribution.
    """
    lower_quantile, upper_quantile = np.quantile(offsets, [0.05, 0.95])
    filtered_offsets = offsets[
        (offsets >= lower_quantile) & (offsets <= upper_quantile)
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_offsets, bins=100)
    plt.title(
        "Relative offset distribution "
        f"(sanctuary period: {max_hour} hours)\n"
        f"Distribution: {distribution_name}"
    )
    plt.xlabel("Offset")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
