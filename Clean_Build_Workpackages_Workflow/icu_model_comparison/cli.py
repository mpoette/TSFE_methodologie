"""Console entry point for the complete experiment pipeline."""

import runpy


def main() -> None:
    """Execute the historical pipeline module with the current CLI arguments."""
    runpy.run_module(
        "ICU_Models_Comparison",
        run_name="__main__",
    )
