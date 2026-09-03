"""Tests for persistent feature-removal provenance."""

from utilitaries.feature_trace import load_feature_trace, record_removed_features


def test_feature_trace_preserves_the_first_removal_stage(tmp_path) -> None:
    """Prevent later stages from overwriting an earlier removal cause."""
    trace_path = tmp_path / "feature_selection_trace.json"

    record_removed_features(
        trace_path,
        "zero_variance",
        ["heart_rate_Mean"],
        reset=True,
    )
    record_removed_features(
        trace_path,
        "correlation",
        ["heart_rate_Mean", "pao2_Median"],
    )
    record_removed_features(
        trace_path,
        "boruta",
        ["heart_rate_Mean", "pao2_Median", "temp_Variance"],
    )

    assert load_feature_trace(trace_path) == {
        "heart_rate_Mean": "zero_variance",
        "pao2_Median": "correlation",
        "temp_Variance": "boruta",
    }
