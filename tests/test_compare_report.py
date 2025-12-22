"""
Tests for comparison report generation.
"""

import json

import pytest

from prompt_evaluator.reporting.compare_report import (
    CompareReportConfig,
    format_delta_sign,
    format_percentage,
    load_comparison_artifact,
    render_comparison_metadata_section,
    render_comparison_report,
    render_improvements_section,
    render_markdown_comparison_report,
    render_regressions_section,
    render_suite_comparison_table,
    render_suite_flags_comparison_table,
)


@pytest.fixture
def sample_comparison_artifact():
    """Create a sample comparison artifact for testing."""
    return {
        "baseline_run_id": "baseline-123",
        "candidate_run_id": "candidate-456",
        "baseline_prompt_version": "v1.0",
        "candidate_prompt_version": "v2.0",
        "comparison_timestamp": "2025-12-22T10:00:00.000000+00:00",
        "has_regressions": True,
        "regression_count": 2,
        "thresholds_config": {
            "metric_threshold": 0.1,
            "flag_threshold": 0.05,
        },
        "metric_deltas": [
            {
                "metric_name": "semantic_fidelity",
                "baseline_mean": 4.0,
                "candidate_mean": 4.3,
                "delta": 0.3,
                "percent_change": 7.5,
                "is_regression": False,
                "threshold_used": 0.1,
            },
            {
                "metric_name": "clarity",
                "baseline_mean": 4.2,
                "candidate_mean": 3.8,
                "delta": -0.4,
                "percent_change": -9.52,
                "is_regression": True,
                "threshold_used": 0.1,
            },
            {
                "metric_name": "new_metric",
                "baseline_mean": None,
                "candidate_mean": 4.0,
                "delta": None,
                "percent_change": None,
                "is_regression": False,
                "threshold_used": 0.1,
            },
        ],
        "flag_deltas": [
            {
                "flag_name": "invented_constraints",
                "baseline_proportion": 0.1,
                "candidate_proportion": 0.05,
                "delta": -0.05,
                "percent_change": -50.0,
                "is_regression": False,
                "threshold_used": 0.05,
            },
            {
                "flag_name": "omitted_constraints",
                "baseline_proportion": 0.05,
                "candidate_proportion": 0.12,
                "delta": 0.07,
                "percent_change": 140.0,
                "is_regression": True,
                "threshold_used": 0.05,
            },
        ],
    }


class TestLoadComparisonArtifact:
    """Tests for loading comparison artifacts."""

    def test_load_valid_artifact(self, tmp_path, sample_comparison_artifact):
        """Test loading a valid comparison artifact file."""
        # Create artifact file
        artifact_file = tmp_path / "comparison.json"
        artifact_file.write_text(json.dumps(sample_comparison_artifact), encoding="utf-8")

        # Load artifact
        loaded = load_comparison_artifact(artifact_file)
        assert loaded["baseline_run_id"] == "baseline-123"
        assert loaded["candidate_run_id"] == "candidate-456"

    def test_load_missing_artifact(self, tmp_path):
        """Test loading from missing file."""
        with pytest.raises(FileNotFoundError, match="Comparison artifact not found"):
            load_comparison_artifact(tmp_path / "missing.json")

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        artifact_file = tmp_path / "invalid.json"
        artifact_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_comparison_artifact(artifact_file)

    def test_load_missing_required_fields(self, tmp_path):
        """Test loading artifact without required fields."""
        artifact_file = tmp_path / "incomplete.json"
        artifact_file.write_text(json.dumps({"some_field": "value"}), encoding="utf-8")

        with pytest.raises(ValueError, match="missing required fields"):
            load_comparison_artifact(artifact_file)


class TestFormatHelpers:
    """Tests for formatting helper functions."""

    def test_format_delta_sign_positive(self):
        """Test formatting positive delta."""
        assert format_delta_sign(0.5) == "+0.500"

    def test_format_delta_sign_negative(self):
        """Test formatting negative delta."""
        assert format_delta_sign(-0.3) == "-0.300"

    def test_format_delta_sign_zero(self):
        """Test formatting zero delta."""
        assert format_delta_sign(0.0) == "0.000"

    def test_format_delta_sign_none(self):
        """Test formatting None delta."""
        assert format_delta_sign(None) == "N/A"

    def test_format_percentage_positive(self):
        """Test formatting positive percentage."""
        assert format_percentage(12.5) == "+12.50%"

    def test_format_percentage_negative(self):
        """Test formatting negative percentage."""
        assert format_percentage(-5.0) == "-5.00%"

    def test_format_percentage_none(self):
        """Test formatting None percentage."""
        assert format_percentage(None) == "N/A"

    def test_format_percentage_infinity(self):
        """Test formatting infinity percentage."""
        assert format_percentage(float("inf")) == "∞"
        assert format_percentage(float("-inf")) == "-∞"


class TestRenderMetadataSection:
    """Tests for rendering comparison metadata section."""

    def test_render_metadata_with_all_fields(self, sample_comparison_artifact):
        """Test rendering metadata with all fields present."""
        section = render_comparison_metadata_section(sample_comparison_artifact)

        assert "# Run Comparison Report" in section
        assert "baseline-123" in section
        assert "candidate-456" in section
        assert "v1.0" in section
        assert "v2.0" in section
        assert "2025-12-22T10:00:00.000000+00:00" in section
        assert "0.1" in section  # metric threshold
        assert "0.05" in section  # flag threshold
        assert "2 regression(s) detected" in section

    def test_render_metadata_no_regressions(self, sample_comparison_artifact):
        """Test rendering metadata when no regressions."""
        sample_comparison_artifact["has_regressions"] = False
        sample_comparison_artifact["regression_count"] = 0

        section = render_comparison_metadata_section(sample_comparison_artifact)
        assert "No regressions detected" in section

    def test_render_metadata_missing_prompt_versions(self, sample_comparison_artifact):
        """Test rendering metadata without prompt versions."""
        del sample_comparison_artifact["baseline_prompt_version"]
        del sample_comparison_artifact["candidate_prompt_version"]

        section = render_comparison_metadata_section(sample_comparison_artifact)
        assert "baseline-123" in section
        # Should not have prompt version lines if both are missing


class TestRenderSuiteComparisonTable:
    """Tests for rendering suite comparison tables."""

    def test_render_metrics_table(self, sample_comparison_artifact):
        """Test rendering metrics comparison table."""
        table = render_suite_comparison_table(sample_comparison_artifact)

        assert "## Suite-Level Metrics Comparison" in table
        assert "semantic_fidelity" in table
        assert "clarity" in table
        assert "new_metric" in table
        assert "4.00" in table  # baseline
        assert "4.30" in table  # candidate
        assert "+0.300" in table  # delta
        assert "Improved" in table
        assert "Regressed" in table
        assert "N/A" in table  # for new_metric

    def test_render_metrics_table_empty(self):
        """Test rendering empty metrics table."""
        data = {
            "metric_deltas": [],
        }
        table = render_suite_comparison_table(data)
        assert "*No metrics available*" in table

    def test_render_flags_table(self, sample_comparison_artifact):
        """Test rendering flags comparison table."""
        table = render_suite_flags_comparison_table(sample_comparison_artifact)

        assert "## Suite-Level Flags Comparison" in table
        assert "invented_constraints" in table
        assert "omitted_constraints" in table
        assert "10.0%" in table  # baseline proportion
        assert "5.0%" in table  # candidate proportion
        assert "pp" in table  # percentage points
        assert "Improved" in table
        assert "Regressed" in table

    def test_render_flags_table_empty(self):
        """Test rendering empty flags table."""
        data = {
            "flag_deltas": [],
        }
        table = render_suite_flags_comparison_table(data)
        assert "*No flags available*" in table


class TestRenderRegressionsSection:
    """Tests for rendering regressions section."""

    def test_render_regressions_with_both_types(self, sample_comparison_artifact):
        """Test rendering regressions with both metrics and flags."""
        section = render_regressions_section(sample_comparison_artifact)

        assert "## Regressions Detected" in section
        assert "### Regressed Metrics" in section
        assert "clarity" in section
        assert "-0.400" in section
        assert "### Regressed Flags" in section
        assert "omitted_constraints" in section

    def test_render_regressions_none(self, sample_comparison_artifact):
        """Test rendering regressions when none present."""
        # Remove regressions
        for delta in sample_comparison_artifact["metric_deltas"]:
            delta["is_regression"] = False
        for delta in sample_comparison_artifact["flag_deltas"]:
            delta["is_regression"] = False

        section = render_regressions_section(sample_comparison_artifact)
        assert "*No regressions detected" in section

    def test_render_regressions_sorting(self):
        """Test that regressions are sorted by severity."""
        data = {
            "metric_deltas": [
                {
                    "metric_name": "metric_a",
                    "baseline_mean": 4.0,
                    "candidate_mean": 3.9,
                    "delta": -0.1,
                    "percent_change": -2.5,
                    "is_regression": True,
                },
                {
                    "metric_name": "metric_b",
                    "baseline_mean": 4.0,
                    "candidate_mean": 3.0,
                    "delta": -1.0,
                    "percent_change": -25.0,
                    "is_regression": True,
                },
            ],
            "flag_deltas": [],
        }

        section = render_regressions_section(data)
        # metric_b should appear before metric_a (larger delta)
        metric_b_pos = section.find("metric_b")
        metric_a_pos = section.find("metric_a")
        assert metric_b_pos < metric_a_pos


class TestRenderImprovementsSection:
    """Tests for rendering improvements section."""

    def test_render_improvements_with_both_types(self, sample_comparison_artifact):
        """Test rendering improvements with both metrics and flags."""
        section = render_improvements_section(sample_comparison_artifact)

        assert "## Improvements Detected" in section
        assert "### Improved Metrics" in section
        assert "semantic_fidelity" in section
        assert "+0.300" in section
        assert "### Improved Flags" in section
        assert "invented_constraints" in section

    def test_render_improvements_none(self):
        """Test rendering improvements when none present."""
        data = {
            "metric_deltas": [
                {
                    "metric_name": "metric_a",
                    "baseline_mean": 4.0,
                    "candidate_mean": 3.8,
                    "delta": -0.2,
                    "percent_change": -5.0,
                    "is_regression": True,
                },
            ],
            "flag_deltas": [
                {
                    "flag_name": "flag_a",
                    "baseline_proportion": 0.1,
                    "candidate_proportion": 0.15,
                    "delta": 0.05,
                    "percent_change": 50.0,
                    "is_regression": False,
                },
            ],
        }

        section = render_improvements_section(data)
        assert "*No improvements detected" in section


class TestRenderFullReport:
    """Tests for rendering complete Markdown report."""

    def test_render_complete_report(self, sample_comparison_artifact):
        """Test rendering a complete comparison report."""
        config = CompareReportConfig()
        report = render_markdown_comparison_report(sample_comparison_artifact, config)

        # Check for all major sections
        assert "# Run Comparison Report" in report
        assert "## Comparison Metadata" in report
        assert "## Suite-Level Metrics Comparison" in report
        assert "## Suite-Level Flags Comparison" in report
        assert "## Regressions Detected" in report
        assert "## Improvements Detected" in report

        # Check for key data
        assert "baseline-123" in report
        assert "candidate-456" in report
        assert "semantic_fidelity" in report
        assert "clarity" in report
        assert "invented_constraints" in report


class TestRenderComparisonReport:
    """Tests for the main render_comparison_report function."""

    def test_render_report_creates_markdown(self, tmp_path, sample_comparison_artifact):
        """Test that render_comparison_report creates a Markdown file."""
        # Create artifact
        artifact_file = tmp_path / "comparison.json"
        artifact_file.write_text(json.dumps(sample_comparison_artifact), encoding="utf-8")

        # Render report
        report_path = render_comparison_report(
            comparison_artifact_path=artifact_file,
            top_cases_per_metric=5,
        )

        # Check file was created
        assert report_path.exists()
        assert report_path.name == "comparison_report.md"

        # Check content
        content = report_path.read_text(encoding="utf-8")
        assert "baseline-123" in content
        assert "candidate-456" in content
        assert "semantic_fidelity" in content

    def test_render_report_custom_output_name(self, tmp_path, sample_comparison_artifact):
        """Test custom output filename."""
        artifact_file = tmp_path / "comparison.json"
        artifact_file.write_text(json.dumps(sample_comparison_artifact), encoding="utf-8")

        report_path = render_comparison_report(
            comparison_artifact_path=artifact_file,
            output_name="custom_comparison.md",
        )

        assert report_path.name == "custom_comparison.md"
        assert report_path.exists()

    def test_render_report_missing_file(self, tmp_path):
        """Test error when comparison artifact doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Comparison artifact not found"):
            render_comparison_report(comparison_artifact_path=tmp_path / "missing.json")

    def test_render_report_not_a_file(self, tmp_path):
        """Test error when path is not a file."""
        with pytest.raises(ValueError, match="not a file"):
            render_comparison_report(comparison_artifact_path=tmp_path)

    def test_render_with_html_option(self, tmp_path, sample_comparison_artifact):
        """Test HTML generation option."""
        artifact_file = tmp_path / "comparison.json"
        artifact_file.write_text(json.dumps(sample_comparison_artifact), encoding="utf-8")

        # Try to render with HTML (may skip if markdown library not available)
        report_path = render_comparison_report(
            comparison_artifact_path=artifact_file,
            generate_html=True,
            html_output_name="test_comparison.html",
        )

        # Markdown should always be created
        assert report_path.exists()

        # HTML may or may not exist depending on markdown library availability
        html_path = tmp_path / "test_comparison.html"
        if html_path.exists():
            html_content = html_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in html_content
            assert "baseline-123" in html_content


class TestEdgeCases:
    """Tests for edge cases in comparison reporting."""

    def test_metrics_with_null_values(self):
        """Test handling metrics with null baseline or candidate."""
        data = {
            "baseline_run_id": "baseline-123",
            "candidate_run_id": "candidate-456",
            "comparison_timestamp": "2025-12-22T10:00:00Z",
            "has_regressions": False,
            "regression_count": 0,
            "thresholds_config": {"metric_threshold": 0.1, "flag_threshold": 0.05},
            "metric_deltas": [
                {
                    "metric_name": "new_metric",
                    "baseline_mean": None,
                    "candidate_mean": 4.0,
                    "delta": None,
                    "percent_change": None,
                    "is_regression": False,
                    "threshold_used": 0.1,
                },
                {
                    "metric_name": "removed_metric",
                    "baseline_mean": 3.5,
                    "candidate_mean": None,
                    "delta": None,
                    "percent_change": None,
                    "is_regression": False,
                    "threshold_used": 0.1,
                },
            ],
            "flag_deltas": [],
        }

        config = CompareReportConfig()
        report = render_markdown_comparison_report(data, config)

        # Check that N/A is shown for missing values
        assert "N/A" in report
        assert "new_metric" in report
        assert "removed_metric" in report

    def test_flags_with_zero_proportions(self):
        """Test handling flags with zero proportions."""
        data = {
            "baseline_run_id": "baseline-123",
            "candidate_run_id": "candidate-456",
            "comparison_timestamp": "2025-12-22T10:00:00Z",
            "has_regressions": False,
            "regression_count": 0,
            "thresholds_config": {"metric_threshold": 0.1, "flag_threshold": 0.05},
            "metric_deltas": [],
            "flag_deltas": [
                {
                    "flag_name": "all_false_flag",
                    "baseline_proportion": 0.0,
                    "candidate_proportion": 0.0,
                    "delta": 0.0,
                    "percent_change": None,
                    "is_regression": False,
                    "threshold_used": 0.05,
                },
            ],
        }

        config = CompareReportConfig()
        report = render_markdown_comparison_report(data, config)

        assert "all_false_flag" in report
        assert "0.0%" in report

    def test_empty_deltas(self):
        """Test handling empty metric and flag deltas."""
        data = {
            "baseline_run_id": "baseline-123",
            "candidate_run_id": "candidate-456",
            "comparison_timestamp": "2025-12-22T10:00:00Z",
            "has_regressions": False,
            "regression_count": 0,
            "thresholds_config": {"metric_threshold": 0.1, "flag_threshold": 0.05},
            "metric_deltas": [],
            "flag_deltas": [],
        }

        config = CompareReportConfig()
        report = render_markdown_comparison_report(data, config)

        assert "*No metrics available*" in report
        assert "*No flags available*" in report
        assert "*No regressions detected" in report
        assert "*No improvements detected" in report
