# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for run report generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from prompt_evaluator.reporting.run_report import (
    ReportConfig,
    identify_unstable_metrics,
    identify_weak_metrics,
    load_run_artifact,
    render_markdown_report,
    render_run_report,
    select_qualitative_samples,
    truncate_text,
)


@pytest.fixture
def sample_run_artifact():
    """Create a sample run artifact for testing."""
    return {
        "run_id": "test-run-123",
        "status": "completed",
        "timestamp_start": "2025-12-21T10:00:00.000000+00:00",
        "timestamp_end": "2025-12-21T10:15:00.000000+00:00",
        "dataset_path": "/path/to/dataset.yaml",
        "dataset_hash": "abc123" * 10,
        "dataset_count": 3,
        "num_samples_per_case": 5,
        "system_prompt_path": "/path/to/system_prompt.txt",
        "prompt_version_id": "v1.0",
        "prompt_hash": "def456" * 10,
        "run_notes": "Test run notes",
        "generator_config": {"model_name": "gpt-4", "temperature": 0.7},
        "judge_config": {"model_name": "gpt-4", "temperature": 0.0},
        "rubric_metadata": {
            "rubric_path": "/path/to/rubric.yaml",
            "rubric_definition": {
                "metrics": [
                    {"name": "semantic_fidelity", "min_score": 1, "max_score": 5},
                    {"name": "clarity", "min_score": 1, "max_score": 5},
                ],
                "flags": [
                    {"name": "invented_constraints", "default": False},
                ],
            },
        },
        "overall_metric_stats": {
            "semantic_fidelity": {
                "mean_of_means": 4.2,
                "std_of_means": 0.3,
                "min_of_means": 3.8,
                "max_of_means": 4.5,
                "num_cases": 3,
            },
            "clarity": {
                "mean_of_means": 4.0,
                "std_of_means": 0.5,
                "min_of_means": 3.5,
                "max_of_means": 4.5,
                "num_cases": 3,
            },
        },
        "overall_flag_stats": {
            "invented_constraints": {
                "true_count": 2,
                "false_count": 13,
                "total_count": 15,
                "true_proportion": 0.1333,
            },
        },
        "test_case_results": [
            {
                "test_case_id": "test-001",
                "test_case_input": "Explain Python",
                "test_case_metadata": {"task": "explanation"},
                "num_samples": 5,
                "status": "completed",
                "per_metric_stats": {
                    "semantic_fidelity": {
                        "mean": 4.5,
                        "std": 0.2,
                        "min": 4.0,
                        "max": 5.0,
                        "count": 5,
                    },
                    "clarity": {
                        "mean": 4.5,
                        "std": 0.3,
                        "min": 4.0,
                        "max": 5.0,
                        "count": 5,
                    },
                },
                "per_flag_stats": {
                    "invented_constraints": {
                        "true_count": 0,
                        "false_count": 5,
                        "total_count": 5,
                        "true_proportion": 0.0,
                    },
                },
                "samples": [
                    {
                        "sample_id": "test-run-123-test-001-sample-1",
                        "input_text": "Explain Python",
                        "generator_output": "Python is a programming language.",
                        "status": "completed",
                        "judge_metrics": {
                            "semantic_fidelity": {"score": 4.5, "rationale": "Good"},
                            "clarity": {"score": 4.5, "rationale": "Clear"},
                        },
                        "judge_flags": {"invented_constraints": False},
                    },
                ],
            },
            {
                "test_case_id": "test-002",
                "test_case_input": "Describe JavaScript",
                "test_case_metadata": {"task": "explanation"},
                "num_samples": 5,
                "status": "completed",
                "per_metric_stats": {
                    "semantic_fidelity": {
                        "mean": 4.0,
                        "std": 1.5,  # High std - should be marked unstable
                        "min": 2.0,
                        "max": 5.0,
                        "count": 5,
                    },
                    "clarity": {
                        "mean": 3.5,
                        "std": 0.4,
                        "min": 3.0,
                        "max": 4.0,
                        "count": 5,
                    },
                },
                "per_flag_stats": {
                    "invented_constraints": {
                        "true_count": 1,
                        "false_count": 4,
                        "total_count": 5,
                        "true_proportion": 0.2,
                    },
                },
                "samples": [
                    {
                        "sample_id": "test-run-123-test-002-sample-1",
                        "input_text": "Describe JavaScript",
                        "generator_output": "JavaScript is used for web development.",
                        "status": "completed",
                        "judge_metrics": {
                            "semantic_fidelity": {"score": 4.0, "rationale": "Okay"},
                            "clarity": {"score": 3.5, "rationale": "Acceptable"},
                        },
                        "judge_flags": {"invented_constraints": False},
                    },
                ],
            },
            {
                "test_case_id": "test-003",
                "test_case_input": "What is Rust?",
                "test_case_metadata": {"task": "explanation"},
                "num_samples": 5,
                "status": "completed",
                "per_metric_stats": {
                    "semantic_fidelity": {
                        "mean": 2.5,  # Low mean - should be marked weak
                        "std": 0.3,
                        "min": 2.0,
                        "max": 3.0,
                        "count": 5,
                    },
                    "clarity": {
                        "mean": 2.8,  # Low mean - should be marked weak
                        "std": 0.2,
                        "min": 2.5,
                        "max": 3.0,
                        "count": 5,
                    },
                },
                "per_flag_stats": {
                    "invented_constraints": {
                        "true_count": 1,
                        "false_count": 4,
                        "total_count": 5,
                        "true_proportion": 0.2,
                    },
                },
                "samples": [
                    {
                        "sample_id": "test-run-123-test-003-sample-1",
                        "input_text": "What is Rust?",
                        "generator_output": "Rust is a language.",
                        "status": "completed",
                        "judge_metrics": {
                            "semantic_fidelity": {"score": 2.5, "rationale": "Too brief"},
                            "clarity": {"score": 2.8, "rationale": "Lacking detail"},
                        },
                        "judge_flags": {"invented_constraints": False},
                    },
                ],
            },
        ],
    }


class TestLoadRunArtifact:
    """Tests for loading run artifacts."""

    def test_load_valid_artifact(self, tmp_path, sample_run_artifact):
        """Test loading a valid artifact file."""
        # Create artifact file
        artifact_file = tmp_path / "dataset_evaluation.json"
        artifact_file.write_text(json.dumps(sample_run_artifact), encoding="utf-8")

        # Load artifact
        loaded = load_run_artifact(tmp_path)
        assert loaded["run_id"] == "test-run-123"
        assert loaded["status"] == "completed"

    def test_load_missing_artifact(self, tmp_path):
        """Test loading from directory without artifact."""
        with pytest.raises(FileNotFoundError, match="Run artifact not found"):
            load_run_artifact(tmp_path)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        artifact_file = tmp_path / "dataset_evaluation.json"
        artifact_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_run_artifact(tmp_path)


class TestIdentifyUnstableMetrics:
    """Tests for identifying unstable metrics."""

    def test_identify_unstable_with_high_std(self, sample_run_artifact):
        """Test identifying metrics with high standard deviation."""
        test_case_results = sample_run_artifact["test_case_results"]
        unstable = identify_unstable_metrics(test_case_results, std_threshold=1.0)

        # test-002 has semantic_fidelity with std=1.5, should be flagged
        assert "test-002" in unstable
        assert "semantic_fidelity" in unstable["test-002"]

    def test_no_unstable_metrics(self, sample_run_artifact):
        """Test when no metrics exceed threshold."""
        test_case_results = [sample_run_artifact["test_case_results"][0]]  # Only test-001
        unstable = identify_unstable_metrics(test_case_results, std_threshold=1.0)

        # test-001 has low std values, should not be flagged
        assert "test-001" not in unstable

    def test_single_sample_not_unstable(self):
        """Test that single-sample cases are never marked unstable."""
        test_case_results = [
            {
                "test_case_id": "single-sample",
                "per_metric_stats": {
                    "metric1": {
                        "mean": 3.0,
                        "std": 0.0,  # No variance with single sample
                        "count": 1,
                    },
                },
            },
        ]

        unstable = identify_unstable_metrics(test_case_results, std_threshold=1.0)
        assert "single-sample" not in unstable


class TestIdentifyWeakMetrics:
    """Tests for identifying weak metrics."""

    def test_identify_weak_with_low_mean(self, sample_run_artifact):
        """Test identifying metrics with low mean scores."""
        test_case_results = sample_run_artifact["test_case_results"]
        weak = identify_weak_metrics(test_case_results, weak_threshold=3.0)

        # test-003 has means below 3.0, should be flagged
        assert "test-003" in weak
        assert "semantic_fidelity" in weak["test-003"]
        assert "clarity" in weak["test-003"]

    def test_no_weak_metrics(self, sample_run_artifact):
        """Test when no metrics below threshold."""
        test_case_results = [sample_run_artifact["test_case_results"][0]]  # Only test-001
        weak = identify_weak_metrics(test_case_results, weak_threshold=3.0)

        # test-001 has high means, should not be flagged
        assert "test-001" not in weak


class TestSelectQualitativeSamples:
    """Tests for selecting qualitative samples."""

    def test_select_worst_cases(self, sample_run_artifact):
        """Test selecting worst-performing cases."""
        samples = select_qualitative_samples(
            sample_run_artifact, count=2, fallback_metric="semantic_fidelity"
        )

        # Should return test-003 (mean=2.5) and test-002 (mean=4.0), worst first
        assert len(samples) == 2
        assert samples[0]["test_case_id"] == "test-003"
        assert samples[1]["test_case_id"] == "test-002"

    def test_select_limited_samples(self, sample_run_artifact):
        """Test selecting when count exceeds available."""
        samples = select_qualitative_samples(
            sample_run_artifact, count=10, fallback_metric="semantic_fidelity"
        )

        # Should return all 3 cases without duplication
        assert len(samples) == 3

    def test_select_with_no_metrics(self):
        """Test selection when no metric data available."""
        run_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {},  # No metrics
                },
            ],
        }

        samples = select_qualitative_samples(run_data, count=1)
        # Should return empty list when no scores available
        assert len(samples) == 0


class TestTruncateText:
    """Tests for text truncation."""

    def test_truncate_long_text(self):
        """Test truncating text that exceeds max length."""
        text = "a" * 100
        truncated = truncate_text(text, max_length=50)
        assert len(truncated) == 50
        assert truncated.endswith("...")

    def test_no_truncation_needed(self):
        """Test text within max length."""
        text = "short text"
        truncated = truncate_text(text, max_length=50)
        assert truncated == text


class TestRenderMarkdownReport:
    """Tests for Markdown report rendering."""

    def test_render_complete_report(self, sample_run_artifact):
        """Test rendering a complete report."""
        config = ReportConfig()
        report = render_markdown_report(sample_run_artifact, config)

        # Check for key sections
        assert "# Evaluation Run Report" in report
        assert "## Metadata" in report
        assert "## Suite-Level Metrics" in report
        assert "## Suite-Level Flags" in report
        assert "## Per-Test-Case Summary" in report
        assert "## Qualitative Examples" in report

        # Check for run ID
        assert "test-run-123" in report

        # Check for metric names
        assert "semantic_fidelity" in report
        assert "clarity" in report

        # Check for unstable annotation (test-002)
        assert "UNSTABLE" in report

        # Check for weak annotation (test-003)
        assert "WEAK" in report

    def test_report_contains_metadata(self, sample_run_artifact):
        """Test that report contains expected metadata."""
        config = ReportConfig()
        report = render_markdown_report(sample_run_artifact, config)

        # Check metadata fields
        assert "test-run-123" in report
        assert "completed" in report
        assert "v1.0" in report
        assert "gpt-4" in report

    def test_report_with_no_test_cases(self):
        """Test rendering report with empty test cases."""
        run_data = {
            "run_id": "empty-run",
            "status": "completed",
            "timestamp_start": "2025-12-21T10:00:00.000000+00:00",
            "timestamp_end": "2025-12-21T10:00:00.000000+00:00",
            "dataset_path": "/path/to/dataset.yaml",
            "dataset_count": 0,
            "num_samples_per_case": 0,
            "generator_config": {"model_name": "gpt-4"},
            "judge_config": {"model_name": "gpt-4"},
            "test_case_results": [],
            "overall_metric_stats": {},
            "overall_flag_stats": {},
        }

        config = ReportConfig()
        report = render_markdown_report(run_data, config)

        assert "empty-run" in report
        assert "*No test cases*" in report


class TestRenderRunReport:
    """Tests for the main render_run_report function."""

    def test_render_report_creates_markdown(self, tmp_path, sample_run_artifact):
        """Test that render_run_report creates a Markdown file."""
        # Create artifact
        artifact_file = tmp_path / "dataset_evaluation.json"
        artifact_file.write_text(json.dumps(sample_run_artifact), encoding="utf-8")

        # Render report
        report_path = render_run_report(
            run_dir=tmp_path,
            std_threshold=1.0,
            weak_score_threshold=3.0,
            qualitative_sample_count=2,
        )

        # Check file was created
        assert report_path.exists()
        assert report_path.name == "report.md"

        # Check content
        content = report_path.read_text(encoding="utf-8")
        assert "test-run-123" in content
        assert "semantic_fidelity" in content

    def test_render_report_custom_output_name(self, tmp_path, sample_run_artifact):
        """Test custom output filename."""
        artifact_file = tmp_path / "dataset_evaluation.json"
        artifact_file.write_text(json.dumps(sample_run_artifact), encoding="utf-8")

        report_path = render_run_report(
            run_dir=tmp_path,
            output_name="custom_report.md",
        )

        assert report_path.name == "custom_report.md"
        assert report_path.exists()

    def test_render_report_missing_directory(self):
        """Test error when run directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            render_run_report(run_dir=Path("/nonexistent/path"))

    def test_render_report_not_a_directory(self, tmp_path):
        """Test error when path is not a directory."""
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(ValueError, match="not a directory"):
            render_run_report(run_dir=file_path)

    def test_render_with_html_option(self, tmp_path, sample_run_artifact):
        """Test HTML generation option."""
        artifact_file = tmp_path / "dataset_evaluation.json"
        artifact_file.write_text(json.dumps(sample_run_artifact), encoding="utf-8")

        # Try to render with HTML (may skip if markdown library not available)
        report_path = render_run_report(
            run_dir=tmp_path,
            generate_html=True,
            html_output_name="test_report.html",
        )

        # Markdown should always be created
        assert report_path.exists()

        # HTML may or may not exist depending on markdown library availability
        html_path = tmp_path / "test_report.html"
        if html_path.exists():
            html_content = html_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in html_content
            assert "test-run-123" in html_content
