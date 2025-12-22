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
Tests for the compare-runs CLI command.

Tests validate CLI parameter parsing, artifact loading, comparison logic,
and output formatting functionality.
"""

import json

import pytest
from typer.testing import CliRunner

from prompt_evaluator.cli import app


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner(mix_stderr=False)


@pytest.fixture
def sample_artifacts(tmp_path):
    """Fixture to create sample baseline and candidate run artifacts."""
    baseline_data = {
        "run_id": "baseline-run-123",
        "prompt_version_id": "v1.0",
        "overall_metric_stats": {
            "semantic_fidelity": {"mean_of_means": 4.0, "min_of_means": 3.5, "max_of_means": 4.5},
            "clarity": {"mean_of_means": 4.2, "min_of_means": 4.0, "max_of_means": 4.5},
        },
        "overall_flag_stats": {
            "invented_constraints": {
                "true_count": 2,
                "false_count": 18,
                "total_count": 20,
                "true_proportion": 0.1,
            },
            "omitted_constraints": {
                "true_count": 1,
                "false_count": 19,
                "total_count": 20,
                "true_proportion": 0.05,
            },
        },
    }

    candidate_data = {
        "run_id": "candidate-run-456",
        "prompt_version_id": "v2.0",
        "overall_metric_stats": {
            "semantic_fidelity": {"mean_of_means": 4.3, "min_of_means": 3.8, "max_of_means": 4.7},
            "clarity": {"mean_of_means": 4.5, "min_of_means": 4.2, "max_of_means": 4.8},
        },
        "overall_flag_stats": {
            "invented_constraints": {
                "true_count": 1,
                "false_count": 19,
                "total_count": 20,
                "true_proportion": 0.05,
            },
            "omitted_constraints": {
                "true_count": 0,
                "false_count": 20,
                "total_count": 20,
                "true_proportion": 0.0,
            },
        },
    }

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline_data, indent=2))
    candidate_path.write_text(json.dumps(candidate_data, indent=2))

    return {
        "baseline": baseline_path,
        "candidate": candidate_path,
        "baseline_data": baseline_data,
        "candidate_data": candidate_data,
    }


class TestCompareRunsCLI:
    """Tests for the compare-runs CLI command."""

    def test_compare_runs_command_exists(self, cli_runner):
        """Test that compare-runs command is available."""
        result = cli_runner.invoke(app, ["compare-runs", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout.lower()

    def test_compare_runs_missing_required_params(self, cli_runner):
        """Test that compare-runs requires baseline and candidate."""
        result = cli_runner.invoke(app, ["compare-runs"])
        assert result.exit_code != 0
        error_output = result.stderr if result.stderr else result.stdout
        assert "Missing option" in error_output or "required" in error_output.lower()

    def test_compare_runs_missing_baseline_file(self, cli_runner, tmp_path):
        """Test error handling when baseline file doesn't exist."""
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps({"run_id": "test"}))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(tmp_path / "nonexistent.json"),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_compare_runs_missing_candidate_file(self, cli_runner, tmp_path):
        """Test error handling when candidate file doesn't exist."""
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"run_id": "test"}))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(tmp_path / "nonexistent.json"),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_compare_runs_successful_no_regressions(self, cli_runner, sample_artifacts):
        """Test successful comparison with no regressions."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
            ],
        )

        assert result.exit_code == 0
        assert "No regressions detected" in result.stderr

        # Verify JSON output
        output_data = json.loads(result.stdout)
        assert output_data["baseline_run_id"] == "baseline-run-123"
        assert output_data["candidate_run_id"] == "candidate-run-456"
        assert output_data["has_regressions"] is False
        assert output_data["regression_count"] == 0
        assert len(output_data["metric_deltas"]) == 2
        assert len(output_data["flag_deltas"]) == 2

    def test_compare_runs_with_regression(self, cli_runner, tmp_path):
        """Test comparison that detects regressions."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 3.5},  # Regression
            },
            "overall_flag_stats": {},
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 1  # Exit with error when regressions detected
        assert "REGRESSION" in result.stderr

        # Verify JSON output
        output_data = json.loads(result.stdout)
        assert output_data["has_regressions"] is True
        assert output_data["regression_count"] == 1

    def test_compare_runs_custom_thresholds(self, cli_runner, tmp_path):
        """Test comparison with custom thresholds."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 3.8},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.12},
            },
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        # With default thresholds (0.1 metric, 0.05 flag), this is a regression
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )
        assert result.exit_code == 1

        # With higher thresholds, no regression
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--metric-threshold",
                "0.3",
                "--flag-threshold",
                "0.1",
            ],
        )
        assert result.exit_code == 0
        assert "No regressions detected" in result.stderr

        # Verify thresholds in output
        output_data = json.loads(result.stdout)
        assert output_data["thresholds_config"]["metric_threshold"] == 0.3
        assert output_data["thresholds_config"]["flag_threshold"] == 0.1

    def test_compare_runs_output_file(self, cli_runner, sample_artifacts, tmp_path):
        """Test saving comparison results to output file."""
        output_file = tmp_path / "comparison.json"

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify file contents match stdout
        file_data = json.loads(output_file.read_text())
        stdout_data = json.loads(result.stdout)
        assert file_data == stdout_data

    def test_compare_runs_invalid_threshold_negative(self, cli_runner, sample_artifacts):
        """Test error handling for negative thresholds."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
                "--metric-threshold",
                "-0.1",
            ],
        )

        assert result.exit_code == 1
        assert "must be non-negative" in result.stderr.lower()

    def test_compare_runs_summary_format(self, cli_runner, sample_artifacts):
        """Test that human-readable summary is properly formatted."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
            ],
        )

        assert result.exit_code == 0

        # Check summary sections
        assert "Run Comparison Summary" in result.stderr
        assert "Baseline Run ID:" in result.stderr
        assert "Candidate Run ID:" in result.stderr
        assert "Thresholds:" in result.stderr
        assert "Metric Deltas" in result.stderr
        assert "Flag Deltas" in result.stderr
        assert "Summary" in result.stderr

        # Check metric display
        assert "semantic_fidelity" in result.stderr
        assert "clarity" in result.stderr

        # Check flag display
        assert "invented_constraints" in result.stderr
        assert "omitted_constraints" in result.stderr

    def test_compare_runs_metric_improvement_display(self, cli_runner, sample_artifacts):
        """Test that metric improvements are shown with positive delta."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
            ],
        )

        # Semantic fidelity improved from 4.0 to 4.3
        assert "+0.3" in result.stderr or "+0.300" in result.stderr

    def test_compare_runs_invalid_json(self, cli_runner, tmp_path):
        """Test error handling for invalid JSON files."""
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text("{ invalid json")
        candidate.write_text(json.dumps({"run_id": "test"}))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 1
        assert "error" in result.stderr.lower()


class TestThresholdParsing:
    """Tests for threshold parsing and validation."""

    def test_compare_runs_zero_threshold(self, cli_runner, tmp_path):
        """Test comparison with zero thresholds."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 3.999},  # Very small regression
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.101},  # Very small regression
            },
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        # With zero thresholds, any negative change is a regression
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--metric-threshold",
                "0.0",
                "--flag-threshold",
                "0.0",
            ],
        )

        assert result.exit_code == 1  # Regressions detected
        assert "REGRESSION" in result.stderr

    def test_compare_runs_very_large_threshold(self, cli_runner, tmp_path):
        """Test comparison with very large thresholds."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 1.0},  # Large regression
            },
            "overall_flag_stats": {},
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        # With very large threshold, even big regressions don't trigger
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--metric-threshold",
                "10.0",
            ],
        )

        assert result.exit_code == 0  # No regressions detected
        assert "No regressions detected" in result.stderr

    def test_compare_runs_fractional_threshold(self, cli_runner, tmp_path):
        """Test comparison with fractional threshold values."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 3.95},  # Delta = -0.05
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.125},  # Delta = 0.025
            },
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        # With thresholds of 0.03, both should regress
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--metric-threshold",
                "0.03",
                "--flag-threshold",
                "0.02",
            ],
        )

        assert result.exit_code == 1  # Regressions detected
        output_data = json.loads(result.stdout)
        assert output_data["regression_count"] == 2

    def test_compare_runs_invalid_threshold_string(self, cli_runner, sample_artifacts):
        """Test error handling for non-numeric threshold values."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
                "--metric-threshold",
                "not-a-number",
            ],
        )

        assert result.exit_code != 0
        # Typer should handle this as a validation error

    def test_compare_runs_default_thresholds(self, cli_runner, sample_artifacts):
        """Test that default thresholds are applied correctly."""
        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(sample_artifacts["baseline"]),
                "--candidate",
                str(sample_artifacts["candidate"]),
            ],
        )

        assert result.exit_code == 0

        # Verify default thresholds in output
        output_data = json.loads(result.stdout)
        assert output_data["thresholds_config"]["metric_threshold"] == 0.1
        assert output_data["thresholds_config"]["flag_threshold"] == 0.05


class TestEdgeCases:
    """Tests for edge cases in comparison logic."""

    def test_compare_runs_delta_on_threshold_boundary(self, cli_runner, tmp_path):
        """Test comparison when delta is exactly on threshold boundary."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 5.0},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.9},  # Delta = -0.1, exactly on threshold
            },
            "overall_flag_stats": {
                "invented_constraints": {
                    "true_proportion": 0.15
                },  # Delta = 0.05, exactly on threshold
            },
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--metric-threshold",
                "0.1",
                "--flag-threshold",
                "0.05",
            ],
        )

        # Delta equal to threshold should NOT be a regression
        assert result.exit_code == 0
        assert "No regressions detected" in result.stderr

        output_data = json.loads(result.stdout)
        assert not output_data["has_regressions"]
        assert output_data["regression_count"] == 0

    def test_compare_runs_empty_metrics_and_flags(self, cli_runner, tmp_path):
        """Test comparison with no metrics or flags."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {},
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {},
            "overall_flag_stats": {},
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 0
        assert "No regressions detected" in result.stderr

        output_data = json.loads(result.stdout)
        assert len(output_data["metric_deltas"]) == 0
        assert len(output_data["flag_deltas"]) == 0

    def test_compare_runs_with_prompt_metadata_fields(self, cli_runner, tmp_path):
        """Test comparison with prompt_version_id, prompt_hash, and run_notes."""
        baseline_data = {
            "run_id": "baseline-123",
            "prompt_version_id": "v1.0.0",
            "prompt_hash": "abc123",
            "run_notes": "Baseline run",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "prompt_version_id": "v2.0.0",
            "prompt_hash": "xyz789",
            "run_notes": "Improved prompt",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.3},
            },
            "overall_flag_stats": {},
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 0

        # Verify metadata in JSON output
        output_data = json.loads(result.stdout)
        assert output_data["baseline_prompt_version"] == "v1.0.0"
        assert output_data["candidate_prompt_version"] == "v2.0.0"

        # Verify metadata in human-readable output
        assert "v1.0.0" in result.stderr
        assert "v2.0.0" in result.stderr


class TestMultipleMetricsAndFlags:
    """Tests for comparison with multiple metrics and flags."""

    def test_compare_runs_many_metrics(self, cli_runner, tmp_path):
        """Test comparison with many different metrics."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.0},
                "metric_2": {"mean_of_means": 3.5},
                "metric_3": {"mean_of_means": 4.2},
                "metric_4": {"mean_of_means": 3.8},
                "metric_5": {"mean_of_means": 4.5},
            },
            "overall_flag_stats": {
                "flag_1": {"true_proportion": 0.1},
                "flag_2": {"true_proportion": 0.05},
                "flag_3": {"true_proportion": 0.15},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.1},  # Improved
                "metric_2": {"mean_of_means": 3.3},  # Regressed
                "metric_3": {"mean_of_means": 4.25},  # Improved
                "metric_4": {"mean_of_means": 3.85},  # Improved
                "metric_5": {"mean_of_means": 4.45},  # Small regression
            },
            "overall_flag_stats": {
                "flag_1": {"true_proportion": 0.08},  # Improved
                "flag_2": {"true_proportion": 0.12},  # Regressed
                "flag_3": {"true_proportion": 0.14},  # Improved
            },
        }

        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps(baseline_data))
        candidate.write_text(json.dumps(candidate_data))

        result = cli_runner.invoke(
            app,
            [
                "compare-runs",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ],
        )

        assert result.exit_code == 1  # Has regressions

        output_data = json.loads(result.stdout)
        assert output_data["has_regressions"]
        # metric_2 regressed by 0.2, flag_2 regressed by 0.07
        assert output_data["regression_count"] == 2

        # Verify all metrics and flags are present
        assert len(output_data["metric_deltas"]) == 5
        assert len(output_data["flag_deltas"]) == 3
