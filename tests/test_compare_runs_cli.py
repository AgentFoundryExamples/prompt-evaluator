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
