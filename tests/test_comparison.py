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
Tests for run comparison and regression detection logic.
"""

import json

import pytest

from prompt_evaluator.comparison import (
    compare_runs,
    compute_flag_delta,
    compute_metric_delta,
    load_run_artifact,
)


class TestLoadRunArtifact:
    """Tests for load_run_artifact function."""

    def test_load_valid_artifact(self, tmp_path):
        """Test loading a valid run artifact."""
        artifact_data = {
            "run_id": "test-run-123",
            "overall_metric_stats": {},
            "overall_flag_stats": {},
        }
        artifact_path = tmp_path / "run.json"
        artifact_path.write_text(json.dumps(artifact_data))

        result = load_run_artifact(artifact_path)

        assert result["run_id"] == "test-run-123"

    def test_load_missing_file(self, tmp_path):
        """Test loading non-existent artifact raises FileNotFoundError."""
        artifact_path = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError, match="Run artifact not found"):
            load_run_artifact(artifact_path)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises ValueError."""
        artifact_path = tmp_path / "invalid.json"
        artifact_path.write_text("{ invalid json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_run_artifact(artifact_path)

    def test_load_missing_required_fields(self, tmp_path):
        """Test loading artifact without required fields raises ValueError."""
        artifact_data = {"some_field": "value"}
        artifact_path = tmp_path / "incomplete.json"
        artifact_path.write_text(json.dumps(artifact_data))

        with pytest.raises(ValueError, match="missing required fields"):
            load_run_artifact(artifact_path)


class TestComputeMetricDelta:
    """Tests for compute_metric_delta function."""

    def test_metric_delta_improvement(self):
        """Test metric delta when candidate improves over baseline."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 4.5}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.metric_name == "test_metric"
        assert delta.baseline_mean == 4.0
        assert delta.candidate_mean == 4.5
        assert delta.delta == 0.5
        assert delta.percent_change == pytest.approx(12.5)
        assert not delta.is_regression
        assert delta.threshold_used == 0.1

    def test_metric_delta_regression_below_threshold(self):
        """Test metric delta when candidate regresses but stays within threshold."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 3.95}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == pytest.approx(-0.05)
        assert not delta.is_regression  # Below threshold

    def test_metric_delta_regression_above_threshold(self):
        """Test metric delta when candidate regresses beyond threshold."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 3.5}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == pytest.approx(-0.5)
        assert delta.is_regression  # Above threshold

    def test_metric_delta_missing_baseline(self):
        """Test metric delta when baseline has no data."""
        baseline_stats = {"mean_of_means": None}
        candidate_stats = {"mean_of_means": 4.0}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.baseline_mean is None
        assert delta.candidate_mean == 4.0
        assert delta.delta is None
        assert delta.percent_change is None
        assert not delta.is_regression

    def test_metric_delta_missing_candidate(self):
        """Test metric delta when candidate has no data."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": None}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.baseline_mean == 4.0
        assert delta.candidate_mean is None
        assert delta.delta is None
        assert delta.percent_change is None
        assert not delta.is_regression

    def test_metric_delta_zero_baseline(self):
        """Test metric delta when baseline mean is zero."""
        baseline_stats = {"mean_of_means": 0.0}
        candidate_stats = {"mean_of_means": 1.0}

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == 1.0
        assert delta.percent_change == float("inf")
        assert not delta.is_regression


class TestComputeFlagDelta:
    """Tests for compute_flag_delta function."""

    def test_flag_delta_improvement(self):
        """Test flag delta when candidate improves (fewer flags)."""
        baseline_stats = {"true_proportion": 0.3}
        candidate_stats = {"true_proportion": 0.2}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.flag_name == "test_flag"
        assert delta.baseline_proportion == 0.3
        assert delta.candidate_proportion == 0.2
        assert delta.delta == pytest.approx(-0.1)
        assert delta.percent_change == pytest.approx(-33.333, rel=0.01)
        assert not delta.is_regression

    def test_flag_delta_regression_below_threshold(self):
        """Test flag delta when candidate regresses but stays within threshold."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.13}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == pytest.approx(0.03)
        assert not delta.is_regression  # Below threshold

    def test_flag_delta_regression_above_threshold(self):
        """Test flag delta when candidate regresses beyond threshold."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.3}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == pytest.approx(0.2)
        assert delta.is_regression  # Above threshold

    def test_flag_delta_zero_baseline(self):
        """Test flag delta when baseline proportion is zero."""
        baseline_stats = {"true_proportion": 0.0}
        candidate_stats = {"true_proportion": 0.1}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == 0.1
        assert delta.percent_change == float("inf")
        assert delta.is_regression  # Above threshold

    def test_flag_delta_missing_proportions(self):
        """Test flag delta when proportions are missing (defaults to 0.0)."""
        baseline_stats = {}
        candidate_stats = {}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.baseline_proportion == 0.0
        assert delta.candidate_proportion == 0.0
        assert delta.delta == 0.0
        assert not delta.is_regression


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_compare_runs_no_regressions(self, tmp_path):
        """Test comparing runs with no regressions."""
        baseline_data = {
            "run_id": "baseline-123",
            "prompt_version_id": "v1",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
                "clarity": {"mean_of_means": 4.2},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
                "omitted_constraints": {"true_proportion": 0.05},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "prompt_version_id": "v2",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.3},
                "clarity": {"mean_of_means": 4.5},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.08},
                "omitted_constraints": {"true_proportion": 0.03},
            },
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        assert result.baseline_run_id == "baseline-123"
        assert result.candidate_run_id == "candidate-456"
        assert result.baseline_prompt_version == "v1"
        assert result.candidate_prompt_version == "v2"
        assert len(result.metric_deltas) == 2
        assert len(result.flag_deltas) == 2
        assert not result.has_regressions
        assert result.regression_count == 0

    def test_compare_runs_with_metric_regression(self, tmp_path):
        """Test comparing runs with metric regression."""
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
                "semantic_fidelity": {"mean_of_means": 3.5},  # Regression > 0.1
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path, metric_threshold=0.1)

        assert result.has_regressions
        assert result.regression_count == 1
        assert result.metric_deltas[0].is_regression

    def test_compare_runs_with_flag_regression(self, tmp_path):
        """Test comparing runs with flag regression."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {},
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {},
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.3},  # Regression > 0.05
            },
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path, flag_threshold=0.05)

        assert result.has_regressions
        assert result.regression_count == 1
        assert result.flag_deltas[0].is_regression

    def test_compare_runs_multiple_regressions(self, tmp_path):
        """Test comparing runs with multiple regressions."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
                "clarity": {"mean_of_means": 4.5},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 3.5},  # Regression
                "clarity": {"mean_of_means": 4.0},  # Regression
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.3},  # Regression
            },
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        assert result.has_regressions
        assert result.regression_count == 3

    def test_compare_runs_custom_thresholds(self, tmp_path):
        """Test comparing runs with custom thresholds."""
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
                "semantic_fidelity": {"mean_of_means": 3.8},  # Delta = -0.2
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # With default threshold (0.1), this is a regression
        result = compare_runs(baseline_path, candidate_path, metric_threshold=0.1)
        assert result.has_regressions

        # With higher threshold (0.3), this is not a regression
        result = compare_runs(baseline_path, candidate_path, metric_threshold=0.3)
        assert not result.has_regressions

    def test_compare_runs_asymmetric_metrics(self, tmp_path):
        """Test comparing runs where metrics differ between runs."""
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
                "semantic_fidelity": {"mean_of_means": 4.2},
                "clarity": {"mean_of_means": 4.5},  # New metric
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        # Should have deltas for both metrics
        assert len(result.metric_deltas) == 2
        metric_names = {d.metric_name for d in result.metric_deltas}
        assert metric_names == {"semantic_fidelity", "clarity"}

        # Clarity should have None baseline
        clarity_delta = next(d for d in result.metric_deltas if d.metric_name == "clarity")
        assert clarity_delta.baseline_mean is None
        assert clarity_delta.candidate_mean == 4.5
