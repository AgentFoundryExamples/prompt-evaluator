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


class TestThresholdEdgeCases:
    """Tests for threshold boundary conditions and edge cases."""

    def test_metric_delta_exactly_on_threshold_positive(self):
        """Test metric delta equal to threshold - should not regress."""
        baseline_stats = {"mean_of_means": 5.0}
        candidate_stats = {"mean_of_means": 4.9}  # Delta = -0.1, exactly on threshold

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == pytest.approx(-0.1)
        assert not delta.is_regression  # Equal to threshold should not be regression

    def test_metric_delta_just_below_threshold(self):
        """Test metric delta just below threshold - should not regress."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 3.91}  # Delta = -0.09, below threshold

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == pytest.approx(-0.09)
        assert not delta.is_regression

    def test_metric_delta_just_above_threshold(self):
        """Test metric delta just above threshold - should regress."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 3.89}  # Delta = -0.11, above threshold

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.1)

        assert delta.delta == pytest.approx(-0.11)
        assert delta.is_regression

    def test_flag_delta_exactly_on_threshold_positive(self):
        """Test flag delta exactly equal to threshold - should not regress."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.15}  # Delta = 0.05, exactly on threshold

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == pytest.approx(0.05)
        assert not delta.is_regression  # Equal to threshold should not be regression

    def test_flag_delta_just_below_threshold(self):
        """Test flag delta just below threshold - should not regress."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.149}  # Delta = 0.049, below threshold

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == pytest.approx(0.049)
        assert not delta.is_regression

    def test_flag_delta_just_above_threshold(self):
        """Test flag delta just above threshold - should regress."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.151}  # Delta = 0.051, above threshold

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == pytest.approx(0.051)
        assert delta.is_regression

    def test_flag_zero_baseline_zero_candidate_no_division_error(self):
        """Test flag with zero baseline and candidate doesn't cause division errors."""
        baseline_stats = {"true_proportion": 0.0}
        candidate_stats = {"true_proportion": 0.0}

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.05)

        assert delta.delta == 0.0
        assert delta.percent_change is None  # Should be None, not inf
        assert not delta.is_regression

    def test_metric_zero_threshold(self):
        """Test metric comparison with zero threshold - any negative change is regression."""
        baseline_stats = {"mean_of_means": 4.0}
        candidate_stats = {"mean_of_means": 3.999}  # Very small negative delta

        delta = compute_metric_delta("test_metric", baseline_stats, candidate_stats, threshold=0.0)

        assert delta.delta == pytest.approx(-0.001)
        assert delta.is_regression  # With zero threshold, any negative delta is regression

    def test_flag_zero_threshold(self):
        """Test flag comparison with zero threshold - any positive change is regression."""
        baseline_stats = {"true_proportion": 0.1}
        candidate_stats = {"true_proportion": 0.101}  # Very small positive delta

        delta = compute_flag_delta("test_flag", baseline_stats, candidate_stats, threshold=0.0)

        assert delta.delta == pytest.approx(0.001)
        assert delta.is_regression  # With zero threshold, any positive delta is regression


class TestMetadataFields:
    """Tests for new metadata fields in run artifacts."""

    def test_compare_runs_with_prompt_metadata(self, tmp_path):
        """Test comparison with prompt_version_id, prompt_hash, and run_notes."""
        baseline_data = {
            "run_id": "baseline-123",
            "prompt_version_id": "v1.0.0",
            "prompt_hash": "abc123def456",
            "run_notes": "Baseline run with original prompt",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "prompt_version_id": "v2.0.0",
            "prompt_hash": "xyz789ghi012",
            "run_notes": "Candidate run with improved prompt",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.3},
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        # Verify metadata is captured in result
        assert result.baseline_prompt_version == "v1.0.0"
        assert result.candidate_prompt_version == "v2.0.0"
        assert not result.has_regressions

    def test_compare_runs_missing_prompt_metadata(self, tmp_path):
        """Test comparison when prompt metadata fields are missing."""
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
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        # Should handle missing metadata gracefully
        assert result.baseline_prompt_version is None
        assert result.candidate_prompt_version is None
        assert not result.has_regressions


class TestPerCaseComparison:
    """Tests for per-case level comparison (future enhancement)."""

    def test_compare_runs_with_per_case_stats(self, tmp_path):
        """Test that comparison works with test_case_results present."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
            },
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "semantic_fidelity": {"mean": 4.0, "std": 0.2, "min": 3.8, "max": 4.2}
                    },
                    "per_flag_stats": {
                        "invented_constraints": {
                            "true_count": 1,
                            "false_count": 4,
                            "total_count": 5,
                            "true_proportion": 0.2,
                        }
                    },
                }
            ],
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.2},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.08},
            },
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "semantic_fidelity": {"mean": 4.2, "std": 0.15, "min": 4.0, "max": 4.4}
                    },
                    "per_flag_stats": {
                        "invented_constraints": {
                            "true_count": 0,
                            "false_count": 5,
                            "total_count": 5,
                            "true_proportion": 0.0,
                        }
                    },
                }
            ],
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # Should compare based on overall stats, not per-case stats
        result = compare_runs(baseline_path, candidate_path)

        assert result.baseline_run_id == "baseline-123"
        assert result.candidate_run_id == "candidate-456"
        assert not result.has_regressions  # Both improved


class TestValidationScenarios:
    """Tests for validation of mismatched or incompatible runs."""

    def test_compare_runs_different_datasets_hash(self, tmp_path):
        """Test comparison with different dataset hashes (should still work but could warn)."""
        baseline_data = {
            "run_id": "baseline-123",
            "dataset_path": "/path/to/dataset.yaml",
            "dataset_hash": "sha256:abc123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "dataset_path": "/path/to/dataset.yaml",
            "dataset_hash": "sha256:xyz789",  # Different hash
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.2},
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # Should succeed but comparisons may not be meaningful
        result = compare_runs(baseline_path, candidate_path)
        assert result.baseline_run_id == "baseline-123"
        assert result.candidate_run_id == "candidate-456"

    def test_compare_runs_different_rubrics(self, tmp_path):
        """Test comparison with different rubrics (should still work but could warn)."""
        baseline_data = {
            "run_id": "baseline-123",
            "rubric_metadata": {
                "rubric_path": "/path/to/rubric_v1.yaml",
                "rubric_hash": "hash123",
            },
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "rubric_metadata": {
                "rubric_path": "/path/to/rubric_v2.yaml",
                "rubric_hash": "hash456",  # Different rubric
            },
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.2},
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # Should succeed but comparisons may not be meaningful
        result = compare_runs(baseline_path, candidate_path)
        assert result.baseline_run_id == "baseline-123"
        assert result.candidate_run_id == "candidate-456"

    def test_compare_runs_different_models(self, tmp_path):
        """Test comparison with different generator models (should still work)."""
        baseline_data = {
            "run_id": "baseline-123",
            "generator_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_completion_tokens": 1024,
            },
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "generator_config": {
                "model_name": "gpt-5.1",  # Different model
                "temperature": 0.7,
                "max_completion_tokens": 1024,
            },
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.2},
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # Should succeed - comparing different models is valid use case
        result = compare_runs(baseline_path, candidate_path)
        assert result.baseline_run_id == "baseline-123"
        assert result.candidate_run_id == "candidate-456"
        assert not result.has_regressions


class TestComplexScenarios:
    """Tests for complex real-world comparison scenarios."""

    def test_compare_runs_mixed_improvements_and_regressions(self, tmp_path):
        """Test comparison with some metrics improving and others regressing."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.0},
                "clarity": {"mean_of_means": 4.5},
                "completeness": {"mean_of_means": 3.8},
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.1},
                "omitted_constraints": {"true_proportion": 0.15},
            },
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "semantic_fidelity": {"mean_of_means": 4.3},  # Improved
                "clarity": {"mean_of_means": 4.2},  # Regressed
                "completeness": {"mean_of_means": 3.75},  # Small regression, below threshold
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.05},  # Improved
                "omitted_constraints": {"true_proportion": 0.25},  # Regressed
            },
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(
            baseline_path, candidate_path, metric_threshold=0.1, flag_threshold=0.05
        )

        assert result.has_regressions
        # clarity regressed by 0.3 (above threshold)
        # completeness regressed by 0.05 (below threshold)
        # omitted_constraints regressed by 0.1 (above threshold)
        assert result.regression_count == 2  # clarity and omitted_constraints

        # Verify specific deltas
        clarity_delta = next(d for d in result.metric_deltas if d.metric_name == "clarity")
        assert clarity_delta.is_regression
        assert clarity_delta.delta == pytest.approx(-0.3)

        completeness_delta = next(
            d for d in result.metric_deltas if d.metric_name == "completeness"
        )
        assert not completeness_delta.is_regression  # Below threshold
        assert completeness_delta.delta == pytest.approx(-0.05)

        omitted_delta = next(d for d in result.flag_deltas if d.flag_name == "omitted_constraints")
        assert omitted_delta.is_regression
        assert omitted_delta.delta == pytest.approx(0.1)

    def test_compare_runs_all_new_metrics(self, tmp_path):
        """Test comparison where candidate has entirely new metrics."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "old_metric_1": {"mean_of_means": 4.0},
                "old_metric_2": {"mean_of_means": 3.5},
            },
            "overall_flag_stats": {},
        }

        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "new_metric_1": {"mean_of_means": 4.2},
                "new_metric_2": {"mean_of_means": 3.8},
            },
            "overall_flag_stats": {},
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        # Should have deltas for all 4 metrics
        assert len(result.metric_deltas) == 4
        metric_names = {d.metric_name for d in result.metric_deltas}
        assert metric_names == {"old_metric_1", "old_metric_2", "new_metric_1", "new_metric_2"}

        # No regressions because all new or removed metrics have None baselines/candidates
        assert not result.has_regressions

    def test_compare_runs_large_threshold_values(self, tmp_path):
        """Test comparison with very large threshold values."""
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
                "semantic_fidelity": {"mean_of_means": 2.0},  # Large regression
            },
            "overall_flag_stats": {
                "invented_constraints": {"true_proportion": 0.9},  # Large regression
            },
        }

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        # With very large thresholds, even big changes don't regress
        result = compare_runs(
            baseline_path, candidate_path, metric_threshold=5.0, flag_threshold=1.0
        )

        assert not result.has_regressions  # Thresholds are too high

    def test_compare_runs_empty_stats(self, tmp_path):
        """Test comparison with empty metric and flag stats."""
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

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))

        result = compare_runs(baseline_path, candidate_path)

        assert len(result.metric_deltas) == 0
        assert len(result.flag_deltas) == 0
        assert not result.has_regressions
        assert result.regression_count == 0
