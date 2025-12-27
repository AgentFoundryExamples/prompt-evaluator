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
Tests for win/loss/tie computation in comparison logic.
"""

import json
from pathlib import Path

import pytest

from prompt_evaluator.comparison import (
    compare_runs,
    compute_test_case_comparisons,
)


class TestComputeTestCaseComparisons:
    """Tests for compute_test_case_comparisons function."""

    def test_candidate_wins_higher_overall_mean(self):
        """Test that candidate wins when overall mean is higher."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.0},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.0},
                        "metric_2": {"mean": 4.5},
                    },
                }
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].winner == "candidate"
        assert test_case_comparisons[0].test_case_id == "test-001"
        assert test_case_comparisons[0].baseline_mean == 3.5  # (3.0 + 4.0) / 2
        assert test_case_comparisons[0].candidate_mean == 4.25  # (4.0 + 4.5) / 2
        assert not test_case_comparisons[0].is_regression
        
        assert win_loss_tie_stats["candidate_wins"] == 1
        assert win_loss_tie_stats["baseline_wins"] == 0
        assert win_loss_tie_stats["ties"] == 0

    def test_baseline_wins_lower_overall_mean(self):
        """Test that baseline wins when candidate has lower overall mean."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.5},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.0},
                        "metric_2": {"mean": 3.5},
                    },
                }
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].winner == "baseline"
        assert win_loss_tie_stats["baseline_wins"] == 1
        assert win_loss_tie_stats["candidate_wins"] == 0

    def test_tie_when_means_equal(self):
        """Test that result is tie when means are equal."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.0},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.0},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].winner == "tie"
        assert win_loss_tie_stats["ties"] == 1
        assert win_loss_tie_stats["candidate_wins"] == 0
        assert win_loss_tie_stats["baseline_wins"] == 0

    def test_tie_with_small_delta(self):
        """Test that result is tie when delta is very small (< 0.01)."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.000},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.005},  # Very small improvement
                    },
                }
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].winner == "tie"
        assert win_loss_tie_stats["ties"] == 1

    def test_regression_detection(self):
        """Test that regressions are detected when metric drops > threshold."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 5.0},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.5},  # Improved
                        "metric_2": {"mean": 2.5},  # Regression: dropped by 1.5
                    },
                }
            ]
        }
        
        test_case_comparisons, _ = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.5
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].is_regression is True

    def test_no_regression_below_threshold(self):
        """Test that small drops below threshold don't count as regression."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.95},  # Small drop
                    },
                }
            ]
        }
        
        test_case_comparisons, _ = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].is_regression is False

    def test_multiple_test_cases(self):
        """Test win/loss/tie computation across multiple test cases."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {"metric_1": {"mean": 3.0}},
                },
                {
                    "test_case_id": "test-002",
                    "per_metric_stats": {"metric_1": {"mean": 4.0}},
                },
                {
                    "test_case_id": "test-003",
                    "per_metric_stats": {"metric_1": {"mean": 4.0}},
                },
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {"metric_1": {"mean": 4.0}},  # Candidate wins
                },
                {
                    "test_case_id": "test-002",
                    "per_metric_stats": {"metric_1": {"mean": 3.0}},  # Baseline wins
                },
                {
                    "test_case_id": "test-003",
                    "per_metric_stats": {"metric_1": {"mean": 4.0}},  # Tie
                },
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 3
        assert win_loss_tie_stats["candidate_wins"] == 1
        assert win_loss_tie_stats["baseline_wins"] == 1
        assert win_loss_tie_stats["ties"] == 1
        assert win_loss_tie_stats["total"] == 3

    def test_per_metric_deltas(self):
        """Test that per-metric deltas are computed correctly."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.0},
                        "metric_2": {"mean": 4.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.5},
                        "metric_2": {"mean": 3.8},
                    },
                }
            ]
        }
        
        test_case_comparisons, _ = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 1
        per_metric_deltas = test_case_comparisons[0].per_metric_deltas
        
        assert per_metric_deltas["metric_1"] == pytest.approx(0.5)
        assert per_metric_deltas["metric_2"] == pytest.approx(-0.2)

    def test_missing_test_cases(self):
        """Test that missing test cases in one run are skipped."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {"metric_1": {"mean": 3.0}},
                },
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-002",
                    "per_metric_stats": {"metric_1": {"mean": 4.0}},
                },
            ]
        }
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        # No test cases matched, so no comparisons
        assert len(test_case_comparisons) == 0
        assert win_loss_tie_stats["total"] == 0

    def test_empty_test_case_results(self):
        """Test handling of empty test case results."""
        baseline_data = {"test_case_results": []}
        candidate_data = {"test_case_results": []}
        
        test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        assert len(test_case_comparisons) == 0
        assert win_loss_tie_stats["total"] == 0

    def test_missing_metrics_in_stats(self):
        """Test handling of missing metrics in per_metric_stats."""
        baseline_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 3.0},
                    },
                }
            ]
        }
        
        candidate_data = {
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_2": {"mean": 4.0},  # Different metric
                    },
                }
            ]
        }
        
        test_case_comparisons, _ = compute_test_case_comparisons(
            baseline_data, candidate_data, metric_threshold=0.1
        )
        
        # Should still create comparison with None deltas for missing metrics
        assert len(test_case_comparisons) == 1
        assert test_case_comparisons[0].per_metric_deltas["metric_1"] is None
        assert test_case_comparisons[0].per_metric_deltas["metric_2"] is None


class TestCompareRunsWithTestCaseComparisons:
    """Tests for compare_runs with test case comparison enabled."""

    def test_compare_runs_includes_test_case_comparisons(self, tmp_path):
        """Test that compare_runs includes test case comparisons when available."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.0},
                    },
                }
            ],
        }
        
        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.5},
            },
            "overall_flag_stats": {},
            "test_case_results": [
                {
                    "test_case_id": "test-001",
                    "per_metric_stats": {
                        "metric_1": {"mean": 4.5},
                    },
                }
            ],
        }
        
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))
        
        result = compare_runs(baseline_path, candidate_path)
        
        # Verify test case comparisons are included
        assert len(result.test_case_comparisons) == 1
        assert result.test_case_comparisons[0].test_case_id == "test-001"
        assert result.test_case_comparisons[0].winner == "candidate"
        
        # Verify win/loss/tie stats
        assert result.win_loss_tie_stats["candidate_wins"] == 1
        assert result.win_loss_tie_stats["total"] == 1

    def test_compare_runs_without_test_case_data(self, tmp_path):
        """Test that compare_runs works without test case data (backward compat)."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.0},
            },
            "overall_flag_stats": {},
        }
        
        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {
                "metric_1": {"mean_of_means": 4.5},
            },
            "overall_flag_stats": {},
        }
        
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))
        
        result = compare_runs(baseline_path, candidate_path)
        
        # Should not have test case comparisons but should still work
        assert len(result.test_case_comparisons) == 0
        # Empty win/loss/tie stats with 0 total is acceptable
        assert result.win_loss_tie_stats.get("total", 0) == 0

    def test_compare_runs_disable_test_case_comparisons(self, tmp_path):
        """Test that test case comparisons can be disabled."""
        baseline_data = {
            "run_id": "baseline-123",
            "overall_metric_stats": {"metric_1": {"mean_of_means": 4.0}},
            "overall_flag_stats": {},
            "test_case_results": [
                {"test_case_id": "test-001", "per_metric_stats": {"metric_1": {"mean": 4.0}}},
            ],
        }
        
        candidate_data = {
            "run_id": "candidate-456",
            "overall_metric_stats": {"metric_1": {"mean_of_means": 4.5}},
            "overall_flag_stats": {},
            "test_case_results": [
                {"test_case_id": "test-001", "per_metric_stats": {"metric_1": {"mean": 4.5}}},
            ],
        }
        
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline_data))
        candidate_path.write_text(json.dumps(candidate_data))
        
        result = compare_runs(
            baseline_path, candidate_path, include_test_case_comparisons=False
        )
        
        # Test case comparisons should be disabled
        assert len(result.test_case_comparisons) == 0
        assert len(result.win_loss_tie_stats) == 0
