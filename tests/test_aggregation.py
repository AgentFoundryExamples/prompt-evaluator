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
Tests for aggregate statistics computation with rubric-aware metrics and flags.
"""

import pytest

from prompt_evaluator.cli import compute_aggregate_statistics, compute_rubric_metadata
from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric, Sample


class TestComputeAggregateStatistics:
    """Tests for compute_aggregate_statistics function."""

    def test_aggregation_without_rubric_legacy_mode(self):
        """Test that legacy statistics are computed when no rubric is provided."""
        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                judge_score=4.5,
                status="completed",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                judge_score=3.0,
                status="completed",
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=None)

        assert stats["mean_score"] == 3.75
        assert stats["min_score"] == 3.0
        assert stats["max_score"] == 4.5
        assert stats["num_successful"] == 2
        assert stats["num_failed"] == 0
        assert "metric_stats" not in stats
        assert "flag_stats" not in stats

    def test_aggregation_with_rubric_computes_metrics(self):
        """Test that per-metric statistics are computed when rubric is provided."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="semantic_fidelity",
                    description="Test metric",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test guidelines",
                ),
                RubricMetric(
                    name="clarity",
                    description="Test metric 2",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test guidelines",
                ),
            ],
            flags=[],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="completed",
                judge_metrics={
                    "semantic_fidelity": {"score": 4.5, "rationale": "Good"},
                    "clarity": {"score": 3.0, "rationale": "OK"},
                },
                judge_flags={},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="completed",
                judge_metrics={
                    "semantic_fidelity": {"score": 5.0, "rationale": "Excellent"},
                    "clarity": {"score": 4.0, "rationale": "Good"},
                },
                judge_flags={},
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Check metric statistics
        assert "metric_stats" in stats
        assert "semantic_fidelity" in stats["metric_stats"]
        assert stats["metric_stats"]["semantic_fidelity"]["mean"] == 4.75
        assert stats["metric_stats"]["semantic_fidelity"]["min"] == 4.5
        assert stats["metric_stats"]["semantic_fidelity"]["max"] == 5.0
        assert stats["metric_stats"]["semantic_fidelity"]["count"] == 2

        assert "clarity" in stats["metric_stats"]
        assert stats["metric_stats"]["clarity"]["mean"] == 3.5
        assert stats["metric_stats"]["clarity"]["min"] == 3.0
        assert stats["metric_stats"]["clarity"]["max"] == 4.0
        assert stats["metric_stats"]["clarity"]["count"] == 2

    def test_aggregation_with_rubric_computes_flags(self):
        """Test that per-flag statistics are computed when rubric is provided."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[
                RubricFlag(name="invented_constraints", description="Test flag 1"),
                RubricFlag(name="omitted_constraints", description="Test flag 2"),
            ],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="completed",
                judge_metrics={"test_metric": {"score": 4.0, "rationale": "Good"}},
                judge_flags={"invented_constraints": True, "omitted_constraints": False},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="completed",
                judge_metrics={"test_metric": {"score": 3.0, "rationale": "OK"}},
                judge_flags={"invented_constraints": False, "omitted_constraints": False},
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                status="completed",
                judge_metrics={"test_metric": {"score": 5.0, "rationale": "Excellent"}},
                judge_flags={"invented_constraints": True, "omitted_constraints": True},
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Check flag statistics
        assert "flag_stats" in stats
        assert "invented_constraints" in stats["flag_stats"]
        assert stats["flag_stats"]["invented_constraints"]["true_count"] == 2
        assert stats["flag_stats"]["invented_constraints"]["false_count"] == 1
        assert stats["flag_stats"]["invented_constraints"]["total_count"] == 3
        assert stats["flag_stats"]["invented_constraints"]["true_proportion"] == pytest.approx(
            2 / 3
        )

        assert "omitted_constraints" in stats["flag_stats"]
        assert stats["flag_stats"]["omitted_constraints"]["true_count"] == 1
        assert stats["flag_stats"]["omitted_constraints"]["false_count"] == 2
        assert stats["flag_stats"]["omitted_constraints"]["total_count"] == 3
        assert stats["flag_stats"]["omitted_constraints"]["true_proportion"] == pytest.approx(1 / 3)

    def test_aggregation_excludes_invalid_judge_responses(self):
        """Test that samples with judge_invalid_response status are excluded."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[RubricFlag(name="test_flag", description="Test")],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="completed",
                judge_metrics={"test_metric": {"score": 4.0, "rationale": "Good"}},
                judge_flags={"test_flag": True},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="judge_invalid_response",
                judge_metrics={},
                judge_flags={},
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                status="completed",
                judge_metrics={"test_metric": {"score": 5.0, "rationale": "Excellent"}},
                judge_flags={"test_flag": False},
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Only 2 samples should be counted (s1 and s3)
        assert stats["metric_stats"]["test_metric"]["count"] == 2
        assert stats["metric_stats"]["test_metric"]["mean"] == 4.5
        assert stats["flag_stats"]["test_flag"]["total_count"] == 2
        assert stats["flag_stats"]["test_flag"]["true_count"] == 1

    def test_aggregation_all_samples_invalid(self):
        """Test aggregation when all samples have invalid judge responses."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[RubricFlag(name="test_flag", description="Test")],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="judge_invalid_response",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="judge_invalid_response",
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Should return zero counts without errors
        assert stats["num_successful"] == 0
        assert stats["num_failed"] == 2
        assert stats["mean_score"] is None
        assert stats["metric_stats"]["test_metric"]["count"] == 0
        assert stats["metric_stats"]["test_metric"]["mean"] is None
        assert stats["flag_stats"]["test_flag"]["total_count"] == 0
        assert stats["flag_stats"]["test_flag"]["true_proportion"] == 0.0

    def test_aggregation_flag_never_true(self):
        """Test that flags with zero true occurrences still appear in stats."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[RubricFlag(name="never_true_flag", description="Test")],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="completed",
                judge_metrics={"test_metric": {"score": 4.0, "rationale": "Good"}},
                judge_flags={"never_true_flag": False},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="completed",
                judge_metrics={"test_metric": {"score": 5.0, "rationale": "Excellent"}},
                judge_flags={"never_true_flag": False},
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Flag should appear with zero true count
        assert "never_true_flag" in stats["flag_stats"]
        assert stats["flag_stats"]["never_true_flag"]["true_count"] == 0
        assert stats["flag_stats"]["never_true_flag"]["false_count"] == 2
        assert stats["flag_stats"]["never_true_flag"]["true_proportion"] == 0.0

    def test_aggregation_mixed_int_float_scores(self):
        """Test that both integer and float scores are handled consistently."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="completed",
                judge_metrics={"test_metric": {"score": 4, "rationale": "Good"}},
                judge_flags={},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="completed",
                judge_metrics={"test_metric": {"score": 4.5, "rationale": "Better"}},
                judge_flags={},
            ),
        ]

        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Should handle both int and float
        assert stats["metric_stats"]["test_metric"]["mean"] == 4.25
        assert stats["metric_stats"]["test_metric"]["min"] == 4
        assert stats["metric_stats"]["test_metric"]["max"] == 4.5

    def test_aggregation_large_rubric_performance(self):
        """Test that aggregation handles large rubrics efficiently."""
        # Create a rubric with many metrics
        metrics = [
            RubricMetric(
                name=f"metric_{i}",
                description=f"Test metric {i}",
                min_score=1.0,
                max_score=5.0,
                guidelines=f"Guidelines {i}",
            )
            for i in range(20)
        ]

        rubric = Rubric(metrics=metrics, flags=[])

        # Create samples with all metrics
        samples = []
        for s_idx in range(10):
            judge_metrics = {
                f"metric_{i}": {"score": 3.0 + (i % 3), "rationale": f"Rationale {i}"}
                for i in range(20)
            }
            samples.append(
                Sample(
                    sample_id=f"s{s_idx}",
                    input_text="test",
                    generator_output=f"output{s_idx}",
                    status="completed",
                    judge_metrics=judge_metrics,
                    judge_flags={},
                )
            )

        # Should complete without performance issues
        stats = compute_aggregate_statistics(samples, rubric=rubric)

        # Verify all metrics are present
        assert len(stats["metric_stats"]) == 20
        for i in range(20):
            assert f"metric_{i}" in stats["metric_stats"]
            assert stats["metric_stats"][f"metric_{i}"]["count"] == 10


class TestComputeRubricMetadata:
    """Tests for compute_rubric_metadata function."""

    def test_rubric_metadata_includes_path_hash_and_definition(self, tmp_path):
        """Test that rubric metadata includes path, hash, and full definition."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test guidelines",
                ),
            ],
            flags=[RubricFlag(name="test_flag", description="Test flag")],
        )

        rubric_path = tmp_path / "test_rubric.yaml"
        rubric_path.write_text("dummy content")

        metadata = compute_rubric_metadata(rubric, rubric_path)

        assert "rubric_path" in metadata
        assert metadata["rubric_path"] == str(rubric_path)
        assert "rubric_hash" in metadata
        assert len(metadata["rubric_hash"]) == 64  # SHA256 hash
        assert "rubric_definition" in metadata
        assert "metrics" in metadata["rubric_definition"]
        assert "flags" in metadata["rubric_definition"]

    def test_rubric_metadata_none_rubric(self):
        """Test that empty dict is returned when rubric is None."""
        metadata = compute_rubric_metadata(None, None)
        assert metadata == {}

    def test_rubric_metadata_hash_changes_with_content(self):
        """Test that rubric hash changes when content changes."""
        rubric1 = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Guidelines 1",
                ),
            ],
            flags=[],
        )

        rubric2 = Rubric(
            metrics=[
                RubricMetric(
                    name="metric2",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Guidelines 2",
                ),
            ],
            flags=[],
        )

        from pathlib import Path

        dummy_path = Path("/dummy/path.yaml")

        metadata1 = compute_rubric_metadata(rubric1, dummy_path)
        metadata2 = compute_rubric_metadata(rubric2, dummy_path)

        # Hashes should be different
        assert metadata1["rubric_hash"] != metadata2["rubric_hash"]

    def test_rubric_metadata_hash_deterministic(self):
        """Test that the same rubric always produces the same hash."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="test_metric",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test guidelines",
                ),
            ],
            flags=[],
        )

        from pathlib import Path

        dummy_path = Path("/dummy/path.yaml")

        metadata1 = compute_rubric_metadata(rubric, dummy_path)
        metadata2 = compute_rubric_metadata(rubric, dummy_path)

        # Hashes should be identical
        assert metadata1["rubric_hash"] == metadata2["rubric_hash"]
