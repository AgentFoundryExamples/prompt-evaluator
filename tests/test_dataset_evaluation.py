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
Tests for dataset evaluation models and orchestration.
"""

from datetime import datetime, timezone

import pytest

from prompt_evaluator.dataset_evaluation import (
    compute_overall_statistics,
    compute_per_case_statistics,
)
from prompt_evaluator.models import (
    DatasetEvaluationRun,
    GeneratorConfig,
    JudgeConfig,
    Rubric,
    RubricFlag,
    RubricMetric,
    Sample,
    TestCaseResult,
)


class TestTestCaseResultModel:
    """Tests for TestCaseResult data model."""

    def test_test_case_result_initialization(self):
        """Test that TestCaseResult can be initialized with valid data."""
        result = TestCaseResult(
            test_case_id="test-001",
            test_case_input="What is Python?",
            test_case_metadata={"task": "explain", "difficulty": "easy"},
            num_samples=5,
            status="completed",
        )

        assert result.test_case_id == "test-001"
        assert result.test_case_input == "What is Python?"
        assert result.test_case_metadata == {"task": "explain", "difficulty": "easy"}
        assert result.num_samples == 5
        assert result.status == "completed"
        assert result.samples == []
        assert result.per_metric_stats == {}
        assert result.per_flag_stats == {}

    def test_test_case_result_invalid_status(self):
        """Test that TestCaseResult rejects invalid status values."""
        with pytest.raises(ValueError, match="status must be one of"):
            TestCaseResult(
                test_case_id="test-001",
                test_case_input="test",
                status="invalid_status",
            )

    def test_test_case_result_valid_statuses(self):
        """Test that TestCaseResult accepts all valid status values."""
        valid_statuses = ["pending", "completed", "failed", "partial"]
        for status in valid_statuses:
            result = TestCaseResult(
                test_case_id="test-001",
                test_case_input="test",
                status=status,
            )
            assert result.status == status

    def test_test_case_result_to_dict(self):
        """Test that TestCaseResult can be serialized to dict."""
        timestamp = datetime.now(timezone.utc)
        result = TestCaseResult(
            test_case_id="test-001",
            test_case_input="What is Python?",
            test_case_metadata={"task": "explain"},
            num_samples=3,
            status="completed",
            timestamp_start=timestamp,
            timestamp_end=timestamp,
        )

        result_dict = result.to_dict()

        assert result_dict["test_case_id"] == "test-001"
        assert result_dict["test_case_input"] == "What is Python?"
        assert result_dict["test_case_metadata"] == {"task": "explain"}
        assert result_dict["num_samples"] == 3
        assert result_dict["status"] == "completed"
        assert result_dict["timestamp_start"] is not None
        assert result_dict["timestamp_end"] is not None
        assert result_dict["samples"] == []

    def test_test_case_result_with_samples(self):
        """Test TestCaseResult with sample data."""
        sample = Sample(
            sample_id="s1",
            input_text="test",
            generator_output="output",
            status="completed",
            judge_score=4.5,
        )

        result = TestCaseResult(
            test_case_id="test-001",
            test_case_input="test",
            num_samples=1,
            samples=[sample],
            status="completed",
        )

        assert len(result.samples) == 1
        assert result.samples[0].sample_id == "s1"

        result_dict = result.to_dict()
        assert len(result_dict["samples"]) == 1
        assert result_dict["samples"][0]["sample_id"] == "s1"


class TestDatasetEvaluationRunModel:
    """Tests for DatasetEvaluationRun data model."""

    def test_dataset_evaluation_run_initialization(self):
        """Test that DatasetEvaluationRun can be initialized with valid data."""
        gen_config = GeneratorConfig(model_name="gpt-4")
        judge_config = JudgeConfig(model_name="gpt-4")

        run = DatasetEvaluationRun(
            run_id="run-123",
            dataset_path="/path/to/dataset.yaml",
            dataset_hash="abc123",
            dataset_count=10,
            num_samples_per_case=5,
            generator_config=gen_config,
            judge_config=judge_config,
        )

        assert run.run_id == "run-123"
        assert run.dataset_path == "/path/to/dataset.yaml"
        assert run.dataset_hash == "abc123"
        assert run.dataset_count == 10
        assert run.num_samples_per_case == 5
        assert run.status == "running"
        assert run.test_case_results == []

    def test_dataset_evaluation_run_invalid_status(self):
        """Test that DatasetEvaluationRun rejects invalid status values."""
        gen_config = GeneratorConfig()
        judge_config = JudgeConfig()

        with pytest.raises(ValueError, match="status must be one of"):
            DatasetEvaluationRun(
                run_id="run-123",
                dataset_path="/path/to/dataset.yaml",
                dataset_hash="abc123",
                dataset_count=10,
                num_samples_per_case=5,
                generator_config=gen_config,
                judge_config=judge_config,
                status="invalid_status",
            )

    def test_dataset_evaluation_run_valid_statuses(self):
        """Test that DatasetEvaluationRun accepts all valid status values."""
        gen_config = GeneratorConfig()
        judge_config = JudgeConfig()
        valid_statuses = ["running", "completed", "failed", "aborted", "partial"]

        for status in valid_statuses:
            run = DatasetEvaluationRun(
                run_id="run-123",
                dataset_path="/path/to/dataset.yaml",
                dataset_hash="abc123",
                dataset_count=10,
                num_samples_per_case=5,
                generator_config=gen_config,
                judge_config=judge_config,
                status=status,
            )
            assert run.status == status

    def test_dataset_evaluation_run_to_dict(self):
        """Test that DatasetEvaluationRun can be serialized to dict."""
        gen_config = GeneratorConfig(model_name="gpt-4")
        judge_config = JudgeConfig(model_name="gpt-4")

        run = DatasetEvaluationRun(
            run_id="run-123",
            dataset_path="/path/to/dataset.yaml",
            dataset_hash="abc123",
            dataset_count=10,
            num_samples_per_case=5,
            generator_config=gen_config,
            judge_config=judge_config,
            status="completed",
        )

        run_dict = run.to_dict()

        assert run_dict["run_id"] == "run-123"
        assert run_dict["dataset_path"] == "/path/to/dataset.yaml"
        assert run_dict["dataset_hash"] == "abc123"
        assert run_dict["dataset_count"] == 10
        assert run_dict["num_samples_per_case"] == 5
        assert run_dict["status"] == "completed"
        assert "generator_config" in run_dict
        assert "judge_config" in run_dict
        assert run_dict["generator_config"]["model_name"] == "gpt-4"

    def test_dataset_evaluation_run_with_results(self):
        """Test DatasetEvaluationRun with test case results."""
        gen_config = GeneratorConfig()
        judge_config = JudgeConfig()

        test_case_result = TestCaseResult(
            test_case_id="test-001",
            test_case_input="test",
            num_samples=3,
            status="completed",
        )

        run = DatasetEvaluationRun(
            run_id="run-123",
            dataset_path="/path/to/dataset.yaml",
            dataset_hash="abc123",
            dataset_count=1,
            num_samples_per_case=3,
            generator_config=gen_config,
            judge_config=judge_config,
            test_case_results=[test_case_result],
        )

        assert len(run.test_case_results) == 1
        assert run.test_case_results[0].test_case_id == "test-001"

        run_dict = run.to_dict()
        assert len(run_dict["test_case_results"]) == 1
        assert run_dict["test_case_results"][0]["test_case_id"] == "test-001"


class TestComputePerCaseStatistics:
    """Tests for compute_per_case_statistics function."""

    def test_compute_per_case_statistics_with_rubric(self):
        """Test per-case statistics computation with rubric."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test metric",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[
                RubricFlag(name="flag1", description="Test flag"),
            ],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="out1",
                status="completed",
                judge_metrics={"metric1": {"score": 4.0, "rationale": "Good"}},
                judge_flags={"flag1": True},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="out2",
                status="completed",
                judge_metrics={"metric1": {"score": 5.0, "rationale": "Excellent"}},
                judge_flags={"flag1": False},
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="out3",
                status="completed",
                judge_metrics={"metric1": {"score": 3.0, "rationale": "OK"}},
                judge_flags={"flag1": True},
            ),
        ]

        per_metric_stats, per_flag_stats = compute_per_case_statistics(samples, rubric)

        # Check metric statistics
        assert "metric1" in per_metric_stats
        assert per_metric_stats["metric1"]["mean"] == 4.0
        assert per_metric_stats["metric1"]["min"] == 3.0
        assert per_metric_stats["metric1"]["max"] == 5.0
        assert per_metric_stats["metric1"]["count"] == 3
        # Sample standard deviation with n-1 denominator
        assert per_metric_stats["metric1"]["std"] == pytest.approx(1.0, rel=0.01)

        # Check flag statistics
        assert "flag1" in per_flag_stats
        assert per_flag_stats["flag1"]["true_count"] == 2
        assert per_flag_stats["flag1"]["false_count"] == 1
        assert per_flag_stats["flag1"]["total_count"] == 3
        assert per_flag_stats["flag1"]["true_proportion"] == pytest.approx(2 / 3)

    def test_compute_per_case_statistics_without_rubric(self):
        """Test that empty dicts are returned when no rubric is provided."""
        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="out1",
                status="completed",
                judge_score=4.0,
            ),
        ]

        per_metric_stats, per_flag_stats = compute_per_case_statistics(samples, rubric=None)

        assert per_metric_stats == {}
        assert per_flag_stats == {}

    def test_compute_per_case_statistics_single_sample(self):
        """Test statistics with a single sample (std should be 0)."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
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
                generator_output="out1",
                status="completed",
                judge_metrics={"metric1": {"score": 4.5, "rationale": "Good"}},
                judge_flags={},
            ),
        ]

        per_metric_stats, per_flag_stats = compute_per_case_statistics(samples, rubric)

        assert per_metric_stats["metric1"]["mean"] == 4.5
        assert per_metric_stats["metric1"]["std"] == 0.0
        assert per_metric_stats["metric1"]["min"] == 4.5
        assert per_metric_stats["metric1"]["max"] == 4.5
        assert per_metric_stats["metric1"]["count"] == 1

    def test_compute_per_case_statistics_excludes_invalid(self):
        """Test that invalid judge responses are excluded."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[RubricFlag(name="flag1", description="Test")],
        )

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="out1",
                status="completed",
                judge_metrics={"metric1": {"score": 4.0, "rationale": "Good"}},
                judge_flags={"flag1": True},
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="out2",
                status="judge_invalid_response",
                judge_metrics={},
                judge_flags={},
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="out3",
                status="completed",
                judge_metrics={"metric1": {"score": 5.0, "rationale": "Excellent"}},
                judge_flags={"flag1": False},
            ),
        ]

        per_metric_stats, per_flag_stats = compute_per_case_statistics(samples, rubric)

        # Only 2 samples should be counted
        assert per_metric_stats["metric1"]["count"] == 2
        assert per_metric_stats["metric1"]["mean"] == 4.5
        assert per_flag_stats["flag1"]["total_count"] == 2


class TestComputeOverallStatistics:
    """Tests for compute_overall_statistics function."""

    def test_compute_overall_statistics_mean_of_means(self):
        """Test that overall statistics compute mean of per-case means."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[],
        )

        # Create test case results with different per-case means
        tc1 = TestCaseResult(
            test_case_id="tc1",
            test_case_input="test1",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 4.0, "std": 0.5, "min": 3.5, "max": 4.5, "count": 3}
            },
            per_flag_stats={},
        )

        tc2 = TestCaseResult(
            test_case_id="tc2",
            test_case_input="test2",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0, "count": 2}
            },
            per_flag_stats={},
        )

        tc3 = TestCaseResult(
            test_case_id="tc3",
            test_case_input="test3",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 3.0, "std": 1.0, "min": 2.0, "max": 4.0, "count": 4}
            },
            per_flag_stats={},
        )

        test_case_results = [tc1, tc2, tc3]

        overall_metric_stats, overall_flag_stats = compute_overall_statistics(
            test_case_results, rubric
        )

        # Mean of means: (4.0 + 5.0 + 3.0) / 3 = 4.0
        assert overall_metric_stats["metric1"]["mean_of_means"] == 4.0
        assert overall_metric_stats["metric1"]["min_of_means"] == 3.0
        assert overall_metric_stats["metric1"]["max_of_means"] == 5.0
        assert overall_metric_stats["metric1"]["num_cases"] == 3

    def test_compute_overall_statistics_aggregates_flags(self):
        """Test that overall flag statistics aggregate across all samples."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[
                RubricFlag(name="flag1", description="Test flag"),
            ],
        )

        # Test case 1: 2 true, 1 false (3 samples)
        tc1 = TestCaseResult(
            test_case_id="tc1",
            test_case_input="test1",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 4.0, "std": 0.0, "min": 4.0, "max": 4.0, "count": 3}
            },
            per_flag_stats={
                "flag1": {
                    "true_count": 2,
                    "false_count": 1,
                    "total_count": 3,
                    "true_proportion": 2 / 3,
                }
            },
        )

        # Test case 2: 1 true, 2 false (3 samples)
        tc2 = TestCaseResult(
            test_case_id="tc2",
            test_case_input="test2",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0, "count": 3}
            },
            per_flag_stats={
                "flag1": {
                    "true_count": 1,
                    "false_count": 2,
                    "total_count": 3,
                    "true_proportion": 1 / 3,
                }
            },
        )

        test_case_results = [tc1, tc2]

        overall_metric_stats, overall_flag_stats = compute_overall_statistics(
            test_case_results, rubric
        )

        # Overall: 3 true, 3 false out of 6 total
        assert overall_flag_stats["flag1"]["true_count"] == 3
        assert overall_flag_stats["flag1"]["false_count"] == 3
        assert overall_flag_stats["flag1"]["total_count"] == 6
        assert overall_flag_stats["flag1"]["true_proportion"] == 0.5

    def test_compute_overall_statistics_without_rubric(self):
        """Test that empty dicts are returned when no rubric is provided."""
        tc1 = TestCaseResult(
            test_case_id="tc1",
            test_case_input="test1",
            status="completed",
        )

        overall_metric_stats, overall_flag_stats = compute_overall_statistics([tc1], rubric=None)

        assert overall_metric_stats == {}
        assert overall_flag_stats == {}

    def test_compute_overall_statistics_excludes_failed_cases(self):
        """Test that failed test cases are excluded from overall statistics."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[],
        )

        tc1 = TestCaseResult(
            test_case_id="tc1",
            test_case_input="test1",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 4.0, "std": 0.0, "min": 4.0, "max": 4.0, "count": 3}
            },
            per_flag_stats={},
        )

        tc2 = TestCaseResult(
            test_case_id="tc2",
            test_case_input="test2",
            status="failed",
            per_metric_stats={
                "metric1": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0, "count": 3}
            },
            per_flag_stats={},
        )

        test_case_results = [tc1, tc2]

        overall_metric_stats, overall_flag_stats = compute_overall_statistics(
            test_case_results, rubric
        )

        # Only tc1 should be included
        assert overall_metric_stats["metric1"]["mean_of_means"] == 4.0
        assert overall_metric_stats["metric1"]["num_cases"] == 1

    def test_compute_overall_statistics_includes_partial_cases(self):
        """Test that partial test cases are included in overall statistics."""
        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="Test",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
            flags=[
                RubricFlag(name="flag1", description="Test flag"),
            ],
        )

        tc1 = TestCaseResult(
            test_case_id="tc1",
            test_case_input="test1",
            status="completed",
            per_metric_stats={
                "metric1": {"mean": 4.0, "std": 0.0, "min": 4.0, "max": 4.0, "count": 3}
            },
            per_flag_stats={
                "flag1": {
                    "true_count": 2,
                    "false_count": 1,
                    "total_count": 3,
                    "true_proportion": 2 / 3,
                }
            },
        )

        tc2 = TestCaseResult(
            test_case_id="tc2",
            test_case_input="test2",
            status="partial",
            per_metric_stats={
                "metric1": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0, "count": 2}
            },
            per_flag_stats={
                "flag1": {
                    "true_count": 1,
                    "false_count": 1,
                    "total_count": 2,
                    "true_proportion": 0.5,
                }
            },
        )

        test_case_results = [tc1, tc2]

        overall_metric_stats, overall_flag_stats = compute_overall_statistics(
            test_case_results, rubric
        )

        # Both tc1 and tc2 should be included
        assert overall_metric_stats["metric1"]["mean_of_means"] == 4.5
        assert overall_metric_stats["metric1"]["num_cases"] == 2
        # Flags: tc1 has 2 true, tc2 has 1 true = 3 total true out of 5 samples
        assert overall_flag_stats["flag1"]["true_count"] == 3
        assert overall_flag_stats["flag1"]["total_count"] == 5
