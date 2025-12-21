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
Tests for judge models and evaluation data structures.

Tests validate JudgeConfig, Sample, SingleEvaluationRun, and judge functionality.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from prompt_evaluator.models import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    GeneratorConfig,
    JudgeConfig,
    Sample,
    SingleEvaluationRun,
    load_judge_prompt,
)
from prompt_evaluator.provider import OpenAIProvider, judge_completion


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_values(self):
        """Test that JudgeConfig has sensible defaults."""
        config = JudgeConfig()
        assert config.model_name == "gpt-5.1"
        assert config.temperature == 0.0
        assert config.max_completion_tokens == 512
        assert config.seed is None

    def test_custom_values(self):
        """Test that JudgeConfig accepts custom values."""
        config = JudgeConfig(
            model_name="gpt-4", temperature=0.3, max_completion_tokens=1024, seed=42
        )
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.3
        assert config.max_completion_tokens == 1024
        assert config.seed == 42

    def test_temperature_validation_negative(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            JudgeConfig(temperature=-0.1)

    def test_temperature_validation_too_high(self):
        """Test that temperature > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            JudgeConfig(temperature=2.1)

    def test_max_completion_tokens_validation_zero(self):
        """Test that max_completion_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            JudgeConfig(max_completion_tokens=0)

    def test_max_completion_tokens_validation_negative(self):
        """Test that negative max_completion_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            JudgeConfig(max_completion_tokens=-100)

    def test_max_completion_tokens_validation_non_integer(self):
        """Test that non-integer max_completion_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be an integer"):
            JudgeConfig(max_completion_tokens=100.5)  # type: ignore[arg-type]


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_creation_minimal(self):
        """Test that Sample can be created with required fields only."""
        sample = Sample(
            sample_id="sample-1",
            input_text="What is Python?",
            generator_output="Python is a programming language.",
        )

        assert sample.sample_id == "sample-1"
        assert sample.input_text == "What is Python?"
        assert sample.generator_output == "Python is a programming language."
        assert sample.judge_score is None
        assert sample.judge_rationale is None
        assert sample.judge_raw_response is None
        assert sample.status == "pending"
        assert sample.task_description is None

    def test_sample_creation_full(self):
        """Test that Sample can be created with all fields."""
        sample = Sample(
            sample_id="sample-2",
            input_text="What is Python?",
            generator_output="Python is a programming language.",
            judge_score=4.5,
            judge_rationale="Good semantic fidelity",
            judge_raw_response='{"semantic_fidelity": 4.5, "rationale": "Good"}',
            status="completed",
            task_description="Explain programming concepts",
        )

        assert sample.sample_id == "sample-2"
        assert sample.judge_score == 4.5
        assert sample.judge_rationale == "Good semantic fidelity"
        assert sample.status == "completed"
        assert sample.task_description == "Explain programming concepts"

    def test_sample_status_validation_invalid(self):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="status must be one of"):
            Sample(
                sample_id="sample-3",
                input_text="test",
                generator_output="test",
                status="invalid",
            )

    def test_sample_status_validation_valid(self):
        """Test that all valid status values are accepted."""
        for status in [
            "pending",
            "completed",
            "judge_error",
            "generation_error",
            "judge_invalid_response",
        ]:
            sample = Sample(
                sample_id=f"sample-{status}",
                input_text="test",
                generator_output="test",
                status=status,
            )
            assert sample.status == status

    def test_sample_judge_score_validation_too_low(self):
        """Test that judge_score < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="judge_score must be between 1.0 and 5.0"):
            Sample(
                sample_id="sample-4",
                input_text="test",
                generator_output="test",
                judge_score=0.5,
            )

    def test_sample_judge_score_validation_too_high(self):
        """Test that judge_score > 5.0 raises ValueError."""
        with pytest.raises(ValueError, match="judge_score must be between 1.0 and 5.0"):
            Sample(
                sample_id="sample-5",
                input_text="test",
                generator_output="test",
                judge_score=5.5,
            )

    def test_sample_judge_score_validation_boundaries(self):
        """Test that judge_score at boundaries (1.0, 5.0) is valid."""
        sample1 = Sample(
            sample_id="sample-6",
            input_text="test",
            generator_output="test",
            judge_score=1.0,
        )
        assert sample1.judge_score == 1.0

        sample2 = Sample(
            sample_id="sample-7",
            input_text="test",
            generator_output="test",
            judge_score=5.0,
        )
        assert sample2.judge_score == 5.0

    def test_sample_to_dict(self):
        """Test that Sample.to_dict() produces JSON-compatible dict."""
        sample = Sample(
            sample_id="sample-8",
            input_text="What is Python?",
            generator_output="Python is a programming language.",
            judge_score=4.0,
            judge_rationale="Good answer",
            judge_raw_response='{"semantic_fidelity": 4, "rationale": "Good answer"}',
            status="completed",
            task_description="Explain programming",
        )

        result = sample.to_dict()

        assert result["sample_id"] == "sample-8"
        assert result["input_text"] == "What is Python?"
        assert result["generator_output"] == "Python is a programming language."
        assert result["judge_score"] == 4.0
        assert result["judge_rationale"] == "Good answer"
        assert result["status"] == "completed"
        assert result["task_description"] == "Explain programming"

    def test_sample_judge_metrics_validation(self):
        """Test that judge_metrics are validated in __post_init__."""
        # Valid metrics should work
        sample = Sample(
            sample_id="sample-9",
            input_text="test",
            generator_output="test",
            judge_metrics={"quality": {"score": 4.0, "rationale": "Good"}},
            status="completed",
        )
        assert sample.judge_metrics["quality"]["score"] == 4.0

        # Missing score field should fail
        with pytest.raises(ValueError, match="must have a 'score' field"):
            Sample(
                sample_id="sample-10",
                input_text="test",
                generator_output="test",
                judge_metrics={"quality": {"rationale": "Good"}},
            )

        # Non-numeric score should fail
        with pytest.raises(ValueError, match="score must be numeric"):
            Sample(
                sample_id="sample-11",
                input_text="test",
                generator_output="test",
                judge_metrics={"quality": {"score": "high", "rationale": "Good"}},
            )

    def test_sample_judge_flags_validation(self):
        """Test that judge_flags are validated in __post_init__."""
        # Valid flags should work
        sample = Sample(
            sample_id="sample-12",
            input_text="test",
            generator_output="test",
            judge_flags={"has_issues": False},
            status="completed",
        )
        assert sample.judge_flags["has_issues"] is False

        # Non-boolean flag should fail
        with pytest.raises(ValueError, match="must be boolean"):
            Sample(
                sample_id="sample-13",
                input_text="test",
                generator_output="test",
                judge_flags={"has_issues": "false"},
            )


class TestSingleEvaluationRun:
    """Tests for SingleEvaluationRun dataclass."""

    def test_single_evaluation_run_creation(self):
        """Test that SingleEvaluationRun can be created with required fields."""
        gen_config = GeneratorConfig()
        judge_config = JudgeConfig()
        timestamp = datetime.now(timezone.utc)

        run = SingleEvaluationRun(
            run_id="run-123",
            timestamp=timestamp,
            num_samples=5,
            generator_config=gen_config,
            judge_config=judge_config,
        )

        assert run.run_id == "run-123"
        assert run.timestamp == timestamp
        assert run.num_samples == 5
        assert run.generator_config == gen_config
        assert run.judge_config == judge_config
        assert run.samples == []

    def test_single_evaluation_run_with_samples(self):
        """Test that SingleEvaluationRun can include samples."""
        gen_config = GeneratorConfig()
        judge_config = JudgeConfig()
        timestamp = datetime.now(timezone.utc)

        sample1 = Sample(
            sample_id="s1", input_text="input1", generator_output="output1"
        )
        sample2 = Sample(
            sample_id="s2", input_text="input2", generator_output="output2"
        )

        run = SingleEvaluationRun(
            run_id="run-456",
            timestamp=timestamp,
            num_samples=2,
            generator_config=gen_config,
            judge_config=judge_config,
            samples=[sample1, sample2],
        )

        assert len(run.samples) == 2
        assert run.samples[0].sample_id == "s1"
        assert run.samples[1].sample_id == "s2"

    def test_single_evaluation_run_to_dict(self):
        """Test that SingleEvaluationRun.to_dict() produces JSON-compatible dict."""
        gen_config = GeneratorConfig(model_name="gpt-4", temperature=0.5)
        judge_config = JudgeConfig(model_name="gpt-4", temperature=0.0)
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        sample = Sample(
            sample_id="s1",
            input_text="input",
            generator_output="output",
            judge_score=4.0,
            status="completed",
        )

        run = SingleEvaluationRun(
            run_id="run-789",
            timestamp=timestamp,
            num_samples=1,
            generator_config=gen_config,
            judge_config=judge_config,
            samples=[sample],
        )

        result = run.to_dict()

        assert result["run_id"] == "run-789"
        assert result["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert result["num_samples"] == 1
        assert result["generator_config"]["model_name"] == "gpt-4"
        assert result["generator_config"]["temperature"] == 0.5
        assert result["judge_config"]["model_name"] == "gpt-4"
        assert result["judge_config"]["temperature"] == 0.0
        assert len(result["samples"]) == 1
        assert result["samples"][0]["sample_id"] == "s1"
        assert result["samples"][0]["judge_score"] == 4.0


class TestDefaultJudgeSystemPrompt:
    """Tests for DEFAULT_JUDGE_SYSTEM_PROMPT constant."""

    def test_default_prompt_exists(self):
        """Test that default judge prompt is defined."""
        assert DEFAULT_JUDGE_SYSTEM_PROMPT is not None
        assert len(DEFAULT_JUDGE_SYSTEM_PROMPT) > 0

    def test_default_prompt_contains_scoring_scale(self):
        """Test that default prompt mentions scoring scale 1-5."""
        assert "1-5" in DEFAULT_JUDGE_SYSTEM_PROMPT
        assert "semantic_fidelity" in DEFAULT_JUDGE_SYSTEM_PROMPT
        assert "rationale" in DEFAULT_JUDGE_SYSTEM_PROMPT

    def test_default_prompt_contains_json_schema(self):
        """Test that default prompt mentions JSON response format."""
        assert "JSON" in DEFAULT_JUDGE_SYSTEM_PROMPT
        assert "{" in DEFAULT_JUDGE_SYSTEM_PROMPT
        assert "}" in DEFAULT_JUDGE_SYSTEM_PROMPT


class TestLoadJudgePrompt:
    """Tests for load_judge_prompt function."""

    def test_load_judge_prompt_default(self):
        """Test that load_judge_prompt returns default when no path provided."""
        prompt = load_judge_prompt()
        assert prompt == DEFAULT_JUDGE_SYSTEM_PROMPT

    def test_load_judge_prompt_none_path(self):
        """Test that load_judge_prompt returns default when path is None."""
        prompt = load_judge_prompt(None)
        assert prompt == DEFAULT_JUDGE_SYSTEM_PROMPT

    def test_load_judge_prompt_from_file(self, tmp_path):
        """Test that load_judge_prompt reads from file."""
        prompt_file = tmp_path / "custom_prompt.txt"
        custom_prompt = "This is a custom judge prompt with scoring instructions."
        prompt_file.write_text(custom_prompt)

        prompt = load_judge_prompt(prompt_file)
        assert prompt == custom_prompt

    def test_load_judge_prompt_file_not_found(self, tmp_path):
        """Test that load_judge_prompt raises FileNotFoundError for missing file."""
        non_existent_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="Judge prompt file not found"):
            load_judge_prompt(non_existent_file)

    def test_load_judge_prompt_empty_file(self, tmp_path):
        """Test that load_judge_prompt raises ValueError for empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="Judge prompt file is empty"):
            load_judge_prompt(empty_file)

    def test_load_judge_prompt_whitespace_only(self, tmp_path):
        """Test that load_judge_prompt raises ValueError for whitespace-only file."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\t\n   ")

        with pytest.raises(ValueError, match="Judge prompt file is empty"):
            load_judge_prompt(whitespace_file)

    def test_load_judge_prompt_strips_whitespace(self, tmp_path):
        """Test that load_judge_prompt strips leading/trailing whitespace."""
        prompt_file = tmp_path / "padded_prompt.txt"
        prompt_file.write_text("  \n  Custom prompt\n\n  ")

        prompt = load_judge_prompt(prompt_file)
        assert prompt == "Custom prompt"


class TestJudgeCompletion:
    """Tests for judge_completion function."""

    def test_judge_completion_success(self):
        """Test successful judge completion with valid JSON response."""
        # Mock provider and generate_completion
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        valid_json = '{"semantic_fidelity": 4.5, "rationale": "Good semantic preservation"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(valid_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="What is Python?",
                generator_output="Python is a programming language.",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 4.5
        assert result["judge_rationale"] == "Good semantic preservation"
        assert result["judge_raw_response"] == valid_json
        assert result["error"] is None

    def test_judge_completion_with_task_description(self):
        """Test judge completion includes task description in user message."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        valid_json = '{"semantic_fidelity": 3.0, "rationale": "Acceptable"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(valid_json, {"tokens_used": 100, "latency_ms": 500}),
        ) as mock_generate:
            result = judge_completion(
                provider=provider,
                input_text="Explain Python",
                generator_output="Python is a language",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
                task_description="Explain programming concepts clearly",
            )

        # Check that task description was included in the call
        call_args = mock_generate.call_args
        user_prompt = call_args[1]["user_prompt"]
        assert "Task: Explain programming concepts clearly" in user_prompt

        assert result["status"] == "completed"

    def test_judge_completion_json_with_extra_text(self):
        """Test that JSON can be extracted from response with extra text."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        response_with_extra = (
            'Here is my evaluation:\n{"semantic_fidelity": 3.5, '
            '"rationale": "Mostly good"}\nHope this helps!'
        )

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(response_with_extra, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test input",
                generator_output="test output",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 3.5
        assert result["judge_rationale"] == "Mostly good"

    def test_judge_completion_score_clamping_below_minimum(self):
        """Test that scores below 1.0 are clamped to 1.0."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        low_score_json = '{"semantic_fidelity": 0.5, "rationale": "Very poor"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(low_score_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 1.0

    def test_judge_completion_score_clamping_above_maximum(self):
        """Test that scores above 5.0 are clamped to 5.0."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        high_score_json = '{"semantic_fidelity": 6.5, "rationale": "Excellent"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(high_score_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 5.0

    def test_judge_completion_invalid_json(self):
        """Test that invalid JSON results in judge_error status."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        invalid_json = "This is not JSON at all"

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(invalid_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert result["judge_score"] is None
        assert result["judge_rationale"] is None
        assert result["judge_raw_response"] == invalid_json
        assert "Failed to parse judge response" in result["error"]

    def test_judge_completion_missing_semantic_fidelity_field(self):
        """Test that missing semantic_fidelity field results in judge_error."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        missing_field_json = '{"rationale": "Good answer"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(missing_field_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert "Missing required field: semantic_fidelity" in result["error"]

    def test_judge_completion_missing_rationale_field(self):
        """Test that missing rationale field results in judge_error."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        missing_field_json = '{"semantic_fidelity": 4.0}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(missing_field_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert "Missing required field: rationale" in result["error"]

    def test_judge_completion_api_exception(self):
        """Test that API exceptions result in judge_error status."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        with patch(
            "prompt_evaluator.provider.generate_completion",
            side_effect=Exception("API connection failed"),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert result["judge_score"] is None
        assert result["judge_rationale"] is None
        assert "Judge API call failed" in result["error"]
        assert "API connection failed" in result["error"]
        assert result["judge_raw_response"] is None

    def test_judge_completion_non_numeric_score(self):
        """Test that non-numeric semantic_fidelity results in judge_error."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        non_numeric_json = '{"semantic_fidelity": "high", "rationale": "Good"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(non_numeric_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert "Failed to parse judge response" in result["error"]

    def test_judge_completion_integer_score(self):
        """Test that integer scores are converted to float correctly."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        integer_score_json = '{"semantic_fidelity": 4, "rationale": "Good answer"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(integer_score_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 4.0
        assert isinstance(result["judge_score"], float)

    def test_judge_completion_json_with_extra_fields(self):
        """Test that JSON with extra fields is handled correctly."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        json_with_extras = (
            '{"semantic_fidelity": 3.5, "rationale": "Acceptable", '
            '"confidence": 0.9, "metadata": {"source": "test"}}'
        )

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(json_with_extras, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "completed"
        assert result["judge_score"] == 3.5
        assert result["judge_rationale"] == "Acceptable"

    def test_judge_completion_malformed_nested_json(self):
        """Test that malformed nested JSON results in judge_error."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        malformed_json = '{"semantic_fidelity": {"value": 4.0}, "rationale": "Test"}'

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(malformed_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert "Failed to parse judge response" in result["error"]

    def test_judge_completion_empty_response(self):
        """Test that empty response results in judge_error."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=("", {"tokens_used": 0, "latency_ms": 100}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert result["judge_raw_response"] == ""
        assert "Failed to parse judge response" in result["error"]

    def test_judge_completion_score_at_exact_boundaries(self):
        """Test scores at exact 1.0 and 5.0 boundaries."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        # Test exact 1.0
        json_min = '{"semantic_fidelity": 1.0, "rationale": "Minimum score"}'
        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(json_min, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )
        assert result["status"] == "completed"
        assert result["judge_score"] == 1.0

        # Test exact 5.0
        json_max = '{"semantic_fidelity": 5.0, "rationale": "Maximum score"}'
        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(json_max, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )
        assert result["status"] == "completed"
        assert result["judge_score"] == 5.0

    def test_judge_completion_preserves_raw_response_on_error(self):
        """Test that raw response is preserved even when parsing fails."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        raw_text = "This is not JSON but should be preserved for debugging"

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(raw_text, {"tokens_used": 50, "latency_ms": 200}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="test input",
                generator_output="test output",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert result["judge_raw_response"] == raw_text
        assert result["judge_score"] is None
        assert result["judge_rationale"] is None

    def test_judge_completion_api_exception_with_response(self):
        """Test that raw response from exception is preserved when available."""
        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        # Create exception with response attribute
        exception = Exception("API error")
        mock_response = MagicMock()
        mock_response.text = "Raw response from failed API call"
        exception.response = mock_response

        with patch(
            "prompt_evaluator.provider.generate_completion",
            side_effect=exception,
        ):
            result = judge_completion(
                provider=provider,
                input_text="test",
                generator_output="test",
                judge_config=judge_config,
                judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )

        assert result["status"] == "judge_error"
        assert result["judge_score"] is None
        assert result["judge_rationale"] is None
        assert result["judge_raw_response"] == "Raw response from failed API call"
        assert "Judge API call failed" in result["error"]


class TestRubricAwareJudge:
    """Tests for rubric-aware judge functionality."""

    def test_build_rubric_judge_prompt(self):
        """Test that rubric-aware prompt is generated correctly."""
        from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric
        from prompt_evaluator.provider import build_rubric_judge_prompt

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="semantic_fidelity",
                    description="Semantic preservation",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5 based on meaning",
                ),
                RubricMetric(
                    name="clarity",
                    description="Output clarity",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate clarity 1-5",
                ),
            ],
            flags=[
                RubricFlag(name="invented_constraints", description="Added constraints"),
            ],
        )

        prompt = build_rubric_judge_prompt(rubric)

        # Check that prompt contains metric information
        assert "semantic_fidelity" in prompt
        assert "Semantic preservation" in prompt
        assert "1.0 (minimum) to 5.0 (maximum)" in prompt
        assert "Rate 1-5 based on meaning" in prompt

        assert "clarity" in prompt
        assert "Output clarity" in prompt

        # Check that prompt contains flag information
        assert "invented_constraints" in prompt
        assert "Added constraints" in prompt

        # Check that prompt contains schema
        assert "metrics" in prompt
        assert "flags" in prompt
        assert "overall_comment" in prompt
        assert "JSON" in prompt or "json" in prompt

    def test_parse_rubric_judge_response_valid(self):
        """Test parsing valid multi-metric response."""
        from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="semantic_fidelity",
                    description="Semantic preservation",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
                RubricMetric(
                    name="clarity",
                    description="Clarity",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
            flags=[
                RubricFlag(name="has_issues", description="Has issues"),
            ],
        )

        response_json = json.dumps(
            {
                "metrics": {
                    "semantic_fidelity": {"score": 4.5, "rationale": "Good preservation"},
                    "clarity": {"score": 3.0, "rationale": "Acceptable clarity"},
                },
                "flags": {"has_issues": False},
                "overall_comment": "Overall good response",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "completed"
        assert result["judge_metrics"]["semantic_fidelity"]["score"] == 4.5
        assert result["judge_metrics"]["semantic_fidelity"]["rationale"] == "Good preservation"
        assert result["judge_metrics"]["clarity"]["score"] == 3.0
        assert result["judge_flags"]["has_issues"] is False
        assert result["judge_overall_comment"] == "Overall good response"
        assert result["error"] is None

    def test_parse_rubric_judge_response_with_markdown(self):
        """Test parsing response wrapped in markdown code fences."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate quality",
                ),
            ],
        )

        response_with_markdown = (
            "Here is my evaluation:\n```json\n"
            + json.dumps(
                {
                    "metrics": {"quality": {"score": 4.0, "rationale": "Good"}},
                    "flags": {},
                    "overall_comment": "Well done",
                }
            )
            + "\n```\nHope this helps!"
        )

        result = parse_rubric_judge_response(response_with_markdown, rubric)

        assert result["status"] == "completed"
        assert result["judge_metrics"]["quality"]["score"] == 4.0

    def test_parse_rubric_judge_response_missing_metric(self):
        """Test that missing metrics are rejected."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="M1",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
                RubricMetric(
                    name="metric2",
                    description="M2",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
        )

        # Response missing metric2
        response_json = json.dumps(
            {
                "metrics": {"metric1": {"score": 3.0, "rationale": "Good"}},
                "flags": {},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "Missing required metrics: metric2" in result["error"]

    def test_parse_rubric_judge_response_extra_metric(self):
        """Test that extra unknown metrics are rejected."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="metric1",
                    description="M1",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test",
                ),
            ],
        )

        # Response with extra unknown metric
        response_json = json.dumps(
            {
                "metrics": {
                    "metric1": {"score": 3.0, "rationale": "Good"},
                    "unknown_metric": {"score": 2.0, "rationale": "Bad"},
                },
                "flags": {},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "Unknown metrics provided: unknown_metric" in result["error"]

    def test_parse_rubric_judge_response_out_of_range_score(self):
        """Test that out-of-range scores are rejected."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
        )

        # Score outside valid range
        response_json = json.dumps(
            {
                "metrics": {"quality": {"score": 6.5, "rationale": "Excellent"}},
                "flags": {},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "out of range" in result["error"]
        assert "6.5" in result["error"]

    def test_parse_rubric_judge_response_non_numeric_score(self):
        """Test that non-numeric scores are rejected."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
        )

        # Non-numeric score
        response_json = json.dumps(
            {
                "metrics": {"quality": {"score": "high", "rationale": "Good"}},
                "flags": {},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "score must be numeric" in result["error"]

    def test_parse_rubric_judge_response_missing_flag(self):
        """Test that missing flags are rejected."""
        from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
            flags=[
                RubricFlag(name="flag1", description="Flag 1"),
                RubricFlag(name="flag2", description="Flag 2"),
            ],
        )

        # Response missing flag2
        response_json = json.dumps(
            {
                "metrics": {"quality": {"score": 4.0, "rationale": "Good"}},
                "flags": {"flag1": True},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "Missing required flags: flag2" in result["error"]

    def test_parse_rubric_judge_response_non_boolean_flag(self):
        """Test that non-boolean flag values are rejected."""
        from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
            flags=[RubricFlag(name="has_issues", description="Has issues")],
        )

        # Flag as string instead of boolean
        response_json = json.dumps(
            {
                "metrics": {"quality": {"score": 4.0, "rationale": "Good"}},
                "flags": {"has_issues": "false"},
                "overall_comment": "Comment",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "must be a boolean" in result["error"]

    def test_parse_rubric_judge_response_missing_overall_comment(self):
        """Test that missing overall_comment is rejected."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate 1-5",
                ),
            ],
        )

        # Response without overall_comment
        response_json = json.dumps(
            {"metrics": {"quality": {"score": 4.0, "rationale": "Good"}}, "flags": {}}
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "judge_invalid_response"
        assert "Missing required field: overall_comment" in result["error"]

    def test_judge_completion_with_rubric(self):
        """Test judge_completion with rubric uses rubric-aware logic."""
        from prompt_evaluator.models import Rubric, RubricMetric

        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate quality 1-5",
                ),
            ],
        )

        valid_json = json.dumps(
            {
                "metrics": {"quality": {"score": 4.0, "rationale": "Good quality"}},
                "flags": {},
                "overall_comment": "Well executed",
            }
        )

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(valid_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="Test input",
                generator_output="Test output",
                judge_config=judge_config,
                judge_system_prompt="<ignored when rubric provided>",
                rubric=rubric,
            )

        assert result["status"] == "completed"
        assert result["judge_metrics"]["quality"]["score"] == 4.0
        assert result["judge_metrics"]["quality"]["rationale"] == "Good quality"
        assert result["judge_overall_comment"] == "Well executed"
        assert result["judge_score"] is None  # Not used in rubric mode
        assert result["judge_rationale"] is None  # Not used in rubric mode

    def test_judge_completion_with_rubric_invalid_response(self):
        """Test that invalid rubric response sets judge_invalid_response status."""
        from prompt_evaluator.models import Rubric, RubricMetric

        provider = MagicMock(spec=OpenAIProvider)
        judge_config = JudgeConfig()

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate quality 1-5",
                ),
            ],
        )

        # Invalid JSON - missing metrics field
        invalid_json = json.dumps({"flags": {}, "overall_comment": "Comment"})

        with patch(
            "prompt_evaluator.provider.generate_completion",
            return_value=(invalid_json, {"tokens_used": 100, "latency_ms": 500}),
        ):
            result = judge_completion(
                provider=provider,
                input_text="Test input",
                generator_output="Test output",
                judge_config=judge_config,
                judge_system_prompt="<ignored>",
                rubric=rubric,
            )

        assert result["status"] == "judge_invalid_response"
        assert "Missing required field: metrics" in result["error"]
        assert result["judge_raw_response"] == invalid_json

    def test_parse_rubric_judge_response_nested_json(self):
        """Test that nested JSON structures are handled correctly."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate quality",
                ),
            ],
        )

        # Response with nested object in commentary (should extract first complete JSON)
        response_with_nested = (
            'Here is my evaluation: {"metrics": {"quality": {"score": 4.0, '
            '"rationale": "Good"}}, "flags": {}, "overall_comment": "Well done"} '
            'And here is some data: {"extra": "info"}'
        )

        result = parse_rubric_judge_response(response_with_nested, rubric)

        assert result["status"] == "completed"
        assert result["judge_metrics"]["quality"]["score"] == 4.0

    def test_parse_rubric_judge_response_escaped_braces(self):
        """Test handling of escaped braces in strings."""
        from prompt_evaluator.models import Rubric, RubricMetric
        from prompt_evaluator.provider import parse_rubric_judge_response

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="quality",
                    description="Quality",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Rate quality",
                ),
            ],
        )

        # JSON with escaped braces in rationale
        response_json = json.dumps(
            {
                "metrics": {
                    "quality": {
                        "score": 4.0,
                        "rationale": "Uses {templates} correctly",
                    }
                },
                "flags": {},
                "overall_comment": "Good use of {syntax}",
            }
        )

        result = parse_rubric_judge_response(response_json, rubric)

        assert result["status"] == "completed"
        assert result["judge_metrics"]["quality"]["score"] == 4.0
        assert "templates" in result["judge_metrics"]["quality"]["rationale"]
