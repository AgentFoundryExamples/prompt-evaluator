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
        expected_msg = (
            "status must be 'pending', 'completed', 'judge_error', or 'generation_error'"
        )
        with pytest.raises(ValueError, match=expected_msg):
            Sample(
                sample_id="sample-3",
                input_text="test",
                generator_output="test",
                status="invalid",
            )

    def test_sample_status_validation_valid(self):
        """Test that all valid status values are accepted."""
        for status in ["pending", "completed", "judge_error", "generation_error"]:
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
