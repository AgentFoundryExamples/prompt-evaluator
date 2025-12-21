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
Basic tests for the prompt_evaluator package.

These tests verify that the package structure is correct and
modules can be imported without errors.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import prompt_evaluator

    assert hasattr(prompt_evaluator, "__version__")
    assert prompt_evaluator.__version__ == "0.1.0"


def test_module_imports():
    """Test that all main modules can be imported."""
    from prompt_evaluator import cli, config, models, provider

    # Verify modules are loaded
    assert cli is not None
    assert config is not None
    assert models is not None
    assert provider is not None


def test_cli_entrypoint_exists():
    """Test that CLI main function exists."""
    from prompt_evaluator.cli import main

    assert callable(main)


def test_config_models_defined():
    """Test that configuration models are defined."""
    from prompt_evaluator.config import EvaluationConfig, ProviderConfig

    assert ProviderConfig is not None
    assert EvaluationConfig is not None


def test_data_models_defined():
    """Test that data models are defined."""
    from prompt_evaluator.models import (
        EvaluationRequest,
        EvaluationResponse,
        EvaluationResult,
        PromptTemplate,
    )

    assert PromptTemplate is not None
    assert EvaluationRequest is not None
    assert EvaluationResponse is not None
    assert EvaluationResult is not None


def test_judge_models_defined():
    """Test that judge models and structures are defined."""
    from prompt_evaluator.models import (
        DEFAULT_JUDGE_SYSTEM_PROMPT,
        JudgeConfig,
        Sample,
        SingleEvaluationRun,
        load_judge_prompt,
    )

    assert JudgeConfig is not None
    assert Sample is not None
    assert SingleEvaluationRun is not None
    assert DEFAULT_JUDGE_SYSTEM_PROMPT is not None
    assert callable(load_judge_prompt)


def test_judge_completion_defined():
    """Test that judge_completion function is defined."""
    from prompt_evaluator.provider import judge_completion

    assert callable(judge_completion)


def test_provider_base_class_defined():
    """Test that provider classes are defined."""
    from prompt_evaluator.provider import BaseProvider, OpenAIProvider, get_provider

    assert BaseProvider is not None
    assert OpenAIProvider is not None
    assert callable(get_provider)


def test_prompt_template_render():
    """Test that PromptTemplate can render templates."""
    from prompt_evaluator.models import PromptTemplate

    template = PromptTemplate(
        template="Hello {name}, you are {age} years old.",
        variables={"name": "User's name", "age": "User's age"},
    )

    result = template.render(name="Alice", age=30)
    assert result == "Hello Alice, you are 30 years old."


def test_prompt_template_validation():
    """Test that PromptTemplate validation prevents unsafe format strings."""
    from prompt_evaluator.models import PromptTemplate

    # Valid template should work
    valid_template = PromptTemplate(template="Hello {name}", variables={"name": "User's name"})
    assert valid_template.template == "Hello {name}"

    # Dangerous patterns should be rejected
    dangerous_templates = [
        "{obj.attr}",  # Attribute access
        "{obj[0]}",  # Indexing
        "{var!r}",  # Conversion
        "{var:03d}",  # Format spec
    ]

    for dangerous in dangerous_templates:
        with pytest.raises(ValueError, match="potentially unsafe format string"):
            PromptTemplate(template=dangerous, variables={})


def test_prompt_template_missing_variable():
    """Test that PromptTemplate raises clear error for missing variables."""
    from prompt_evaluator.models import PromptTemplate

    template = PromptTemplate(
        template="Hello {name} and {friend}",
        variables={"name": "User's name", "friend": "Friend's name"},
    )

    with pytest.raises(KeyError, match="Missing required template variable"):
        template.render(name="Alice")  # Missing 'friend'


def test_get_provider_openai():
    """Test that OpenAI provider can be retrieved."""
    from prompt_evaluator.provider import OpenAIProvider, get_provider

    # Provider can be instantiated even without API key (will fail on actual API calls)
    provider = get_provider("openai", api_key="test-key")
    assert isinstance(provider, OpenAIProvider)


def test_get_provider_invalid():
    """Test that invalid provider name raises error."""
    from prompt_evaluator.provider import get_provider

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("invalid_provider")


class TestAggregationStatistics:
    """Tests for aggregate statistics calculation from samples."""

    def test_aggregation_all_successful_samples(self):
        """Test statistics calculation with all successful samples."""
        from prompt_evaluator.models import Sample

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
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                judge_score=5.0,
                status="completed",
            ),
        ]

        # Calculate statistics as done in CLI
        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 3
        assert sum(successful_scores) / len(successful_scores) == pytest.approx(4.166666, rel=1e-3)
        assert min(successful_scores) == 3.0
        assert max(successful_scores) == 5.0

    def test_aggregation_mixed_success_failure(self):
        """Test statistics with mix of successful and failed samples."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                judge_score=4.0,
                status="completed",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="judge_error",
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                judge_score=3.5,
                status="completed",
            ),
            Sample(
                sample_id="s4",
                input_text="test",
                generator_output="",
                status="generation_error",
            ),
        ]

        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 2
        assert sum(successful_scores) / len(successful_scores) == 3.75
        assert min(successful_scores) == 3.5
        assert max(successful_scores) == 4.0

    def test_aggregation_zero_successful_samples(self):
        """Test statistics when all samples fail using actual CLI aggregation logic."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                status="judge_error",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="",
                status="generation_error",
            ),
        ]

        # Use same aggregation logic as CLI evaluate_single command
        successful_scores = [
            s.judge_score for s in samples
            if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 0

        # Apply same conditional logic as CLI
        if successful_scores:
            stats = {
                "mean_score": sum(successful_scores) / len(successful_scores),
                "min_score": min(successful_scores),
                "max_score": max(successful_scores),
                "num_successful": len(successful_scores),
                "num_failed": len(samples) - len(successful_scores),
            }
        else:
            stats = {
                "mean_score": None,
                "min_score": None,
                "max_score": None,
                "num_successful": 0,
                "num_failed": len(samples),
            }

        assert stats["mean_score"] is None
        assert stats["min_score"] is None
        assert stats["max_score"] is None
        assert stats["num_successful"] == 0
        assert stats["num_failed"] == 2

    def test_aggregation_floating_point_precision(self):
        """Test that floating-point calculations are precise."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id=f"s{i}",
                input_text="test",
                generator_output=f"output{i}",
                judge_score=score,
                status="completed",
            )
            for i, score in enumerate([4.123, 3.456, 2.789, 4.567, 3.891])
        ]

        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        mean = sum(successful_scores) / len(successful_scores)
        # Mean should be (4.123 + 3.456 + 2.789 + 4.567 + 3.891) / 5 = 3.7652
        assert mean == pytest.approx(3.7652, rel=1e-3)

        # Test rounding to 3 decimal places
        mean_rounded = round(mean, 3)
        assert mean_rounded == 3.765

    def test_aggregation_single_successful_sample(self):
        """Test statistics with only one successful sample."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                judge_score=4.2,
                status="completed",
            ),
        ]

        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 1
        assert sum(successful_scores) / len(successful_scores) == 4.2
        assert min(successful_scores) == 4.2
        assert max(successful_scores) == 4.2

    def test_aggregation_boundary_scores(self):
        """Test aggregation with scores at boundaries (1.0 and 5.0)."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                judge_score=1.0,
                status="completed",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                judge_score=5.0,
                status="completed",
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                judge_score=3.0,
                status="completed",
            ),
        ]

        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 3
        assert sum(successful_scores) / len(successful_scores) == 3.0
        assert min(successful_scores) == 1.0
        assert max(successful_scores) == 5.0

    def test_aggregation_identical_scores(self):
        """Test aggregation when all scores are identical."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id=f"s{i}",
                input_text="test",
                generator_output=f"output{i}",
                judge_score=4.0,
                status="completed",
            )
            for i in range(5)
        ]

        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        assert len(successful_scores) == 5
        assert sum(successful_scores) / len(successful_scores) == 4.0
        assert min(successful_scores) == 4.0
        assert max(successful_scores) == 4.0

    def test_aggregation_excludes_invalid_judge_responses(self):
        """Test that samples with judge_invalid_response status are excluded from aggregation."""
        from prompt_evaluator.models import Sample

        samples = [
            Sample(
                sample_id="s1",
                input_text="test",
                generator_output="output1",
                judge_score=4.0,
                status="completed",
            ),
            Sample(
                sample_id="s2",
                input_text="test",
                generator_output="output2",
                status="judge_invalid_response",
            ),
            Sample(
                sample_id="s3",
                input_text="test",
                generator_output="output3",
                judge_score=5.0,
                status="completed",
            ),
            Sample(
                sample_id="s4",
                input_text="test",
                generator_output="output4",
                status="judge_invalid_response",
            ),
        ]

        # Use same aggregation logic as CLI
        successful_scores = [
            s.judge_score for s in samples if s.status == "completed" and s.judge_score is not None
        ]

        # Only the 2 completed samples should be included
        assert len(successful_scores) == 2
        assert sum(successful_scores) / len(successful_scores) == 4.5
        assert min(successful_scores) == 4.0
        assert max(successful_scores) == 5.0
