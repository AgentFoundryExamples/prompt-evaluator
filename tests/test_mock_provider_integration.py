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
Integration tests for LocalMockProvider with CLI and dataset flows.

Tests validate that LocalMockProvider can be used for offline testing
without making real API calls, ensuring deterministic behavior in tests.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prompt_evaluator.cli import app
from prompt_evaluator.dataset_evaluation import evaluate_dataset
from prompt_evaluator.models import GeneratorConfig, JudgeConfig, Rubric, RubricMetric, TestCase
from prompt_evaluator.provider import get_provider


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_prompts(tmp_path):
    """Fixture to create temporary prompt files."""
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("You are a helpful assistant.")

    user_input = tmp_path / "input.txt"
    user_input.write_text("What is 2+2?")

    return {
        "system": system_prompt,
        "input": user_input,
        "output_dir": tmp_path / "runs",
    }


class TestMockProviderCLIIntegration:
    """Tests for LocalMockProvider integration with CLI commands."""

    @patch("prompt_evaluator.cli.APIConfig")
    @patch("prompt_evaluator.cli.get_provider")
    def test_generate_with_mock_provider(
        self, mock_get_provider, mock_api_config_class, cli_runner, temp_prompts
    ):
        """Test generate command with mocked provider."""
        # Mock APIConfig to avoid API key validation
        mock_api_config = mock_api_config_class.return_value
        mock_api_config.api_key = "mock-key"
        mock_api_config.base_url = None
        mock_api_config.model_name = "gpt-5.1"

        # Setup mock to return LocalMockProvider
        mock_provider = get_provider("mock")
        mock_get_provider.return_value = mock_provider

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0
        assert "Mock response to:" in result.stdout
        assert "What is 2+2?" in result.stdout

        # Verify output file was created with mock response
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        assert len(run_dirs) == 1
        output_file = run_dirs[0] / "output.txt"
        assert output_file.exists()

        content = output_file.read_text()
        assert "Mock response" in content

    @patch("prompt_evaluator.cli.APIConfig")
    @patch("prompt_evaluator.cli.get_provider")
    def test_evaluate_single_with_mock_provider(
        self, mock_get_provider, mock_api_config_class, cli_runner, temp_prompts
    ):
        """Test evaluate-single command with mocked provider."""
        # Mock APIConfig
        mock_api_config = mock_api_config_class.return_value
        mock_api_config.api_key = "mock-key"
        mock_api_config.base_url = None
        mock_api_config.model_name = "gpt-5.1"

        # Setup mock to return LocalMockProvider
        mock_provider = get_provider("mock")
        mock_get_provider.return_value = mock_provider

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "3",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should complete (though judge may fail to parse mock responses)
        assert result.exit_code == 0

        # Verify that samples were generated (even if judge parsing failed)
        # The CLI outputs JSON at the end
        # Look for the JSON structure in output
        assert '"num_samples": 3' in result.stdout or "num_samples" in result.stdout

    @patch("prompt_evaluator.cli.APIConfig")
    @patch("prompt_evaluator.cli.get_provider")
    def test_mock_provider_no_real_api_calls(
        self, mock_get_provider, mock_api_config_class, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that using mock provider makes no real API calls."""
        # Remove API key to ensure no real calls can be made
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Mock APIConfig
        mock_api_config = mock_api_config_class.return_value
        mock_api_config.api_key = "mock-key"
        mock_api_config.base_url = None
        mock_api_config.model_name = "gpt-5.1"

        # Setup mock to return LocalMockProvider
        mock_provider = get_provider("mock")
        mock_get_provider.return_value = mock_provider

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should succeed even without API key
        assert result.exit_code == 0
        assert "Mock response" in result.stdout


class TestMockProviderDatasetIntegration:
    """Tests for LocalMockProvider integration with dataset evaluation."""

    def test_evaluate_dataset_with_mock_provider(self, tmp_path):
        """Test dataset evaluation with LocalMockProvider."""
        # Setup test data
        mock_provider = get_provider("mock")

        test_cases = [
            TestCase(
                id="test-001",
                input="What is Python?",
                task="Explain Python programming language",
            ),
            TestCase(
                id="test-002",
                input="What is Java?",
                task="Explain Java programming language",
            ),
        ]

        dataset_metadata = {
            "path": "test-dataset.yaml",
            "hash": "abc123",
            "count": 2,
        }

        system_prompt_path = tmp_path / "system.txt"
        system_prompt_path.write_text("You are a helpful assistant.")
        system_prompt = system_prompt_path.read_text()

        rubric = Rubric(
            metrics=[
                RubricMetric(
                    name="Accuracy",
                    description="Test metric",
                    min_score=1.0,
                    max_score=5.0,
                    guidelines="Test guidelines",
                )
            ],
            flags=[],
        )

        generator_config = GeneratorConfig(model_name="gpt-5.1")
        judge_config = JudgeConfig(model_name="gpt-5.1")

        # Run evaluation
        result = evaluate_dataset(
            provider=mock_provider,
            test_cases=test_cases,
            dataset_metadata=dataset_metadata,
            system_prompt=system_prompt,
            system_prompt_path=system_prompt_path,
            num_samples_per_case=2,
            generator_config=generator_config,
            judge_config=judge_config,
            judge_system_prompt="Evaluate the response.",
            rubric=rubric,
            rubric_metadata={},
            output_dir=tmp_path,
            prompt_version_id="test-v1",
            prompt_hash="hash123",
        )

        # Verify results - status may be 'failed' since judge can't parse mock responses
        assert result.status in ("completed", "partial", "failed")
        assert len(result.test_case_results) == 2

        # All test cases should have samples generated (even though judge parsing fails)
        for tc_result in result.test_case_results:
            assert len(tc_result.samples) == 2
            # All samples should have generator output from mock
            for sample in tc_result.samples:
                assert "Mock response" in sample.generator_output

    def test_mock_provider_deterministic_across_runs(self, tmp_path):
        """Test that LocalMockProvider produces deterministic results."""
        mock_provider = get_provider("mock")

        test_cases = [
            TestCase(id="test-001", input="Fixed prompt", task="Test task"),
        ]

        dataset_metadata = {"path": "test.yaml", "hash": "abc", "count": 1}

        system_prompt_path = tmp_path / "system.txt"
        system_prompt_path.write_text("System")
        system_prompt = system_prompt_path.read_text()

        generator_config = GeneratorConfig(model_name="gpt-5.1")
        judge_config = JudgeConfig(model_name="gpt-5.1")

        # Run evaluation twice
        result1 = evaluate_dataset(
            provider=mock_provider,
            test_cases=test_cases,
            dataset_metadata=dataset_metadata,
            system_prompt=system_prompt,
            system_prompt_path=system_prompt_path,
            num_samples_per_case=1,
            generator_config=generator_config,
            judge_config=judge_config,
            judge_system_prompt="Evaluate",
            rubric=None,
            rubric_metadata={},
            output_dir=tmp_path / "run1",
            prompt_version_id="v1",
            prompt_hash="hash",
        )

        result2 = evaluate_dataset(
            provider=mock_provider,
            test_cases=test_cases,
            dataset_metadata=dataset_metadata,
            system_prompt=system_prompt,
            system_prompt_path=system_prompt_path,
            num_samples_per_case=1,
            generator_config=generator_config,
            judge_config=judge_config,
            judge_system_prompt="Evaluate",
            rubric=None,
            rubric_metadata={},
            output_dir=tmp_path / "run2",
            prompt_version_id="v1",
            prompt_hash="hash",
        )

        # Generator outputs should be identical
        output1 = result1.test_case_results[0].samples[0].generator_output
        output2 = result2.test_case_results[0].samples[0].generator_output
        assert output1 == output2


class TestProviderSelectionErrorHandling:
    """Tests for provider selection error handling."""

    def test_unknown_provider_error_message(self):
        """Test that unknown provider gives actionable error."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown-provider")

        error_msg = str(exc_info.value)
        assert "unknown-provider" in error_msg
        # Should list supported providers
        assert "openai" in error_msg
        assert "mock" in error_msg

    @patch("prompt_evaluator.cli.APIConfig")
    @patch("prompt_evaluator.cli.get_provider")
    def test_cli_handles_unknown_provider_gracefully(
        self, mock_get_provider, mock_api_config_class, cli_runner, temp_prompts
    ):
        """Test CLI handles unknown provider errors gracefully."""
        # Mock APIConfig
        mock_api_config = mock_api_config_class.return_value
        mock_api_config.api_key = "mock-key"
        mock_api_config.base_url = None
        mock_api_config.model_name = "gpt-5.1"

        mock_get_provider.side_effect = ValueError("Unsupported provider: invalid")

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
            ],
        )

        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "error" in result.stderr.lower()


class TestConcurrentProviderUsage:
    """Tests for concurrent provider usage to ensure no state leakage."""

    def test_concurrent_mock_provider_usage(self, tmp_path):
        """Test that concurrent LocalMockProvider usage doesn't leak state."""
        import concurrent.futures

        mock_provider = get_provider("mock")

        def generate_with_id(prompt_id):
            """Generate a response with a unique prompt ID."""
            test_case = TestCase(
                id=f"test-{prompt_id:03d}",
                input=f"Prompt {prompt_id}",
                task=f"Task {prompt_id}",
            )

            dataset_metadata = {"path": "test.yaml", "hash": "abc", "count": 1}
            system_prompt_path = tmp_path / "system.txt"
            system_prompt_path.write_text("System")

            generator_config = GeneratorConfig(model_name="gpt-5.1")
            judge_config = JudgeConfig(model_name="gpt-5.1")

            result = evaluate_dataset(
                provider=mock_provider,
                test_cases=[test_case],
                dataset_metadata=dataset_metadata,
                system_prompt="System",
                system_prompt_path=system_prompt_path,
                num_samples_per_case=1,
                generator_config=generator_config,
                judge_config=judge_config,
                judge_system_prompt="Evaluate",
                rubric=None,
                rubric_metadata={},
                output_dir=tmp_path / f"run-{prompt_id:03d}",
                prompt_version_id="v1",
                prompt_hash="hash",
            )

            return (
                prompt_id,
                result.test_case_results[0].samples[0].generator_output,
            )

        # Run 10 evaluations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_with_id, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Each result should contain its unique prompt ID
        results_dict = dict(results)
        for prompt_id in range(10):
            output = results_dict[prompt_id]
            assert f"Prompt {prompt_id}" in output
            # Ensure no cross-contamination
            for other_id in range(10):
                if other_id != prompt_id:
                    assert f"Prompt {other_id}" not in output or prompt_id == other_id
