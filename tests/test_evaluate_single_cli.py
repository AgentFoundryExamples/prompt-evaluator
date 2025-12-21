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
Tests for the evaluate-single CLI command.

Tests validate CLI parameter parsing, sample generation, judging,
aggregate statistics computation, and output file generation.
"""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prompt_evaluator.cli import app


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
    user_input.write_text("What is Python?")

    return {
        "system": system_prompt,
        "input": user_input,
        "output_dir": tmp_path / "runs",
    }


class TestEvaluateSingleCLI:
    """Tests for the evaluate-single CLI command."""

    def test_evaluate_single_command_exists(self, cli_runner):
        """Test that evaluate-single command is available."""
        result = cli_runner.invoke(app, ["evaluate-single", "--help"])
        assert result.exit_code == 0
        assert "Evaluate a prompt by generating N samples" in result.stdout

    def test_evaluate_single_missing_required_params(self, cli_runner):
        """Test that evaluate-single requires all required parameters."""
        result = cli_runner.invoke(app, ["evaluate-single"])
        assert result.exit_code != 0
        assert "Missing option" in result.stdout or "required" in result.stdout.lower()

    def test_evaluate_single_missing_system_prompt_file(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Test error handling when system prompt file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        input_file = tmp_path / "input.txt"
        input_file.write_text("test input")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(tmp_path / "nonexistent.txt"),
                "--input",
                str(input_file),
                "--num-samples",
                "2",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_evaluate_single_missing_input_file(self, cli_runner, tmp_path, monkeypatch):
        """Test error handling when input file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a helpful assistant.")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_file),
                "--input",
                str(tmp_path / "nonexistent.txt"),
                "--num-samples",
                "2",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_evaluate_single_invalid_num_samples(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test error handling when num_samples is non-positive."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_basic(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test basic evaluate-single with successful samples."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock generator responses
        mock_generate.side_effect = [
            ("Python is a programming language.", {"tokens_used": 10, "latency_ms": 100}),
            ("Python is a high-level language.", {"tokens_used": 12, "latency_ms": 110}),
        ]

        # Mock judge responses
        mock_judge.side_effect = [
            {
                "status": "completed",
                "judge_score": 4.5,
                "judge_rationale": "Good answer",
                "judge_raw_response": '{"semantic_fidelity": 4.5, "rationale": "Good"}',
                "error": None,
            },
            {
                "status": "completed",
                "judge_score": 4.0,
                "judge_rationale": "Decent answer",
                "judge_raw_response": '{"semantic_fidelity": 4.0, "rationale": "Decent"}',
                "error": None,
            },
        ]

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "2",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0
        assert mock_generate.call_count == 2
        assert mock_judge.call_count == 2

        # Check that output directory was created
        assert temp_prompts["output_dir"].exists()

        # Find the run directory
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Check evaluation file
        evaluation_file = run_dir / "evaluate-single.json"
        assert evaluation_file.exists()
        evaluation = json.loads(evaluation_file.read_text())

        assert "run_id" in evaluation
        assert "timestamp" in evaluation
        assert evaluation["num_samples"] == 2
        assert len(evaluation["samples"]) == 2

        # Check aggregate stats
        assert "aggregate_stats" in evaluation
        stats = evaluation["aggregate_stats"]
        assert stats["num_successful"] == 2
        assert stats["num_failed"] == 0
        assert stats["mean_score"] == 4.25  # (4.5 + 4.0) / 2
        assert stats["min_score"] == 4.0
        assert stats["max_score"] == 4.5

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_judge_errors(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test evaluate-single handles judge errors gracefully."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock generator responses (all successful)
        mock_generate.side_effect = [
            ("Response 1", {"tokens_used": 10, "latency_ms": 100}),
            ("Response 2", {"tokens_used": 10, "latency_ms": 100}),
            ("Response 3", {"tokens_used": 10, "latency_ms": 100}),
        ]

        # Mock judge responses (some errors)
        mock_judge.side_effect = [
            {
                "status": "completed",
                "judge_score": 4.0,
                "judge_rationale": "Good",
                "judge_raw_response": '{"semantic_fidelity": 4.0, "rationale": "Good"}',
                "error": None,
            },
            {
                "status": "judge_error",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "Invalid JSON",
                "error": "Failed to parse judge response",
            },
            {
                "status": "completed",
                "judge_score": 3.5,
                "judge_rationale": "Okay",
                "judge_raw_response": '{"semantic_fidelity": 3.5, "rationale": "Okay"}',
                "error": None,
            },
        ]

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

        assert result.exit_code == 0

        # Check evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        evaluation_file = run_dirs[0] / "evaluate-single.json"
        evaluation = json.loads(evaluation_file.read_text())

        # Check aggregate stats
        stats = evaluation["aggregate_stats"]
        assert stats["num_successful"] == 2
        assert stats["num_failed"] == 1
        assert stats["mean_score"] == 3.75  # (4.0 + 3.5) / 2
        assert stats["min_score"] == 3.5
        assert stats["max_score"] == 4.0

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_all_judge_failures(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test evaluate-single with all judge failures."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock generator responses
        mock_generate.side_effect = [
            ("Response 1", {"tokens_used": 10, "latency_ms": 100}),
            ("Response 2", {"tokens_used": 10, "latency_ms": 100}),
        ]

        # Mock all judge failures
        mock_judge.side_effect = [
            {
                "status": "judge_error",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "Error 1",
                "error": "Judge failed",
            },
            {
                "status": "judge_error",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "Error 2",
                "error": "Judge failed",
            },
        ]

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "2",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0

        # Check evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        evaluation_file = run_dirs[0] / "evaluate-single.json"
        evaluation = json.loads(evaluation_file.read_text())

        # Check aggregate stats - should be null with failures
        stats = evaluation["aggregate_stats"]
        assert stats["num_successful"] == 0
        assert stats["num_failed"] == 2
        assert stats["mean_score"] is None
        assert stats["min_score"] is None
        assert stats["max_score"] is None

        # Check that error message is displayed
        # Note: CliRunner combines stdout and stderr in result.stdout
        assert "No successful samples" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_custom_judge_prompt(
        self, mock_judge, mock_generate, cli_runner, tmp_path, monkeypatch
    ):
        """Test evaluate-single with custom judge prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create prompt files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are helpful.")

        input_file = tmp_path / "input.txt"
        input_file.write_text("Test question")

        judge_prompt = tmp_path / "judge.txt"
        judge_prompt.write_text("Custom judge instructions")

        output_dir = tmp_path / "runs"

        # Mock responses
        mock_generate.return_value = ("Answer", {"tokens_used": 5, "latency_ms": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 5.0,
            "judge_rationale": "Perfect",
            "judge_raw_response": '{"semantic_fidelity": 5.0, "rationale": "Perfect"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(input_file),
                "--num-samples",
                "1",
                "--judge-system-prompt",
                str(judge_prompt),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Verify judge was called with custom prompt
        assert mock_judge.called
        call_kwargs = mock_judge.call_args[1]
        assert call_kwargs["judge_system_prompt"] == "Custom judge instructions"

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_all_parameters(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test evaluate-single with all optional parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Answer", {"tokens_used": 5, "latency_ms": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.0,
            "judge_rationale": "Good",
            "judge_raw_response": '{"semantic_fidelity": 4.0, "rationale": "Good"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
                "--generator-model",
                "gpt-4",
                "--judge-model",
                "gpt-4",
                "--seed",
                "42",
                "--temperature",
                "0.5",
                "--max-tokens",
                "500",
                "--task-description",
                "Explain programming concepts",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0

        # Verify generator was called with correct parameters
        assert mock_generate.called
        gen_kwargs = mock_generate.call_args[1]
        assert gen_kwargs["model"] == "gpt-4"
        assert gen_kwargs["temperature"] == 0.5
        assert gen_kwargs["max_completion_tokens"] == 500
        assert gen_kwargs["seed"] == 42

        # Verify judge was called with task description
        assert mock_judge.called
        judge_kwargs = mock_judge.call_args[1]
        assert judge_kwargs["task_description"] == "Explain programming concepts"

        # Check evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        evaluation_file = run_dirs[0] / "evaluate-single.json"
        evaluation = json.loads(evaluation_file.read_text())

        # Check configs
        assert evaluation["generator_config"]["model_name"] == "gpt-4"
        assert evaluation["generator_config"]["temperature"] == 0.5
        assert evaluation["generator_config"]["seed"] == 42
        assert evaluation["judge_config"]["model_name"] == "gpt-4"

    def test_evaluate_single_missing_api_key(self, cli_runner, temp_prompts, monkeypatch):
        """Test error handling when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
            ],
        )
        assert result.exit_code == 1
        assert "API key is required" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_handles_generation_exceptions(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that generation exceptions are handled gracefully."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock generator to raise an exception
        mock_generate.side_effect = Exception("API connection failed")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # The command should still complete (not crash)
        assert result.exit_code == 0

        # Check that error was recorded
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        evaluation_file = run_dirs[0] / "evaluate-single.json"
        evaluation = json.loads(evaluation_file.read_text())

        # Sample should have error status
        assert len(evaluation["samples"]) == 1
        assert evaluation["samples"][0]["status"] == "generation_error"

        # Stats should show failure
        stats = evaluation["aggregate_stats"]
        assert stats["num_successful"] == 0
        assert stats["num_failed"] == 1

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_num_samples_boundary_values(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test num-samples with boundary values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Answer", {"tokens_used": 5, "latency_ms": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.0,
            "judge_rationale": "Good",
            "judge_raw_response": '{"semantic_fidelity": 4.0, "rationale": "Good"}',
            "error": None,
        }

        # Test num-samples = 1 (minimum valid value)
        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )
        assert result.exit_code == 0
        assert mock_generate.call_count == 1

    def test_evaluate_single_negative_num_samples(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that negative num-samples is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "-5",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_seed_parameter_wiring(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that seed parameter is wired correctly through evaluate-single."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Answer", {"tokens_used": 5, "latency_ms": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.5,
            "judge_rationale": "Good",
            "judge_raw_response": '{"semantic_fidelity": 4.5, "rationale": "Good"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
                "--seed",
                "42",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0

        # Verify seed was passed to generator
        gen_kwargs = mock_generate.call_args[1]
        assert gen_kwargs["seed"] == 42

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_no_real_api_calls(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that mocked tests don't make real API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-no-real-calls")

        mock_generate.return_value = ("Mocked output", {"tokens_used": 10, "latency_ms": 100})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 3.5,
            "judge_rationale": "Mocked evaluation",
            "judge_raw_response": '{"semantic_fidelity": 3.5, "rationale": "Mocked"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "2",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should succeed with mocks, no real API calls
        assert result.exit_code == 0
        assert mock_generate.call_count == 2
        assert mock_judge.call_count == 2

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_config_flags_wiring(
        self, mock_judge, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that all config flags are properly wired without real API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Generated text", {"tokens_used": 20, "latency_ms": 200})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.8,
            "judge_rationale": "Excellent",
            "judge_raw_response": '{"semantic_fidelity": 4.8, "rationale": "Excellent"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--num-samples",
                "1",
                "--generator-model",
                "test-gen-model",
                "--judge-model",
                "test-judge-model",
                "--temperature",
                "0.8",
                "--max-tokens",
                "750",
                "--seed",
                "999",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0

        # Verify all parameters were passed correctly
        gen_kwargs = mock_generate.call_args[1]
        assert gen_kwargs["model"] == "test-gen-model"
        assert gen_kwargs["temperature"] == 0.8
        assert gen_kwargs["max_completion_tokens"] == 750
        assert gen_kwargs["seed"] == 999

        judge_kwargs = mock_judge.call_args[1]
        # Judge config model name is passed through judge_config
        assert mock_judge.called

    def test_evaluate_single_unreadable_prompt_path(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Test error handling for unreadable prompt files."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a directory where file is expected (can't read directory as file)
        bad_path = tmp_path / "bad_prompt"
        bad_path.mkdir()

        input_file = tmp_path / "input.txt"
        input_file.write_text("test")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(bad_path),
                "--input",
                str(input_file),
                "--num-samples",
                "1",
            ],
        )

        # Should fail with clear error
        assert result.exit_code == 1
