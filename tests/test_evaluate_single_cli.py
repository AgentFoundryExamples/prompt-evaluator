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

    def test_evaluate_single_missing_system_prompt_file(self, cli_runner, tmp_path, monkeypatch):
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

    def test_evaluate_single_invalid_num_samples(self, cli_runner, temp_prompts, monkeypatch):
        """Test error handling when num_samples is non-positive (zero or negative)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Test with zero
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

        # Test with negative value
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

        # Verify judge was called and check judge_config details
        assert mock_judge.called
        judge_kwargs = mock_judge.call_args[1]
        judge_config = judge_kwargs["judge_config"]
        assert judge_config.model_name == "test-judge-model"
        assert judge_config.temperature == 0.0  # Judge uses deterministic temperature

    def test_evaluate_single_unreadable_prompt_path(self, cli_runner, tmp_path, monkeypatch):
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

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_rubric_metrics_and_flags(
        self, mock_judge, mock_generate, cli_runner, tmp_path, monkeypatch
    ):
        """Test evaluate-single with rubric-based evaluation including metrics and flags."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create prompt files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are helpful.")

        input_file = tmp_path / "input.txt"
        input_file.write_text("Test question")

        output_dir = tmp_path / "runs"

        # Mock generator response
        mock_generate.return_value = ("Answer", {"tokens_used": 5, "latency_ms": 50})

        # Mock judge response with rubric-based metrics and flags
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": None,  # Legacy field not used with rubric
            "judge_rationale": None,
            "judge_raw_response": '{"metrics": {...}, "flags": {...}}',
            "judge_metrics": {
                "semantic_fidelity": {"score": 4.5, "rationale": "Excellent preservation"},
                "decomposition_quality": {"score": 3.0, "rationale": "Adequate structure"},
                "constraint_adherence": {"score": 5.0, "rationale": "Perfect adherence"},
            },
            "judge_flags": {
                "invented_constraints": False,
                "omitted_constraints": False,
            },
            "judge_overall_comment": "Good response overall",
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
                "2",
                "--rubric",
                "default",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Check that output directory was created
        assert output_dir.exists()

        # Find the run directory
        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Check evaluation file
        evaluation_file = run_dir / "evaluate-single.json"
        assert evaluation_file.exists()

        import json

        evaluation = json.loads(evaluation_file.read_text())

        # Verify rubric metadata is present
        assert "rubric_metadata" in evaluation
        assert "rubric_path" in evaluation["rubric_metadata"]
        assert "rubric_hash" in evaluation["rubric_metadata"]
        assert "rubric_definition" in evaluation["rubric_metadata"]

        # Verify aggregate stats include metric_stats and flag_stats
        assert "aggregate_stats" in evaluation
        stats = evaluation["aggregate_stats"]

        assert "metric_stats" in stats
        assert "semantic_fidelity" in stats["metric_stats"]
        assert stats["metric_stats"]["semantic_fidelity"]["mean"] == 4.5
        assert stats["metric_stats"]["semantic_fidelity"]["count"] == 2

        assert "flag_stats" in stats
        assert "invented_constraints" in stats["flag_stats"]
        assert stats["flag_stats"]["invented_constraints"]["true_count"] == 0
        assert stats["flag_stats"]["invented_constraints"]["false_count"] == 2

        # Verify samples have rubric fields populated
        assert len(evaluation["samples"]) == 2
        for sample in evaluation["samples"]:
            assert "judge_metrics" in sample
            assert "judge_flags" in sample
            assert "judge_overall_comment" in sample
            assert sample["judge_overall_comment"] == "Good response overall"

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_rubric_aggregation_with_mixed_results(
        self, mock_judge, mock_generate, cli_runner, tmp_path, monkeypatch
    ):
        """Test rubric aggregation with mix of successful and invalid responses."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are helpful.")

        input_file = tmp_path / "input.txt"
        input_file.write_text("Test question")

        output_dir = tmp_path / "runs"

        # Mock generator responses
        mock_generate.side_effect = [
            ("Answer 1", {"tokens_used": 5, "latency_ms": 50}),
            ("Answer 2", {"tokens_used": 5, "latency_ms": 50}),
            ("Answer 3", {"tokens_used": 5, "latency_ms": 50}),
        ]

        # Mock judge responses - mix of successful and invalid
        mock_judge.side_effect = [
            {
                "status": "completed",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "{}",
                "judge_metrics": {"test_metric": {"score": 4.0, "rationale": "Good"}},
                "judge_flags": {"test_flag": True},
                "judge_overall_comment": "Good",
                "error": None,
            },
            {
                "status": "judge_invalid_response",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "Invalid JSON",
                "judge_metrics": {},
                "judge_flags": {},
                "judge_overall_comment": None,
                "error": "Failed to parse",
            },
            {
                "status": "completed",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": "{}",
                "judge_metrics": {"test_metric": {"score": 5.0, "rationale": "Excellent"}},
                "judge_flags": {"test_flag": False},
                "judge_overall_comment": "Excellent",
                "error": None,
            },
        ]

        # Create a minimal rubric for testing
        rubric_file = tmp_path / "test_rubric.yaml"
        rubric_file.write_text("""
metrics:
  - name: test_metric
    description: Test metric
    min_score: 1
    max_score: 5
    guidelines: Test guidelines
flags:
  - name: test_flag
    description: Test flag
    default: false
""")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(input_file),
                "--num-samples",
                "3",
                "--rubric",
                str(rubric_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Check evaluation file
        run_dirs = list(output_dir.iterdir())
        evaluation_file = run_dirs[0] / "evaluate-single.json"

        import json

        evaluation = json.loads(evaluation_file.read_text())

        # Verify rubric metadata is present
        assert "rubric_metadata" in evaluation
        assert "rubric_path" in evaluation["rubric_metadata"]
        assert "rubric_hash" in evaluation["rubric_metadata"]
        assert "rubric_definition" in evaluation["rubric_metadata"]
        # Verify rubric definition contains expected structure
        rubric_def = evaluation["rubric_metadata"]["rubric_definition"]
        assert "metrics" in rubric_def
        assert "flags" in rubric_def
        assert len(rubric_def["metrics"]) == 1
        assert rubric_def["metrics"][0]["name"] == "test_metric"
        assert len(rubric_def["flags"]) == 1
        assert rubric_def["flags"][0]["name"] == "test_flag"

        # Check that invalid sample is excluded from aggregation
        stats = evaluation["aggregate_stats"]
        assert stats["metric_stats"]["test_metric"]["count"] == 2  # Only 2 valid samples
        assert stats["metric_stats"]["test_metric"]["mean"] == 4.5  # (4.0 + 5.0) / 2
        assert stats["flag_stats"]["test_flag"]["total_count"] == 2
        assert stats["flag_stats"]["test_flag"]["true_count"] == 1
        assert stats["flag_stats"]["test_flag"]["true_proportion"] == 0.5

    def test_evaluate_single_uses_config_defaults(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that evaluate-single uses config defaults when flags are omitted."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a config file with defaults
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            """
defaults:
  generator:
    provider: mock
    model: config-model
    temperature: 0.8
  judge:
    provider: mock
    model: config-judge-model
  rubric: default
  run_directory: config_runs
"""
        )

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
                "--config",
                str(config_file),
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        # Check that config defaults were used
        assert "Using provider from config: mock" in result.stdout
        assert "Using output directory from config: config_runs" in result.stdout
        assert "Using default rubric from config: default" in result.stdout

    def test_evaluate_single_cli_overrides_config_defaults(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that CLI flags override config defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a config file with defaults
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            """
defaults:
  generator:
    provider: anthropic
    model: config-model
  run_directory: config_runs
"""
        )

        # Create a minimal rubric file for testing
        rubric_file = temp_prompts["system"].parent / "test_rubric.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test_metric
    description: A test metric
    min_score: 1
    max_score: 5
    guidelines: Test guidelines
flags: []
"""
        )

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
                "--provider",
                "mock",
                "--output-dir",
                str(temp_prompts["output_dir"]),
                "--rubric",
                str(rubric_file),
                "--config",
                str(config_file),
            ],
        )

        # Should succeed with CLI overrides
        assert result.exit_code == 0
        # Should NOT show config messages since CLI flags were provided
        assert "Using provider from config:" not in result.stdout
        assert "Using output directory from config:" not in result.stdout


class TestABTestingEvaluateSingle:
    """Tests for A/B testing mode in evaluate-single command."""

    @patch("prompt_evaluator.cli.judge_completion")
    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_ab_test_doubles_samples(
        self, mock_generate, mock_judge, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that A/B testing mode doubles the number of samples."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock generate and judge responses
        mock_generate.return_value = ("Generated response", {"tokens_used": 10, "latency_ms": 100.0})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.5,
            "judge_rationale": "Good response",
            "judge_raw_response": '{"score": 4.5, "rationale": "Good"}',
            "judge_metrics": {"quality": {"score": 4.5, "rationale": "Good"}},
            "judge_flags": {},
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
                "--ab-test-system-prompt",
            ],
        )

        assert result.exit_code == 0

        # With --num-samples=2 and A/B testing, should generate 4 total samples
        # (2 for with_prompt, 2 for no_prompt)
        assert mock_generate.call_count == 4
        assert mock_judge.call_count == 4

    @patch("prompt_evaluator.cli.judge_completion")
    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_ab_test_tags_variants(
        self, mock_generate, mock_judge, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that A/B testing properly tags samples with variant metadata."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Response", {"tokens_used": 10, "latency_ms": 100.0})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.0,
            "judge_rationale": "Good",
            "judge_raw_response": "{}",
            "judge_metrics": {},
            "judge_flags": {},
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
                "--output-dir",
                str(temp_prompts["output_dir"]),
                "--ab-test-system-prompt",
            ],
        )

        assert result.exit_code == 0

        # Load the saved evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        assert len(run_dirs) == 1
        eval_file = run_dirs[0] / "evaluate-single.json"
        assert eval_file.exists()

        eval_data = json.loads(eval_file.read_text())

        # Check that we have 2 samples with proper variant tags
        assert len(eval_data["samples"]) == 2
        variants = [s["ab_variant"] for s in eval_data["samples"]]
        assert "with_prompt" in variants
        assert "no_prompt" in variants

    @patch("prompt_evaluator.cli.judge_completion")
    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_ab_test_shows_warning(
        self, mock_generate, mock_judge, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that A/B testing shows warning about doubled API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Response", {"tokens_used": 10, "latency_ms": 100.0})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.0,
            "judge_rationale": "Good",
            "judge_raw_response": "{}",
            "judge_metrics": {},
            "judge_flags": {},
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
                "--ab-test-system-prompt",
            ],
        )

        assert result.exit_code == 0

        # Check for warning message
        assert "WARNING" in result.stdout or "warning" in result.stdout.lower()
        assert "DOUBLE" in result.stdout or "double" in result.stdout.lower()

    @patch("prompt_evaluator.cli.judge_completion")
    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_ab_test_variant_statistics(
        self, mock_generate, mock_judge, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that A/B testing computes and displays variant statistics."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Response", {"tokens_used": 10, "latency_ms": 100.0})

        # Different scores for different variants to verify statistics
        mock_judge.side_effect = [
            {  # with_prompt sample 1
                "status": "completed",
                "judge_score": 5.0,
                "judge_rationale": "Excellent",
                "judge_raw_response": "{}",
                "judge_metrics": {},
                "judge_flags": {},
            },
            {  # with_prompt sample 2
                "status": "completed",
                "judge_score": 4.5,
                "judge_rationale": "Very good",
                "judge_raw_response": "{}",
                "judge_metrics": {},
                "judge_flags": {},
            },
            {  # no_prompt sample 1
                "status": "completed",
                "judge_score": 3.0,
                "judge_rationale": "Okay",
                "judge_raw_response": "{}",
                "judge_metrics": {},
                "judge_flags": {},
            },
            {  # no_prompt sample 2
                "status": "completed",
                "judge_score": 3.5,
                "judge_rationale": "Fair",
                "judge_raw_response": "{}",
                "judge_metrics": {},
                "judge_flags": {},
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
                "--ab-test-system-prompt",
            ],
        )

        assert result.exit_code == 0

        # Load the saved evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        eval_file = run_dirs[0] / "evaluate-single.json"
        eval_data = json.loads(eval_file.read_text())

        # Check that variant statistics are computed
        assert "variant_stats" in eval_data
        assert "with_prompt" in eval_data["variant_stats"]
        assert "no_prompt" in eval_data["variant_stats"]

        # Verify statistics for each variant
        with_prompt_stats = eval_data["variant_stats"]["with_prompt"]
        no_prompt_stats = eval_data["variant_stats"]["no_prompt"]

        # with_prompt should have higher mean score (5.0 + 4.5) / 2 = 4.75
        assert with_prompt_stats["mean_score"] == 4.75
        assert with_prompt_stats["num_successful"] == 2

        # no_prompt should have lower mean score (3.0 + 3.5) / 2 = 3.25
        assert no_prompt_stats["mean_score"] == 3.25
        assert no_prompt_stats["num_successful"] == 2

    @patch("prompt_evaluator.cli.judge_completion")
    @patch("prompt_evaluator.cli.generate_completion")
    def test_evaluate_single_without_ab_test_single_execution(
        self, mock_generate, mock_judge, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that without A/B mode, only single execution happens."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Response", {"tokens_used": 10, "latency_ms": 100.0})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.0,
            "judge_rationale": "Good",
            "judge_raw_response": "{}",
            "judge_metrics": {},
            "judge_flags": {},
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
                # No --ab-test-system-prompt flag
            ],
        )

        assert result.exit_code == 0

        # Without A/B testing, should only generate 2 samples (not 4)
        assert mock_generate.call_count == 2
        assert mock_judge.call_count == 2

        # Load the saved evaluation file
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        eval_file = run_dirs[0] / "evaluate-single.json"
        eval_data = json.loads(eval_file.read_text())

        # Should only have 2 samples
        assert len(eval_data["samples"]) == 2

        # Should NOT have variant_stats
        assert "variant_stats" not in eval_data or not eval_data.get("variant_stats")

        # Samples should not have ab_variant tags (or should be None)
        for sample in eval_data["samples"]:
            assert sample.get("ab_variant") is None
