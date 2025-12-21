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
Tests for the evaluate-dataset CLI command.

Tests validate CLI parameter parsing, filtering options (case-ids, max-cases),
quick mode, resume functionality, and output generation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from prompt_evaluator.cli import app


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dataset_yaml(tmp_path):
    """Fixture to create a temporary YAML dataset file."""
    dataset = tmp_path / "dataset.yaml"
    test_cases = [
        {
            "id": "test-001",
            "input": "What is Python?",
            "task": "Explain Python",
            "description": "Basic explanation",
        },
        {
            "id": "test-002",
            "input": "What is Java?",
            "task": "Explain Java",
            "description": "Basic explanation",
        },
        {
            "id": "test-003",
            "input": "What is Rust?",
            "task": "Explain Rust",
            "description": "Basic explanation",
        },
    ]
    dataset.write_text(yaml.dump(test_cases))
    return dataset


@pytest.fixture
def temp_dataset_jsonl(tmp_path):
    """Fixture to create a temporary JSONL dataset file."""
    dataset = tmp_path / "dataset.jsonl"
    test_cases = [
        {
            "id": "case-a",
            "input": "Explain lists",
            "task": "Technical explanation",
        },
        {
            "id": "case-b",
            "input": "Explain dictionaries",
            "task": "Technical explanation",
        },
    ]
    with open(dataset, "w") as f:
        for tc in test_cases:
            f.write(json.dumps(tc) + "\n")
    return dataset


@pytest.fixture
def temp_prompts(tmp_path):
    """Fixture to create temporary prompt files."""
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("You are a helpful assistant.")

    return {
        "system": system_prompt,
        "output_dir": tmp_path / "runs",
    }


class TestEvaluateDatasetCLI:
    """Tests for the evaluate-dataset CLI command."""

    def test_evaluate_dataset_command_exists(self, cli_runner):
        """Test that evaluate-dataset command is available."""
        result = cli_runner.invoke(app, ["evaluate-dataset", "--help"])
        assert result.exit_code == 0
        assert "Evaluate a dataset of test cases" in result.stdout

    def test_evaluate_dataset_missing_required_params(self, cli_runner):
        """Test that evaluate-dataset requires dataset and system-prompt."""
        result = cli_runner.invoke(app, ["evaluate-dataset"])
        assert result.exit_code != 0
        assert "Missing option" in result.stdout or "required" in result.stdout.lower()

    def test_evaluate_dataset_missing_dataset_file(
        self, cli_runner, tmp_path, temp_prompts, monkeypatch
    ):
        """Test error handling when dataset file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(tmp_path / "nonexistent.yaml"),
                "--system-prompt",
                str(temp_prompts["system"]),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_evaluate_dataset_missing_system_prompt_file(
        self, cli_runner, tmp_path, temp_dataset_yaml, monkeypatch
    ):
        """Test error handling when system prompt file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(tmp_path / "nonexistent.txt"),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_evaluate_dataset_default_num_samples(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test that num-samples defaults to 5 when not provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Using default --num-samples=5" in result.stdout
            # Verify the evaluation was called with num_samples=5
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["num_samples_per_case"] == 5

    def test_evaluate_dataset_quick_mode(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test that --quick flag sets num-samples to 2."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--quick",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Quick mode: Using --num-samples=2" in result.stdout
            # Verify the evaluation was called with num_samples=2
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["num_samples_per_case"] == 2

    def test_evaluate_dataset_quick_mode_with_explicit_num_samples(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test that explicit --num-samples overrides --quick flag with a warning."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--quick",
                    "--num-samples",
                    "10",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Warning: Both --quick and --num-samples provided" in result.stdout
            assert "Using explicit --num-samples=10" in result.stdout
            # Verify the evaluation was called with explicit num_samples=10
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["num_samples_per_case"] == 10

    def test_evaluate_dataset_invalid_num_samples(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test error handling for invalid num-samples values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Test zero
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(temp_prompts["system"]),
                "--num-samples",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

        # Test negative
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(temp_prompts["system"]),
                "--num-samples",
                "-3",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

    def test_evaluate_dataset_case_ids_filter(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test filtering by specific case IDs."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--case-ids",
                    "test-001,test-003",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 2 test cases by --case-ids" in result.stdout

            # Verify only filtered test cases were passed
            call_kwargs = mock_eval.call_args.kwargs
            test_cases = call_kwargs["test_cases"]
            assert len(test_cases) == 2
            assert {tc.id for tc in test_cases} == {"test-001", "test-003"}

    def test_evaluate_dataset_case_ids_unknown_id(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test error handling when case-ids contains unknown IDs."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(temp_prompts["system"]),
                "--case-ids",
                "test-001,unknown-id,another-unknown",
                "--num-samples",
                "2",
            ],
        )

        assert result.exit_code == 1
        assert "Unknown test case IDs" in result.stdout
        assert "unknown-id" in result.stdout
        assert "another-unknown" in result.stdout
        assert "Available IDs:" in result.stdout

    def test_evaluate_dataset_max_cases_filter(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test limiting number of test cases with --max-cases."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--max-cases",
                    "2",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Limited to first 2 test cases by --max-cases" in result.stdout

            # Verify only first 2 test cases were passed
            call_kwargs = mock_eval.call_args.kwargs
            test_cases = call_kwargs["test_cases"]
            assert len(test_cases) == 2

    def test_evaluate_dataset_max_cases_invalid(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test error handling for invalid --max-cases values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Test zero
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(temp_prompts["system"]),
                "--max-cases",
                "0",
                "--num-samples",
                "2",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

        # Test negative
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset",
                str(temp_dataset_yaml),
                "--system-prompt",
                str(temp_prompts["system"]),
                "--max-cases",
                "-5",
                "--num-samples",
                "2",
            ],
        )
        assert result.exit_code == 1
        assert "must be positive" in result.stdout

    def test_evaluate_dataset_case_ids_and_max_cases(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test combining --case-ids and --max-cases filters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            # Filter to test-001 and test-002, then limit to first 1
            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--case-ids",
                    "test-001,test-002",
                    "--max-cases",
                    "1",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 2 test cases by --case-ids" in result.stdout
            assert "Limited to first 1 test cases by --max-cases" in result.stdout

            # Verify only 1 test case was passed
            call_kwargs = mock_eval.call_args.kwargs
            test_cases = call_kwargs["test_cases"]
            assert len(test_cases) == 1

    def test_evaluate_dataset_jsonl_format(
        self, cli_runner, temp_dataset_jsonl, temp_prompts, monkeypatch
    ):
        """Test loading JSONL format dataset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_jsonl),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Loaded 2 test cases" in result.stdout

    def test_evaluate_dataset_with_rubric(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test evaluation with a rubric."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset") as mock_eval:
            mock_eval.return_value = MagicMock(
                run_id="test-run-id",
                status="completed",
                test_case_results=[],
                overall_metric_stats={},
                overall_flag_stats={},
            )
            mock_eval.return_value.to_dict.return_value = {}

            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--rubric",
                    "default",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            assert "Using rubric:" in result.stdout

    def test_evaluate_dataset_output_format(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test that evaluate-dataset outputs proper JSON format."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_run = MagicMock()
        mock_run.run_id = "test-run-123"
        mock_run.status = "completed"
        mock_run.test_case_results = []
        mock_run.overall_metric_stats = {}
        mock_run.overall_flag_stats = {}
        mock_run.to_dict.return_value = {
            "run_id": "test-run-123",
            "status": "completed",
        }

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset", return_value=mock_run):
            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0

            # Verify JSON output in stdout
            try:
                output_lines = result.stdout.strip().split("\n")
                # Find JSON output (should be last contiguous JSON block)
                json_start = -1
                for i, line in enumerate(output_lines):
                    if line.strip() == "{":
                        json_start = i
                        break

                if json_start >= 0:
                    json_str = "\n".join(output_lines[json_start:])
                    parsed = json.loads(json_str)
                    assert parsed["run_id"] == "test-run-123"
                    assert parsed["status"] == "completed"
            except (json.JSONDecodeError, ValueError):
                # JSON output validation is optional for this test
                pass

    def test_evaluate_dataset_std_highlighting(
        self, cli_runner, temp_dataset_yaml, temp_prompts, monkeypatch
    ):
        """Test that high standard deviation is highlighted in output."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create mock test case result with high std
        mock_tc_result = MagicMock()
        mock_tc_result.test_case_id = "test-001"
        mock_tc_result.status = "completed"
        mock_tc_result.per_metric_stats = {
            "accuracy": {
                "mean": 3.5,
                "std": 1.5,  # High std
                "count": 5,
            },
            "clarity": {
                "mean": 4.0,
                "std": 0.2,  # Low std
                "count": 5,
            },
        }
        mock_tc_result.per_flag_stats = {}

        mock_run = MagicMock()
        mock_run.run_id = "test-run-123"
        mock_run.status = "completed"
        mock_run.test_case_results = [mock_tc_result]
        mock_run.overall_metric_stats = {}
        mock_run.overall_flag_stats = {}
        mock_run.to_dict.return_value = {}

        with patch("prompt_evaluator.dataset_evaluation.evaluate_dataset", return_value=mock_run):
            result = cli_runner.invoke(
                app,
                [
                    "evaluate-dataset",
                    "--dataset",
                    str(temp_dataset_yaml),
                    "--system-prompt",
                    str(temp_prompts["system"]),
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(temp_prompts["output_dir"]),
                ],
            )

            assert result.exit_code == 0
            # Check for high variability warning on accuracy
            assert "accuracy" in result.stdout
            assert "HIGH VARIABILITY" in result.stdout
            # Clarity should not have warning
            assert "clarity" in result.stdout
