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
Tests for the compare-prompts CLI command.

Tests validate dual-prompt evaluation workflow, CLI parameter parsing,
and comparison generation.
"""

import json

import pytest
import yaml
from typer.testing import CliRunner

from prompt_evaluator.cli import app


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner(mix_stderr=False)


@pytest.fixture
def sample_dataset(tmp_path):
    """Fixture to create a sample dataset for testing."""
    dataset_content = [
        {
            "id": "test-001",
            "input": "Write a hello world program",
            "task": "Generate code",
            "description": "Basic hello world test",
        },
        {
            "id": "test-002",
            "input": "Explain recursion",
            "task": "Explain concept",
            "description": "Recursion explanation test",
        },
    ]
    dataset_path = tmp_path / "test_dataset.yaml"
    import yaml
    dataset_path.write_text(yaml.dump(dataset_content))
    return dataset_path


@pytest.fixture
def sample_prompts(tmp_path):
    """Fixture to create sample prompt files."""
    prompt_a = tmp_path / "prompt_a.txt"
    prompt_b = tmp_path / "prompt_b.txt"
    
    prompt_a.write_text("You are a helpful assistant. Be concise.")
    prompt_b.write_text("You are a helpful assistant. Be detailed and thorough.")
    
    return prompt_a, prompt_b


@pytest.fixture
def identical_prompts(tmp_path):
    """Fixture to create identical prompt files for sanity check testing."""
    prompt_a = tmp_path / "prompt_identical_a.txt"
    prompt_b = tmp_path / "prompt_identical_b.txt"
    
    prompt_content = "You are a helpful assistant."
    prompt_a.write_text(prompt_content)
    prompt_b.write_text(prompt_content)
    
    return prompt_a, prompt_b


class TestComparePromptsCLI:
    """Tests for the compare-prompts CLI command."""

    def test_compare_prompts_command_exists(self, cli_runner):
        """Test that compare-prompts command is available."""
        result = cli_runner.invoke(app, ["compare-prompts", "--help"])
        assert result.exit_code == 0
        # Check for key terms in help output (case-insensitive, handle ANSI codes)
        output = result.stdout.lower().replace('\x1b[0m', '').replace('\x1b[1m', '').replace('\x1b[2m', '').replace('\x1b[1;33m', '').replace('\x1b[1;36m', '')
        assert "compare" in output
        assert "prompt-a" in output
        assert "prompt-b" in output

    def test_compare_prompts_missing_required_params(self, cli_runner):
        """Test that compare-prompts requires dataset, prompt-a, and prompt-b."""
        result = cli_runner.invoke(app, ["compare-prompts"])
        assert result.exit_code != 0
        error_output = result.stderr if result.stderr else result.stdout
        assert "Missing option" in error_output or "required" in error_output.lower()

    def test_compare_prompts_missing_dataset(self, cli_runner, sample_prompts):
        """Test error handling when dataset is missing."""
        prompt_a, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
            ],
        )
        assert result.exit_code != 0
        error_output = result.stderr if result.stderr else result.stdout
        assert "Missing option" in error_output or "dataset" in error_output.lower()

    def test_compare_prompts_missing_prompt_a(self, cli_runner, sample_dataset, sample_prompts):
        """Test error handling when prompt-a is missing."""
        _, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-b",
                str(prompt_b),
            ],
        )
        assert result.exit_code != 0
        error_output = result.stderr if result.stderr else result.stdout
        assert "Missing option" in error_output or "prompt-a" in error_output.lower()

    def test_compare_prompts_missing_prompt_b(self, cli_runner, sample_dataset, sample_prompts):
        """Test error handling when prompt-b is missing."""
        prompt_a, _ = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
            ],
        )
        assert result.exit_code != 0
        error_output = result.stderr if result.stderr else result.stdout
        assert "Missing option" in error_output or "prompt-b" in error_output.lower()

    def test_compare_prompts_nonexistent_dataset(self, cli_runner, sample_prompts, tmp_path):
        """Test error handling when dataset file doesn't exist."""
        prompt_a, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(tmp_path / "nonexistent.yaml"),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_compare_prompts_nonexistent_prompt_a(self, cli_runner, sample_dataset, sample_prompts, tmp_path):
        """Test error handling when prompt-a file doesn't exist."""
        _, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(tmp_path / "nonexistent.txt"),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_compare_prompts_nonexistent_prompt_b(self, cli_runner, sample_dataset, sample_prompts, tmp_path):
        """Test error handling when prompt-b file doesn't exist."""
        prompt_a, _ = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(tmp_path / "nonexistent.txt"),
                "--provider",
                "mock",
                "--quick",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_compare_prompts_successful_with_mock_provider(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path, monkeypatch
    ):
        """Test successful dual-prompt comparison with mock provider."""
        # Set dummy API key for config loading
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        # Should succeed with mock provider
        assert result.exit_code == 0
        
        # Verify JSON output structure
        output_data = json.loads(result.stdout)
        assert "prompt_a_run_id" in output_data
        assert "prompt_b_run_id" in output_data
        assert "prompt_a_status" in output_data
        assert "prompt_b_status" in output_data
        assert "comparison" in output_data
        assert "prompt_a_artifact_path" in output_data
        assert "prompt_b_artifact_path" in output_data
        assert "comparison_artifact_path" in output_data
        
        # Verify stderr contains progress messages
        assert "Phase 1/3: Evaluating Prompt A" in result.stderr
        assert "Phase 2/3: Evaluating Prompt B" in result.stderr
        assert "Phase 3/3: Comparing results" in result.stderr
        assert "Dual-Prompt Comparison Complete!" in result.stderr

    def test_compare_prompts_identical_prompts_warning(
        self, cli_runner, sample_dataset, identical_prompts, tmp_path
    ):
        """Test that warning is shown when comparing identical prompts."""
        prompt_a, prompt_b = identical_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        # Should succeed
        assert result.exit_code == 0
        
        # Should warn about identical prompts
        assert "identical content" in result.stderr.lower() or "WARNING" in result.stderr

    def test_compare_prompts_with_custom_num_samples(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with custom number of samples."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--num-samples",
                "3",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Samples per Case: 3" in result.stderr

    def test_compare_prompts_with_case_ids_filter(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with case-ids filter."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--case-ids",
                "test-001",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Filtered to 1 test cases" in result.stderr

    def test_compare_prompts_with_max_cases(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with max-cases limit."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--max-cases",
                "1",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Limited to first 1 test cases" in result.stderr

    def test_compare_prompts_with_custom_thresholds(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with custom regression thresholds."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--metric-threshold",
                "0.5",
                "--flag-threshold",
                "0.1",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0

    def test_compare_prompts_with_prompt_versions(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with custom prompt version identifiers."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--prompt-a-version",
                "v1.0",
                "--prompt-b-version",
                "v2.0",
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Version: v1.0" in result.stderr
        assert "Version: v2.0" in result.stderr

    def test_compare_prompts_with_run_note(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test compare-prompts with run note."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--run-note",
                "Testing concise vs detailed prompts",
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0

    def test_compare_prompts_invalid_num_samples(self, cli_runner, sample_dataset, sample_prompts):
        """Test error handling for invalid num-samples."""
        prompt_a, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--num-samples",
                "-1",
            ],
        )
        
        assert result.exit_code == 1
        assert "must be positive" in result.stderr.lower()

    def test_compare_prompts_invalid_metric_threshold(self, cli_runner, sample_dataset, sample_prompts):
        """Test error handling for negative metric threshold."""
        prompt_a, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--metric-threshold",
                "-0.1",
            ],
        )
        
        assert result.exit_code == 1
        assert "must be non-negative" in result.stderr.lower()

    def test_compare_prompts_invalid_flag_threshold(self, cli_runner, sample_dataset, sample_prompts):
        """Test error handling for negative flag threshold."""
        prompt_a, prompt_b = sample_prompts
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--flag-threshold",
                "-0.05",
            ],
        )
        
        assert result.exit_code == 1
        assert "must be non-negative" in result.stderr.lower()


class TestComparePromptsArtifacts:
    """Tests for artifact generation in compare-prompts command."""

    def test_compare_prompts_creates_separate_run_directories(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test that separate directories are created for each prompt."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        
        # Parse output to get run IDs
        output_data = json.loads(result.stdout)
        prompt_a_run_id = output_data["prompt_a_run_id"]
        prompt_b_run_id = output_data["prompt_b_run_id"]
        
        # Verify directories exist
        assert (output_dir / prompt_a_run_id).exists()
        assert (output_dir / prompt_b_run_id).exists()
        
        # Verify artifacts exist
        assert (output_dir / prompt_a_run_id / "dataset_evaluation.json").exists()
        assert (output_dir / prompt_b_run_id / "dataset_evaluation.json").exists()

    def test_compare_prompts_creates_comparison_directory(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test that comparison directory and artifacts are created."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        
        # Parse output to get comparison path
        output_data = json.loads(result.stdout)
        comparison_artifact_path = output_data["comparison_artifact_path"]
        
        # Verify comparison artifact exists
        assert Path(comparison_artifact_path).exists()
        
        # Verify comparison report exists
        comparison_dir = Path(comparison_artifact_path).parent
        report_path = comparison_dir / "comparison_report.md"
        assert report_path.exists()


class TestComparePromptsQuickMode:
    """Tests for quick mode in compare-prompts command."""

    def test_compare_prompts_quick_mode_sets_num_samples(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test that quick mode sets num-samples to 2."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Quick mode: Using --num-samples=2" in result.stderr

    def test_compare_prompts_quick_mode_overridden_by_num_samples(
        self, cli_runner, sample_dataset, sample_prompts, tmp_path
    ):
        """Test that explicit num-samples overrides quick mode."""
        prompt_a, prompt_b = sample_prompts
        output_dir = tmp_path / "runs"
        
        result = cli_runner.invoke(
            app,
            [
                "compare-prompts",
                "--dataset",
                str(sample_dataset),
                "--prompt-a",
                str(prompt_a),
                "--prompt-b",
                str(prompt_b),
                "--provider",
                "mock",
                "--quick",
                "--num-samples",
                "5",
                "--output-dir",
                str(output_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Using explicit --num-samples=5" in result.stderr
