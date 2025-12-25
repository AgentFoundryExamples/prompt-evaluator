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
Tests for the generate CLI command.

Tests validate CLI parameter parsing, file reading, config merging,
and output writing functionality.
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
    user_input.write_text("What is 2+2?")

    return {
        "system": system_prompt,
        "input": user_input,
        "output_dir": tmp_path / "runs",
    }


class TestGenerateCLI:
    """Tests for the generate CLI command."""

    def test_generate_command_exists(self, cli_runner):
        """Test that generate command is available."""
        result = cli_runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate a completion" in result.stdout

    def test_generate_missing_required_params(self, cli_runner):
        """Test that generate command requires system-prompt and input."""
        result = cli_runner.invoke(app, ["generate"])
        assert result.exit_code != 0
        assert "Missing option" in result.stdout or "required" in result.stdout.lower()

    def test_generate_missing_system_prompt_file(self, cli_runner, tmp_path, monkeypatch):
        """Test error handling when system prompt file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        input_file = tmp_path / "input.txt"
        input_file.write_text("test input")

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(tmp_path / "nonexistent.txt"),
                "--input",
                str(input_file),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_generate_missing_input_file(self, cli_runner, tmp_path, monkeypatch):
        """Test error handling when input file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a helpful assistant.")

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(system_file),
                "--input",
                str(tmp_path / "nonexistent.txt"),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_with_files(self, mock_generate, cli_runner, temp_prompts, monkeypatch):
        """Test generate command with valid file inputs."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock the generate_completion function
        mock_generate.return_value = (
            "Paris is the capital of France.",
            {"tokens_used": 10, "latency_ms": 100.5},
        )

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
        assert "Paris is the capital of France." in result.stdout
        assert mock_generate.called

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_with_overrides(self, mock_generate, cli_runner, temp_prompts, monkeypatch):
        """Test generate command with parameter overrides."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Test response", {"tokens_used": 5, "latency_ms": 50.0})

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--model",
                "gpt-4",
                "--temperature",
                "0.5",
                "--max-tokens",
                "500",
                "--seed",
                "42",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0
        assert mock_generate.called

        # Verify the correct parameters were passed
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_completion_tokens"] == 500
        assert call_kwargs["seed"] == 42

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_creates_output_files(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that generate command creates output and metadata files."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        response_text = "This is the response."
        mock_generate.return_value = (response_text, {"tokens_used": 8, "latency_ms": 75.0})

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

        # Check that output directory was created
        assert temp_prompts["output_dir"].exists()

        # Find the run directory (should be a UUID)
        run_dirs = list(temp_prompts["output_dir"].iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Check output file
        output_file = run_dir / "output.txt"
        assert output_file.exists()
        assert output_file.read_text() == response_text

        # Check metadata file
        metadata_file = run_dir / "metadata.json"
        assert metadata_file.exists()
        metadata = json.loads(metadata_file.read_text())
        assert "id" in metadata
        assert "timestamp" in metadata
        assert metadata["tokens_used"] == 8
        assert metadata["latency_ms"] == 75.0

    def test_generate_missing_api_key(self, cli_runner, temp_prompts, monkeypatch):
        """Test error handling when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
        assert "API key is required" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_with_stdin(self, mock_generate, cli_runner, temp_prompts, monkeypatch):
        """Test generate command with stdin input."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Response from stdin", {"tokens_used": 6, "latency_ms": 60.0})

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                "-",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
            input="What is the meaning of life?",
        )

        assert result.exit_code == 0
        assert "Response from stdin" in result.stdout
        assert mock_generate.called

        # Verify stdin content was passed
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["user_prompt"] == "What is the meaning of life?"

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_validates_temperature_bounds(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that temperature validation works in CLI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Test temperature too low
        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--temperature",
                "-0.1",
            ],
        )
        assert result.exit_code == 1
        assert "temperature must be between 0.0 and 2.0" in result.stdout

        # Test temperature too high
        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--temperature",
                "2.5",
            ],
        )
        assert result.exit_code == 1
        assert "temperature must be between 0.0 and 2.0" in result.stdout

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_seed_parameter_wiring(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that seed parameter is properly wired through CLI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Test response", {"tokens_used": 5, "latency_ms": 50.0})

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--seed",
                "12345",
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        assert result.exit_code == 0
        assert mock_generate.called

        # Verify seed was passed correctly
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["seed"] == 12345

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_without_seed_uses_none(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that without seed parameter, None is used."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_generate.return_value = ("Test response", {"tokens_used": 5, "latency_ms": 50.0})

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
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["seed"] is None

    @patch("prompt_evaluator.cli.generate_completion")
    def test_generate_no_real_api_calls_with_mock(
        self, mock_generate, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that mocked tests don't make real API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-12345")

        mock_generate.return_value = ("Mocked response", {"tokens_used": 10, "latency_ms": 100})

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

        # Should succeed with mock, no real API call
        assert result.exit_code == 0
        assert "Mocked response" in result.stdout
        # Verify mock was called instead of real API
        assert mock_generate.call_count == 1

    def test_generate_uses_config_defaults_for_provider(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that generate uses config default for provider when flag is omitted."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a config file with custom provider default (using mock provider)
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            """
defaults:
  generator:
    provider: mock
    model: test-model-from-config
    temperature: 0.5
    max_completion_tokens: 2048
  run_directory: custom_runs
"""
        )

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should succeed and use config provider
        assert result.exit_code == 0
        # Check that config provider was used (mock)
        assert "Using provider from config: mock" in result.stdout
        # Mock provider should return a mock response
        assert "Mock response" in result.stdout

    def test_generate_cli_override_takes_precedence_over_config(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that CLI flag for provider overrides config default."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a config file with default provider (anthropic)
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            """
defaults:
  generator:
    provider: anthropic
    model: test-model
"""
        )

        # Explicitly provide --provider to override config (use mock)
        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--provider",
                "mock",
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should succeed with explicit provider
        assert result.exit_code == 0
        # Should NOT show "Using provider from config" since CLI flag was provided
        assert "Using provider from config:" not in result.stdout
        # Mock provider should be used (CLI override)
        assert "Mock response" in result.stdout

    def test_generate_uses_config_defaults_for_output_dir(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that generate uses config default for output_dir when flag is omitted."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a config file with custom run_directory
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            """
defaults:
  generator:
    provider: mock
    model: test-model
  run_directory: config_custom_output
"""
        )

        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                str(temp_prompts["system"]),
                "--input",
                str(temp_prompts["input"]),
                "--config",
                str(config_file),
            ],
        )

        # Should succeed and use config output directory
        assert result.exit_code == 0
        assert "Using output directory from config: config_custom_output" in result.stdout

    def test_generate_resolves_prompt_template_key(
        self, cli_runner, temp_prompts, monkeypatch
    ):
        """Test that generate resolves prompt template keys from config."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a system prompt file
        system_prompt_file = temp_prompts["system"].parent / "my_system_prompt.txt"
        system_prompt_file.write_text("You are a coding assistant.")

        # Create a config file with prompt template mapping
        config_file = temp_prompts["system"].parent / "test_config.yaml"
        config_file.write_text(
            f"""
defaults:
  generator:
    provider: mock
    model: test-model
prompt_templates:
  my_template: {system_prompt_file.name}
"""
        )

        # Use template key instead of file path
        result = cli_runner.invoke(
            app,
            [
                "generate",
                "--system-prompt",
                "my_template",
                "--input",
                str(temp_prompts["input"]),
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_prompts["output_dir"]),
            ],
        )

        # Should succeed and resolve template key
        assert result.exit_code == 0
        # Check that mock provider was used and template was resolved
        assert "Mock response" in result.stdout
