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
Tests for rubric CLI functionality.

Tests validate rubric path resolution, preset handling, show-rubric command,
and integration with evaluate-single command.
"""

import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prompt_evaluator.cli import app
from prompt_evaluator.config import RUBRIC_PRESETS, resolve_rubric_path


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_rubric(tmp_path):
    """Fixture to create a temporary rubric file."""
    rubric_content = """
metrics:
  - name: test_metric
    description: A test metric
    min_score: 1
    max_score: 5
    guidelines: |
      Score 1: Poor
      Score 5: Excellent

flags:
  - name: test_flag
    description: A test flag
    default: false
"""
    rubric_file = tmp_path / "test_rubric.yaml"
    rubric_file.write_text(rubric_content)
    return rubric_file


class TestResolveRubricPath:
    """Tests for resolve_rubric_path function."""

    def test_resolve_none_returns_default(self):
        """Test that None input returns default rubric path."""
        result = resolve_rubric_path(None)
        assert result.exists()
        assert result.name == "default.yaml"

    def test_resolve_default_preset(self):
        """Test resolving 'default' preset alias."""
        result = resolve_rubric_path("default")
        assert result.exists()
        assert result.name == "default.yaml"

    def test_resolve_content_quality_preset(self):
        """Test resolving 'content-quality' preset alias."""
        result = resolve_rubric_path("content-quality")
        assert result.exists()
        assert result.name == "content_quality.yaml"

    def test_resolve_code_review_preset(self):
        """Test resolving 'code-review' preset alias."""
        result = resolve_rubric_path("code-review")
        assert result.exists()
        assert result.name == "code_review.json"

    def test_resolve_relative_file_path(self, temp_rubric):
        """Test resolving a relative file path."""
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_rubric.parent)
            result = resolve_rubric_path(temp_rubric.name)
            assert result.exists()
            assert result.resolve() == temp_rubric.resolve()
        finally:
            os.chdir(original_cwd)

    def test_resolve_absolute_file_path(self, temp_rubric):
        """Test resolving an absolute file path."""
        result = resolve_rubric_path(str(temp_rubric))
        assert result.exists()
        assert result.resolve() == temp_rubric.resolve()

    def test_resolve_nonexistent_file_raises_error(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_rubric_path(str(nonexistent))
        assert "not found" in str(exc_info.value).lower()
        assert "preset" in str(exc_info.value).lower()

    def test_resolve_path_traversal_rejected(self, tmp_path):
        """Test that path traversal attempts are rejected."""
        # Create a rubric file in temp directory
        rubric_file = tmp_path / "test.yaml"
        rubric_content = (
            "metrics:\n  - name: test\n    description: test\n"
            "    min_score: 1\n    max_score: 5\n    guidelines: test\n"
        )
        rubric_file.write_text(rubric_content)

        # Try to access it with path traversal patterns
        # Note: This might not trigger on all systems due to path normalization
        # but it demonstrates the intent
        traversal_path = str(tmp_path / ".." / tmp_path.name / "test.yaml")

        # The function should either reject it or resolve it safely
        # We're testing that it doesn't allow arbitrary path traversal
        try:
            result = resolve_rubric_path(traversal_path)
            # If it succeeds, ensure it resolved to the correct canonical path
            assert result.resolve() == rubric_file.resolve()
        except (ValueError, FileNotFoundError):
            # It's also acceptable to reject path traversal attempts
            pass

    def test_resolve_directory_raises_error(self, tmp_path):
        """Test that directory path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_rubric_path(str(tmp_path))
        assert "directory" in str(exc_info.value).lower()

    @pytest.mark.skipif(
        os.name == 'nt',
        reason="File permission tests not reliable on Windows"
    )
    def test_resolve_unreadable_file_raises_error(self, temp_rubric):
        """Test that unreadable file raises ValueError."""
        # Make file unreadable (Unix-like systems only)
        original_mode = temp_rubric.stat().st_mode
        try:
            temp_rubric.chmod(0o000)
            with pytest.raises(ValueError) as exc_info:
                resolve_rubric_path(str(temp_rubric))
            assert "not readable" in str(exc_info.value).lower()
        finally:
            # Restore original permissions even if test fails
            try:
                temp_rubric.chmod(original_mode)
            except (OSError, PermissionError):
                # If we can't restore permissions, at least try to make it writable
                try:
                    temp_rubric.chmod(0o644)
                except (OSError, PermissionError):
                    pass  # Best effort cleanup

    def test_all_presets_exist(self):
        """Test that all defined presets resolve to existing files."""
        for preset_name in RUBRIC_PRESETS.keys():
            result = resolve_rubric_path(preset_name)
            assert result.exists(), f"Preset '{preset_name}' file does not exist"


class TestShowRubricCommand:
    """Tests for the show-rubric CLI command."""

    def test_show_rubric_command_exists(self, cli_runner):
        """Test that show-rubric command is available."""
        result = cli_runner.invoke(app, ["show-rubric", "--help"])
        assert result.exit_code == 0
        assert "Display the effective rubric" in result.stdout

    def test_show_rubric_default(self, cli_runner):
        """Test show-rubric with default (no --rubric option)."""
        result = cli_runner.invoke(app, ["show-rubric"])
        assert result.exit_code == 0

        # Parse JSON output
        output = json.loads(result.stdout)
        assert "rubric_path" in output
        assert "metrics" in output
        assert "flags" in output
        assert len(output["metrics"]) > 0
        assert "default.yaml" in output["rubric_path"]

    def test_show_rubric_with_preset(self, cli_runner):
        """Test show-rubric with preset alias."""
        result = cli_runner.invoke(app, ["show-rubric", "--rubric", "content-quality"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "content_quality.yaml" in output["rubric_path"]
        assert len(output["metrics"]) > 0

    def test_show_rubric_with_file_path(self, cli_runner, temp_rubric):
        """Test show-rubric with file path."""
        result = cli_runner.invoke(app, ["show-rubric", "--rubric", str(temp_rubric)])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert temp_rubric.name in output["rubric_path"]
        assert len(output["metrics"]) == 1
        assert output["metrics"][0]["name"] == "test_metric"

    def test_show_rubric_invalid_preset(self, cli_runner):
        """Test show-rubric with invalid preset."""
        result = cli_runner.invoke(app, ["show-rubric", "--rubric", "nonexistent-preset"])
        assert result.exit_code == 1
        assert "Error loading rubric" in result.stdout

    def test_show_rubric_missing_file(self, cli_runner, tmp_path):
        """Test show-rubric with missing file."""
        missing_file = tmp_path / "missing.yaml"
        result = cli_runner.invoke(app, ["show-rubric", "--rubric", str(missing_file)])
        assert result.exit_code == 1
        assert "Error loading rubric" in result.stdout

    def test_show_rubric_directory_path(self, cli_runner, tmp_path):
        """Test show-rubric with directory path."""
        result = cli_runner.invoke(app, ["show-rubric", "--rubric", str(tmp_path)])
        assert result.exit_code == 1
        assert "Error loading rubric" in result.stdout
        assert "directory" in result.stdout.lower()

    def test_show_rubric_no_api_key_required(self, cli_runner, monkeypatch):
        """Test that show-rubric works without API key."""
        # Remove API key from environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = cli_runner.invoke(app, ["show-rubric"])
        # Should succeed without API key
        assert result.exit_code == 0


class TestEvaluateSingleWithRubric:
    """Tests for evaluate-single command with --rubric option."""

    def test_evaluate_single_rubric_option_in_help(self, cli_runner):
        """Test that --rubric option appears in evaluate-single help."""
        result = cli_runner.invoke(app, ["evaluate-single", "--help"])
        assert result.exit_code == 0
        # Strip ANSI codes for cleaner comparison
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        assert "--rubric" in clean_output
        assert "preset alias" in clean_output or "preset" in clean_output.lower()

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_default_rubric(
        self,
        mock_judge,
        mock_generate,
        cli_runner,
        tmp_path,
        monkeypatch,
    ):
        """Test evaluate-single uses default rubric when not specified."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Setup temporary files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are a helpful assistant.")
        user_input = tmp_path / "input.txt"
        user_input.write_text("What is Python?")
        output_dir = tmp_path / "runs"

        # Mock responses
        mock_generate.return_value = ("Sample output", {"latency_ms": 100, "tokens_used": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.5,
            "judge_rationale": "Good quality",
            "judge_raw_response": '{"semantic_fidelity": 4.5, "rationale": "Good quality"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(user_input),
                "--num-samples",
                "1",
                "--output-dir",
                str(output_dir),
            ],
        )

        # Check that command succeeded
        assert result.exit_code == 0, f"Command failed with: {result.stdout}"

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_preset_rubric(
        self,
        mock_judge,
        mock_generate,
        cli_runner,
        tmp_path,
        monkeypatch,
    ):
        """Test evaluate-single with preset rubric."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Setup temporary files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are a helpful assistant.")
        user_input = tmp_path / "input.txt"
        user_input.write_text("What is Python?")
        output_dir = tmp_path / "runs"

        # Mock responses
        mock_generate.return_value = ("Sample output", {"latency_ms": 100, "tokens_used": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.5,
            "judge_rationale": "Good quality",
            "judge_raw_response": '{"semantic_fidelity": 4.5, "rationale": "Good quality"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(user_input),
                "--num-samples",
                "1",
                "--rubric",
                "content-quality",
                "--output-dir",
                str(output_dir),
            ],
        )

        # Check that command succeeded
        assert result.exit_code == 0, f"Command failed with: {result.stdout}"
        # Verify rubric was passed to judge_completion
        assert mock_judge.called
        call_kwargs = mock_judge.call_args[1]
        assert "rubric" in call_kwargs
        assert call_kwargs["rubric"] is not None

    @patch("prompt_evaluator.cli.generate_completion")
    @patch("prompt_evaluator.cli.judge_completion")
    def test_evaluate_single_with_custom_rubric_file(
        self,
        mock_judge,
        mock_generate,
        cli_runner,
        tmp_path,
        temp_rubric,
        monkeypatch,
    ):
        """Test evaluate-single with custom rubric file."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Setup temporary files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are a helpful assistant.")
        user_input = tmp_path / "input.txt"
        user_input.write_text("What is Python?")
        output_dir = tmp_path / "runs"

        # Mock responses
        mock_generate.return_value = ("Sample output", {"latency_ms": 100, "tokens_used": 50})
        mock_judge.return_value = {
            "status": "completed",
            "judge_score": 4.5,
            "judge_rationale": "Good quality",
            "judge_raw_response": '{"semantic_fidelity": 4.5, "rationale": "Good quality"}',
            "error": None,
        }

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(user_input),
                "--num-samples",
                "1",
                "--rubric",
                str(temp_rubric),
                "--output-dir",
                str(output_dir),
            ],
        )

        # Check that command succeeded
        assert result.exit_code == 0, f"Command failed with: {result.stdout}"

    def test_evaluate_single_invalid_rubric(self, cli_runner, tmp_path, monkeypatch):
        """Test evaluate-single with invalid rubric fails gracefully."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Setup temporary files
        system_prompt = tmp_path / "system.txt"
        system_prompt.write_text("You are a helpful assistant.")
        user_input = tmp_path / "input.txt"
        user_input.write_text("What is Python?")
        output_dir = tmp_path / "runs"

        result = cli_runner.invoke(
            app,
            [
                "evaluate-single",
                "--system-prompt",
                str(system_prompt),
                "--input",
                str(user_input),
                "--num-samples",
                "1",
                "--rubric",
                "invalid-preset",
                "--output-dir",
                str(output_dir),
            ],
        )

        # Should fail with rubric error
        assert result.exit_code == 1
        assert "Error loading rubric" in result.stdout
