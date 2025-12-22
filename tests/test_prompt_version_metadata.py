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
Tests for prompt version metadata functionality.

Tests validate prompt hashing, version metadata computation,
and integration with CLI commands.
"""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from prompt_evaluator.cli import app, compute_prompt_metadata
from prompt_evaluator.models import Rubric, RubricMetric
from prompt_evaluator.provider import OpenAIProvider


# Helper to strip ANSI color codes from CLI output
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text."""
    return ANSI_ESCAPE.sub('', text)


@pytest.fixture
def cli_runner():
    """Fixture for Typer CLI runner."""
    return CliRunner()


class TestComputePromptMetadata:
    """Tests for compute_prompt_metadata function."""

    def test_compute_prompt_metadata_with_version(self, tmp_path):
        """Test computing metadata with user-provided version."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a helpful assistant.")
        
        version_id, prompt_hash = compute_prompt_metadata(prompt_file, "v1.0")
        
        assert version_id == "v1.0"
        assert len(prompt_hash) == 64  # SHA-256 hex digest
        # Hash should be consistent for the same content
        expected_hash = hashlib.sha256(b"You are a helpful assistant.").hexdigest()
        assert prompt_hash == expected_hash

    def test_compute_prompt_metadata_without_version(self, tmp_path):
        """Test computing metadata without user-provided version (uses hash)."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_content = "You are a helpful assistant."
        prompt_file.write_text(prompt_content)
        
        version_id, prompt_hash = compute_prompt_metadata(prompt_file, None)
        
        # When no version provided, version_id should equal prompt_hash
        assert version_id == prompt_hash
        assert len(prompt_hash) == 64  # SHA-256 hex digest

    def test_compute_prompt_metadata_consistent_hash(self, tmp_path):
        """Test that same content produces same hash."""
        prompt_file1 = tmp_path / "prompt1.txt"
        prompt_file2 = tmp_path / "prompt2.txt"
        
        content = "You are a helpful assistant."
        prompt_file1.write_text(content)
        prompt_file2.write_text(content)
        
        _, hash1 = compute_prompt_metadata(prompt_file1, None)
        _, hash2 = compute_prompt_metadata(prompt_file2, None)
        
        assert hash1 == hash2

    def test_compute_prompt_metadata_different_content(self, tmp_path):
        """Test that different content produces different hashes."""
        prompt_file1 = tmp_path / "prompt1.txt"
        prompt_file2 = tmp_path / "prompt2.txt"
        
        prompt_file1.write_text("Content A")
        prompt_file2.write_text("Content B")
        
        _, hash1 = compute_prompt_metadata(prompt_file1, None)
        _, hash2 = compute_prompt_metadata(prompt_file2, None)
        
        assert hash1 != hash2

    def test_compute_prompt_metadata_file_not_found(self, tmp_path):
        """Test error handling when prompt file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            compute_prompt_metadata(nonexistent_file, None)
        
        assert "not found" in str(exc_info.value).lower()

    def test_compute_prompt_metadata_empty_file(self, tmp_path):
        """Test error handling when prompt file is empty."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        with pytest.raises(ValueError) as exc_info:
            compute_prompt_metadata(empty_file, None)
        
        assert "empty" in str(exc_info.value).lower()

    def test_compute_prompt_metadata_with_binary_content(self, tmp_path):
        """Test that binary content is hashed correctly."""
        prompt_file = tmp_path / "prompt.txt"
        # Write some binary content
        prompt_file.write_bytes(b"Binary content with special chars: \x00\xff")
        
        version_id, prompt_hash = compute_prompt_metadata(prompt_file, None)
        
        assert len(prompt_hash) == 64
        assert version_id == prompt_hash


class TestEvaluateSingleWithPromptMetadata:
    """Tests for evaluate-single command with prompt metadata."""

    def test_evaluate_single_help_shows_prompt_version(self, cli_runner):
        """Test that --prompt-version option appears in help."""
        result = cli_runner.invoke(app, ["evaluate-single", "--help"])
        
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--prompt-version" in clean_output

    def test_evaluate_single_help_shows_run_note(self, cli_runner):
        """Test that --run-note option appears in help."""
        result = cli_runner.invoke(app, ["evaluate-single", "--help"])
        
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--run-note" in clean_output

    def test_evaluate_single_integration_with_metadata(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Integration test that verifies metadata appears in output JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a helpful assistant.")
        
        input_file = tmp_path / "input.txt"
        input_file.write_text("What is Python?")
        
        output_dir = tmp_path / "runs"
        
        # Mock the provider and functions to avoid real API calls
        with patch("prompt_evaluator.cli.get_provider") as mock_get_provider, \
             patch("prompt_evaluator.cli.generate_completion") as mock_generate, \
             patch("prompt_evaluator.cli.judge_completion") as mock_judge, \
             patch("prompt_evaluator.config.load_rubric") as mock_rubric, \
             patch("prompt_evaluator.models.load_judge_prompt") as mock_judge_prompt, \
             patch("prompt_evaluator.config.resolve_rubric_path") as mock_resolve:
            
            # Setup provider mock to return an OpenAIProvider instance
            mock_provider = MagicMock(spec=OpenAIProvider)
            mock_get_provider.return_value = mock_provider
            
            # Setup other mocks
            mock_generate.return_value = ("Test output", {"latency_ms": 100})
            mock_judge.return_value = {
                "status": "completed",
                "judge_metrics": {"test_metric": {"score": 4.0, "rationale": "Good"}},
                "judge_flags": {},
            }
            # Create proper Rubric mock using actual Rubric class
            mock_rubric_obj = Rubric(
                metrics=[
                    RubricMetric(
                        name="test_metric",
                        description="Test metric",
                        min_score=1.0,
                        max_score=5.0,
                        guidelines="Test guidelines"
                    )
                ],
                flags=[]
            )
            mock_rubric.return_value = mock_rubric_obj
            mock_resolve.return_value = Path("fake_rubric.yaml")
            mock_judge_prompt.return_value = "Judge prompt"
            
            result = cli_runner.invoke(
                app,
                [
                    "evaluate-single",
                    "--system-prompt", str(system_file),
                    "--input", str(input_file),
                    "--num-samples", "1",
                    "--prompt-version", "v1.0.0",
                    "--run-note", "Integration test run",
                    "--output-dir", str(output_dir),
                ],
            )
            
            # Verify command succeeded
            assert result.exit_code == 0, f"Command failed with output: {result.stdout}"
            
            # Find the output file
            run_dirs = list(output_dir.glob("*"))
            assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
            
            output_file = run_dirs[0] / "evaluate-single.json"
            assert output_file.exists(), f"Output file not found at {output_file}"
            
            # Parse and verify JSON content
            with open(output_file) as f:
                data = json.load(f)
            
            # Verify metadata fields exist and have correct values
            assert "prompt_version_id" in data, "prompt_version_id missing from output"
            assert data["prompt_version_id"] == "v1.0.0", \
                f"Expected prompt_version_id='v1.0.0', got '{data['prompt_version_id']}'"
            
            assert "prompt_hash" in data, "prompt_hash missing from output"
            assert len(data["prompt_hash"]) == 64, \
                f"Expected 64-char hash, got {len(data['prompt_hash'])} chars"
            
            # Verify the hash matches the actual file content
            expected_hash = hashlib.sha256(b"You are a helpful assistant.").hexdigest()
            assert data["prompt_hash"] == expected_hash, \
                f"Hash mismatch: expected {expected_hash}, got {data['prompt_hash']}"
            
            assert "run_notes" in data, "run_notes missing from output"
            assert data["run_notes"] == "Integration test run", \
                f"Expected run_notes='Integration test run', got '{data['run_notes']}'"

    def test_evaluate_single_integration_without_version(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Integration test that verifies auto-generated hash when version not provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        system_file = tmp_path / "system.txt"
        system_file.write_text("Test prompt content")
        
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test input")
        
        output_dir = tmp_path / "runs"
        
        # Mock the provider and functions
        with patch("prompt_evaluator.cli.get_provider") as mock_get_provider, \
             patch("prompt_evaluator.cli.generate_completion") as mock_generate, \
             patch("prompt_evaluator.cli.judge_completion") as mock_judge, \
             patch("prompt_evaluator.config.load_rubric") as mock_rubric, \
             patch("prompt_evaluator.models.load_judge_prompt") as mock_judge_prompt, \
             patch("prompt_evaluator.config.resolve_rubric_path") as mock_resolve:
            
            # Setup provider mock
            mock_provider = MagicMock(spec=OpenAIProvider)
            mock_get_provider.return_value = mock_provider
            
            # Setup other mocks
            mock_generate.return_value = ("Output", {"latency_ms": 100})
            mock_judge.return_value = {
                "status": "completed",
                "judge_metrics": {"metric": {"score": 5.0, "rationale": "Perfect"}},
                "judge_flags": {},
            }
            # Create proper Rubric mock using actual Rubric class
            mock_rubric_obj = Rubric(
                metrics=[
                    RubricMetric(
                        name="metric",
                        description="Test metric",
                        min_score=1.0,
                        max_score=5.0,
                        guidelines="Test guidelines"
                    )
                ],
                flags=[]
            )
            mock_rubric.return_value = mock_rubric_obj
            mock_resolve.return_value = Path("fake_rubric.yaml")
            mock_judge_prompt.return_value = "Judge prompt"
            
            result = cli_runner.invoke(
                app,
                [
                    "evaluate-single",
                    "--system-prompt", str(system_file),
                    "--input", str(input_file),
                    "--num-samples", "1",
                    "--output-dir", str(output_dir),
                ],
            )
            
            assert result.exit_code == 0, f"Command failed: {result.stdout}"
            
            # Find and parse output
            run_dirs = list(output_dir.glob("*"))
            assert len(run_dirs) == 1
            
            output_file = run_dirs[0] / "evaluate-single.json"
            with open(output_file) as f:
                data = json.load(f)
            
            # When no version provided, version_id should equal hash
            assert "prompt_version_id" in data
            assert "prompt_hash" in data
            assert data["prompt_version_id"] == data["prompt_hash"], \
                "Without --prompt-version, version_id should equal hash"
            
            # Verify hash is correct
            expected_hash = hashlib.sha256(b"Test prompt content").hexdigest()
            assert data["prompt_hash"] == expected_hash
            
            # run_notes should be None when not provided
            assert data["run_notes"] is None


class TestEvaluateDatasetWithPromptMetadata:
    """Tests for evaluate-dataset command with prompt metadata."""

    def test_evaluate_dataset_with_prompt_version_option(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Test that --prompt-version option is accepted for dataset evaluation."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a helpful assistant.")
        
        dataset_file = tmp_path / "dataset.yaml"
        dataset_file.write_text("- id: test-1\n  input: What is Python?\n")
        
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--dataset", str(dataset_file),
                "--system-prompt", str(system_file),
                "--num-samples", "1",
                "--prompt-version", "v2.0",
                "--output-dir", str(tmp_path / "runs"),
                "--help",
            ],
        )
        
        # Just checking that the option is recognized (help should work)
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--prompt-version" in clean_output

    def test_evaluate_dataset_with_run_note_option(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Test that --run-note option is accepted for dataset evaluation."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        result = cli_runner.invoke(
            app,
            [
                "evaluate-dataset",
                "--help",
            ],
        )
        
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--run-note" in clean_output
