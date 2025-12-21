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
Tests for configuration loading and data models.

Tests validate GeneratorConfig, PromptRun, and config loading functionality.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from prompt_evaluator.config import APIConfig, load_api_config
from prompt_evaluator.models import GeneratorConfig, PromptRun


class TestGeneratorConfig:
    """Tests for GeneratorConfig dataclass."""

    def test_default_values(self):
        """Test that GeneratorConfig has sensible defaults."""
        config = GeneratorConfig()
        assert config.model_name == "gpt-5.1"
        assert config.temperature == 0.7
        assert config.max_completion_tokens == 1024
        assert config.seed is None

    def test_custom_values(self):
        """Test that GeneratorConfig accepts custom values."""
        config = GeneratorConfig(
            model_name="gpt-4", temperature=0.5, max_completion_tokens=2048, seed=42
        )
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_completion_tokens == 2048
        assert config.seed == 42

    def test_temperature_validation_negative(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            GeneratorConfig(temperature=-0.1)

    def test_temperature_validation_too_high(self):
        """Test that temperature > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            GeneratorConfig(temperature=2.1)

    def test_temperature_validation_valid_boundaries(self):
        """Test that temperature at boundaries (0.0, 2.0) is valid."""
        config1 = GeneratorConfig(temperature=0.0)
        assert config1.temperature == 0.0

        config2 = GeneratorConfig(temperature=2.0)
        assert config2.temperature == 2.0

    def test_max_completion_tokens_validation_zero(self):
        """Test that max_completion_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            GeneratorConfig(max_completion_tokens=0)

    def test_max_completion_tokens_validation_negative(self):
        """Test that negative max_completion_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            GeneratorConfig(max_completion_tokens=-100)

    def test_max_completion_tokens_validation_non_integer(self):
        """Test that non-integer max_completion_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_completion_tokens must be an integer"):
            GeneratorConfig(max_completion_tokens=100.5)  # type: ignore[arg-type]


class TestPromptRun:
    """Tests for PromptRun dataclass."""

    def test_prompt_run_creation(self):
        """Test that PromptRun can be created with required fields."""
        config = GeneratorConfig()
        timestamp = datetime.now(timezone.utc)
        run = PromptRun(
            id="run-123",
            timestamp=timestamp,
            system_prompt_path=Path("/path/to/system.txt"),
            input_path=Path("/path/to/input.txt"),
            model_config=config,
            raw_output_path=Path("/path/to/output.txt"),
        )

        assert run.id == "run-123"
        assert run.timestamp == timestamp
        assert run.system_prompt_path == Path("/path/to/system.txt")
        assert run.input_path == Path("/path/to/input.txt")
        assert run.model_config == config
        assert run.raw_output_path == Path("/path/to/output.txt")

    def test_prompt_run_to_dict(self):
        """Test that PromptRun.to_dict() produces JSON-compatible dict."""
        config = GeneratorConfig(model_name="gpt-4", temperature=0.5)
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        run = PromptRun(
            id="run-456",
            timestamp=timestamp,
            system_prompt_path=Path("/system.txt"),
            input_path=Path("/input.txt"),
            model_config=config,
            raw_output_path=Path("/output.txt"),
        )

        result = run.to_dict()

        assert result["id"] == "run-456"
        assert result["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert result["system_prompt_path"] == "/system.txt"
        assert result["input_path"] == "/input.txt"
        assert result["raw_output_path"] == "/output.txt"
        assert result["model_config"]["model_name"] == "gpt-4"
        assert result["model_config"]["temperature"] == 0.5
        assert result["model_config"]["max_completion_tokens"] == 1024
        assert result["model_config"]["seed"] is None

    def test_prompt_run_to_dict_with_relative_paths(self):
        """Test that PromptRun.to_dict() handles relative paths consistently."""
        config = GeneratorConfig()
        timestamp = datetime.now(timezone.utc)
        run = PromptRun(
            id="run-789",
            timestamp=timestamp,
            system_prompt_path=Path("relative/system.txt"),
            input_path=Path("relative/input.txt"),
            model_config=config,
            raw_output_path=Path("relative/output.txt"),
        )

        result = run.to_dict()

        assert result["system_prompt_path"] == "relative/system.txt"
        assert result["input_path"] == "relative/input.txt"
        assert result["raw_output_path"] == "relative/output.txt"


class TestAPIConfig:
    """Tests for APIConfig class."""

    def test_api_config_with_env_vars(self, monkeypatch):
        """Test that APIConfig reads from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.openai.com")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")

        config = APIConfig()

        assert config.api_key == "test-key-123"
        assert config.base_url == "https://test.openai.com"
        assert config.model_name == "gpt-4"

    def test_api_config_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            APIConfig()

    def test_api_config_default_model(self, monkeypatch):
        """Test that default model is used when env var not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        config = APIConfig()

        assert config.model_name == "gpt-5.1"

    def test_api_config_with_yaml_file(self, monkeypatch, tmp_path):
        """Test that APIConfig reads from YAML config file."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
api_key: yaml-key-456
base_url: https://yaml.openai.com
model_name: gpt-4-turbo
""")

        config = APIConfig(config_file_path=config_file)

        assert config.api_key == "yaml-key-456"
        assert config.base_url == "https://yaml.openai.com"
        assert config.model_name == "gpt-4-turbo"

    def test_api_config_with_toml_file(self, monkeypatch, tmp_path):
        """Test that APIConfig reads from TOML config file."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config_file = tmp_path / "config.toml"
        config_file.write_text("""
api_key = "toml-key-789"
base_url = "https://toml.openai.com"
model_name = "gpt-4-vision"
""")

        config = APIConfig(config_file_path=config_file)

        assert config.api_key == "toml-key-789"
        assert config.base_url == "https://toml.openai.com"
        assert config.model_name == "gpt-4-vision"

    def test_api_config_file_overrides_env(self, monkeypatch, tmp_path):
        """Test that config file values override environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
api_key: file-key
model_name: gpt-4
""")

        config = APIConfig(config_file_path=config_file)

        assert config.api_key == "file-key"
        assert config.model_name == "gpt-4"

    def test_api_config_missing_file_falls_back_to_env(self, monkeypatch, tmp_path):
        """Test that missing config file gracefully falls back to env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")

        non_existent_file = tmp_path / "nonexistent.yaml"

        config = APIConfig(config_file_path=non_existent_file)

        assert config.api_key == "fallback-key"
        assert config.model_name == "gpt-5.1"

    def test_api_config_malformed_yaml_raises_error(self, monkeypatch, tmp_path):
        """Test that malformed YAML raises clear error."""
        monkeypatch.setenv("OPENAI_API_KEY", "backup-key")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Failed to load config file"):
            APIConfig(config_file_path=config_file)

    def test_api_config_unsupported_format_raises_error(self, monkeypatch, tmp_path):
        """Test that unsupported config format raises error."""
        monkeypatch.setenv("OPENAI_API_KEY", "backup-key")

        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "json-key"}')

        with pytest.raises(ValueError, match="Unsupported config file format"):
            APIConfig(config_file_path=config_file)

    def test_api_config_to_dict(self, monkeypatch):
        """Test that APIConfig.to_dict() produces correct dictionary."""
        monkeypatch.setenv("OPENAI_API_KEY", "dict-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://dict.openai.com")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")

        config = APIConfig()
        result = config.to_dict()

        assert result == {
            "api_key": "dict-key",
            "base_url": "https://dict.openai.com",
            "model_name": "gpt-4",
        }

    def test_load_api_config_function(self, monkeypatch):
        """Test the load_api_config convenience function."""
        monkeypatch.setenv("OPENAI_API_KEY", "func-key")

        config = load_api_config()

        assert config.api_key == "func-key"
        assert isinstance(config, APIConfig)

    def test_api_config_explicit_parameters_override_all(self, monkeypatch, tmp_path):
        """Test that explicit parameters override both env and file."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("api_key: file-key\nmodel_name: gpt-4")

        config = APIConfig(
            api_key="explicit-key",
            base_url="https://explicit.com",
            model_name="explicit-model",
            config_file_path=config_file,
        )

        assert config.api_key == "explicit-key"
        assert config.base_url == "https://explicit.com"
        assert config.model_name == "explicit-model"

    def test_api_config_partial_config_file(self, monkeypatch, tmp_path):
        """Test that partial config file merges with env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.openai.com")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_name: file-model")

        config = APIConfig(config_file_path=config_file)

        assert config.api_key == "env-key"
        assert config.base_url == "https://env.openai.com"
        assert config.model_name == "file-model"
