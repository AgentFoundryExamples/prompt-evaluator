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

from prompt_evaluator.config import (
    APIConfig,
    PromptEvaluatorConfig,
    load_api_config,
    load_prompt_evaluator_config,
    locate_config_file,
)
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


class TestPromptEvaluatorConfig:
    """Tests for PromptEvaluatorConfig and related models."""

    def test_default_values(self):
        """Test that PromptEvaluatorConfig has sensible defaults."""
        config = PromptEvaluatorConfig()

        assert config.defaults.generator.provider == "openai"
        assert config.defaults.generator.model == "gpt-5.1"
        assert config.defaults.generator.temperature == 0.7
        assert config.defaults.generator.max_completion_tokens == 1024

        assert config.defaults.judge.provider == "openai"
        assert config.defaults.judge.model == "gpt-5.1"
        assert config.defaults.judge.temperature == 0.0
        assert config.defaults.judge.max_completion_tokens == 1024
        assert config.defaults.judge.top_p is None
        assert config.defaults.judge.system_instructions is None

        assert config.defaults.rubric is None
        assert config.defaults.run_directory == "runs"

        assert config.prompt_templates == {}
        assert config.dataset_paths == {}

    def test_custom_defaults(self):
        """Test that custom default values are accepted."""
        config_data = {
            "defaults": {
                "generator": {
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "temperature": 0.5,
                    "max_completion_tokens": 2048
                },
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_completion_tokens": 4096,
                    "top_p": 0.9,
                    "system_instructions": "Custom judge instructions"
                },
                "rubric": "content-quality",
                "run_directory": "custom_runs"
            }
        }

        config = PromptEvaluatorConfig(**config_data)

        assert config.defaults.generator.provider == "anthropic"
        assert config.defaults.generator.model == "claude-3-opus"
        assert config.defaults.generator.temperature == 0.5
        assert config.defaults.generator.max_completion_tokens == 2048

        assert config.defaults.judge.provider == "openai"
        assert config.defaults.judge.model == "gpt-4"
        assert config.defaults.judge.temperature == 0.1
        assert config.defaults.judge.max_completion_tokens == 4096
        assert config.defaults.judge.top_p == 0.9
        assert config.defaults.judge.system_instructions == "Custom judge instructions"

        assert config.defaults.rubric == "content-quality"
        assert config.defaults.run_directory == "custom_runs"

    def test_prompt_templates_mapping(self):
        """Test that prompt templates can be defined."""
        config_data = {
            "prompt_templates": {
                "checkout_compiler": "prompts/checkout.txt",
                "default_system": "prompts/system.txt"
            }
        }

        config = PromptEvaluatorConfig(**config_data)

        assert len(config.prompt_templates) == 2
        assert config.prompt_templates["checkout_compiler"] == "prompts/checkout.txt"
        assert config.prompt_templates["default_system"] == "prompts/system.txt"

    def test_dataset_paths_mapping(self):
        """Test that dataset paths can be defined."""
        config_data = {
            "dataset_paths": {
                "sample": "datasets/sample.yaml",
                "production": "datasets/prod.jsonl"
            }
        }

        config = PromptEvaluatorConfig(**config_data)

        assert len(config.dataset_paths) == 2
        assert config.dataset_paths["sample"] == "datasets/sample.yaml"
        assert config.dataset_paths["production"] == "datasets/prod.jsonl"

    def test_empty_template_key_validation(self):
        """Test that empty template keys are rejected."""
        config_data = {
            "prompt_templates": {
                "": "some/path.txt"
            }
        }

        with pytest.raises(ValueError, match="Prompt template keys cannot be empty"):
            PromptEvaluatorConfig(**config_data)

    def test_empty_template_value_validation(self):
        """Test that empty template values are rejected."""
        config_data = {
            "prompt_templates": {
                "test": ""
            }
        }

        with pytest.raises(ValueError, match="Prompt template path for key 'test' cannot be empty"):
            PromptEvaluatorConfig(**config_data)

    def test_empty_dataset_key_validation(self):
        """Test that empty dataset keys are rejected."""
        config_data = {
            "dataset_paths": {
                "": "some/path.yaml"
            }
        }

        with pytest.raises(ValueError, match="Dataset keys cannot be empty"):
            PromptEvaluatorConfig(**config_data)

    def test_empty_dataset_value_validation(self):
        """Test that empty dataset values are rejected."""
        config_data = {
            "dataset_paths": {
                "test": ""
            }
        }

        with pytest.raises(ValueError, match="Dataset path for key 'test' cannot be empty"):
            PromptEvaluatorConfig(**config_data)

    def test_invalid_generator_provider(self):
        """Test that invalid generator provider is rejected."""
        config_data = {
            "defaults": {
                "generator": {
                    "provider": "invalid_provider",
                    "model": "some-model"
                }
            }
        }

        with pytest.raises(ValueError, match="Invalid provider 'invalid_provider'"):
            PromptEvaluatorConfig(**config_data)

    def test_invalid_judge_provider(self):
        """Test that invalid judge provider is rejected."""
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "invalid_provider",
                    "model": "some-model"
                }
            }
        }

        with pytest.raises(ValueError, match="Invalid provider 'invalid_provider'"):
            PromptEvaluatorConfig(**config_data)

    def test_resolve_absolute_path(self, tmp_path):
        """Test that absolute paths are resolved correctly."""
        config = PromptEvaluatorConfig()
        config._config_dir = tmp_path

        abs_path = tmp_path / "test.txt"
        resolved = config.resolve_path(str(abs_path))

        assert resolved == abs_path

    def test_resolve_relative_path_with_config_dir(self, tmp_path):
        """Test that relative paths are resolved relative to config dir."""
        config = PromptEvaluatorConfig()
        config._config_dir = tmp_path

        resolved = config.resolve_path("subdir/test.txt")

        assert resolved == (tmp_path / "subdir/test.txt").resolve()

    def test_resolve_relative_path_without_config_dir(self, monkeypatch, tmp_path):
        """Test that relative paths fall back to cwd when config dir not set."""
        monkeypatch.chdir(tmp_path)

        config = PromptEvaluatorConfig()
        # Don't set config._config_dir

        resolved = config.resolve_path("test.txt")

        assert resolved == (tmp_path / "test.txt").resolve()

    def test_get_prompt_template_path_success(self, tmp_path):
        """Test getting prompt template path successfully."""
        # Create a test file
        template_file = tmp_path / "template.txt"
        template_file.write_text("test template")

        config_data = {
            "prompt_templates": {
                "test_template": str(template_file)
            }
        }

        config = PromptEvaluatorConfig(**config_data)
        config._config_dir = tmp_path

        path = config.get_prompt_template_path("test_template")

        assert path == template_file
        assert path.exists()

    def test_get_prompt_template_path_key_not_found(self):
        """Test that missing template key raises KeyError."""
        config_data = {
            "prompt_templates": {
                "existing": "some/path.txt"
            }
        }

        config = PromptEvaluatorConfig(**config_data)

        with pytest.raises(KeyError, match="Prompt template key 'missing' not found"):
            config.get_prompt_template_path("missing")

    def test_get_prompt_template_path_file_not_found(self, tmp_path):
        """Test that missing template file raises FileNotFoundError."""
        config_data = {
            "prompt_templates": {
                "test": "nonexistent.txt"
            }
        }

        config = PromptEvaluatorConfig(**config_data)
        config._config_dir = tmp_path

        with pytest.raises(FileNotFoundError, match="Prompt template file not found"):
            config.get_prompt_template_path("test")

    def test_get_dataset_path_success(self, tmp_path):
        """Test getting dataset path successfully."""
        # Create a test file
        dataset_file = tmp_path / "dataset.yaml"
        dataset_file.write_text("test: data")

        config_data = {
            "dataset_paths": {
                "test_dataset": str(dataset_file)
            }
        }

        config = PromptEvaluatorConfig(**config_data)
        config._config_dir = tmp_path

        path = config.get_dataset_path("test_dataset")

        assert path == dataset_file
        assert path.exists()

    def test_get_dataset_path_key_not_found(self):
        """Test that missing dataset key raises KeyError."""
        config_data = {
            "dataset_paths": {
                "existing": "some/path.yaml"
            }
        }

        config = PromptEvaluatorConfig(**config_data)

        with pytest.raises(KeyError, match="Dataset key 'missing' not found"):
            config.get_dataset_path("missing")

    def test_get_dataset_path_file_not_found(self, tmp_path):
        """Test that missing dataset file raises FileNotFoundError."""
        config_data = {
            "dataset_paths": {
                "test": "nonexistent.yaml"
            }
        }

        config = PromptEvaluatorConfig(**config_data)
        config._config_dir = tmp_path

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            config.get_dataset_path("test")


class TestLocateConfigFile:
    """Tests for locate_config_file function."""

    def test_cli_path_takes_precedence(self, tmp_path, monkeypatch):
        """Test that CLI path has highest precedence."""
        cli_path = tmp_path / "cli_config.yaml"
        env_path = tmp_path / "env_config.yaml"
        default_path = tmp_path / "prompt_evaluator.yaml"

        # Create all files
        cli_path.write_text("cli: config")
        env_path.write_text("env: config")
        default_path.write_text("default: config")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(env_path))

        result = locate_config_file(cli_path=cli_path)

        assert result == cli_path

    def test_env_var_second_precedence(self, tmp_path, monkeypatch):
        """Test that env var is checked when CLI path not provided."""
        env_path = tmp_path / "env_config.yaml"
        default_path = tmp_path / "prompt_evaluator.yaml"

        env_path.write_text("env: config")
        default_path.write_text("default: config")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(env_path))

        result = locate_config_file()

        assert result == env_path

    def test_default_path_lowest_precedence(self, tmp_path, monkeypatch):
        """Test that default path is used as fallback."""
        default_path = tmp_path / "prompt_evaluator.yaml"
        default_path.write_text("default: config")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)

        result = locate_config_file()

        assert result == default_path

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        """Test that None is returned when no config found."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)

        result = locate_config_file()

        assert result is None


class TestLoadPromptEvaluatorConfig:
    """Tests for load_prompt_evaluator_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: openai
    model: gpt-4
    temperature: 0.8
    max_completion_tokens: 2048
  judge:
    provider: anthropic
    model: claude-3
    temperature: 0.0
  rubric: default
  run_directory: custom_runs

prompt_templates:
  test_template: prompts/test.txt

dataset_paths:
  test_dataset: datasets/test.yaml
""")

        config = load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)

        assert config is not None
        assert config.defaults.generator.provider == "openai"
        assert config.defaults.generator.model == "gpt-4"
        assert config.defaults.generator.temperature == 0.8
        assert config.defaults.judge.provider == "anthropic"
        assert config.defaults.rubric == "default"
        assert config.prompt_templates["test_template"] == "prompts/test.txt"
        assert config.dataset_paths["test_dataset"] == "datasets/test.yaml"

    def test_load_minimal_config(self, tmp_path):
        """Test loading a minimal configuration with defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# Minimal config\n")

        config = load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)

        assert config is not None
        # Should have all defaults
        assert config.defaults.generator.provider == "openai"
        assert config.defaults.generator.model == "gpt-5.1"
        assert config.prompt_templates == {}

    def test_load_config_with_relative_paths(self, tmp_path):
        """Test that config directory is stored for path resolution."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
prompt_templates:
  rel_template: relative/path.txt
""")

        config = load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)

        assert config is not None
        assert config._config_dir == tmp_path.resolve()

    def test_load_missing_config_with_warning(self, tmp_path, monkeypatch):
        """Test that missing config returns None with warning when not explicitly provided."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)

        # No config file exists, and we're not providing an explicit path
        with pytest.warns(UserWarning, match="No prompt_evaluator.yaml config file found"):
            config = load_prompt_evaluator_config(warn_if_missing=True)

        assert config is None

    def test_load_explicit_missing_config_raises(self, tmp_path):
        """Test that explicitly provided missing config raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_prompt_evaluator_config(config_path=nonexistent, warn_if_missing=True)

    def test_load_missing_config_without_warning(self, tmp_path, monkeypatch):
        """Test that missing config returns None without warning when disabled."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)

        # Should not raise any warnings
        config = load_prompt_evaluator_config(warn_if_missing=False)

        assert config is None

    def test_load_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: [unclosed")

        with pytest.raises(ValueError, match="Failed to parse YAML configuration"):
            load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)

    def test_load_invalid_schema(self, tmp_path):
        """Test that invalid config schema raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: invalid_provider
    model: test
""")

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)

    def test_locate_via_env_var(self, tmp_path, monkeypatch):
        """Test that config is located via environment variable."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("defaults: {}")

        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(config_file))

        config = load_prompt_evaluator_config(warn_if_missing=False)

        assert config is not None
        assert config._config_dir == tmp_path.resolve()

    def test_locate_default_file(self, tmp_path, monkeypatch):
        """Test that default config file is found in cwd."""
        config_file = tmp_path / "prompt_evaluator.yaml"
        config_file.write_text("defaults: {}")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)

        config = load_prompt_evaluator_config(warn_if_missing=False)

        assert config is not None
        assert config._config_dir == tmp_path.resolve()

    def test_judge_max_completion_tokens_validation(self):
        """Test that judge max_completion_tokens is validated."""
        # Valid max_completion_tokens
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "max_completion_tokens": 4096
                }
            }
        }
        config = PromptEvaluatorConfig(**config_data)
        assert config.defaults.judge.max_completion_tokens == 4096

        # Zero max_completion_tokens should fail
        config_data_zero = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "max_completion_tokens": 0
                }
            }
        }
        with pytest.raises(ValueError, match="greater than 0"):
            PromptEvaluatorConfig(**config_data_zero)

        # Negative max_completion_tokens should fail
        config_data_neg = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "max_completion_tokens": -100
                }
            }
        }
        with pytest.raises(ValueError, match="greater than 0"):
            PromptEvaluatorConfig(**config_data_neg)

    def test_judge_top_p_validation(self):
        """Test that judge top_p is validated."""
        # Valid top_p
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "top_p": 0.9
                }
            }
        }
        config = PromptEvaluatorConfig(**config_data)
        assert config.defaults.judge.top_p == 0.9

        # Boundary values should work
        config_data_min = {
            "defaults": {"judge": {"provider": "openai", "model": "gpt-4", "top_p": 0.0}}
        }
        config_min = PromptEvaluatorConfig(**config_data_min)
        assert config_min.defaults.judge.top_p == 0.0

        config_data_max = {
            "defaults": {"judge": {"provider": "openai", "model": "gpt-4", "top_p": 1.0}}
        }
        config_max = PromptEvaluatorConfig(**config_data_max)
        assert config_max.defaults.judge.top_p == 1.0

        # top_p > 1.0 should fail
        config_data_high = {
            "defaults": {"judge": {"provider": "openai", "model": "gpt-4", "top_p": 1.5}}
        }
        with pytest.raises(ValueError, match="less than or equal to 1"):
            PromptEvaluatorConfig(**config_data_high)

        # Negative top_p should fail
        config_data_neg = {
            "defaults": {"judge": {"provider": "openai", "model": "gpt-4", "top_p": -0.1}}
        }
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            PromptEvaluatorConfig(**config_data_neg)

    def test_judge_system_instructions(self):
        """Test that judge system_instructions is accepted."""
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "system_instructions": "You are a helpful evaluator. Score carefully."
                }
            }
        }
        config = PromptEvaluatorConfig(**config_data)
        assert config.defaults.judge.system_instructions == "You are a helpful evaluator. Score carefully."

    def test_judge_config_backward_compatibility(self):
        """Test that configs without new judge fields still work."""
        # Old-style config without max_completion_tokens, top_p, or system_instructions
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0
                }
            }
        }
        config = PromptEvaluatorConfig(**config_data)
        
        # Should use default values
        assert config.defaults.judge.max_completion_tokens == 1024
        assert config.defaults.judge.top_p is None
        assert config.defaults.judge.system_instructions is None
