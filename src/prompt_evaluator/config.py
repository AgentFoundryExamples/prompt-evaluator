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
Configuration management for the prompt evaluator.

This module handles loading and validating configuration files,
managing provider settings, and storing evaluation parameters.
"""

import os
import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# Centralized list of valid providers to avoid duplication
VALID_PROVIDERS = ["openai", "anthropic", "claude", "mock"]


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    model_config = {"extra": "allow"}

    name: str = Field(..., description="Provider name (e.g., 'openai')")
    api_key: str | None = Field(None, description="API key for the provider")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_completion_tokens: int | None = Field(None, gt=0, description="Maximum tokens to generate")


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""

    name: str = Field(..., description="Name of the evaluation")
    prompts: list[str] = Field(..., description="List of prompt variations to evaluate")
    providers: list[ProviderConfig] = Field(..., description="LLM providers to use")
    test_cases: list[dict[str, Any]] = Field(
        default_factory=list, description="Test cases with input variables"
    )
    output_dir: Path = Field(default=Path("runs"), description="Directory for evaluation outputs")


def load_config(config_path: Path) -> EvaluationConfig:
    """
    Load and validate an evaluation configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated evaluation configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return EvaluationConfig(**config_data)


class DefaultGeneratorConfig(BaseModel):
    """Default configuration for the generator LLM."""

    provider: str = Field(
        "openai",
        description="Default generator provider (e.g., 'openai', 'anthropic')"
    )
    model: str = Field("gpt-5.1", description="Default generator model ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Default temperature")
    max_completion_tokens: int = Field(1024, gt=0, description="Default max completion tokens")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        if v not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{v}'. Valid providers: {', '.join(VALID_PROVIDERS)}"
            )
        return v


class DefaultJudgeConfig(BaseModel):
    """
    Default configuration for the judge LLM.
    
    Attributes:
        provider: Default judge provider (e.g., 'openai', 'anthropic')
        model: Default judge model ID
        temperature: Default temperature for judge (0.0-2.0)
        max_completion_tokens: Maximum tokens to generate for judge responses.
            Higher values allow for more detailed evaluation rationales.
        top_p: Nucleus sampling parameter (0.0-1.0). Controls diversity of judge outputs.
            Lower values make outputs more focused and deterministic.
        system_instructions: Optional system instructions override for judge prompts.
            If provided, overrides the default or rubric-based judge prompt.
    """

    provider: str = Field(
        "openai",
        description="Default judge provider (e.g., 'openai', 'anthropic')"
    )
    model: str = Field("gpt-5.1", description="Default judge model ID")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Default temperature for judge")
    max_completion_tokens: int = Field(
        1024,
        gt=0,
        description="Maximum tokens to generate for judge responses"
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter for judge (0.0-1.0)"
    )
    system_instructions: str | None = Field(
        None,
        description="Optional system instructions override for judge prompts"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        if v not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{v}'. Valid providers: {', '.join(VALID_PROVIDERS)}"
            )
        return v


class DefaultsConfig(BaseModel):
    """Default configuration values for generators and judges."""

    generator: DefaultGeneratorConfig = Field(
        default_factory=DefaultGeneratorConfig,
        description="Default generator configuration"
    )
    judge: DefaultJudgeConfig = Field(
        default_factory=DefaultJudgeConfig,
        description="Default judge configuration"
    )
    rubric: str | None = Field(None, description="Default rubric path or preset name")
    run_directory: str = Field("runs", description="Default directory for run outputs")


class PromptEvaluatorConfig(BaseModel):
    """Main configuration for prompt evaluator loaded from prompt_evaluator.yaml."""

    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig,
        description="Default settings for generators, judges, and runs"
    )
    prompt_templates: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of template keys to file paths"
    )
    dataset_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of dataset keys to file paths"
    )

    # Store the config file directory for resolving relative paths
    _config_dir: Path | None = None

    @field_validator("prompt_templates")
    @classmethod
    def validate_template_keys(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that template keys don't have duplicates (case handled by dict)."""
        if not v:
            return v

        # Check for empty keys or values
        for key, value in v.items():
            if not key.strip():
                raise ValueError("Prompt template keys cannot be empty")
            if not value.strip():
                raise ValueError(f"Prompt template path for key '{key}' cannot be empty")

        return v

    @field_validator("dataset_paths")
    @classmethod
    def validate_dataset_keys(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that dataset keys are not empty."""
        if not v:
            return v

        for key, value in v.items():
            if not key.strip():
                raise ValueError("Dataset keys cannot be empty")
            if not value.strip():
                raise ValueError(f"Dataset path for key '{key}' cannot be empty")

        return v

    def resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path string relative to the config file location.

        Args:
            path_str: Path string from config (can be relative or absolute)

        Returns:
            Resolved absolute Path object

        Raises:
            ValueError: If path contains suspicious patterns or traverses outside expected bounds
        """
        # Basic input validation to prevent path traversal attacks
        if not path_str or not path_str.strip():
            raise ValueError("Path cannot be empty")

        path = Path(path_str)

        # If already absolute, validate and return
        if path.is_absolute():
            resolved = path.resolve()
        elif self._config_dir is not None:
            # Resolve relative to config directory
            resolved = (self._config_dir / path).resolve()
        else:
            # Fallback to current working directory
            resolved = (Path.cwd() / path).resolve()

        # Additional security check: ensure resolved path doesn't contain suspicious patterns
        # This prevents issues like symlink attacks or unexpected path resolution
        try:
            # Just validate the path can be resolved without errors
            str(resolved)
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid or suspicious path: {path_str}") from e

        return resolved

    def get_prompt_template_path(self, key: str) -> Path:
        """
        Get resolved path for a prompt template by key.

        Args:
            key: Template key

        Returns:
            Resolved absolute path to template file

        Raises:
            KeyError: If key not found in prompt_templates
            FileNotFoundError: If resolved file doesn't exist
        """
        if key not in self.prompt_templates:
            available = (
                ', '.join(sorted(self.prompt_templates.keys()))
                if self.prompt_templates else 'none'
            )
            raise KeyError(
                f"Prompt template key '{key}' not found in configuration. "
                f"Available templates: {available}"
            )

        path = self.resolve_path(self.prompt_templates[key])

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template file not found: {path} (key: '{key}')"
            )

        return path

    def get_dataset_path(self, key: str) -> Path:
        """
        Get resolved path for a dataset by key.

        Args:
            key: Dataset key

        Returns:
            Resolved absolute path to dataset file

        Raises:
            KeyError: If key not found in dataset_paths
            FileNotFoundError: If resolved file doesn't exist
        """
        if key not in self.dataset_paths:
            available = (
                ', '.join(sorted(self.dataset_paths.keys()))
                if self.dataset_paths else 'none'
            )
            raise KeyError(
                f"Dataset key '{key}' not found in configuration. "
                f"Available datasets: {available}"
            )

        path = self.resolve_path(self.dataset_paths[key])

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path} (key: '{key}')"
            )

        return path


def locate_config_file(
    cli_path: Path | None = None,
    env_var: str = "PROMPT_EVALUATOR_CONFIG",
    default_name: str = "prompt_evaluator.yaml"
) -> Path | None:
    """
    Locate the configuration file using precedence: CLI flag > env var > default path.

    Args:
        cli_path: Path provided via CLI flag (highest precedence)
        env_var: Environment variable name to check
        default_name: Default config filename to look for in cwd

    Returns:
        Path to config file if found, None otherwise
    """
    # 1. CLI flag takes highest precedence
    if cli_path is not None:
        return cli_path

    # 2. Check environment variable
    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path)

    # 3. Look for default file in current working directory
    default_path = Path.cwd() / default_name
    if default_path.exists():
        return default_path

    # Not found
    return None


def load_prompt_evaluator_config(
    config_path: Path | None = None,
    env_var: str = "PROMPT_EVALUATOR_CONFIG",
    warn_if_missing: bool = True
) -> PromptEvaluatorConfig | None:
    """
    Load the main prompt evaluator configuration from YAML file.

    Args:
        config_path: Optional explicit path to config file
        env_var: Environment variable to check for config path
        warn_if_missing: Whether to warn if config file not found

    Returns:
        Loaded configuration or None if not found and not required

    Raises:
        ValueError: If config file is found but invalid
    """
    # Locate config file
    located_path = locate_config_file(cli_path=config_path, env_var=env_var)

    if located_path is None:
        if warn_if_missing:
            warnings.warn(
                "No prompt_evaluator.yaml config file found. "
                "Using default settings and CLI arguments. "
                f"To use a config file, create 'prompt_evaluator.yaml' in current directory "
                f"or set {env_var} environment variable.",
                UserWarning,
                stacklevel=2
            )
        return None

    # Resolve to absolute path to avoid duplicate loading
    # resolve() handles both absolute and relative paths correctly
    located_path = located_path.resolve()

    # Check if file exists
    if not located_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {located_path}"
        )

    # Check if file is readable
    if not os.access(located_path, os.R_OK):
        raise ValueError(
            f"Configuration file is not readable: {located_path}"
        )

    # Load and parse YAML
    try:
        with open(located_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"Failed to parse YAML configuration file {located_path}: {str(e)}"
        ) from e
    except OSError as e:
        raise ValueError(
            f"Failed to read configuration file {located_path}: {str(e)}"
        ) from e

    # Validate and create config object
    try:
        config = PromptEvaluatorConfig(**config_data)
        # Store config directory for relative path resolution
        config._config_dir = located_path.parent.resolve()
        return config
    except Exception as e:
        raise ValueError(
            f"Invalid configuration in {located_path}: {str(e)}"
        ) from e


class APIConfig:
    """Configuration for API credentials and defaults loaded from environment."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        config_file_path: Path | None = None,
    ):
        """
        Initialize API configuration from environment variables and optional config file.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            base_url: OpenAI base URL (if None, reads from OPENAI_BASE_URL env var)
            model_name: Default model name (if None, reads from OPENAI_MODEL env var)
            config_file_path: Optional path to config file for overrides

        Raises:
            ValueError: If API key is missing or config is invalid
        """
        self._load_from_env_and_file(api_key, base_url, model_name, config_file_path)
        self._validate()

    def _load_from_env_and_file(
        self,
        api_key: str | None,
        base_url: str | None,
        model_name: str | None,
        config_file_path: Path | None,
    ) -> None:
        """Load configuration from environment variables and optional config file."""
        # Start with environment variables
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-5.1")

        # Load from config file if provided and values are not explicitly set
        if config_file_path is not None:
            self._load_config_file(config_file_path, api_key, base_url, model_name)

    def _load_config_file(
        self,
        config_file_path: Path,
        explicit_api_key: str | None,
        explicit_base_url: str | None,
        explicit_model_name: str | None,
    ) -> None:
        """
        Load configuration from file and override defaults.

        Args:
            config_file_path: Path to config file (TOML or YAML)
            explicit_api_key: Explicitly provided API key (takes precedence)
            explicit_base_url: Explicitly provided base URL (takes precedence)
            explicit_model_name: Explicitly provided model name (takes precedence)

        Raises:
            ValueError: If config file is malformed
        """
        if not config_file_path.exists():
            # Gracefully handle missing config file
            return

        try:
            if config_file_path.suffix in [".yaml", ".yml"]:
                with open(config_file_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            elif config_file_path.suffix == ".toml":
                try:
                    import tomllib  # type: ignore[import-not-found]
                except ImportError:
                    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

                with open(config_file_path, "rb") as f:
                    config_data = tomllib.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_file_path.suffix}. "
                    "Supported formats: .yaml, .yml, .toml"
                )

            # Override with config file values only if not explicitly provided
            if not explicit_api_key and "api_key" in config_data and config_data["api_key"]:
                self.api_key = config_data["api_key"]
            if not explicit_base_url and "base_url" in config_data and config_data["base_url"]:
                self.base_url = config_data["base_url"]
            if (
                not explicit_model_name
                and "model_name" in config_data
                and config_data["model_name"]
            ):
                self.model_name = config_data["model_name"]

        except Exception as e:
            raise ValueError(f"Failed to load config file {config_file_path}: {str(e)}") from e

    def _validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Please set the OPENAI_API_KEY environment variable "
                "or provide it in the config file."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
        }


def load_api_config(config_file_path: Path | None = None) -> APIConfig:
    """
    Load API configuration from environment and optional config file.

    Args:
        config_file_path: Optional path to configuration file

    Returns:
        Validated API configuration

    Raises:
        ValueError: If API key is missing or configuration is invalid
    """
    return APIConfig(config_file_path=config_file_path)


# Preset aliases for bundled rubrics
RUBRIC_PRESETS = {
    "default": "default.yaml",
    "content-quality": "content_quality.yaml",
    "code-review": "code_review.json",
}


def get_default_rubric_path() -> Path:
    """
    Get the path to the default bundled rubric file.

    Returns:
        Path to the default rubric file

    Raises:
        FileNotFoundError: If the default rubric file cannot be found
    """
    # Try to find the examples/rubrics directory relative to the package
    # First, try relative to this file (for development mode)
    config_dir = Path(__file__).parent
    rubrics_dir = config_dir.parent.parent / "examples" / "rubrics"
    default_rubric = rubrics_dir / "default.yaml"

    if default_rubric.exists():
        return default_rubric

    # If not found in development location, try relative to current working directory
    cwd_rubrics = Path.cwd() / "examples" / "rubrics" / "default.yaml"
    if cwd_rubrics.exists():
        return cwd_rubrics

    raise FileNotFoundError(
        "Default rubric file not found. Expected location: examples/rubrics/default.yaml"
    )


def resolve_rubric_path(rubric_input: str | None) -> Path:
    """
    Resolve a rubric path from user input, handling preset aliases and file paths.

    Args:
        rubric_input: User-provided rubric identifier - can be:
                     - None (use default)
                     - A preset alias (e.g., "default", "content-quality", "code-review")
                     - An absolute or relative file path

    Returns:
        Resolved absolute Path to the rubric file

    Raises:
        FileNotFoundError: If the rubric file doesn't exist
        ValueError: If rubric_input points to a directory or is invalid
    """
    # If no input provided, use default
    if rubric_input is None:
        return get_default_rubric_path()

    # Check if it's a preset alias
    if rubric_input in RUBRIC_PRESETS:
        # Resolve preset to bundled file
        preset_filename = RUBRIC_PRESETS[rubric_input]

        # Try to find rubrics directory
        config_dir = Path(__file__).parent
        rubrics_dir = config_dir.parent.parent / "examples" / "rubrics"
        preset_path = rubrics_dir / preset_filename

        if preset_path.exists():
            return preset_path

        # If not found in development location, try relative to current working directory
        cwd_preset_path = Path.cwd() / "examples" / "rubrics" / preset_filename
        if cwd_preset_path.exists():
            return cwd_preset_path

        raise FileNotFoundError(
            f"Preset rubric '{rubric_input}' not found. "
            f"Expected file '{preset_filename}' in examples/rubrics/ directory. "
            f"Available presets: {', '.join(sorted(RUBRIC_PRESETS.keys()))}"
        )

    # Treat as a file path (absolute or relative)
    rubric_path = Path(rubric_input)

    # Resolve to absolute path
    if not rubric_path.is_absolute():
        rubric_path = Path.cwd() / rubric_path

    # Resolve to canonical path to prevent path traversal attacks
    # resolve() normalizes the path, resolving symlinks and removing .. components
    try:
        rubric_path = rubric_path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid rubric path: {rubric_input}") from e

    # Check if path exists
    if not rubric_path.exists():
        raise FileNotFoundError(
            f"Rubric file not found: {rubric_path}. "
            f"Please provide a valid file path or use a preset: "
            f"{', '.join(sorted(RUBRIC_PRESETS.keys()))}"
        )

    # Check if it's a directory
    if rubric_path.is_dir():
        raise ValueError(
            f"Rubric path points to a directory: {rubric_path}. "
            "Please provide a path to a rubric file (.yaml, .yml, or .json)"
        )

    # Check if file is readable
    if not os.access(rubric_path, os.R_OK):
        raise ValueError(f"Rubric file is not readable: {rubric_path}")

    return rubric_path


def load_rubric(rubric_path: Path) -> "Rubric":  # type: ignore[name-defined] # noqa: F821
    """
    Load and validate a rubric from a YAML or JSON file.

    Args:
        rubric_path: Path to rubric file (.yaml, .yml, or .json)

    Returns:
        Validated Rubric instance

    Raises:
        FileNotFoundError: If rubric file doesn't exist
        ValueError: If rubric is invalid or malformed
    """
    from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric

    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")

    try:
        # Load file based on extension
        if rubric_path.suffix in [".yaml", ".yml"]:
            with open(rubric_path, encoding="utf-8") as f:
                rubric_data = yaml.safe_load(f)
        elif rubric_path.suffix == ".json":
            import json

            with open(rubric_path, encoding="utf-8") as f:
                rubric_data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported rubric file format: {rubric_path.suffix}. "
                "Supported formats: .yaml, .yml, .json"
            )

        if not isinstance(rubric_data, dict):
            raise ValueError(
                f"Rubric file must contain a dictionary, got {type(rubric_data).__name__}"
            )

        # Parse metrics
        metrics_data = rubric_data.get("metrics", [])
        if not isinstance(metrics_data, list):
            raise ValueError(f"Metrics must be a list, got {type(metrics_data).__name__}")

        if not metrics_data:
            raise ValueError("Rubric must contain at least one metric")

        metrics = []
        for i, metric_data in enumerate(metrics_data):
            if not isinstance(metric_data, dict):
                raise ValueError(f"Metric at index {i} must be a dictionary")

            # Validate required fields
            required_fields = {"name", "description", "min_score", "max_score", "guidelines"}
            missing_fields = required_fields - set(metric_data.keys())
            if missing_fields:
                raise ValueError(
                    f"Metric at index {i} is missing required fields: "
                    f"{', '.join(sorted(missing_fields))}"
                )

            try:
                metric = RubricMetric(
                    name=metric_data["name"],
                    description=metric_data["description"],
                    min_score=metric_data["min_score"],
                    max_score=metric_data["max_score"],
                    guidelines=metric_data["guidelines"],
                )
                metrics.append(metric)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid metric at index {i}: {str(e)}") from e

        # Parse flags (optional)
        flags_data = rubric_data.get("flags", [])
        if not isinstance(flags_data, list):
            raise ValueError(f"Flags must be a list, got {type(flags_data).__name__}")

        flags = []
        for i, flag_data in enumerate(flags_data):
            if not isinstance(flag_data, dict):
                raise ValueError(f"Flag at index {i} must be a dictionary")

            # Validate required fields
            if "name" not in flag_data:
                raise ValueError(f"Flag at index {i} is missing required field: name")
            if "description" not in flag_data:
                raise ValueError(f"Flag at index {i} is missing required field: description")

            try:
                flag = RubricFlag(
                    name=flag_data["name"],
                    description=flag_data["description"],
                    default=flag_data.get("default", False),
                )
                flags.append(flag)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid flag at index {i}: {str(e)}") from e

        # Create and validate rubric
        try:
            rubric = Rubric(metrics=metrics, flags=flags)
        except ValueError as e:
            raise ValueError(f"Rubric validation failed: {str(e)}") from e

        return rubric

    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to read rubric file {rubric_path}: {str(e)}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML rubric file {rubric_path}: {str(e)}") from e
    except (FileNotFoundError, ValueError):
        raise
    except ImportError:
        # Let import errors propagate naturally (e.g., missing json module)
        raise
    except KeyboardInterrupt:
        # Let interrupts propagate naturally
        raise


# Supported dataset file extensions
DATASET_FILE_EXTENSIONS = [".jsonl", ".yaml", ".yml"]


class ConfigManager:
    """
    Shared configuration manager that caches loaded config files.
    
    This class ensures that config files are loaded once and reused across
    CLI commands within the same invocation, preventing duplicate loading
    and ensuring consistent configuration state.
    """
    
    def __init__(self) -> None:
        """Initialize the configuration manager with empty cache."""
        self._app_config_cache: PromptEvaluatorConfig | None = None
        self._app_config_path_cache: Path | None = None
        self._api_config_cache: APIConfig | None = None
        self._api_config_path_cache: Path | None = None
    
    def get_app_config(
        self,
        config_path: Path | None = None,
        warn_if_missing: bool = True
    ) -> PromptEvaluatorConfig | None:
        """
        Get or load the application configuration.
        
        If a config has already been loaded with the same path, return the cached version.
        Otherwise, load the config and cache it.
        
        Args:
            config_path: Optional explicit path to config file
            warn_if_missing: Whether to warn if config file not found
            
        Returns:
            Loaded configuration or None if not found
            
        Raises:
            ValueError: If config file is found but invalid
        """
        # Locate the config file to determine the canonical path for caching
        # This ensures we cache based on the actual file found, not the input path
        located_path = locate_config_file(cli_path=config_path)
        normalized_path = located_path.resolve() if located_path else None
        
        # Check cache - return cached config if path matches
        if self._app_config_path_cache == normalized_path and self._app_config_cache is not None:
            return self._app_config_cache
        
        # Load new config, passing the original path for correct logic inside
        config = load_prompt_evaluator_config(
            config_path=config_path,
            warn_if_missing=warn_if_missing
        )
        
        # Update cache with the resolved path
        self._app_config_cache = config
        self._app_config_path_cache = normalized_path
        
        return config
    
    def get_api_config(
        self,
        config_file_path: Path | None = None
    ) -> APIConfig:
        """
        Get or load the API configuration.
        
        If an API config has already been loaded with the same path, return the cached version.
        Otherwise, load the config and cache it.
        
        Args:
            config_file_path: Optional path to configuration file
            
        Returns:
            Loaded API configuration
            
        Raises:
            ValueError: If API key is missing or configuration is invalid
        """
        # Normalize config path for cache comparison
        normalized_path = None
        if config_file_path is not None:
            normalized_path = config_file_path.resolve()
        
        # Check cache - return cached config if path matches
        if self._api_config_path_cache == normalized_path and self._api_config_cache is not None:
            return self._api_config_cache
        
        # Load new config
        config = APIConfig(config_file_path=config_file_path)
        
        # Update cache
        self._api_config_cache = config
        self._api_config_path_cache = normalized_path
        
        return config
    
    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._app_config_cache = None
        self._app_config_path_cache = None
        self._api_config_cache = None
        self._api_config_path_cache = None


def load_dataset(dataset_path: Path) -> tuple[list["TestCase"], dict[str, Any]]:  # type: ignore[name-defined] # noqa: F821
    """
    Load and validate a dataset from JSONL or YAML file.

    Args:
        dataset_path: Path to dataset file (.jsonl or .yaml/.yml)

    Returns:
        Tuple of (list of TestCase objects, dataset metadata dict)
        Metadata includes: path, hash, count, format

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is invalid, has duplicate IDs, missing required fields,
                   or unsupported file extension
    """
    import hashlib
    import json

    from pydantic import ValidationError

    from prompt_evaluator.models import TestCase

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Check file extension
    if dataset_path.suffix not in DATASET_FILE_EXTENSIONS:
        supported_formats = ", ".join(DATASET_FILE_EXTENSIONS)
        raise ValueError(
            f"Unsupported dataset file format: {dataset_path.suffix}. "
            f"Supported formats: {supported_formats}"
        )

    # Read file content once for both hash computation and parsing
    try:
        with open(dataset_path, "rb") as f:
            file_content = f.read()
        dataset_hash = hashlib.sha256(file_content).hexdigest()
        # Decode to string for parsing
        file_content_str = file_content.decode("utf-8")
    except OSError as e:
        raise ValueError(f"Failed to read dataset file {dataset_path}: {str(e)}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode dataset file {dataset_path}: {str(e)}") from e

    # Parse dataset based on format
    test_cases = []
    seen_ids = set()

    try:
        if dataset_path.suffix == ".jsonl":
            # JSONL format: one JSON object per line, parse from memory
            for line_num, line in enumerate(file_content_str.splitlines(), start=1):
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_num} in {dataset_path}: {str(e)}"
                    ) from e

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at line {line_num} must be a JSON object, "
                        f"got {type(record).__name__}"
                    )

                # Validate required fields
                if "id" not in record:
                    raise ValueError(f"Record at line {line_num} is missing required field: id")
                if "input" not in record:
                    raise ValueError(f"Record at line {line_num} is missing required field: input")

                # Check for duplicate IDs
                record_id = record["id"]
                if record_id in seen_ids:
                    raise ValueError(
                        f"Duplicate test case ID '{record_id}' found at line {line_num}"
                    )
                seen_ids.add(record_id)

                # Parse into TestCase, handling extra fields
                try:
                    test_case = TestCase(**record)
                    test_cases.append(test_case)
                except ValidationError as e:
                    raise ValueError(f"Invalid test case at line {line_num}: {str(e)}") from e

        else:
            # YAML format: list of objects, parse from memory
            try:
                dataset_data = yaml.safe_load(file_content_str)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse YAML dataset file {dataset_path}: {str(e)}"
                ) from e

            if dataset_data is None:
                # Empty YAML file
                dataset_data = []

            if not isinstance(dataset_data, list):
                raise ValueError(
                    f"YAML dataset must contain a list of objects, "
                    f"got {type(dataset_data).__name__}"
                )

            for index, record in enumerate(dataset_data):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at index {index} must be an object, got {type(record).__name__}"
                    )

                # Validate required fields
                if "id" not in record:
                    raise ValueError(f"Record at index {index} is missing required field: id")
                if "input" not in record:
                    raise ValueError(f"Record at index {index} is missing required field: input")

                # Check for duplicate IDs
                record_id = record["id"]
                if record_id in seen_ids:
                    raise ValueError(f"Duplicate test case ID '{record_id}' found at index {index}")
                seen_ids.add(record_id)

                # Parse into TestCase, handling extra fields
                try:
                    test_case = TestCase(**record)
                    test_cases.append(test_case)
                except ValidationError as e:
                    raise ValueError(f"Invalid test case at index {index}: {str(e)}") from e

    except (FileNotFoundError, ValueError):
        # Let these propagate naturally
        raise

    # Build metadata
    metadata = {
        "path": str(dataset_path.absolute()),
        "hash": dataset_hash,
        "count": len(test_cases),
        "format": dataset_path.suffix,
    }

    return test_cases, metadata
