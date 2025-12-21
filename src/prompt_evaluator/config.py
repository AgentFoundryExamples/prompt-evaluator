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
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


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

    # Read file content for hash computation
    try:
        with open(dataset_path, "rb") as f:
            file_content = f.read()
        dataset_hash = hashlib.sha256(file_content).hexdigest()
    except OSError as e:
        raise ValueError(f"Failed to read dataset file {dataset_path}: {str(e)}") from e

    # Parse dataset based on format
    test_cases = []
    seen_ids = set()

    try:
        if dataset_path.suffix == ".jsonl":
            # JSONL format: one JSON object per line
            with open(dataset_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
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
                        raise ValueError(
                            f"Record at line {line_num} is missing required field: input"
                        )

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
                    except Exception as e:
                        raise ValueError(
                            f"Invalid test case at line {line_num}: {str(e)}"
                        ) from e

        else:
            # YAML format: list of objects
            with open(dataset_path, encoding="utf-8") as f:
                dataset_data = yaml.safe_load(f)

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
                    raise ValueError(
                        f"Duplicate test case ID '{record_id}' found at index {index}"
                    )
                seen_ids.add(record_id)

                # Parse into TestCase, handling extra fields
                try:
                    test_case = TestCase(**record)
                    test_cases.append(test_case)
                except Exception as e:
                    raise ValueError(f"Invalid test case at index {index}: {str(e)}") from e

    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to read dataset file {dataset_path}: {str(e)}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML dataset file {dataset_path}: {str(e)}") from e

    # Build metadata
    metadata = {
        "path": str(dataset_path.absolute()),
        "hash": dataset_hash,
        "count": len(test_cases),
        "format": dataset_path.suffix,
    }

    return test_cases, metadata
