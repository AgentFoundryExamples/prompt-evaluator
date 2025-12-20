"""
Configuration management for the prompt evaluator.

This module handles loading and validating configuration files,
managing provider settings, and storing evaluation parameters.
"""

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
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens to generate")


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""

    name: str = Field(..., description="Name of the evaluation")
    prompts: list[str] = Field(..., description="List of prompt variations to evaluate")
    providers: list[ProviderConfig] = Field(..., description="LLM providers to use")
    test_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Test cases with input variables"
    )
    output_dir: Path = Field(
        default=Path("runs"),
        description="Directory for evaluation outputs"
    )


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
