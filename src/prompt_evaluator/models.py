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
Data models for prompt evaluation.

This module defines the core data structures used throughout the
evaluation process, including prompts, responses, and results.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class PromptTemplate(BaseModel):
    """A prompt template with variable substitution."""

    template: str = Field(..., description="Prompt template with {variable} placeholders")
    variables: dict[str, str] = Field(
        default_factory=dict, description="Variable names and descriptions"
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """
        Validate template format strings to prevent potential injection attacks.

        Only allows simple variable placeholders like {variable_name}.
        Disallows format specs, attribute access, and indexing.
        """
        # Check for dangerous format string patterns
        dangerous_patterns = [
            r"\{[^}]*\.[^}]+\}",  # Attribute access: {obj.attr}
            r"\{[^}]*\[[^}]+\]\}",  # Indexing: {obj[0]}
            r"\{[^}]*![^}]+\}",  # Conversion: {var!r}
            r"\{[^}]+:[^}]+\}",  # Format spec: {var:03d} (requires both var and spec)
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v):
                raise ValueError(
                    "Template contains potentially unsafe format string pattern. "
                    "Only simple variable names like {variable} are allowed."
                )

        return v

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with provided variable values.

        The template validation ensures only simple {variable} placeholders
        are used, preventing format string injection attacks.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If required template variable is not provided
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Missing required template variable: {e}. "
                f"Available variables: {list(kwargs.keys())}"
            ) from e


class EvaluationRequest(BaseModel):
    """A request to evaluate a prompt with a specific provider."""

    prompt: str = Field(..., description="The prompt to evaluate")
    provider: str = Field(..., description="Provider identifier")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    """Response from an LLM provider evaluation."""

    request: EvaluationRequest = Field(..., description="Original request")
    response_text: str = Field(..., description="Generated text response")
    tokens_used: int | None = Field(None, description="Number of tokens used")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = Field(None, description="Error message if request failed")


class EvaluationResult(BaseModel):
    """Complete results from an evaluation run."""

    run_id: str = Field(..., description="Unique identifier for this run")
    config_name: str = Field(..., description="Name of the evaluation configuration")
    responses: list[EvaluationResponse] = Field(
        default_factory=list, description="All responses from this evaluation"
    )
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = Field(None)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total duration of the evaluation run."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class GeneratorConfig:
    """Configuration for LLM generation parameters."""

    model_name: str = "gpt-5.1"
    temperature: float = 0.7
    max_completion_tokens: int = 1024
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.max_completion_tokens, int):
            raise ValueError(
                f"max_completion_tokens must be an integer, "
                f"got {type(self.max_completion_tokens).__name__}"
            )
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_completion_tokens <= 0:
            raise ValueError(
                f"max_completion_tokens must be positive, got {self.max_completion_tokens}"
            )


@dataclass
class PromptRun:
    """Metadata for a single prompt evaluation run."""

    id: str
    timestamp: datetime
    system_prompt_path: Path
    input_path: Path
    model_config: GeneratorConfig
    raw_output_path: Path

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PromptRun to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "system_prompt_path": str(self.system_prompt_path),
            "input_path": str(self.input_path),
            "model_config": asdict(self.model_config),
            "raw_output_path": str(self.raw_output_path),
        }


@dataclass
class JudgeConfig:
    """Configuration for LLM judge parameters."""

    model_name: str = "gpt-5.1"
    temperature: float = 0.0
    max_completion_tokens: int = 512
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.max_completion_tokens, int):
            raise ValueError(
                f"max_completion_tokens must be an integer, "
                f"got {type(self.max_completion_tokens).__name__}"
            )
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_completion_tokens <= 0:
            raise ValueError(
                f"max_completion_tokens must be positive, got {self.max_completion_tokens}"
            )


# Default judge system prompt for semantic fidelity scoring
DEFAULT_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the semantic \
fidelity of AI-generated responses.

Your task is to evaluate how well a generated output preserves the semantic meaning \
and intent of the original input text.

Scoring scale (1-5):
- 1: Completely unfaithful - Output contradicts or has no relation to the input
- 2: Mostly unfaithful - Major semantic deviations or omissions
- 3: Partially faithful - Some key information preserved but with notable gaps
- 4: Mostly faithful - Minor deviations but core semantics preserved
- 5: Completely faithful - Perfect preservation of semantic meaning and intent

You must respond with a valid JSON object containing exactly two fields:
{
  "semantic_fidelity": <number between 1 and 5>,
  "rationale": "<string explaining your score in 1-3 sentences>"
}

Do not include any text before or after the JSON object."""


@dataclass
class Sample:
    """
    A single evaluation sample with input, output, and judge scoring.

    Attributes:
        sample_id: Unique identifier for this sample
        input_text: Original input text
        generator_output: Text generated by the model
        judge_score: Semantic fidelity score (1-5) from judge, or None if not yet scored
        judge_rationale: Explanation for the judge's score, or None if not yet scored
        judge_raw_response: Raw response from judge model for debugging
        status: Status of evaluation ("pending", "completed", "judge_error")
        task_description: Optional description of the task for context
    """

    sample_id: str
    input_text: str
    generator_output: str
    judge_score: float | None = None
    judge_rationale: str | None = None
    judge_raw_response: str | None = None
    status: str = "pending"
    task_description: str | None = None

    def __post_init__(self) -> None:
        """Validate sample fields."""
        if self.status not in ("pending", "completed", "judge_error"):
            raise ValueError(
                f"status must be 'pending', 'completed', or 'judge_error', got '{self.status}'"
            )
        if self.judge_score is not None and (self.judge_score < 1.0 or self.judge_score > 5.0):
            raise ValueError(f"judge_score must be between 1.0 and 5.0, got {self.judge_score}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Sample to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return asdict(self)


@dataclass
class SingleEvaluationRun:
    """
    Complete evaluation run with metadata and samples.

    Attributes:
        run_id: Unique identifier for this evaluation run
        timestamp: When the evaluation was started
        num_samples: Total number of samples in this run
        generator_config: Configuration for the generator model
        judge_config: Configuration for the judge model
        samples: List of Sample objects with results
    """

    run_id: str
    timestamp: datetime
    num_samples: int
    generator_config: GeneratorConfig
    judge_config: JudgeConfig
    samples: list[Sample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SingleEvaluationRun to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "num_samples": self.num_samples,
            "generator_config": asdict(self.generator_config),
            "judge_config": asdict(self.judge_config),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def load_judge_prompt(prompt_path: Path | None = None) -> str:
    """
    Load judge system prompt from file or return default.

    Args:
        prompt_path: Optional path to custom judge prompt file.
                    If None, returns DEFAULT_JUDGE_SYSTEM_PROMPT.

    Returns:
        Judge system prompt text

    Raises:
        FileNotFoundError: If prompt_path is specified but file doesn't exist
        ValueError: If prompt file is empty or unreadable
    """
    if prompt_path is None:
        return DEFAULT_JUDGE_SYSTEM_PROMPT

    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt file not found: {prompt_path}")

    try:
        with open(prompt_path, encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise ValueError(f"Judge prompt file is empty: {prompt_path}")

        return content
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to read judge prompt file {prompt_path}: {str(e)}") from e
