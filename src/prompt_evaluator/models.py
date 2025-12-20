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
from datetime import datetime, timezone
from enum import Enum
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
        default_factory=dict,
        description="Variable names and descriptions"
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
            r'\{[^}]*\.[^}]+\}',  # Attribute access: {obj.attr}
            r'\{[^}]*\[[^}]+\]\}',  # Indexing: {obj[0]}
            r'\{[^}]*![^}]+\}',  # Conversion: {var!r}
            r'\{[^}]+:[^}]+\}',  # Format spec: {var:03d} (requires both var and spec)
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
    max_tokens: int | None = Field(None, gt=0)
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
        default_factory=list,
        description="All responses from this evaluation"
    )
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = Field(None)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total duration of the evaluation run."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
