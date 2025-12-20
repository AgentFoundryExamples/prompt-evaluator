"""
Data models for prompt evaluation.

This module defines the core data structures used throughout the
evaluation process, including prompts, responses, and results.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with provided variable values.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Rendered prompt string
        """
        return self.template.format(**kwargs)


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = Field(None, description="Error message if request failed")


class EvaluationResult(BaseModel):
    """Complete results from an evaluation run."""

    run_id: str = Field(..., description="Unique identifier for this run")
    config_name: str = Field(..., description="Name of the evaluation configuration")
    responses: list[EvaluationResponse] = Field(
        default_factory=list,
        description="All responses from this evaluation"
    )
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime | None = Field(None)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total duration of the evaluation run."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
