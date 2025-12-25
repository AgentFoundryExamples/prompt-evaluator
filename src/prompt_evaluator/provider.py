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
LLM provider integrations.

This module provides a unified interface for interacting with different
LLM providers (OpenAI, Anthropic, etc.) and handles API communication.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, OpenAIError

from prompt_evaluator.models import EvaluationRequest, EvaluationResponse

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for LLM provider requests."""

    model: str
    temperature: float = 0.7
    max_completion_tokens: int = 1024
    seed: int | None = None
    additional_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")


@dataclass
class ProviderResult:
    """Result from LLM provider generation."""

    text: str
    usage: dict[str, int | None]
    latency_ms: float
    model: str
    finish_reason: str | None = None
    error: str | None = None


class LLMProvider(ABC):
    """
    Abstract base class defining the canonical interface for LLM providers.

    All providers must implement the generate() method which accepts system
    and user prompts along with provider-specific configuration and returns
    normalized text and usage metadata.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str | list[str],
        config: ProviderConfig,
    ) -> ProviderResult:
        """
        Generate a completion from the LLM.

        Args:
            system_prompt: Optional system prompt to set context
            user_prompt: User prompt or list of user prompts (for multi-turn)
            config: Provider-specific configuration including model, temperature, etc.

        Returns:
            ProviderResult with generated text, usage metadata, and latency

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider API call fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate provider configuration (API keys, etc.) before use.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        pass


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers (legacy interface).

    Note: This is maintained for backward compatibility.
    New code should use LLMProvider instead.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication (if None, uses environment variable)
        """
        self.api_key = api_key

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """
        Evaluate a prompt with the provider.

        Args:
            request: The evaluation request

        Returns:
            Evaluation response with generated text and metadata
        """
        pass


class OpenAIProvider(BaseProvider, LLMProvider):
    """Provider implementation for OpenAI models."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            base_url: Optional custom base URL for OpenAI API
        """
        super().__init__(api_key)
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def validate_config(self) -> None:
        """
        Validate OpenAI provider configuration.

        Raises:
            ValueError: If API key is not configured
        """
        # Check if API key is available (either passed or in environment)
        if not self.api_key and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str | list[str],
        config: ProviderConfig,
    ) -> ProviderResult:
        """
        Generate a completion using OpenAI API.

        Args:
            system_prompt: Optional system prompt to set context
            user_prompt: User prompt (string) or list of user prompts for multi-turn
            config: Provider configuration including model, temperature, etc.

        Returns:
            ProviderResult with generated text, usage metadata, and latency

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If API call fails
        """
        start_time = time.time()

        try:
            # Build messages
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Handle single or multiple user prompts
            if isinstance(user_prompt, str):
                messages.append({"role": "user", "content": user_prompt})
            else:
                for prompt in user_prompt:
                    messages.append({"role": "user", "content": prompt})

            # Build API parameters
            params: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_completion_tokens": config.max_completion_tokens,
            }

            # Add seed if provided
            if config.seed is not None:
                params["seed"] = config.seed

            # Add additional parameters if provided
            if config.additional_params:
                params.update(config.additional_params)

            # Make API call
            completion = self.client.chat.completions.create(**params)  # type: ignore[call-overload]

            response_text = completion.choices[0].message.content or ""
            tokens_used = completion.usage.total_tokens if completion.usage else None
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
            completion_tokens = completion.usage.completion_tokens if completion.usage else None
            finish_reason = completion.choices[0].finish_reason if completion.choices else None

            latency_ms = (time.time() - start_time) * 1000

            return ProviderResult(
                text=response_text,
                usage={
                    "total_tokens": tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                latency_ms=latency_ms,
                model=config.model,
                finish_reason=finish_reason,
            )

        except OpenAIError as e:
            logger.error("OpenAI API request failed: %s", e, exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return ProviderResult(
                text="",
                usage={"total_tokens": None, "prompt_tokens": None, "completion_tokens": None},
                latency_ms=latency_ms,
                model=config.model,
                error=f"OpenAI API error: {str(e)}",
            )
        except Exception as e:
            logger.error("Unexpected error during generation: %s", e, exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return ProviderResult(
                text="",
                usage={"total_tokens": None, "prompt_tokens": None, "completion_tokens": None},
                latency_ms=latency_ms,
                model=config.model,
                error=f"Unexpected error: {str(e)}",
            )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """
        Evaluate a prompt using OpenAI API.

        Args:
            request: The evaluation request

        Returns:
            Evaluation response with generated text and metadata
        """
        start_time = time.time()
        error_msg: str | None = None
        response_text = ""
        tokens_used: int | None = None

        try:
            # Build API parameters, excluding None values
            params = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": request.temperature,
            }
            if request.max_completion_tokens is not None:
                params["max_completion_tokens"] = request.max_completion_tokens

            completion = self.client.chat.completions.create(**params)  # type: ignore[call-overload]

            response_text = completion.choices[0].message.content or ""
            tokens_used = completion.usage.total_tokens if completion.usage else None

        except OpenAIError as e:
            # Catch specific OpenAI errors for better error handling
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error("OpenAI API request failed: %s", e, exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error during evaluation: %s", e, exc_info=True)

        latency_ms = (time.time() - start_time) * 1000

        return EvaluationResponse(
            request=request,
            response_text=response_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            error=error_msg,
        )


class LocalMockProvider(LLMProvider):
    """
    Mock provider for testing and offline operation.

    Provides deterministic stubbed outputs without making real API calls.
    Thread-safe and suitable for concurrent operations.
    """

    def __init__(self, response_template: str = "Mock response to: {prompt}"):
        """
        Initialize local mock provider.

        Args:
            response_template: Template for mock responses. Use {prompt} placeholder
                             to include the user prompt in the response.
        """
        self.response_template = response_template

    def validate_config(self) -> None:
        """
        Validate mock provider configuration.

        Mock provider doesn't require any configuration, so this is a no-op.
        """
        pass  # No configuration required for mock provider

    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str | list[str],
        config: ProviderConfig,
    ) -> ProviderResult:
        """
        Generate a mock completion deterministically.

        Args:
            system_prompt: Optional system prompt (included in mock metadata)
            user_prompt: User prompt (string) or list of user prompts
            config: Provider configuration (model name used in result)

        Returns:
            ProviderResult with deterministic mock text and metadata
        """
        start_time = time.time()

        # Construct prompt text for template
        if isinstance(user_prompt, str):
            prompt_text = user_prompt
        else:
            prompt_text = " | ".join(user_prompt)

        # Generate deterministic mock response
        response_text = self.response_template.format(prompt=prompt_text[:100])

        # Add system prompt context if provided
        if system_prompt:
            response_text = f"[System: {system_prompt[:50]}...] {response_text}"

        # Compute deterministic mock token counts based on text length
        mock_prompt_tokens = len(prompt_text.split()) + (
            len(system_prompt.split()) if system_prompt else 0
        )
        mock_completion_tokens = len(response_text.split())
        mock_total_tokens = mock_prompt_tokens + mock_completion_tokens

        # Simulate minimal latency
        latency_ms = (time.time() - start_time) * 1000 + 10.0  # Add 10ms base

        return ProviderResult(
            text=response_text,
            usage={
                "total_tokens": mock_total_tokens,
                "prompt_tokens": mock_prompt_tokens,
                "completion_tokens": mock_completion_tokens,
            },
            latency_ms=latency_ms,
            model=f"mock-{config.model}",
            finish_reason="stop",
        )


def get_provider(
    provider_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    validate: bool = True,
) -> LLMProvider:
    """
    Factory function to get a provider instance by name.

    This is the canonical way to instantiate providers. It supports provider
    selection via name string and validates configuration before returning.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'mock')
        api_key: Optional API key (for providers that require it)
        base_url: Optional custom base URL (for providers that support it)
        validate: Whether to validate provider configuration (default: True)

    Returns:
        LLMProvider instance ready for use

    Raises:
        ValueError: If provider name is not supported or configuration is invalid
    """
    provider_name_lower = provider_name.lower()

    # Registry of available providers
    provider: LLMProvider
    if provider_name_lower == "openai":
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    elif provider_name_lower == "mock" or provider_name_lower == "local-mock":
        provider = LocalMockProvider()
    else:
        supported = ["openai", "mock", "local-mock"]
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: {supported}"
        )

    # Validate provider configuration if requested
    if validate:
        try:
            provider.validate_config()
        except ValueError as e:
            raise ValueError(f"Provider configuration invalid for '{provider_name}': {e}") from e

    return provider


def generate_completion(
    provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    seed: int | None = None,
) -> tuple[str, dict[str, float | int | None]]:
    """
    Generate a completion using an LLM provider with system and user prompts.

    This is a convenience wrapper around the provider's generate() method that
    maintains backward compatibility with the existing codebase signature.

    Args:
        provider: The LLM provider to use (OpenAIProvider, LocalMockProvider, etc.)
        system_prompt: System prompt to set context
        user_prompt: User prompt/input
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_completion_tokens: Maximum tokens to generate
        seed: Optional seed for reproducibility

    Returns:
        Tuple of (response_text, metadata) where metadata contains tokens_used and latency_ms

    Raises:
        RuntimeError: If the provider API call fails
    """
    # Create provider config
    config = ProviderConfig(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        seed=seed,
    )

    # Generate using the provider
    result = provider.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
    )

    # Check for errors
    if result.error:
        raise RuntimeError(result.error)

    # Format metadata for backward compatibility
    metadata = {
        "tokens_used": result.usage.get("total_tokens"),
        "latency_ms": result.latency_ms,
    }

    return result.text, metadata


def build_rubric_judge_prompt(rubric: Any) -> str:
    """
    Build a rubric-aware judge system prompt that enumerates all metrics and flags.

    Args:
        rubric: Rubric object with metrics and flags

    Returns:
        System prompt string instructing judge to evaluate based on rubric
    """

    # Helper function to sanitize text for prompt inclusion
    def sanitize_text(text: str) -> str:
        """Remove control characters and limit length for safe prompt inclusion."""
        # Remove control characters except newlines and tabs
        sanitized = "".join(char for char in text if char.isprintable() or char in "\n\t")
        # Limit length to prevent excessively long prompts
        max_length = 5000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized

    prompt_parts = [
        "You are an expert evaluator assessing AI-generated responses using a structured rubric.",
        "",
        "Your task is to evaluate the generated output based on multiple "
        "metrics and flags defined below.",
        "",
        "## METRICS",
        "",
    ]

    # Enumerate each metric
    for metric in rubric.metrics:
        prompt_parts.append(f"### {sanitize_text(metric.name)}")
        prompt_parts.append(f"**Description:** {sanitize_text(metric.description)}")
        prompt_parts.append(
            f"**Score Range:** {metric.min_score} (minimum) to {metric.max_score} (maximum)"
        )
        prompt_parts.append("**Guidelines:**")
        prompt_parts.append(sanitize_text(metric.guidelines))
        prompt_parts.append("")

    # Enumerate flags if present
    if rubric.flags:
        prompt_parts.append("## FLAGS")
        prompt_parts.append("")
        for flag in rubric.flags:
            prompt_parts.append(f"### {sanitize_text(flag.name)}")
            prompt_parts.append(f"**Description:** {sanitize_text(flag.description)}")
            prompt_parts.append("**Type:** Boolean (true/false)")
            prompt_parts.append("")

    # Build JSON schema
    metrics_schema = {}
    for metric in rubric.metrics:
        # Use sanitized metric name as key
        metric_name = sanitize_text(metric.name)
        metrics_schema[metric_name] = {
            "score": f"<number between {metric.min_score} and {metric.max_score}>",
            "rationale": "<string explaining your score in 1-3 sentences>",
        }

    flags_schema = {sanitize_text(flag.name): "<boolean true or false>" for flag in rubric.flags}

    schema_example = {
        "metrics": metrics_schema,
        "flags": flags_schema,
        "overall_comment": "<string with overall assessment>",
    }

    # Add response format instructions
    prompt_parts.append("## RESPONSE FORMAT")
    prompt_parts.append("")
    prompt_parts.append(
        "You must respond with a valid JSON object containing exactly three fields:"
    )
    prompt_parts.append("")
    prompt_parts.append("```json")
    prompt_parts.append(json.dumps(schema_example, indent=2))
    prompt_parts.append("```")
    prompt_parts.append("")
    prompt_parts.append("**Requirements:**")
    prompt_parts.append("1. Each metric must have a numeric 'score' within the specified range")
    prompt_parts.append("2. Each metric must have a 'rationale' explaining the score")
    prompt_parts.append("3. Each flag must have a boolean value (true or false)")
    prompt_parts.append("4. Include an 'overall_comment' summarizing your evaluation")
    prompt_parts.append("5. Do not include any text before or after the JSON object")

    return "\n".join(prompt_parts)


def parse_rubric_judge_response(response_text: str, rubric: Any) -> dict[str, Any]:
    """
    Parse and validate judge response against rubric schema.

    Args:
        response_text: Raw JSON response from judge model
        rubric: Rubric object with expected metrics and flags

    Returns:
        Dictionary with:
        - status: "completed" or "judge_invalid_response"
        - judge_metrics: dict[str, dict] with score/rationale per metric (if valid)
        - judge_flags: dict[str, bool] with boolean per flag (if valid)
        - judge_overall_comment: str with overall comment (if valid)
        - judge_raw_response: str with raw response
        - error: str with error details (if invalid)
    """
    raw_response = response_text

    try:
        # Try to extract JSON from response
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON object from text with markdown fencing or extra content
            # Remove markdown code fences
            cleaned = response_text
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            # Try to extract valid JSON by finding matching braces
            # Use a stack-based approach to handle nested structures correctly
            start_index = cleaned.find("{")
            if start_index == -1:
                raise ValueError("No JSON object found in response")

            # Track brace depth to find the matching closing brace
            brace_count = 0
            end_index = -1
            in_string = False
            escape_next = False

            for i in range(start_index, len(cleaned)):
                char = cleaned[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_index = i
                            break

            if end_index == -1:
                raise ValueError("No complete JSON object found in response")

            json_str = cleaned[start_index : end_index + 1]
            parsed = json.loads(json_str)

        # Validate top-level structure
        if not isinstance(parsed, dict):
            raise ValueError(f"Response must be a JSON object, got {type(parsed).__name__}")

        # Validate metrics field
        if "metrics" not in parsed:
            raise ValueError("Missing required field: metrics")

        metrics_data = parsed["metrics"]
        if not isinstance(metrics_data, dict):
            raise ValueError(f"metrics must be a dictionary, got {type(metrics_data).__name__}")

        # Validate all expected metrics are present
        expected_metric_names = {m.name for m in rubric.metrics}
        provided_metric_names = set(metrics_data.keys())

        missing_metrics = expected_metric_names - provided_metric_names
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {', '.join(sorted(missing_metrics))}")

        extra_metrics = provided_metric_names - expected_metric_names
        if extra_metrics:
            raise ValueError(f"Unknown metrics provided: {', '.join(sorted(extra_metrics))}")

        # Validate each metric
        judge_metrics = {}
        for metric in rubric.metrics:
            metric_name = metric.name
            metric_value = metrics_data[metric_name]

            if not isinstance(metric_value, dict):
                raise ValueError(
                    f"Metric '{metric_name}' must be a dictionary with 'score' and 'rationale'"
                )

            if "score" not in metric_value:
                raise ValueError(f"Metric '{metric_name}' missing required field: score")
            if "rationale" not in metric_value:
                raise ValueError(f"Metric '{metric_name}' missing required field: rationale")

            # Validate score is numeric and in range
            try:
                score = float(metric_value["score"])
            except (TypeError, ValueError) as e:
                score_type = type(metric_value["score"]).__name__
                raise ValueError(
                    f"Metric '{metric_name}' score must be numeric, got {score_type}"
                ) from e

            if score < metric.min_score or score > metric.max_score:
                raise ValueError(
                    f"Metric '{metric_name}' score {score} is out of range "
                    f"[{metric.min_score}, {metric.max_score}]"
                )

            # Validate rationale is a string
            rationale = str(metric_value["rationale"])

            judge_metrics[metric_name] = {"score": score, "rationale": rationale}

        # Validate flags field (optional if no flags in rubric)
        judge_flags = {}
        if rubric.flags:
            if "flags" not in parsed:
                raise ValueError("Missing required field: flags")

            flags_data = parsed["flags"]
            if not isinstance(flags_data, dict):
                raise ValueError(f"flags must be a dictionary, got {type(flags_data).__name__}")

            # Validate all expected flags are present
            expected_flag_names = {f.name for f in rubric.flags}
            provided_flag_names = set(flags_data.keys())

            missing_flags = expected_flag_names - provided_flag_names
            if missing_flags:
                raise ValueError(f"Missing required flags: {', '.join(sorted(missing_flags))}")

            extra_flags = provided_flag_names - expected_flag_names
            if extra_flags:
                raise ValueError(f"Unknown flags provided: {', '.join(sorted(extra_flags))}")

            # Validate each flag
            for flag in rubric.flags:
                flag_name = flag.name
                flag_value = flags_data[flag_name]

                if not isinstance(flag_value, bool):
                    raise ValueError(
                        f"Flag '{flag_name}' must be a boolean, got {type(flag_value).__name__}"
                    )

                judge_flags[flag_name] = flag_value

        # Validate overall_comment
        if "overall_comment" not in parsed:
            raise ValueError("Missing required field: overall_comment")

        overall_comment = str(parsed["overall_comment"])

        return {
            "status": "completed",
            "judge_metrics": judge_metrics,
            "judge_flags": judge_flags,
            "judge_overall_comment": overall_comment,
            "judge_raw_response": raw_response,
            "judge_score": None,  # Legacy field, not used with rubric
            "judge_rationale": None,  # Legacy field, not used with rubric
            "error": None,
        }

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # Parsing or validation failed
        error_msg = f"Failed to parse or validate judge response: {str(e)}"
        logger.warning("%s. Raw response: %s", error_msg, raw_response[:200])

        return {
            "status": "judge_invalid_response",
            "judge_metrics": {},
            "judge_flags": {},
            "judge_overall_comment": None,
            "judge_raw_response": raw_response,
            "judge_score": None,
            "judge_rationale": None,
            "error": error_msg,
        }


def judge_completion(
    provider: LLMProvider,
    input_text: str,
    generator_output: str,
    judge_config: Any,
    judge_system_prompt: str,
    task_description: str | None = None,
    rubric: Any | None = None,
) -> dict[str, Any]:
    """
    Call judge model to evaluate generator output and return structured results.

    This function builds system+user messages, executes the provider API with the
    configured judge model, enforces JSON schema, and returns structured results
    or a judge_error status with raw text preserved.

    Args:
        provider: The LLM provider to use for judge model
        input_text: Original input text
        generator_output: Output from the generator model to evaluate
        judge_config: JudgeConfig with model settings
        judge_system_prompt: System prompt instructing judge on scoring
        task_description: Optional description of the task for context
        rubric: Optional Rubric object for structured evaluation criteria.
               Currently passed through but not yet integrated into judge prompt.
               Reserved for future enhancement to support multi-dimensional scoring.

    Returns:
        Dictionary with keys:
        - status: "completed" or "judge_error"
        - judge_score: float (1-5) if status is "completed", None otherwise
        - judge_rationale: str if status is "completed", None otherwise
        - judge_raw_response: str with raw model output
        - error: str with error details if status is "judge_error", None otherwise

    Note:
        The rubric parameter is currently reserved for future implementation of
        multi-dimensional scoring. When fully implemented, it will be used to
        dynamically generate judge prompts based on rubric metrics and flags.
    """
    # Use rubric-aware prompt if rubric is provided
    if rubric is not None:
        system_prompt = build_rubric_judge_prompt(rubric)
    else:
        system_prompt = judge_system_prompt

    # Build user message with input, output, and optional task description
    user_parts = []

    if task_description:
        user_parts.append(f"Task: {task_description}\n")

    user_parts.append(f"Original Input:\n{input_text}\n")
    user_parts.append(f"Generated Output:\n{generator_output}\n")

    if rubric is not None:
        user_parts.append("Please evaluate the generated output according to the rubric.")
    else:
        user_parts.append("Please evaluate the semantic fidelity of the generated output.")

    user_message = "\n".join(user_parts)

    # Make API call
    try:
        response_text, metadata = generate_completion(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_message,
            model=judge_config.model_name,
            temperature=judge_config.temperature,
            max_completion_tokens=judge_config.max_completion_tokens,
            seed=judge_config.seed,
        )

        raw_response = response_text

        # Use rubric-aware parsing if rubric is provided
        if rubric is not None:
            return parse_rubric_judge_response(response_text, rubric)

        # Try to extract and parse JSON from response
        try:
            # First try direct parsing
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON object from text that may have additional content
                # Find first { and last } to capture the entire JSON object
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                if start_index == -1 or end_index == -1 or end_index < start_index:
                    raise ValueError("No JSON object found in response")

                json_str = response_text[start_index : end_index + 1]
                parsed = json.loads(json_str)

            # Validate required fields
            if "semantic_fidelity" not in parsed:
                raise ValueError("Missing required field: semantic_fidelity")
            if "rationale" not in parsed:
                raise ValueError("Missing required field: rationale")

            # Extract and validate score
            score = float(parsed["semantic_fidelity"])

            # Clamp score to valid range [1, 5]
            if score < 1.0:
                logger.warning("Judge score %s below minimum, clamping to 1.0", score)
                score = 1.0
            elif score > 5.0:
                logger.warning("Judge score %s above maximum, clamping to 5.0", score)
                score = 5.0

            rationale = str(parsed["rationale"])

            return {
                "status": "completed",
                "judge_score": score,
                "judge_rationale": rationale,
                "judge_raw_response": raw_response,
                "judge_metrics": {},  # Empty for legacy mode
                "judge_flags": {},  # Empty for legacy mode
                "judge_overall_comment": None,  # Not used in legacy mode
                "error": None,
            }

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            # JSON parsing or validation failed
            error_msg = f"Failed to parse judge response: {str(e)}"
            logger.warning("%s. Raw response: %s", error_msg, raw_response[:200])

            return {
                "status": "judge_error",
                "judge_score": None,
                "judge_rationale": None,
                "judge_raw_response": raw_response,
                "judge_metrics": {},
                "judge_flags": {},
                "judge_overall_comment": None,
                "error": error_msg,
            }

    except Exception as e:
        # API call failed
        error_msg = f"Judge API call failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Try to preserve any raw response data from the exception
        raw_response = None
        if hasattr(e, "response"):
            response_obj = getattr(e, "response")
            if hasattr(response_obj, "text"):
                raw_response = getattr(response_obj, "text")

        return {
            "status": "judge_error",
            "judge_score": None,
            "judge_rationale": None,
            "judge_raw_response": raw_response,
            "judge_metrics": {},
            "judge_flags": {},
            "judge_overall_comment": None,
            "error": error_msg,
        }
