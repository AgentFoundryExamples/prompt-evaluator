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

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI, OpenAIError

from prompt_evaluator.models import EvaluationRequest, EvaluationResponse

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

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


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI models."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            base_url: Optional custom base URL for OpenAI API
        """
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

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
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens

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


def get_provider(
    provider_name: str, api_key: str | None = None, base_url: str | None = None
) -> BaseProvider:
    """
    Factory function to get a provider instance.

    Args:
        provider_name: Name of the provider (e.g., 'openai')
        api_key: Optional API key
        base_url: Optional custom base URL for the provider

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is not supported
    """
    providers = {
        "openai": OpenAIProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unsupported provider: {provider_name}. Supported providers: {list(providers.keys())}"
        )

    # OpenAI provider supports base_url parameter
    if provider_name.lower() == "openai":
        return provider_class(api_key=api_key, base_url=base_url)

    # For other providers that don't support base_url
    return provider_class(api_key=api_key)  # type: ignore[call-arg]


def generate_completion(
    provider: BaseProvider,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    seed: int | None = None,
) -> tuple[str, dict[str, float | int | None]]:
    """
    Generate a completion using the provider with system and user prompts.

    Args:
        provider: The LLM provider to use
        system_prompt: System prompt to set context
        user_prompt: User prompt/input
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        seed: Optional seed for reproducibility

    Returns:
        Tuple of (response_text, metadata) where metadata contains tokens_used and latency_ms

    Raises:
        ValueError: If the provider doesn't support the operation
        OpenAIError: If the API call fails
    """
    # TODO: Make this more extensible to support multiple providers
    # Currently only OpenAI provider is supported
    if not isinstance(provider, OpenAIProvider):
        raise ValueError("Only OpenAI provider is currently supported for generate_completion")

    start_time = time.time()

    try:
        # Build messages with system and user prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build API parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add seed if provided
        if seed is not None:
            params["seed"] = seed

        completion = provider.client.chat.completions.create(**params)  # type: ignore[call-overload]

        response_text = completion.choices[0].message.content or ""
        tokens_used = completion.usage.total_tokens if completion.usage else None
        latency_ms = (time.time() - start_time) * 1000

        metadata = {
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
        }

        return response_text, metadata

    except OpenAIError as e:
        logger.error("OpenAI API request failed: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error during completion: %s", e, exc_info=True)
        raise
