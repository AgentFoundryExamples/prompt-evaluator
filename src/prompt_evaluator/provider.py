"""
LLM provider integrations.

This module provides a unified interface for interacting with different
LLM providers (OpenAI, Anthropic, etc.) and handles API communication.
"""

import logging
import time
from abc import ABC, abstractmethod

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

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)

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

            completion = self.client.chat.completions.create(**params)

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


def get_provider(provider_name: str, api_key: str | None = None) -> BaseProvider:
    """
    Factory function to get a provider instance.

    Args:
        provider_name: Name of the provider (e.g., 'openai')
        api_key: Optional API key

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
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: {list(providers.keys())}"
        )

    return provider_class(api_key=api_key)
