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
Tests for LLM provider abstraction and implementations.

Tests validate the pluggable provider interface, provider selection logic,
LocalMockProvider behavior, and integration with CLI and dataset flows.
"""

import threading
from unittest.mock import patch

import pytest

from prompt_evaluator.provider import (
    LocalMockProvider,
    OpenAIProvider,
    ProviderConfig,
    ProviderResult,
    get_provider,
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_defaults(self):
        """Test that ProviderConfig has sensible defaults."""
        config = ProviderConfig(model="gpt-5.1")
        assert config.model == "gpt-5.1"
        assert config.temperature == 0.7
        assert config.max_completion_tokens == 1024
        assert config.seed is None
        assert config.additional_params is None

    def test_provider_config_custom_values(self):
        """Test that ProviderConfig accepts custom values."""
        config = ProviderConfig(
            model="gpt-4",
            temperature=0.5,
            max_completion_tokens=2048,
            seed=42,
            additional_params={"top_p": 0.9},
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_completion_tokens == 2048
        assert config.seed == 42
        assert config.additional_params == {"top_p": 0.9}

    def test_provider_config_validates_temperature(self):
        """Test that ProviderConfig validates temperature bounds."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            ProviderConfig(model="gpt-5.1", temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            ProviderConfig(model="gpt-5.1", temperature=2.5)

    def test_provider_config_validates_max_tokens(self):
        """Test that ProviderConfig validates max_completion_tokens."""
        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            ProviderConfig(model="gpt-5.1", max_completion_tokens=0)

        with pytest.raises(ValueError, match="max_completion_tokens must be positive"):
            ProviderConfig(model="gpt-5.1", max_completion_tokens=-10)


class TestProviderResult:
    """Tests for ProviderResult dataclass."""

    def test_provider_result_structure(self):
        """Test that ProviderResult has expected structure."""
        result = ProviderResult(
            text="Test response",
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
            latency_ms=123.45,
            model="gpt-5.1",
            finish_reason="stop",
        )
        assert result.text == "Test response"
        assert result.usage["total_tokens"] == 100
        assert result.latency_ms == 123.45
        assert result.model == "gpt-5.1"
        assert result.finish_reason == "stop"
        assert result.error is None

    def test_provider_result_with_error(self):
        """Test ProviderResult with error."""
        result = ProviderResult(
            text="",
            usage={"total_tokens": None, "prompt_tokens": None, "completion_tokens": None},
            latency_ms=50.0,
            model="gpt-5.1",
            error="API call failed",
        )
        assert result.text == ""
        assert result.error == "API call failed"
        assert result.usage["total_tokens"] is None


class TestGetProvider:
    """Tests for get_provider factory function."""

    def test_get_provider_openai(self, monkeypatch):
        """Test that get_provider returns OpenAIProvider for 'openai'."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_openai_case_insensitive(self, monkeypatch):
        """Test that provider name is case-insensitive."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = get_provider("OpenAI")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_mock(self):
        """Test that get_provider returns LocalMockProvider for 'mock'."""
        provider = get_provider("mock")
        assert isinstance(provider, LocalMockProvider)

    def test_get_provider_local_mock(self):
        """Test that get_provider returns LocalMockProvider for 'local-mock'."""
        provider = get_provider("local-mock")
        assert isinstance(provider, LocalMockProvider)

    def test_get_provider_unknown_provider(self):
        """Test that get_provider raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unsupported provider: unknown"):
            get_provider("unknown")

        # Verify error message includes supported providers
        try:
            get_provider("unknown")
        except ValueError as e:
            assert "openai" in str(e)
            assert "mock" in str(e)

    def test_get_provider_validates_config_by_default(self, monkeypatch):
        """Test that get_provider validates configuration by default."""
        # Remove any existing API key from environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # OpenAI provider without API key should fail validation
        # Note: OpenAI client throws error during initialization, not during validate_config
        with pytest.raises((ValueError, Exception), match="API key|OPENAI_API_KEY"):
            get_provider("openai")

    def test_get_provider_skip_validation(self):
        """Test that get_provider can skip validation."""
        # Pass a dummy key - even with validate=False, OpenAI client needs a key to initialize
        provider = get_provider("openai", api_key="sk-dummy-key-for-testing", validate=False)
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "sk-dummy-key-for-testing"

    def test_get_provider_passes_parameters(self, monkeypatch):
        """Test that get_provider passes api_key and base_url."""
        monkeypatch.setenv("OPENAI_API_KEY", "default-key")
        provider = get_provider("openai", api_key="custom-key", base_url="https://custom.url")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "custom-key"
        assert provider.base_url == "https://custom.url"


class TestLocalMockProvider:
    """Tests for LocalMockProvider implementation."""

    def test_mock_provider_initialization(self):
        """Test LocalMockProvider initialization."""
        provider = LocalMockProvider()
        assert provider.response_template == "Mock response to: {prompt}"

    def test_mock_provider_custom_template(self):
        """Test LocalMockProvider with custom template."""
        provider = LocalMockProvider(response_template="Custom: {prompt}")
        assert provider.response_template == "Custom: {prompt}"

    def test_mock_provider_validate_config(self):
        """Test that LocalMockProvider.validate_config doesn't raise."""
        provider = LocalMockProvider()
        provider.validate_config()  # Should not raise

    def test_mock_provider_generate_simple(self):
        """Test basic generation with LocalMockProvider."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt=None,
            user_prompt="What is Python?",
            config=config,
        )

        assert isinstance(result, ProviderResult)
        assert "What is Python?" in result.text
        assert result.model == "mock-gpt-5.1"
        assert result.finish_reason == "stop"
        assert result.error is None
        assert result.usage["total_tokens"] > 0
        assert result.latency_ms > 0

    def test_mock_provider_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is Python?",
            config=config,
        )

        assert "[System:" in result.text
        assert "What is Python?" in result.text
        assert result.usage["prompt_tokens"] > 0  # Should count system + user tokens

    def test_mock_provider_generate_multi_turn(self):
        """Test generation with list of user prompts."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt=None,
            user_prompt=["First message", "Second message", "Third message"],
            config=config,
        )

        # Should concatenate prompts
        assert "First message" in result.text
        assert "Second message" in result.text
        assert "Third message" in result.text

    def test_mock_provider_deterministic_token_counts(self):
        """Test that mock token counts are deterministic based on text length."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")

        # Same prompt should give same token counts
        result1 = provider.generate(
            system_prompt="System",
            user_prompt="Test prompt",
            config=config,
        )
        result2 = provider.generate(
            system_prompt="System",
            user_prompt="Test prompt",
            config=config,
        )

        assert result1.usage["prompt_tokens"] == result2.usage["prompt_tokens"]
        assert result1.usage["completion_tokens"] == result2.usage["completion_tokens"]
        assert result1.usage["total_tokens"] == result2.usage["total_tokens"]

    def test_mock_provider_truncates_long_prompts(self):
        """Test that mock provider truncates very long prompts."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")

        # Very long prompt (more than 100 chars)
        long_prompt = "x" * 200

        result = provider.generate(
            system_prompt=None,
            user_prompt=long_prompt,
            config=config,
        )

        # Template should truncate to 100 chars
        assert len(result.text) < len(long_prompt)

    def test_mock_provider_thread_safety(self):
        """Test that LocalMockProvider is thread-safe."""
        provider = LocalMockProvider()
        config = ProviderConfig(model="gpt-5.1")
        results = []
        errors = []

        def generate_in_thread(prompt_id):
            try:
                result = provider.generate(
                    system_prompt=None,
                    user_prompt=f"Prompt {prompt_id}",
                    config=config,
                )
                results.append((prompt_id, result))
            except Exception as e:
                errors.append((prompt_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=generate_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All threads should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # Each result should be unique and correct
        for prompt_id, result in results:
            assert f"Prompt {prompt_id}" in result.text

    def test_mock_provider_respects_config_model(self):
        """Test that mock provider uses model from config."""
        provider = LocalMockProvider()

        config1 = ProviderConfig(model="gpt-5.1")
        result1 = provider.generate(
            system_prompt=None,
            user_prompt="Test",
            config=config1,
        )
        assert result1.model == "mock-gpt-5.1"

        config2 = ProviderConfig(model="gpt-4")
        result2 = provider.generate(
            system_prompt=None,
            user_prompt="Test",
            config=config2,
        )
        assert result2.model == "mock-gpt-4"


class TestOpenAIProviderNewInterface:
    """Tests for OpenAIProvider using new LLMProvider interface."""

    def test_openai_provider_validate_config_with_env_key(self, monkeypatch):
        """Test OpenAI provider validation with environment API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIProvider()
        provider.validate_config()  # Should not raise

    def test_openai_provider_validate_config_with_passed_key(self, monkeypatch):
        """Test OpenAI provider validation with passed API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(api_key="test-key")
        provider.validate_config()  # Should not raise

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_validate_config_missing_key(self, mock_openai_class, monkeypatch):
        """Test OpenAI provider validation fails without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Mock OpenAI client to bypass initialization check, but set api_key to None
        mock_client = mock_openai_class.return_value
        mock_client.api_key = None

        provider = OpenAIProvider(api_key=None)

        # The validation should check environment which has no key
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            provider.validate_config()

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_validate_invalid_base_url(self, mock_openai_class, monkeypatch):
        """Test OpenAI provider validation fails with invalid base_url format."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create provider with invalid base_url
        provider = OpenAIProvider(base_url="not-a-url")

        with pytest.raises(ValueError, match="Invalid base_url format"):
            provider.validate_config()

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_validate_valid_base_url(self, mock_openai_class, monkeypatch):
        """Test OpenAI provider validation succeeds with valid base_url."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Test http and https URLs
        for base_url in ["http://localhost:8000", "https://api.example.com"]:
            provider = OpenAIProvider(base_url=base_url)
            provider.validate_config()  # Should not raise

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_generate_success(self, mock_openai_class, monkeypatch):
        """Test successful generation with OpenAIProvider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI client response
        mock_client = mock_openai_class.return_value
        mock_completion = mock_client.chat.completions.create.return_value
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage.total_tokens = 100
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 50

        provider = OpenAIProvider()
        config = ProviderConfig(model="gpt-5.1", temperature=0.5, seed=42)

        result = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is Python?",
            config=config,
        )

        assert result.text == "Test response"
        assert result.model == "gpt-5.1"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 100
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 50
        assert result.error is None
        assert result.latency_ms > 0

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.1"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["seed"] == 42
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_generate_without_system_prompt(self, mock_openai_class, monkeypatch):
        """Test generation without system prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = mock_openai_class.return_value
        mock_completion = mock_client.chat.completions.create.return_value
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage.total_tokens = 50
        mock_completion.usage.prompt_tokens = 25
        mock_completion.usage.completion_tokens = 25

        provider = OpenAIProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt=None,
            user_prompt="What is Python?",
            config=config,
        )

        assert result.text == "Test response"

        # Verify only user message was sent
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_generate_multi_turn(self, mock_openai_class, monkeypatch):
        """Test generation with multiple user prompts."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = mock_openai_class.return_value
        mock_completion = mock_client.chat.completions.create.return_value
        mock_completion.choices[0].message.content = "Multi-turn response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = None  # Test case with no usage info

        provider = OpenAIProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt="System",
            user_prompt=["First", "Second", "Third"],
            config=config,
        )

        # Verify all user prompts were sent
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 4  # 1 system + 3 user
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["content"] == "First"
        assert call_kwargs["messages"][2]["content"] == "Second"
        assert call_kwargs["messages"][3]["content"] == "Third"

        # Test that None usage is handled
        assert result.usage["total_tokens"] is None

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_generate_with_error(self, mock_openai_class, monkeypatch):
        """Test that OpenAI provider returns error result on failure."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.side_effect = Exception("API error")

        provider = OpenAIProvider()
        config = ProviderConfig(model="gpt-5.1")

        result = provider.generate(
            system_prompt=None,
            user_prompt="Test",
            config=config,
        )

        assert result.text == ""
        assert result.error is not None
        assert "Unexpected error" in result.error
        assert result.usage["total_tokens"] is None
        assert result.latency_ms > 0  # Should still track latency

    @patch("prompt_evaluator.provider.OpenAI")
    def test_openai_provider_passes_additional_params(self, mock_openai_class, monkeypatch):
        """Test that additional_params are passed through to the API call."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = mock_openai_class.return_value
        mock_completion = mock_client.chat.completions.create.return_value
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage.total_tokens = 50
        mock_completion.usage.prompt_tokens = 25
        mock_completion.usage.completion_tokens = 25

        provider = OpenAIProvider()

        # Create config with additional params
        config = ProviderConfig(
            model="gpt-5.1",
            temperature=0.7,
            additional_params={"top_p": 0.9, "frequency_penalty": 0.5}
        )

        result = provider.generate(
            system_prompt="System",
            user_prompt="Test",
            config=config,
        )

        assert result.text == "Test response"

        # Verify additional_params were passed to the API
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # Check that additional params were included
        assert "top_p" in call_kwargs
        assert call_kwargs["top_p"] == 0.9
        assert "frequency_penalty" in call_kwargs
        assert call_kwargs["frequency_penalty"] == 0.5

        # Check that standard params are still there
        assert call_kwargs["model"] == "gpt-5.1"
        assert call_kwargs["temperature"] == 0.7
