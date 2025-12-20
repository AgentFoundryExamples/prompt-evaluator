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
Basic tests for the prompt_evaluator package.

These tests verify that the package structure is correct and
modules can be imported without errors.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import prompt_evaluator

    assert hasattr(prompt_evaluator, "__version__")
    assert prompt_evaluator.__version__ == "0.1.0"


def test_module_imports():
    """Test that all main modules can be imported."""
    from prompt_evaluator import cli, config, models, provider

    # Verify modules are loaded
    assert cli is not None
    assert config is not None
    assert models is not None
    assert provider is not None


def test_cli_entrypoint_exists():
    """Test that CLI main function exists."""
    from prompt_evaluator.cli import main

    assert callable(main)


def test_config_models_defined():
    """Test that configuration models are defined."""
    from prompt_evaluator.config import EvaluationConfig, ProviderConfig

    assert ProviderConfig is not None
    assert EvaluationConfig is not None


def test_data_models_defined():
    """Test that data models are defined."""
    from prompt_evaluator.models import (
        EvaluationRequest,
        EvaluationResponse,
        EvaluationResult,
        PromptTemplate,
    )

    assert PromptTemplate is not None
    assert EvaluationRequest is not None
    assert EvaluationResponse is not None
    assert EvaluationResult is not None


def test_provider_base_class_defined():
    """Test that provider classes are defined."""
    from prompt_evaluator.provider import BaseProvider, OpenAIProvider, get_provider

    assert BaseProvider is not None
    assert OpenAIProvider is not None
    assert callable(get_provider)


def test_prompt_template_render():
    """Test that PromptTemplate can render templates."""
    from prompt_evaluator.models import PromptTemplate

    template = PromptTemplate(
        template="Hello {name}, you are {age} years old.",
        variables={"name": "User's name", "age": "User's age"}
    )

    result = template.render(name="Alice", age=30)
    assert result == "Hello Alice, you are 30 years old."


def test_prompt_template_validation():
    """Test that PromptTemplate validation prevents unsafe format strings."""
    from prompt_evaluator.models import PromptTemplate

    # Valid template should work
    valid_template = PromptTemplate(
        template="Hello {name}",
        variables={"name": "User's name"}
    )
    assert valid_template.template == "Hello {name}"

    # Dangerous patterns should be rejected
    dangerous_templates = [
        "{obj.attr}",  # Attribute access
        "{obj[0]}",    # Indexing
        "{var!r}",     # Conversion
        "{var:03d}",   # Format spec
    ]

    for dangerous in dangerous_templates:
        with pytest.raises(ValueError, match="potentially unsafe format string"):
            PromptTemplate(template=dangerous, variables={})


def test_prompt_template_missing_variable():
    """Test that PromptTemplate raises clear error for missing variables."""
    from prompt_evaluator.models import PromptTemplate

    template = PromptTemplate(
        template="Hello {name} and {friend}",
        variables={"name": "User's name", "friend": "Friend's name"}
    )

    with pytest.raises(KeyError, match="Missing required template variable"):
        template.render(name="Alice")  # Missing 'friend'


def test_get_provider_openai():
    """Test that OpenAI provider can be retrieved."""
    from prompt_evaluator.provider import OpenAIProvider, get_provider

    # Provider can be instantiated even without API key (will fail on actual API calls)
    provider = get_provider("openai", api_key="test-key")
    assert isinstance(provider, OpenAIProvider)


def test_get_provider_invalid():
    """Test that invalid provider name raises error."""
    from prompt_evaluator.provider import get_provider

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("invalid_provider")
