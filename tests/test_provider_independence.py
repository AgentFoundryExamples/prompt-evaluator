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
Tests for independent generator and judge provider configuration.
"""

from pathlib import Path

import pytest


class TestProviderIndependence:
    """Tests for independent generator and judge provider configuration."""

    def test_config_has_separate_provider_fields(self):
        """Test that config has separate generator and judge provider fields."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config_data = {
            "defaults": {
                "generator": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7
                },
                "judge": {
                    "provider": "anthropic",
                    "model": "claude-3",
                    "temperature": 0.0
                }
            }
        }
        
        config = PromptEvaluatorConfig(**config_data)
        
        assert config.defaults.generator.provider == "openai"
        assert config.defaults.judge.provider == "anthropic"
        # Ensure they are truly independent
        assert config.defaults.generator.provider != config.defaults.judge.provider

    def test_config_default_providers_are_independent(self):
        """Test that default providers can be different."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        # Use defaults
        config = PromptEvaluatorConfig()
        
        # Both should exist but can be changed independently
        assert hasattr(config.defaults.generator, 'provider')
        assert hasattr(config.defaults.judge, 'provider')

    def test_config_generator_change_does_not_affect_judge(self):
        """Test that changing generator provider doesn't affect judge."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config_data = {
            "defaults": {
                "generator": {
                    "provider": "openai",
                    "model": "gpt-4"
                },
                "judge": {
                    "provider": "anthropic",
                    "model": "claude-3"
                }
            }
        }
        
        config = PromptEvaluatorConfig(**config_data)
        
        original_judge_provider = config.defaults.judge.provider
        
        # Modify generator config (simulating a config update)
        config.defaults.generator.provider = "mock"
        
        # Judge provider should remain unchanged
        assert config.defaults.judge.provider == original_judge_provider
        assert config.defaults.judge.provider == "anthropic"

    def test_dataset_evaluation_accepts_separate_providers(self):
        """Test that dataset_evaluation function accepts separate provider parameters."""
        from prompt_evaluator.dataset_evaluation import evaluate_dataset
        import inspect
        
        # Get function signature
        sig = inspect.signature(evaluate_dataset)
        params = list(sig.parameters.keys())
        
        # Should have separate generator_provider and judge_provider parameters
        assert "generator_provider" in params
        assert "judge_provider" in params
        # Should not have a single "provider" parameter anymore
        assert "provider" not in params


class TestConfigPrecedence:
    """Tests for configuration precedence rules."""

    def test_explicit_config_path_resolves_correctly(self, tmp_path):
        """Test that explicit config path is properly resolved and cached."""
        from prompt_evaluator.config import ConfigManager
        
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: anthropic
    model: claude-3
  judge:
    provider: openai
    model: gpt-4
""")
        
        manager = ConfigManager()
        config = manager.get_app_config(config_path=config_file, warn_if_missing=False)
        
        assert config is not None
        assert config.defaults.generator.provider == "anthropic"
        assert config.defaults.judge.provider == "openai"

    def test_relative_config_path_resolution(self, tmp_path, monkeypatch):
        """Test that relative config paths resolve from CWD."""
        monkeypatch.chdir(tmp_path)
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: openai
""")
        
        from prompt_evaluator.config import load_prompt_evaluator_config
        
        # Load with relative path
        config = load_prompt_evaluator_config(
            config_path=Path("config.yaml"),
            warn_if_missing=False
        )
        
        assert config is not None
        assert config.defaults.generator.provider == "openai"

    def test_missing_config_path_emits_error(self, tmp_path):
        """Test that missing config path raises FileNotFoundError."""
        from prompt_evaluator.config import load_prompt_evaluator_config
        
        nonexistent = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_prompt_evaluator_config(config_path=nonexistent, warn_if_missing=False)

    def test_unreadable_config_path_emits_error(self, tmp_path):
        """Test that unreadable config path raises ValueError."""
        from prompt_evaluator.config import load_prompt_evaluator_config
        import os
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text("defaults: {}")
        
        # Make file unreadable
        os.chmod(config_file, 0o000)
        
        try:
            with pytest.raises(ValueError, match="not readable"):
                load_prompt_evaluator_config(config_path=config_file, warn_if_missing=False)
        finally:
            # Restore permissions for cleanup
            os.chmod(config_file, 0o644)

    def test_config_only_generator_section_falls_back_to_judge_defaults(self):
        """Test that config with only generator section uses judge defaults."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config_data = {
            "defaults": {
                "generator": {
                    "provider": "anthropic",
                    "model": "claude-3"
                }
                # No judge section specified
            }
        }
        
        config = PromptEvaluatorConfig(**config_data)
        
        # Generator should use specified values
        assert config.defaults.generator.provider == "anthropic"
        assert config.defaults.generator.model == "claude-3"
        
        # Judge should use defaults
        assert config.defaults.judge.provider == "openai"
        assert config.defaults.judge.model == "gpt-5.1"

    def test_config_only_judge_section_falls_back_to_generator_defaults(self):
        """Test that config with only judge section uses generator defaults."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config_data = {
            "defaults": {
                "judge": {
                    "provider": "anthropic",
                    "model": "claude-3"
                }
                # No generator section specified
            }
        }
        
        config = PromptEvaluatorConfig(**config_data)
        
        # Judge should use specified values
        assert config.defaults.judge.provider == "anthropic"
        assert config.defaults.judge.model == "claude-3"
        
        # Generator should use defaults
        assert config.defaults.generator.provider == "openai"
        assert config.defaults.generator.model == "gpt-5.1"


class TestProviderPrecedenceLogic:
    """Tests for provider precedence logic in CLI commands."""

    @staticmethod
    def _apply_precedence_logic(generator_provider, judge_provider, provider, config):
        """Helper to apply provider precedence logic as done in CLI."""
        final_generator_provider = generator_provider or provider
        if final_generator_provider is None:
            final_generator_provider = config.defaults.generator.provider
        
        final_judge_provider = judge_provider or provider
        if final_judge_provider is None:
            final_judge_provider = config.defaults.judge.provider
        
        return final_generator_provider, final_judge_provider

    def test_precedence_both_flags_none_uses_config(self, tmp_path):
        """Test that when both CLI flags are None, config values are used."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        # Simulate config with different providers
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic(None, None, None, config)
        
        assert final_gen == "anthropic"
        assert final_judge == "openai"

    def test_precedence_only_provider_flag_applies_to_both(self, tmp_path):
        """Test that --provider flag applies to both generator and judge."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic(None, None, "mock", config)
        
        # Both should use the --provider value
        assert final_gen == "mock"
        assert final_judge == "mock"

    def test_precedence_only_generator_provider_set(self, tmp_path):
        """Test that --generator-provider only affects generator."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic("mock", None, None, config)
        
        # Generator uses CLI flag, judge uses config
        assert final_gen == "mock"
        assert final_judge == "openai"

    def test_precedence_only_judge_provider_set(self, tmp_path):
        """Test that --judge-provider only affects judge."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic(None, "mock", None, config)
        
        # Generator uses config, judge uses CLI flag
        assert final_gen == "anthropic"
        assert final_judge == "mock"

    def test_precedence_both_specific_flags_set(self, tmp_path):
        """Test that specific flags override --provider and config."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic("mock", "mock", "anthropic", config)
        
        # Both should use specific flags
        assert final_gen == "mock"
        assert final_judge == "mock"

    def test_precedence_generator_provider_overrides_generic_provider(self, tmp_path):
        """Test that --generator-provider takes precedence over --provider."""
        from prompt_evaluator.config import PromptEvaluatorConfig
        
        config = PromptEvaluatorConfig(**{
            "defaults": {
                "generator": {"provider": "anthropic", "model": "claude-3"},
                "judge": {"provider": "openai", "model": "gpt-4"}
            }
        })
        
        final_gen, final_judge = self._apply_precedence_logic("mock", None, "anthropic", config)
        
        # Generator uses specific flag, judge uses --provider
        assert final_gen == "mock"
        assert final_judge == "anthropic"
