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
Tests for ConfigManager to ensure proper config caching and loading.
"""

from pathlib import Path

import pytest

from prompt_evaluator.config import ConfigManager


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_config_manager_initialization(self):
        """Test that ConfigManager initializes with empty caches."""
        manager = ConfigManager()
        
        assert manager._app_config_cache is None
        assert manager._app_config_path_cache is None
        assert manager._api_config_cache is None
        assert manager._api_config_path_cache is None

    def test_app_config_caching(self, tmp_path, monkeypatch):
        """Test that app config is cached and reused."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: openai
    model: gpt-4
  judge:
    provider: anthropic
    model: claude-3
""")
        
        manager = ConfigManager()
        
        # First load
        config1 = manager.get_app_config(config_path=config_file, warn_if_missing=False)
        assert config1 is not None
        assert config1.defaults.generator.provider == "openai"
        assert config1.defaults.judge.provider == "anthropic"
        
        # Second load should return cached config
        config2 = manager.get_app_config(config_path=config_file, warn_if_missing=False)
        assert config2 is config1  # Same object, not just equal

    def test_app_config_cache_invalidation_on_path_change(self, tmp_path):
        """Test that cache is invalidated when path changes."""
        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("""
defaults:
  generator:
    provider: openai
    model: gpt-4
""")
        
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("""
defaults:
  generator:
    provider: anthropic
    model: claude-3
""")
        
        manager = ConfigManager()
        
        # Load first config
        config1 = manager.get_app_config(config_path=config_file1, warn_if_missing=False)
        assert config1.defaults.generator.provider == "openai"
        
        # Load second config should invalidate cache
        config2 = manager.get_app_config(config_path=config_file2, warn_if_missing=False)
        assert config2.defaults.generator.provider == "anthropic"
        assert config2 is not config1

    def test_app_config_cache_with_relative_and_absolute_paths(self, tmp_path, monkeypatch):
        """Test that relative and absolute paths to the same file use the same cache."""
        monkeypatch.chdir(tmp_path)
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: openai
""")
        
        manager = ConfigManager()
        
        # Load with relative path
        config1 = manager.get_app_config(config_path=Path("config.yaml"), warn_if_missing=False)
        
        # Load with absolute path should return cached config
        config2 = manager.get_app_config(config_path=config_file, warn_if_missing=False)
        assert config2 is config1

    def test_api_config_caching(self, tmp_path, monkeypatch):
        """Test that API config is cached and reused."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
api_key: test-key-123
model_name: gpt-4
""")
        
        manager = ConfigManager()
        
        # First load
        config1 = manager.get_api_config(config_file_path=config_file)
        assert config1.api_key == "test-key-123"
        assert config1.model_name == "gpt-4"
        
        # Second load should return cached config
        config2 = manager.get_api_config(config_file_path=config_file)
        assert config2 is config1  # Same object

    def test_api_config_cache_invalidation_on_path_change(self, tmp_path):
        """Test that API config cache is invalidated when path changes."""
        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("""
api_key: key1
model_name: gpt-4
""")
        
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("""
api_key: key2
model_name: gpt-5
""")
        
        manager = ConfigManager()
        
        # Load first config
        config1 = manager.get_api_config(config_file_path=config_file1)
        assert config1.api_key == "key1"
        
        # Load second config should invalidate cache
        config2 = manager.get_api_config(config_file_path=config_file2)
        assert config2.api_key == "key2"
        assert config2 is not config1

    def test_clear_cache(self, tmp_path):
        """Test that clear_cache removes all cached configs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
api_key: test-key
defaults:
  generator:
    provider: openai
""")
        
        manager = ConfigManager()
        
        # Load configs
        app_config = manager.get_app_config(config_path=config_file, warn_if_missing=False)
        api_config = manager.get_api_config(config_file_path=config_file)
        
        assert manager._app_config_cache is not None
        assert manager._api_config_cache is not None
        
        # Clear cache
        manager.clear_cache()
        
        assert manager._app_config_cache is None
        assert manager._app_config_path_cache is None
        assert manager._api_config_cache is None
        assert manager._api_config_path_cache is None

    def test_app_config_without_path_returns_none(self, tmp_path, monkeypatch):
        """Test that app config without path returns None (no default in cwd)."""
        # Change to a directory without a default config file
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROMPT_EVALUATOR_CONFIG", raising=False)
        
        manager = ConfigManager()
        
        # Should return None when no config file exists and warn_if_missing=False
        config = manager.get_app_config(config_path=None, warn_if_missing=False)
        assert config is None

    def test_api_config_cache_with_none_path(self, monkeypatch):
        """Test that API config with None path is cached."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        
        manager = ConfigManager()
        
        # Load without file path (uses env vars)
        config1 = manager.get_api_config(config_file_path=None)
        assert config1.api_key == "env-key"
        
        # Second load should return cached config
        config2 = manager.get_api_config(config_file_path=None)
        assert config2 is config1

    def test_app_config_respects_default_file_discovery(self, tmp_path, monkeypatch):
        """Test that config manager respects default file discovery via env var."""
        monkeypatch.chdir(tmp_path)
        
        # Create a config file
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text("""
defaults:
  generator:
    provider: anthropic
    model: claude-3
""")
        
        # Set environment variable to point to this file
        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(config_file))
        
        manager = ConfigManager()
        
        # First call with None should find config via env var
        config1 = manager.get_app_config(config_path=None, warn_if_missing=False)
        assert config1 is not None
        assert config1.defaults.generator.provider == "anthropic"
        
        # Second call should return cached config
        config2 = manager.get_app_config(config_path=None, warn_if_missing=False)
        assert config2 is config1

    def test_app_config_cache_changes_when_env_var_changes(self, tmp_path, monkeypatch):
        """Test that cache is invalidated when env var points to different file."""
        monkeypatch.chdir(tmp_path)
        
        # Create two config files
        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("""
defaults:
  generator:
    provider: openai
""")
        
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("""
defaults:
  generator:
    provider: anthropic
""")
        
        manager = ConfigManager()
        
        # Load first config via env var
        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(config_file1))
        config1 = manager.get_app_config(config_path=None, warn_if_missing=False)
        assert config1.defaults.generator.provider == "openai"
        
        # Change env var to point to second config
        monkeypatch.setenv("PROMPT_EVALUATOR_CONFIG", str(config_file2))
        config2 = manager.get_app_config(config_path=None, warn_if_missing=False)
        assert config2.defaults.generator.provider == "anthropic"
        assert config2 is not config1  # Should be a different instance
