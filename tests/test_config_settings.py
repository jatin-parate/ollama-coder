"""Tests for configuration settings module."""

import os
from pathlib import Path

import pytest

from src.ollama_coder.config.settings import OllamaSettings, AppSettings, load_settings


class TestOllamaSettings:
    """Test OllamaSettings class."""

    def test_create_with_defaults(self):
        """Test creating OllamaSettings with default values."""
        settings = OllamaSettings(model_id="llama3.2")
        assert settings.model_id == "llama3.2"
        assert settings.temperature == 0.0
        assert settings.num_ctx == 4096
        assert settings.history_token_budget == 1024
        assert settings.keep_alive is None
        assert settings.enable_cache is True

    def test_create_with_custom_values(self):
        """Test creating OllamaSettings with custom values."""
        settings = OllamaSettings(
            model_id="custom-model",
            temperature=0.7,
            num_ctx=2048,
            history_token_budget=512,
            keep_alive="5m",
            enable_cache=False,
        )
        assert settings.model_id == "custom-model"
        assert settings.temperature == 0.7
        assert settings.num_ctx == 2048
        assert settings.history_token_budget == 512
        assert settings.keep_alive == "5m"
        assert settings.enable_cache is False

    def test_from_env_with_defaults(self, monkeypatch):
        """Test loading from environment with defaults."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)

        settings = OllamaSettings.from_env()
        assert settings.model_id == "llama3.2"
        assert settings.temperature == 0.0
        assert settings.num_ctx == 4096

    def test_from_env_with_custom_values(self, monkeypatch):
        """Test loading from environment with custom values."""
        monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")

        settings = OllamaSettings.from_env()
        assert settings.model_id == "custom-model"
        assert settings.temperature == 0.5
        assert settings.num_ctx == 2048

    def test_cache_disabled_with_non_zero_temperature(self, monkeypatch):
        """Test that cache is disabled when temperature is non-zero."""
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.7")
        monkeypatch.setenv("OLLAMA_EXACT_CACHE", "1")

        settings = OllamaSettings.from_env()
        assert settings.temperature == 0.7
        assert settings.enable_cache is False

    def test_history_token_budget_capped_at_num_ctx(self, monkeypatch):
        """Test that history token budget is capped at num_ctx."""
        monkeypatch.setenv("OLLAMA_NUM_CTX", "1000")
        monkeypatch.setenv("OLLAMA_HISTORY_TOKENS", "5000")

        settings = OllamaSettings.from_env()
        assert settings.history_token_budget == 1000

    def test_keep_alive_none_when_minus_one(self, monkeypatch):
        """Test that keep_alive is None when set to -1."""
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "-1")

        settings = OllamaSettings.from_env()
        assert settings.keep_alive is None

    def test_keep_alive_preserved(self, monkeypatch):
        """Test that keep_alive is preserved when not -1."""
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")

        settings = OllamaSettings.from_env()
        assert settings.keep_alive == "5m"


class TestAppSettings:
    """Test AppSettings class."""

    def test_create_with_defaults(self):
        """Test creating AppSettings with default values."""
        settings = AppSettings()
        assert settings.log_level == 20  # INFO level
        assert settings.log_file == Path("/tmp/ollama_coder.log")
        assert settings.max_file_content_size == 100000

    def test_create_with_custom_values(self):
        """Test creating AppSettings with custom values."""
        settings = AppSettings(
            log_level="DEBUG",
            log_file="/tmp/app.log",
            max_file_content_size=50000,
        )
        assert settings.log_level == "DEBUG"
        assert settings.log_file == "/tmp/app.log"
        assert settings.max_file_content_size == 50000


class TestLoadSettings:
    """Test load_settings function."""

    def test_load_settings_returns_tuple(self, monkeypatch):
        """Test that load_settings returns a tuple of (OllamaSettings, AppSettings)."""
        monkeypatch.setenv("OLLAMA_MODEL", "test-model")

        ollama_settings, app_settings = load_settings("test-model")
        assert isinstance(ollama_settings, OllamaSettings)
        assert isinstance(app_settings, AppSettings)

    def test_load_settings_with_model_id(self, monkeypatch):
        """Test load_settings with explicit model_id."""
        ollama_settings, app_settings = load_settings("custom-model")
        assert ollama_settings.model_id == "custom-model"

    def test_load_settings_with_none_model_id(self, monkeypatch):
        """Test load_settings with None model_id uses environment."""
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")

        ollama_settings, app_settings = load_settings(None)
        assert ollama_settings.model_id == "env-model"
