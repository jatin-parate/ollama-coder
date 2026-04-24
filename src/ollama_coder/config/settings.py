"""Application settings and configuration management."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OllamaSettings:
    """Settings for Ollama model configuration."""

    model_id: str
    temperature: float = 0.0
    num_ctx: int = 4096
    history_token_budget: int = 1024
    keep_alive: Optional[str] = None
    enable_cache: bool = True

    @classmethod
    def from_env(cls, model_id: Optional[str] = None) -> "OllamaSettings":
        """Create settings from environment variables."""
        model = model_id or os.environ.get("OLLAMA_MODEL", "llama3.2")
        temperature = cls._get_float("OLLAMA_TEMPERATURE", 0.0)
        num_ctx = cls._get_int("OLLAMA_NUM_CTX", 4096)
        history_tokens = cls._get_int("OLLAMA_HISTORY_TOKENS", max(1024, int(num_ctx * 0.6)))
        keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
        enable_cache = os.environ.get("OLLAMA_EXACT_CACHE", "1") != "0"

        # Disable cache if temperature is non-deterministic
        if enable_cache and temperature != 0:
            enable_cache = False

        return cls(
            model_id=model,
            temperature=temperature,
            num_ctx=num_ctx,
            history_token_budget=min(history_tokens, num_ctx),
            keep_alive=keep_alive if keep_alive != "-1" else None,
            enable_cache=enable_cache,
        )

    @staticmethod
    def _get_int(env_name: str, default: int) -> int:
        """Parse integer environment variable with safe fallback."""
        raw = os.environ.get(env_name)
        if raw is None:
            return default
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError("Must be positive")
            return value
        except ValueError:
            return default

    @staticmethod
    def _get_float(env_name: str, default: float) -> float:
        """Parse float environment variable with safe fallback."""
        raw = os.environ.get(env_name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default


@dataclass
class AppSettings:
    """Application-wide settings."""

    log_level: int = 20  # INFO
    log_file: Path = Path("/tmp/ollama_coder.log")
    max_tool_retries: int = 3
    file_write_timeout: int = 30
    max_memory_facts: int = 6
    max_transcript_size: int = 12000
    max_file_content_size: int = 100000  # 100KB


def load_settings(model_id: Optional[str] = None) -> tuple[OllamaSettings, AppSettings]:
    """Load all settings from environment and defaults."""
    ollama_settings = OllamaSettings.from_env(model_id)
    app_settings = AppSettings()
    return ollama_settings, app_settings
