"""Logging configuration for Ollama Coder."""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: int = logging.DEBUG,
    log_file: Optional[Path] = None,
    suppress_warnings: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        log_level: The minimum log level to record.
        log_file: Optional file path to write logs to.
        suppress_warnings: Whether to suppress verbose warnings from dependencies.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure handlers
    handlers = []
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Setup root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Suppress verbose logging from dependencies
    if suppress_warnings:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Suppress langchain token counting fallback warning
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Using fallback GPT-2 tokenizer for token counting",
        category=UserWarning,
        module="langchain_core.language_models.base",
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
