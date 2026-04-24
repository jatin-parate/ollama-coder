"""Custom exceptions for Ollama Coder."""


class OllamaCoderError(Exception):
    """Base exception for Ollama Coder errors."""


class ConfigurationError(OllamaCoderError):
    """Raised when there's a configuration issue."""


class ToolExecutionError(OllamaCoderError):
    """Raised when a tool fails to execute."""


class MemoryError(OllamaCoderError):
    """Raised when there's an issue with project memory operations."""


class ValidationError(OllamaCoderError):
    """Raised when input validation fails."""


class FileOperationError(OllamaCoderError):
    """Raised when file operations fail."""
