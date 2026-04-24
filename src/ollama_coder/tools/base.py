"""Base tool interface and utilities."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """Abstract base class for all tools."""

    description: ClassVar[str] = "Base tool"

    class Config:
        """Pydantic configuration."""
        extra = "forbid"

    @abstractmethod
    def execute(self) -> str:
        """Execute the tool and return the result."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the tool parameters before execution."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return self.model_dump()
