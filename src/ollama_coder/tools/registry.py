"""Tool registry for managing available tools."""

import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool as LangchainTool
from pydantic import BaseModel

from .base import BaseTool
from .bash_tool import BashTool
from .file_tool import ReadFileTool, WriteFileTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._langchain_tools: Dict[str, Type[LangchainTool]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        self.register_tool(BashTool)
        self.register_tool(ReadFileTool)
        self.register_tool(WriteFileTool)

    def register_tool(self, tool_class: Type[BaseTool]) -> None:
        """Register a new tool class."""
        tool_name = tool_class.__name__
        self._tools[tool_name] = tool_class
        logger.debug(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool by name."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Tool not found for unregistration: {tool_name}")

    def get_tool(self, tool_name: str) -> Type[BaseTool]:
        """Get a tool class by name."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool not found: {tool_name}")
        return self._tools[tool_name]

    def create_tool(self, tool_name: str, **kwargs: Any) -> BaseTool:
        """Create an instance of a tool by name."""
        tool_class = self.get_tool(tool_name)
        return tool_class(**kwargs)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_langchain_tools(self) -> List[Type[LangchainTool]]:
        """Get all tools as Langchain tool classes for model binding."""
        return list(self._tools.values())

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the JSON schema for a tool."""
        tool_class = self.get_tool(tool_name)
        return tool_class.model_json_schema()

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get JSON schemas for all registered tools."""
        return {name: tool.model_json_schema() for name, tool in self._tools.items()}

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
