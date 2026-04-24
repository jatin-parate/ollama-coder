"""Tests for tools module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ollama_coder.tools.bash_tool import BashTool
from src.ollama_coder.tools.file_tool import ReadFileTool, WriteFileTool
from src.ollama_coder.tools.registry import ToolRegistry
from src.ollama_coder.exceptions import ToolExecutionError


class TestBashTool:
    """Test BashTool class."""

    def test_execute_simple_command(self):
        """Test executing a simple bash command."""
        tool = BashTool(command="echo hello")
        result = tool.execute()
        assert "hello" in result

    def test_execute_command_with_error(self):
        """Test executing a command that produces an error."""
        tool = BashTool(command="ls /nonexistent/path")
        result = tool.execute()
        assert "Error" in result or "cannot access" in result.lower()

    def test_execute_command_timeout(self):
        """Test that command timeout is handled."""
        tool = BashTool(command="sleep 60")
        result = tool.execute()
        assert "timed out" in result.lower()

    def test_cd_command_changes_directory(self, temp_project_dir):
        """Test that cd command changes directory."""
        tool = BashTool(command=f"cd {temp_project_dir}")
        result = tool.execute()
        assert "Changed directory" in result

    def test_validate_empty_command(self):
        """Test validation fails for empty command."""
        tool = BashTool(command="")
        assert tool.validate() is False

    def test_validate_whitespace_command(self):
        """Test validation fails for whitespace-only command."""
        tool = BashTool(command="   ")
        assert tool.validate() is False

    def test_validate_valid_command(self):
        """Test validation passes for valid command."""
        tool = BashTool(command="echo test")
        assert tool.validate() is True


class TestReadFileTool:
    """Test ReadFileTool class."""

    def test_read_existing_file(self, sample_test_file):
        """Test reading an existing file."""
        tool = ReadFileTool(file_path=str(sample_test_file))
        result = tool.execute()
        assert "def hello():" in result
        assert "return 'world'" in result

    def test_read_nonexistent_file(self):
        """Test reading a nonexistent file."""
        tool = ReadFileTool(file_path="/nonexistent/file.txt")
        result = tool.execute()
        assert "Error" in result or "not found" in result.lower()

    def test_read_directory_fails(self, temp_project_dir):
        """Test that reading a directory fails."""
        tool = ReadFileTool(file_path=str(temp_project_dir))
        result = tool.execute()
        assert "Error" in result or "not a file" in result.lower()

    def test_validate_empty_path(self):
        """Test validation fails for empty path."""
        tool = ReadFileTool(file_path="")
        assert tool.validate() is False

    def test_validate_valid_path(self, sample_test_file):
        """Test validation passes for valid path."""
        tool = ReadFileTool(file_path=str(sample_test_file))
        assert tool.validate() is True


class TestWriteFileTool:
    """Test WriteFileTool class."""

    def test_create_new_file(self, temp_project_dir, monkeypatch):
        """Test creating a new file."""
        # Mock os.chdir to avoid actual directory changes
        monkeypatch.setattr("os.getcwd", lambda: str(temp_project_dir))
        monkeypatch.setattr("os.chdir", lambda x: None)
        
        file_path = temp_project_dir / "new_file.txt"
        tool = WriteFileTool(
            file_path=str(file_path),
            new_string="Hello, World!",
        )
        result = tool.execute()
        assert "Success" in result
        assert file_path.exists()
        assert file_path.read_text() == "Hello, World!"

    def test_create_nested_directories(self, temp_project_dir, monkeypatch):
        """Test creating nested directories."""
        monkeypatch.setattr("os.getcwd", lambda: str(temp_project_dir))
        monkeypatch.setattr("os.chdir", lambda x: None)
        
        file_path = temp_project_dir / "nested" / "dir" / "file.txt"
        tool = WriteFileTool(
            file_path=str(file_path),
            new_string="Content",
        )
        result = tool.execute()
        assert "Success" in result
        assert file_path.exists()

    def test_edit_existing_file(self, sample_test_file, monkeypatch):
        """Test editing an existing file."""
        monkeypatch.setattr("os.getcwd", lambda: str(sample_test_file.parent))
        monkeypatch.setattr("os.chdir", lambda x: None)
        
        original_content = sample_test_file.read_text()
        tool = WriteFileTool(
            file_path=str(sample_test_file),
            old_string="def hello():\n    return 'world'",
            new_string="def hello():\n    return 'universe'",
        )
        result = tool.execute()
        assert "Success" in result
        assert "universe" in sample_test_file.read_text()
        assert "world" not in sample_test_file.read_text()

    def test_edit_with_nonunique_string(self, temp_project_dir):
        """Test that editing with non-unique string fails."""
        file_path = temp_project_dir / "test.txt"
        file_path.write_text("hello hello hello")
        tool = WriteFileTool(
            file_path=str(file_path),
            old_string="hello",
            new_string="goodbye",
        )
        result = tool.execute()
        assert "Error" in result or "not unique" in result.lower()

    def test_edit_with_missing_string(self, sample_test_file):
        """Test that editing with missing string fails."""
        tool = WriteFileTool(
            file_path=str(sample_test_file),
            old_string="nonexistent string",
            new_string="replacement",
        )
        result = tool.execute()
        assert "Error" in result or "not found" in result.lower()

    def test_append_to_file(self, sample_test_file, monkeypatch):
        """Test appending to an existing file."""
        monkeypatch.setattr("os.getcwd", lambda: str(sample_test_file.parent))
        monkeypatch.setattr("os.chdir", lambda x: None)
        
        original_content = sample_test_file.read_text()
        tool = WriteFileTool(
            file_path=str(sample_test_file),
            new_string="\n\ndef goodbye():\n    pass",
            append=True,
        )
        result = tool.execute()
        assert "Success" in result
        new_content = sample_test_file.read_text()
        assert original_content in new_content
        assert "def goodbye():" in new_content

    def test_append_with_old_string_fails(self, sample_test_file):
        """Test that append with old_string fails."""
        tool = WriteFileTool(
            file_path=str(sample_test_file),
            old_string="something",
            new_string="something else",
            append=True,
        )
        result = tool.execute()
        assert "Error" in result

    def test_validate_empty_path(self):
        """Test validation fails for empty path."""
        tool = WriteFileTool(file_path="", new_string="content")
        assert tool.validate() is False

    def test_validate_append_with_old_string(self, sample_test_file):
        """Test validation fails for append with old_string."""
        tool = WriteFileTool(
            file_path=str(sample_test_file),
            old_string="old",
            new_string="new",
            append=True,
        )
        assert tool.validate() is False

    def test_validate_valid_create(self, temp_project_dir):
        """Test validation passes for valid create."""
        file_path = temp_project_dir / "new.txt"
        tool = WriteFileTool(file_path=str(file_path), new_string="content")
        assert tool.validate() is True


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_default_tools_registered(self):
        """Test that default tools are registered."""
        registry = ToolRegistry()
        assert registry.has_tool("BashTool")
        assert registry.has_tool("ReadFileTool")
        assert registry.has_tool("WriteFileTool")

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert "BashTool" in tools
        assert "ReadFileTool" in tools
        assert "WriteFileTool" in tools

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool_class = registry.get_tool("BashTool")
        assert tool_class == BashTool

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool raises KeyError."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get_tool("NonexistentTool")

    def test_create_tool(self):
        """Test creating a tool instance."""
        registry = ToolRegistry()
        tool = registry.create_tool("BashTool", command="echo test")
        assert isinstance(tool, BashTool)
        assert tool.command == "echo test"

    def test_register_custom_tool(self):
        """Test registering a custom tool."""
        registry = ToolRegistry()
        registry.register_tool(BashTool)
        assert registry.has_tool("BashTool")

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.unregister_tool("BashTool")
        assert not registry.has_tool("BashTool")

    def test_get_tool_schema(self):
        """Test getting tool schema."""
        registry = ToolRegistry()
        schema = registry.get_tool_schema("BashTool")
        assert isinstance(schema, dict)
        assert "properties" in schema or "type" in schema

    def test_get_all_schemas(self):
        """Test getting all tool schemas."""
        registry = ToolRegistry()
        schemas = registry.get_all_schemas()
        assert isinstance(schemas, dict)
        assert "BashTool" in schemas
        assert "ReadFileTool" in schemas
        assert "WriteFileTool" in schemas
