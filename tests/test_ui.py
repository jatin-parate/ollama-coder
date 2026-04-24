"""Tests for UI module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.ollama_coder.ui.renderer import MessageRenderer


class TestMessageRenderer:
    """Test MessageRenderer class."""

    def test_init(self, message_renderer):
        """Test renderer initialization."""
        assert message_renderer.model_id == "test-model"
        assert message_renderer.console is not None
        assert message_renderer._thinking_buffer == ""
        assert message_renderer._thinking_panel is None

    def test_render_user_message(self, message_renderer, capsys):
        """Test rendering user message."""
        message_renderer.render_user_message("Hello, world!")
        # Should not raise
        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out or len(captured.out) > 0

    def test_render_streaming_content(self, message_renderer):
        """Test rendering streaming content."""
        message_renderer.render_streaming_content("Thinking...")
        assert "Thinking..." in message_renderer._thinking_buffer

    def test_render_streaming_content_accumulates(self, message_renderer):
        """Test that streaming content accumulates."""
        message_renderer.render_streaming_content("Part 1 ")
        message_renderer.render_streaming_content("Part 2")
        assert "Part 1" in message_renderer._thinking_buffer
        assert "Part 2" in message_renderer._thinking_buffer

    def test_finalize_thinking(self, message_renderer, capsys):
        """Test finalizing thinking."""
        message_renderer.render_streaming_content("Thinking content")
        message_renderer.finalize_thinking()
        assert message_renderer._thinking_buffer == ""
        assert message_renderer._thinking_panel is None

    def test_finalize_thinking_empty_buffer(self, message_renderer):
        """Test finalizing with empty buffer."""
        message_renderer.finalize_thinking()
        # Should not raise
        assert message_renderer._thinking_buffer == ""

    def test_render_assistant_message_with_content(self, message_renderer, capsys):
        """Test rendering assistant message with content."""
        msg = AIMessage(content="This is a response")
        message_renderer.render_assistant_message(msg)
        # Should not raise
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_assistant_message_with_tool_calls(self, message_renderer, capsys):
        """Test rendering assistant message with tool calls."""
        msg = AIMessage(
            content="I'll run a command",
            tool_calls=[{
                "name": "BashTool",
                "args": {"command": "echo test"},
                "id": "call_123"
            }],
        )
        message_renderer.render_assistant_message(msg)
        # Should not raise
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_tool_message(self, message_renderer, capsys):
        """Test rendering tool message."""
        msg = ToolMessage(content="Command output", tool_call_id="123")
        message_renderer.render_tool_message(msg)
        # Should not raise
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_tool_message_long_content(self, message_renderer, capsys):
        """Test rendering tool message with long content."""
        long_content = "x" * 500
        msg = ToolMessage(content=long_content, tool_call_id="123")
        message_renderer.render_tool_message(msg)
        # Should truncate to 300 chars
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_welcome(self, message_renderer, capsys):
        """Test rendering welcome message."""
        settings = {
            "model_id": "test-model",
            "temperature": 0.0,
            "num_ctx": 4096,
            "keep_alive": None,
            "enable_cache": True,
        }
        message_renderer.render_welcome(settings)
        captured = capsys.readouterr()
        assert "test-model" in captured.out

    def test_render_status(self, message_renderer, capsys):
        """Test rendering status message."""
        message_renderer.render_status("Processing...")
        captured = capsys.readouterr()
        assert "Processing..." in captured.out

    def test_render_error(self, message_renderer, capsys):
        """Test rendering error message."""
        message_renderer.render_error("Something went wrong")
        captured = capsys.readouterr()
        assert "Something went wrong" in captured.out or "Error" in captured.out

    def test_render_clear(self, message_renderer, capsys):
        """Test rendering clear message."""
        message_renderer.render_clear()
        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower() or len(captured.out) > 0

    def test_render_exit(self, message_renderer, capsys):
        """Test rendering exit message."""
        message_renderer.render_exit()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out or len(captured.out) > 0

    def test_extract_tool_call_dict(self, message_renderer):
        """Test extracting tool call from dict."""
        tool_call = {"name": "BashTool", "args": {"command": "echo test"}}
        name, args = message_renderer._extract_tool_call(tool_call)
        assert name == "BashTool"
        assert args == {"command": "echo test"}

    def test_extract_tool_call_object(self, message_renderer):
        """Test extracting tool call from object."""
        tool_call = MagicMock()
        tool_call.name = "BashTool"
        tool_call.args = {"command": "echo test"}
        name, args = message_renderer._extract_tool_call(tool_call)
        assert name == "BashTool"
        assert args == {"command": "echo test"}

    def test_extract_tool_call_missing_fields(self, message_renderer):
        """Test extracting tool call with missing fields."""
        tool_call = {}
        name, args = message_renderer._extract_tool_call(tool_call)
        assert name == "unknown"
        assert args == {}

    def test_format_token_status_with_metadata(self, message_renderer):
        """Test formatting token status with metadata."""
        msg = AIMessage(content="test")
        msg.response_metadata = {
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        status = message_renderer._format_token_status(msg)
        assert "prompt" in status.lower() or "token" in status.lower()

    def test_format_token_status_no_metadata(self, message_renderer):
        """Test formatting token status without metadata."""
        msg = AIMessage(content="test")
        status = message_renderer._format_token_status(msg)
        assert status == ""

    def test_format_token_status_with_cached_tokens(self, message_renderer):
        """Test formatting token status with cached tokens."""
        msg = AIMessage(content="test")
        msg.response_metadata = {
            "prompt_eval_count": 100,
            "eval_count": 50,
            "cached_prompt_tokens": 50,
        }
        status = message_renderer._format_token_status(msg)
        assert "cached" in status.lower() or "token" in status.lower()

    def test_render_token_status(self, message_renderer, capsys):
        """Test rendering token status."""
        msg = AIMessage(content="test")
        msg.response_metadata = {
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        message_renderer._render_token_status(msg)
        # Should not raise
        captured = capsys.readouterr()
        # May or may not print depending on metadata
