"""Tests for core modules (agent, context_builder, system_prompt)."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.ollama_coder.core.agent import AgentExecutor
from src.ollama_coder.core.context_builder import ContextBuilder
from src.ollama_coder.core.system_prompt import SystemPromptBuilder
from src.ollama_coder.exceptions import ToolExecutionError


class TestSystemPromptBuilder:
    """Test SystemPromptBuilder class."""

    def test_build_base_prompt(self, system_prompt_builder):
        """Test building base prompt."""
        prompt = system_prompt_builder._build_base_prompt()
        assert "expert AI coding assistant" in prompt
        assert "BashTool" in prompt
        assert "ReadFileTool" in prompt
        assert "WriteFileTool" in prompt

    def test_build_with_memory_context(self, system_prompt_builder):
        """Test building prompt with memory context."""
        memory_context = "\n\n## Stored Project Facts\n- Fact 1\n- Fact 2"
        prompt = system_prompt_builder.build(memory_context)
        assert "Fact 1" in prompt
        assert "Fact 2" in prompt

    def test_build_includes_working_directory(self, system_prompt_builder):
        """Test that prompt includes working directory."""
        prompt = system_prompt_builder._build_base_prompt()
        assert "Current Working Directory" in prompt

    def test_build_with_custom_project_root(self, temp_project_dir):
        """Test building with custom project root."""
        builder = SystemPromptBuilder(temp_project_dir)
        prompt = builder._build_base_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestContextBuilder:
    """Test ContextBuilder class."""

    def test_build_context_empty_message(self, context_builder):
        """Test building context with empty message."""
        system_prompt, messages = context_builder.build_context("", [])
        assert isinstance(system_prompt, str)
        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_build_context_with_message(self, context_builder):
        """Test building context with a message."""
        system_prompt, messages = context_builder.build_context("Hello", [])
        assert isinstance(system_prompt, str)
        assert isinstance(messages, list)

    def test_build_context_with_history(self, context_builder):
        """Test building context with message history."""
        history = [
            HumanMessage(content="First message"),
            AIMessage(content="Response"),
        ]
        system_prompt, messages = context_builder.build_context("Second message", history)
        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_process_message_with_files_no_files(self, context_builder):
        """Test processing message without file references."""
        message = "This is a regular message"
        result = context_builder._process_message_with_files(message)
        assert result == message

    def test_process_message_with_files_existing_file(self, context_builder, sample_test_file):
        """Test processing message with existing file reference."""
        message = f"Check this file: {sample_test_file}"
        result = context_builder._process_message_with_files(message)
        assert str(sample_test_file) in result
        assert "def hello():" in result

    def test_process_message_with_files_nonexistent_file(self, context_builder):
        """Test processing message with nonexistent file."""
        message = "Check /nonexistent/file.txt"
        result = context_builder._process_message_with_files(message)
        # Should return original message if file doesn't exist
        assert isinstance(result, str)

    def test_process_message_with_files_size_limit(self, context_builder, temp_project_dir):
        """Test that large files are skipped."""
        large_file = temp_project_dir / "large.txt"
        large_file.write_text("x" * 200000)  # Larger than default limit
        
        message = f"Check {large_file}"
        result = context_builder._process_message_with_files(message)
        # Large file should be skipped
        assert "x" * 1000 not in result

    def test_build_memory_context_no_store(self, system_prompt_builder):
        """Test building memory context without store."""
        builder = ContextBuilder(
            system_prompt_builder=system_prompt_builder,
            memory_store=None,
        )
        context = builder._build_memory_context("query")
        assert "No project memory configured" in context

    def test_build_memory_context_with_store(self, context_builder, project_memory_store):
        """Test building memory context with store."""
        project_memory_store.upsert_facts(["Test fact"], "context")
        context = context_builder._build_memory_context("test")
        assert "Stored Project Facts" in context

    def test_extract_active_query_from_messages(self, context_builder):
        """Test extracting active query from messages."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="Response"),
            HumanMessage(content="Second question"),
        ]
        query = context_builder.extract_active_query(messages)
        assert query == "Second question"

    def test_extract_active_query_empty_messages(self, context_builder):
        """Test extracting query from empty messages."""
        query = context_builder.extract_active_query([])
        assert query == ""

    def test_process_messages_prepends_system_message(self, context_builder):
        """Test that system message is prepended."""
        messages = context_builder._process_messages(
            [],
            "User message",
            "System prompt",
        )
        assert len(messages) > 0
        assert isinstance(messages[0], SystemMessage)


class TestAgentExecutor:
    """Test AgentExecutor class."""

    def test_build_workflow(self, agent_executor):
        """Test that workflow is built correctly."""
        assert agent_executor.app is not None
        assert agent_executor.workflow is not None

    def test_build_chat_context(self, agent_executor):
        """Test building chat context."""
        state = {"messages": [HumanMessage(content="Hello")]}
        system_prompt, messages = agent_executor._build_chat_context(state)
        assert isinstance(system_prompt, str)
        assert isinstance(messages, list)

    def test_build_chat_context_empty_state(self, agent_executor):
        """Test building context with empty state."""
        state = {"messages": []}
        system_prompt, messages = agent_executor._build_chat_context(state)
        assert isinstance(system_prompt, str)
        assert isinstance(messages, list)

    def test_extract_tool_call_field_dict(self, agent_executor):
        """Test extracting field from dict tool call."""
        tool_call = {"name": "BashTool", "args": {"command": "echo test"}}
        name = agent_executor._extract_tool_call_field(tool_call, "name")
        assert name == "BashTool"

    def test_extract_tool_call_field_object(self, agent_executor):
        """Test extracting field from object tool call."""
        tool_call = MagicMock()
        tool_call.name = "BashTool"
        name = agent_executor._extract_tool_call_field(tool_call, "name")
        assert name == "BashTool"

    def test_extract_tool_call_field_default(self, agent_executor):
        """Test extracting field with default value."""
        tool_call = {}
        result = agent_executor._extract_tool_call_field(tool_call, "missing", "default")
        assert result == "default"

    def test_should_continue_with_tool_calls(self, agent_executor):
        """Test should_continue returns 'tools' when tool calls present."""
        msg = AIMessage(
            content="test",
            tool_calls=[{
                "name": "BashTool",
                "args": {"command": "echo test"},
                "id": "call_123"
            }]
        )
        state = {"messages": [msg]}
        result = agent_executor._should_continue(state)
        assert result == "tools"

    def test_should_continue_without_tool_calls(self, agent_executor):
        """Test should_continue returns END when no tool calls."""
        msg = AIMessage(content="test")
        state = {"messages": [msg]}
        result = agent_executor._should_continue(state)
        assert result == "__end__"

    def test_trim_messages(self, agent_executor):
        """Test message trimming."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
        ]
        trimmed = agent_executor._trim_messages(messages)
        assert isinstance(trimmed, list)

    def test_execute_tool_call_success(self, agent_executor):
        """Test successful tool execution."""
        tool_call = {"name": "BashTool", "args": {"command": "echo test"}}
        result = agent_executor._execute_tool_call(tool_call)
        assert result is not None
        assert "test" in result

    def test_execute_tool_call_unknown_tool(self, agent_executor):
        """Test execution with unknown tool raises error."""
        tool_call = {"name": "UnknownTool", "args": {}}
        with pytest.raises(ToolExecutionError):
            agent_executor._execute_tool_call(tool_call)

    def test_execute_tool_call_validation_failure(self, agent_executor):
        """Test execution with validation failure raises error."""
        tool_call = {"name": "BashTool", "args": {"command": ""}}
        with pytest.raises(ToolExecutionError):
            agent_executor._execute_tool_call(tool_call)

    def test_tool_node_no_tool_calls(self, agent_executor):
        """Test tool node with no tool calls."""
        msg = AIMessage(content="test")
        state = {"messages": [msg]}
        result = agent_executor._tool_node(state)
        assert result == {"messages": []}

    def test_tool_node_with_tool_calls(self, agent_executor):
        """Test tool node with tool calls."""
        msg = AIMessage(
            content="test",
            tool_calls=[{
                "name": "BashTool",
                "args": {"command": "echo test"},
                "id": "call_123"
            }],
        )
        state = {"messages": [msg]}
        result = agent_executor._tool_node(state)
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_tool_node_handles_execution_error(self, agent_executor):
        """Test tool node handles execution errors gracefully."""
        msg = AIMessage(
            content="test",
            tool_calls=[{
                "name": "UnknownTool",
                "args": {},
                "id": "call_456"
            }],
        )
        state = {"messages": [msg]}
        result = agent_executor._tool_node(state)
        assert "messages" in result
        # Should have error message
        assert len(result["messages"]) > 0

    def test_queue_memory_update(self, agent_executor):
        """Test queuing memory update."""
        messages = [
            AIMessage(content="Response"),
        ]
        # Should not raise
        agent_executor.queue_memory_update("User input", messages)

    def test_queue_memory_update_empty_response(self, agent_executor):
        """Test queuing memory update with empty response."""
        messages = [
            AIMessage(content=""),
        ]
        # Should not raise
        agent_executor.queue_memory_update("User input", messages)

    def test_attach_token_metadata(self, agent_executor):
        """Test attaching token metadata."""
        messages = [HumanMessage(content="test")]
        response = AIMessage(content="response")
        agent_executor._attach_token_metadata(messages, response)
        # Should have response_metadata
        assert hasattr(response, "response_metadata")
