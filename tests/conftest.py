"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from src.ollama_coder.config.settings import OllamaSettings, AppSettings
from src.ollama_coder.core.agent import AgentExecutor
from src.ollama_coder.core.context_builder import ContextBuilder
from src.ollama_coder.core.system_prompt import SystemPromptBuilder
from src.ollama_coder.memory.extractor import ProjectMemoryExtractor
from src.ollama_coder.memory.store import ProjectMemoryStore
from src.ollama_coder.tools.registry import ToolRegistry
from src.ollama_coder.ui.renderer import MessageRenderer


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def ollama_settings():
    """Create test Ollama settings."""
    return OllamaSettings(
        model_id="test-model",
        temperature=0.0,
        num_ctx=4096,
        history_token_budget=1024,
        keep_alive=None,
        enable_cache=True,
    )


@pytest.fixture
def app_settings():
    """Create test app settings."""
    return AppSettings(
        log_level="DEBUG",
        log_file=None,
        max_file_content_size=100000,
    )


@pytest.fixture
def mock_chat_model():
    """Create a mock ChatOllama model."""
    model = MagicMock(spec=ChatOllama)
    model.invoke = MagicMock()
    model.stream = MagicMock()
    model.get_num_tokens_from_messages = MagicMock(return_value=100)
    model.bind_tools = MagicMock(return_value=model)
    return model


@pytest.fixture
def tool_registry():
    """Create a tool registry with default tools."""
    return ToolRegistry()


@pytest.fixture
def project_memory_store(temp_project_dir):
    """Create a project memory store."""
    return ProjectMemoryStore(temp_project_dir)


@pytest.fixture
def memory_extractor(mock_chat_model):
    """Create a memory extractor."""
    return ProjectMemoryExtractor(mock_chat_model)


@pytest.fixture
def system_prompt_builder(temp_project_dir):
    """Create a system prompt builder."""
    return SystemPromptBuilder(temp_project_dir)


@pytest.fixture
def context_builder(system_prompt_builder, project_memory_store):
    """Create a context builder."""
    return ContextBuilder(
        system_prompt_builder=system_prompt_builder,
        memory_store=project_memory_store,
        max_file_content_size=100000,
    )


@pytest.fixture
def memory_saver():
    """Create a memory saver."""
    return MemorySaver()


@pytest.fixture
def agent_executor(mock_chat_model, tool_registry, context_builder, memory_saver, project_memory_store, memory_extractor):
    """Create an agent executor."""
    return AgentExecutor(
        model=mock_chat_model,
        tool_registry=tool_registry,
        context_builder=context_builder,
        memory=memory_saver,
        project_memory=project_memory_store,
        project_memory_extractor=memory_extractor,
        history_token_budget=1024,
    )


@pytest.fixture
def message_renderer():
    """Create a message renderer."""
    return MessageRenderer(model_id="test-model")


@pytest.fixture
def sample_test_file(temp_project_dir):
    """Create a sample test file."""
    test_file = temp_project_dir / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    return test_file
