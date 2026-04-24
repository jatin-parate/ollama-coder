# Ollama Coder CLI - Architecture Documentation

## Overview

This document describes the low-level design (LLD) and architecture of the Ollama Coder CLI application after refactoring for scalability and maintainability.

## Architecture Principles

1. **Single Responsibility Principle**: Each module has one clear responsibility
2. **Dependency Injection**: Components receive dependencies via constructor
3. **Interface-Based Design**: Clear interfaces for tools, memory, UI
4. **Separation of Concerns**: UI, business logic, and data layers separated
5. **Testability**: Each component can be unit tested independently
6. **Extensibility**: Easy to add new tools, memory strategies, UI renderers

## Directory Structure

```
src/ollama_coder/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m ollama_coder`
├── cli.py                   # Main CLI orchestration
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py          # Settings classes and loading
├── core/                    # Core business logic
│   ├── __init__.py
│   ├── agent.py             # LangGraph workflow orchestration
│   ├── system_prompt.py     # System prompt generation
│   └── context_builder.py   # Context building (memory, files)
├── exceptions/              # Custom exceptions
│   ├── __init__.py
│   └── errors.py            # Exception definitions
├── memory/                  # Project memory management
│   ├── __init__.py
│   ├── store.py             # SQLite persistence
│   └── extractor.py         # Fact extraction logic
├── tools/                   # Tool implementations
│   ├── __init__.py
│   ├── base.py              # Base tool interface
│   ├── bash_tool.py         # Bash execution tool
│   ├── file_tool.py         # File read/write tools
│   └── registry.py          # Tool registry
├── ui/                      # UI rendering and interaction
│   ├── __init__.py
│   ├── renderer.py          # Message rendering
│   ├── completer.py         # File completion
│   └── styles.py            # UI styling
└── utils/                   # Utility functions
    ├── __init__.py
    └── logging.py           # Logging setup
```

## Module Breakdown

### 1. Configuration Layer (`config/`)

**Purpose**: Centralized configuration management

**Key Classes**:
- `OllamaSettings`: Model-specific settings (temperature, context window, etc.)
- `AppSettings`: Application-wide settings (logging, timeouts, etc.)
- `load_settings()`: Factory function to load settings from environment

**Features**:
- Environment variable-based configuration
- Safe parsing with fallback defaults
- Validation of numeric values

### 2. Tools Layer (`tools/`)

**Purpose**: Tool execution and management

**Key Classes**:
- `BaseTool`: Abstract base class for all tools
- `BashTool`: Executes bash commands
- `ReadFileTool`: Reads file contents
- `WriteFileTool`: Creates/edits/appends files
- `ToolRegistry`: Centralized tool registration and discovery

**Features**:
- Interface-based design for extensibility
- Tool validation before execution
- Security checks for file operations
- Automatic tool discovery via registry

### 3. Memory Layer (`memory/`)

**Purpose**: Project memory persistence and retrieval

**Key Classes**:
- `ProjectMemoryStore`: SQLite-based persistence
- `ProjectMemoryExtractor`: Extracts durable facts from conversations

**Features**:
- Keyword-based fact search
- Automatic fact extraction from conversations
- Heuristic fallback for common patterns
- Timestamp-based ranking

### 4. UI Layer (`ui/`)

**Purpose**: User interface rendering and interaction

**Key Classes**:
- `MessageRenderer`: Renders messages to console
- `FileCompleter`: File path completion
- `UIStyles`: UI styling configuration

**Features**:
- Rich text formatting with panels
- Token usage status display
- File completion with @ prefix
- Custom keybindings (Tab, Escape)

### 5. Core Layer (`core/`)

**Purpose**: Core business logic and orchestration

**Key Classes**:
- `AgentExecutor`: LangGraph workflow orchestration
- `SystemPromptBuilder`: System prompt generation
- `ContextBuilder`: Message context building

**Features**:
- LangGraph workflow management
- Message trimming for context budget
- Memory context injection
- Tool execution coordination

### 6. Utils Layer (`utils/`)

**Purpose**: Utility functions

**Key Classes**:
- `setup_logging()`: Logging configuration

**Features**:
- File-based logging
- Dependency warning suppression
- Consistent log format

### 7. Exceptions Layer (`exceptions/`)

**Purpose**: Custom exception hierarchy

**Key Classes**:
- `OllamaCoderError`: Base exception
- `ConfigurationError`: Configuration issues
- `ToolExecutionError`: Tool execution failures
- `MemoryError`: Memory operations
- `ValidationError`: Input validation
- `FileOperationError`: File operations

## Data Flow

```
User Input → CLI → AgentExecutor → LangGraph Workflow
                                    ↓
                            ┌───────┴───────┐
                            ↓               ↓
                       Chat Node        Tool Node
                            ↓               ↓
                    Context Builder    Tool Registry
                            ↓               ↓
                    System Prompt      Tool Execution
                            ↓               ↓
                    Memory Context      Tool Results
                            ↓
                       Model Response
                            ↓
                    Message Renderer
                            ↓
                       User Display
```

## Key Design Patterns

### 1. Dependency Injection

Components receive dependencies via constructor:

```python
agent = AgentExecutor(
    model=chat_model,
    tool_registry=tool_registry,
    context_builder=context_builder,
    memory=memory_saver,
    project_memory=project_memory,
    project_memory_extractor=extractor,
)
```

### 2. Strategy Pattern

Different memory extraction strategies can be implemented:

```python
class ProjectMemoryExtractor:
    def extract_facts(self, transcript: str) -> List[str]:
        # Can be extended with different strategies
        pass
```

### 3. Registry Pattern

Tools are registered and discovered via the registry:

```python
registry = ToolRegistry()
registry.register_tool(BashTool)
registry.register_tool(ReadFileTool)
registry.register_tool(WriteFileTool)
```

### 4. Builder Pattern

System prompts are built incrementally:

```python
builder = SystemPromptBuilder()
prompt = builder.build(memory_context)
```

## Extensibility Points

### Adding New Tools

1. Create a new tool class extending `BaseTool`
2. Implement `execute()` and `validate()` methods
3. Register in `ToolRegistry`

### Adding New Memory Strategies

1. Implement a new extractor class
2. Register in `AgentExecutor`

### Adding New UI Renderers

1. Implement a new renderer class
2. Register in `MessageRenderer`

## Testing Strategy

### Unit Tests

Each module can be tested independently:

```python
# Test tools
def test_bash_tool_execute():
    tool = BashTool(command="echo hello")
    result = tool.execute()
    assert "hello" in result

# Test memory
def test_memory_store_search():
    store = ProjectMemoryStore(project_root)
    store.upsert_facts(["test fact"], "context")
    results = store.search("test")
    assert "test fact" in results

# Test renderer
def test_renderer_user_message():
    renderer = MessageRenderer()
    # Test rendering logic
```

### Integration Tests

Test the full workflow:

```python
def test_full_workflow():
    cli = OllamaChatCLI()
    response = cli.agent.invoke("What files are in the project?")
    assert response is not None
```

## Performance Considerations

1. **Caching**: Response caching for exact matches
2. **Message Trimming**: Dynamic history trimming based on token budget
3. **Background Memory Updates**: Memory persistence in background threads
4. **File Size Limits**: Maximum file content size to prevent memory issues

## Security Considerations

1. **File Path Validation**: All file operations validated against current directory
2. **Command Timeout**: Bash commands have 30-second timeout
3. **Tool Validation**: All tools validate parameters before execution
4. **Input Sanitization**: User input sanitized before processing

## Future Enhancements

1. **Plugin System**: Dynamic tool loading
2. **Multiple Memory Stores**: Support for different backends
3. **Advanced UI**: TUI with panels and splits
4. **Remote Execution**: Execute tools on remote servers
5. **Collaborative Memory**: Shared project memory across team
