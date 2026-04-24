# Migration Guide: Monolithic to Modular Architecture

## Overview

This guide helps you understand the changes made during the refactoring from a monolithic CLI to a modular architecture.

## What Changed

### Before (Monolithic)
- All code in a single `cli.py` file (1286 lines)
- No clear separation of concerns
- Difficult to test individual components
- Hard to extend with new features

### After (Modular)
- Code organized into logical modules
- Clear separation of concerns
- Easy to test individual components
- Extensible with new features

## File Structure Changes

### New Files Created

| File | Purpose |
|------|---------|
| `config/settings.py` | Configuration management |
| `config/__init__.py` | Config module exports |
| `utils/logging.py` | Logging setup |
| `utils/__init__.py` | Utils module exports |
| `exceptions/__init__.py` | Custom exceptions |
| `tools/base.py` | Base tool interface |
| `tools/bash_tool.py` | Bash execution tool |
| `tools/file_tool.py` | File read/write tools |
| `tools/registry.py` | Tool registry |
| `tools/__init__.py` | Tools module exports |
| `memory/store.py` | Project memory persistence |
| `memory/extractor.py` | Fact extraction |
| `memory/__init__.py` | Memory module exports |
| `ui/renderer.py` | Message rendering |
| `ui/completer.py` | File completion |
| `ui/styles.py` | UI styling |
| `ui/__init__.py` | UI module exports |
| `core/agent.py` | Agent orchestration |
| `core/system_prompt.py` | System prompt generation |
| `core/context_builder.py` | Context building |
| `core/__init__.py` | Core module exports |

### Modified Files

| File | Changes |
|------|---------|
| `cli.py` | Refactored to use modular components (now ~300 lines) |
| `__init__.py` | Added CLI class export |

## API Changes

### Import Path Changes

**Before:**
```python
from ollama_coder.cli import OllamaChatCLI
```

**After:**
```python
from ollama_coder.cli import OllamaChatCLI  # Same import
```

The public API remains the same. Internal imports changed to use relative imports.

### Configuration

**Before:**
```python
# Settings loaded in __init__
self.model_id = os.environ.get("OLLAMA_MODEL", "llama3.2")
self.temperature = self._get_float_env("OLLAMA_TEMPERATURE", 0.0)
```

**After:**
```python
# Settings loaded from config module
self.ollama_settings, self.app_settings = load_settings(model_id)
```

### Tool Usage

**Before:**
```python
# Tools defined in cli.py
class BashTool(BaseModel):
    command: str
    def execute(self) -> str:
        ...
```

**After:**
```python
# Tools in tools/ directory
from tools.bash_tool import BashTool
from tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(BashTool)
```

### Memory Usage

**Before:**
```python
# Memory store in cli.py
class ProjectMemoryStore:
    def __init__(self, project_root):
        ...
```

**After:**
```python
# Memory in memory/ directory
from memory.store import ProjectMemoryStore
from memory.extractor import ProjectMemoryExtractor

store = ProjectMemoryStore(project_root)
extractor = ProjectMemoryExtractor(model)
```

### UI Usage

**Before:**
```python
# UI methods in cli.py
def display_message(self, message, sender: str = "user"):
    ...
```

**After:**
```python
# UI in ui/ directory
from ui.renderer import MessageRenderer

renderer = MessageRenderer(model_id="llama3.2")
renderer.render_user_message("Hello")
```

## Backward Compatibility

### Public API

The public API remains unchanged:
- `OllamaChatCLI` class works the same way
- `main()` function works the same way
- Command-line interface unchanged

### Environment Variables

All environment variables work the same:
- `OLLAMA_MODEL`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_NUM_CTX`
- `OLLAMA_HISTORY_TOKENS`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_EXACT_CACHE`

## Testing

### Unit Testing

**Before:**
```python
# Hard to test individual components
def test_bash_tool():
    # Had to mock the entire CLI class
    cli = OllamaChatCLI()
    # ...
```

**After:**
```python
# Easy to test individual components
def test_bash_tool():
    tool = BashTool(command="echo hello")
    result = tool.execute()
    assert "hello" in result

def test_memory_store():
    store = ProjectMemoryStore(project_root)
    store.upsert_facts(["test"], "context")
    results = store.search("test")
    assert "test" in results
```

### Integration Testing

```python
def test_full_workflow():
    cli = OllamaChatCLI()
    response = cli.agent.invoke("Test message")
    assert response is not None
```

## Migration Steps

### For Users

No migration needed. The CLI works the same way.

### For Developers

1. **Update imports** to use relative imports:
   ```python
   # Old
   from config.settings import load_settings
   
   # New
   from .config.settings import load_settings
   ```

2. **Use the new module structure**:
   ```python
   # Add new tools to tools/ directory
   # Add new memory strategies to memory/ directory
   # Add new UI components to ui/ directory
   ```

3. **Follow the architecture principles**:
   - Single responsibility
   - Dependency injection
   - Interface-based design
   - Separation of concerns

## Benefits

1. **Maintainability**: Clear module boundaries
2. **Testability**: Easy to test individual components
3. **Extensibility**: Easy to add new features
4. **Readability**: Code organized by concern
5. **Reusability**: Components can be reused in other contexts

## Troubleshooting

### Import Errors

If you see import errors, ensure you're using relative imports:
```python
# Correct
from .config.settings import load_settings

# Incorrect
from config.settings import load_settings
```

### Missing Dependencies

Ensure all modules are properly initialized:
```python
# Check that __init__.py files exist
ls src/ollama_coder/*/__init__.py
```

### Configuration Issues

Settings are now loaded from `config/settings.py`:
```python
from .config.settings import load_settings
settings = load_settings()
```

## Support

For questions or issues:
1. Check the architecture documentation
2. Review the unit tests
3. Consult the original monolithic implementation for reference
