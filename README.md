# Ollama Coder CLI

A beautiful terminal chat application using Ollama local instance.

## Architecture

This project follows a modular architecture with clear separation of concerns:

```
src/ollama_coder/
├── config/          # Configuration management
├── core/            # Core business logic
├── exceptions/      # Custom exceptions
├── memory/          # Project memory management
├── tools/           # Tool implementations
├── ui/              # UI rendering and interaction
└── utils/           # Utility functions
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Setup

1. Make sure Ollama is running locally:
```bash
ollama serve
```

2. Set the model environment variable:
```bash
export OLLAMA_MODEL=llama3
```

3. Run the chat:
```bash
uv run ollama-coder
```

## Features

- Elegant terminal UI using Rich
- Conversation history maintained
- Token-cache-aware Ollama settings with bounded history trimming
- Project-specific memory persisted to `memroy.db` in the directory where the CLI starts
- Stored facts are reused on later prompts and future sessions to guide tool and command choices
- Clear command to reset chat
- Exit with `exit` or `quit`

## Project Memory

The CLI now keeps a lightweight, project-local memory database named `memroy.db` in the current working directory when you start `ollama-coder`.

After a task completes successfully, the CLI extracts durable repository facts in the background and stores them in SQLite. These facts are meant to capture conventions that should influence future tasks, for example:

- which package manager the repo uses
- whether the project is a workspace/monorepo
- common test or build commands
- stable tooling conventions such as `uv`, `pytest`, `yarn`, or `pnpm`

When you send a new prompt later in the same session, or in a new session started from the same project root, the CLI searches `memroy.db` for relevant facts and injects them back into the agent context.

Example flow:

1. You tell the agent that the repo uses Yarn workspaces.
2. A task finishes successfully.
3. The CLI stores a fact such as "prefer yarn over npm for package scripts".
4. On a later prompt, the agent recalls that fact and prefers `yarn` commands automatically.

## Token Caching

The CLI now keeps Ollama models warm and trims older conversation history to preserve a stable prompt prefix, which improves KV-cache reuse across turns.

Exact-response caching is only enabled for deterministic runs. If `OLLAMA_TEMPERATURE` is non-zero, the app disables exact caching automatically and relies on Ollama's prefix KV cache instead.

Optional environment variables:

- `OLLAMA_EXACT_CACHE` default: `1`
- `OLLAMA_KEEP_ALIVE` default: `-1`
- `OLLAMA_NUM_CTX` default: `4096`
- `OLLAMA_HISTORY_TOKENS` default: `60%` of `OLLAMA_NUM_CTX`
- `OLLAMA_TEMPERATURE` default: `0`

Recommended server-side setting for strongest KV-cache reuse:

```bash
export OLLAMA_NUM_PARALLEL=1
```

This must be set for the Ollama server process, not just the CLI process.

Example:

```bash
export OLLAMA_MODEL=llama3
export OLLAMA_EXACT_CACHE=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_CTX=8192
export OLLAMA_HISTORY_TOKENS=4096
export OLLAMA_TEMPERATURE=0
uv run ollama-coder
```

## Extending the CLI

### Adding New Tools

1. Create a new tool class in `src/ollama_coder/tools/`:
```python
from tools.base import BaseTool

class MyTool(BaseTool):
    description = "My custom tool"
    
    def execute(self) -> str:
        # Your implementation
        return "Result"
    
    def validate(self) -> bool:
        return True
```

2. Register the tool in `tools/registry.py`:
```python
self.register_tool(MyTool)
```

### Adding New Memory Strategies

1. Create a new extractor in `src/ollama_coder/memory/`:
```python
class MyExtractor:
    def extract_facts(self, transcript: str) -> List[str]:
        # Your extraction logic
        return facts
```

2. Use the extractor in `core/agent.py`:
```python
extractor = MyExtractor()
facts = extractor.extract_facts(transcript)
```

## Development

### Running Tests

```bash
pytest src/ollama_coder/
```

### Code Quality

```bash
# Linting
ruff check src/ollama_coder/

# Type checking
mypy src/ollama_coder/
```

### Building

```bash
uv build
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Detailed architecture documentation
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Migration from monolithic to modular

## License

MIT
