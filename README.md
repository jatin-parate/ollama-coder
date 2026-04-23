# Ollama Coder CLI

A terminal chat application using Ollama and LangChain/LangGraph with an elegant Rich-based UI.

## Features

- Elegant terminal UI similar to qwen-coder-cli and claude-cli
- Uses LangGraph for agentic workflow management
- Supports any Ollama model via environment variable
- Conversation history management
- Clean markdown rendering for model responses

## Requirements

- Python 3.12+
- Ollama running locally

## Installation

```bash
uv sync
```

## Usage

1. Make sure Ollama is running locally:

```bash
ollama serve
```

2. Pull a model (if not already pulled):

```bash
ollama pull llama3.2
```

3. Run the CLI with the model ID:

```bash
export OLLAMA_MODEL=llama3.2
python -m ollama_coder.cli
```

Or use the installed command:

```bash
export OLLAMA_MODEL=llama3.2
ollama-coder
```

## Commands

- `exit`, `quit`, `q` - Exit the application
- `clear` - Clear the conversation history

## Development

```bash
# Run linter
uv run ruff check .

# Format code
uv run ruff format .
```
