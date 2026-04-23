# Ollama Coder CLI

A beautiful terminal chat application using Ollama local instance.

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
- Clear command to reset chat
- Exit with `exit` or `quit`
