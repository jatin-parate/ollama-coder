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
- Token-cache-aware Ollama settings with bounded history trimming
- Clear command to reset chat
- Exit with `exit` or `quit`

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
