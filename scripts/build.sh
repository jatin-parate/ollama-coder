#!/bin/bash
set -e

echo "Building ollama-coder executable..."

# Create dist directory if it doesn't exist
mkdir -p dist

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec

# Build the executable
echo "Building executable..."
uv run pyinstaller --clean --onefile --name ollama-coder \
    --add-data "src/ollama_coder:ollama_coder" \
    --hidden-import=langchain_ollama \
    --hidden-import=langgraph \
    --hidden-import=langgraph.checkpoint.memory \
    --hidden-import=prompt_toolkit \
    --hidden-import=rich \
    --hidden-import=httpx \
    --hidden-import=pydantic \
    --hidden-import=langchain_core.messages \
    --hidden-import=langchain_core.tools \
    --console \
    src/ollama_coder/__main__.py

echo ""
echo "Build complete! Executable is located at: dist/ollama-coder"
echo ""
echo "To test the executable:"
echo "  ./dist/ollama-coder --help"
