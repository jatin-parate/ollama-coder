.PHONY: build build-mac build-linux build-win clean install

# Default build target
build: build-mac

# Build for macOS (current platform)
build-mac:
	uv run pyinstaller --clean --onefile --name ollama-coder \
		--add-data "src/ollama_coder:ollama_coder" \
		--hidden-import=langchain_ollama \
		--hidden-import=langgraph \
		--hidden-import=langgraph.checkpoint.memory \
		--hidden-import=prompt_toolkit \
		--hidden-import=rich \
		--hidden-import=httpx \
		--hidden-import=pydantic \
		src/ollama_coder/__main__.py

# Build for Linux
build-linux:
	uv run pyinstaller --clean --onefile --name ollama-coder \
		--add-data "src/ollama_coder:ollama_coder" \
		--hidden-import=langchain_ollama \
		--hidden-import=langgraph \
		--hidden-import=langgraph.checkpoint.memory \
		--hidden-import=prompt_toolkit \
		--hidden-import=rich \
		--hidden-import=httpx \
		--hidden-import=pydantic \
		--strip \
		src/ollama_coder/__main__.py

# Build for Windows
build-win:
	uv run pyinstaller --clean --onefile --name ollama-coder.exe \
		--add-data "src/ollama_coder;ollama_coder" \
		--hidden-import=langchain_ollama \
		--hidden-import=langgraph \
		--hidden-import=langgraph.checkpoint.memory \
		--hidden-import=prompt_toolkit \
		--hidden-import=rich \
		--hidden-import=httpx \
		--hidden-import=pydantic \
		--console \
		src/ollama_coder/__main__.py

# Clean build artifacts
clean:
	rm -rf build dist *.spec
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install the package in development mode
install:
	uv pip install -e .

# Run the application (for testing)
run:
	uv run python -m ollama_coder.cli
