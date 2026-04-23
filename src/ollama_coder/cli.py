"""Main CLI application for Ollama chat."""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Tuple

from langchain_core.caches import BaseCache
from langchain_core.messages import trim_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Setup logging - only to file, not to console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/ollama_coder.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from other libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)


class SessionResponseCache(BaseCache):
    """Simple in-memory cache for exact prompt and model matches."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def lookup(self, prompt: str, llm_string: str) -> Any:
        """Return a cached generation list for an exact request match."""
        return self._cache.get((prompt, llm_string))

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """Store generation results for later exact-match reuse."""
        self._cache[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        """Clear cached generations."""
        self._cache.clear()


class BashTool(BaseModel):
    """Tool to execute bash commands."""

    command: str = Field(
        description="The bash command to execute. Use cd <path> to change directories."
    )

    def execute(self) -> str:
        """Execute the bash command and return the result."""
        try:
            # Handle cd command specially
            if self.command.strip().startswith("cd "):
                path = self.command.strip()[3:].strip()
                new_dir = Path(path).resolve()
                new_dir.mkdir(parents=True, exist_ok=True)
                os.chdir(new_dir)
                return f"Changed directory to {new_dir}"
            
            # Execute other commands
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nError: {result.stderr}"
            return output or "Command executed successfully"
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description": "Execute a bash command in the current directory. "
                          f"Current directory: {Path.cwd()}"
        }


class ReadFileTool(BaseModel):
    """Tool to read file contents."""

    file_path: str = Field(
        description="The path to the file to read. Can be relative or absolute path."
    )

    def execute(self) -> str:
        """Read and return the file contents."""
        try:
            file_path = Path(self.file_path).resolve()
            
            # Security check: ensure the file is within the workspace
            if not file_path.exists():
                return f"Error: File not found: {self.file_path}"
            
            if not file_path.is_file():
                return f"Error: Path is not a file: {self.file_path}"
            
            # Read the file
            content = file_path.read_text()
            logger.info(f"Read file: {file_path}, size: {len(content)} bytes")
            return content
        except PermissionError:
            return f"Error: Permission denied reading file: {self.file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description": "Read the contents of a file. "
                          f"Current directory: {Path.cwd()}"
        }


class WriteFileTool(BaseModel):
    """Tool to write/edit file contents by replacing unique strings, creating new files, or appending."""

    file_path: str = Field(
        description="The path to the file to edit, create, or append to. Can be relative or absolute path. "
                   "Must be within the current working directory."
    )
    old_string: str = Field(
        default="",
        description="The exact string to find and replace. Must be unique in the file. "
                   "Can be multi-line. Include enough context to make it unique. "
                   "Leave empty to create a new file or append to an existing file."
    )
    new_string: str = Field(
        description="The new string to replace with, the complete file content if creating a new file, "
                   "or the text to append if old_string is empty and file exists. Can be multi-line."
    )
    append: bool = Field(
        default=False,
        description="If True and file exists, append new_string to the end of the file instead of replacing. "
                   "Only used when old_string is empty."
    )

    def execute(self) -> str:
        """Replace old_string with new_string, create a new file, or append to a file."""
        try:
            file_path = Path(self.file_path).resolve()
            current_dir = Path.cwd().resolve()
            
            # Security check: ensure the file is within the current directory
            try:
                file_path.relative_to(current_dir)
            except ValueError:
                logger.error(f"Attempted to write outside current directory: {file_path}")
                return (
                    f"Error: Cannot write to {self.file_path}. "
                    f"File must be within the current working directory: {current_dir}"
                )
            
            # Case 1: Append to an existing file
            if self.append and file_path.exists():
                logger.info(f"Appending to file: {file_path}")
                
                if self.old_string:
                    return (
                        "Error: Cannot use old_string when appending. "
                        "Leave old_string empty when append=True."
                    )
                
                # Read the current file
                content = file_path.read_text()
                original_size = len(content)
                
                # Append the new string
                new_content = content + self.new_string
                
                # Write back to file
                file_path.write_text(new_content)
                logger.info(f"Successfully appended to file: {file_path}")
                
                return (
                    f"Success: Text appended to {self.file_path}\n"
                    f"Appended {len(self.new_string)} characters.\n"
                    f"File size changed from {original_size} to {len(new_content)} bytes."
                )
            
            # Case 2: Create a new file
            elif not file_path.exists():
                logger.info(f"Creating new file: {file_path}")
                
                if self.old_string:
                    return (
                        f"Error: Cannot use old_string when creating a new file. "
                        f"File {self.file_path} does not exist. "
                        f"Leave old_string empty to create a new file with the content in new_string."
                    )
                
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the new file
                file_path.write_text(self.new_string)
                logger.info(f"Successfully created file: {file_path}, size: {len(self.new_string)} bytes")
                
                return (
                    f"Success: New file created at {self.file_path}\n"
                    f"File size: {len(self.new_string)} bytes\n"
                    f"Location: {file_path}"
                )
            
            # Case 3: Edit an existing file
            else:
                logger.info(f"Editing existing file: {file_path}")
                
                if not self.old_string:
                    return (
                        f"Error: old_string is required when editing an existing file. "
                        f"File {self.file_path} already exists. "
                        f"Provide the exact string to replace in old_string, or set append=True to append instead."
                    )
                
                # Read the file
                content = file_path.read_text()
                logger.info(f"File size: {len(content)} bytes")
                logger.info(f"Looking for string: {self.old_string[:100]}")
                
                # Check if old_string exists in the file
                if self.old_string not in content:
                    logger.warning(f"String not found in file: {file_path}")
                    return (
                        f"Error: The string to replace was not found in {self.file_path}.\n"
                        f"Please provide the exact string including proper indentation and context.\n"
                        f"String to find:\n{repr(self.old_string)}"
                    )
                
                # Count occurrences to ensure uniqueness
                occurrences = content.count(self.old_string)
                logger.info(f"Found {occurrences} occurrence(s) of the string")
                
                if occurrences > 1:
                    logger.warning(f"String is not unique: {occurrences} occurrences found")
                    return (
                        f"Error: The string to replace is not unique in the file. "
                        f"Found {occurrences} occurrences.\n"
                        f"Please provide a more unique string with additional context "
                        f"(e.g., include surrounding lines or function names).\n"
                        f"Current string:\n{repr(self.old_string)}"
                    )
                
                # Replace the string
                new_content = content.replace(self.old_string, self.new_string)
                
                # Write back to file
                file_path.write_text(new_content)
                logger.info(f"Successfully wrote to file: {file_path}")
                
                return (
                    f"Success: File {self.file_path} has been updated.\n"
                    f"Replaced {len(self.old_string)} characters with {len(self.new_string)} characters.\n"
                    f"File size changed from {len(content)} to {len(new_content)} bytes."
                )
        
        except PermissionError:
            logger.error(f"Permission denied writing to file: {self.file_path}")
            return f"Error: Permission denied writing to file: {self.file_path}"
        except Exception as e:
            logger.error(f"Error writing to file: {str(e)}")
            return f"Error writing to file: {str(e)}"

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description": "Edit an existing file by replacing a unique string, create a new file, or append to a file. "
                          "For editing: old_string must be unique in the file. "
                          "For creating: leave old_string empty and provide file content in new_string. "
                          "For appending: set append=True, leave old_string empty, and provide text to append in new_string. "
                          "Files must be created/edited within the current working directory. "
                          f"Current directory: {Path.cwd()}"
        }


class ProjectMemoryStore:
    """Persist and retrieve durable project facts from a local SQLite database."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.db_path = self.project_root / "memroy.db"
        self._initialize()

    def _initialize(self) -> None:
        """Create the database schema if it does not exist."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL UNIQUE,
                    keywords TEXT NOT NULL,
                    source_context TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.commit()

    def search(self, query: str, limit: int = 6) -> list[str]:
        """Return the most relevant stored facts for a query."""
        normalized_query = query.strip()
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT fact, keywords, updated_at FROM facts ORDER BY updated_at DESC"
            ).fetchall()

        if not normalized_query:
            return [fact for fact, _, _ in rows[:limit]]

        query_terms = self._tokenize(normalized_query)

        ranked_rows: list[tuple[int, float, str]] = []
        recent_facts: list[str] = []
        for fact, keywords, updated_at in rows:
            recent_facts.append(fact)
            keyword_terms = set(filter(None, keywords.split(" ")))
            overlap = len(query_terms & keyword_terms)
            if overlap == 0:
                continue
            ranked_rows.append((overlap, updated_at, fact))

        ranked_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_facts = [fact for _, _, fact in ranked_rows[:limit]]
        for fact in recent_facts:
            if len(selected_facts) >= limit:
                break
            if fact not in selected_facts:
                selected_facts.append(fact)
        return selected_facts

    def upsert_facts(self, facts: list[str], source_context: str) -> int:
        """Insert new facts or refresh timestamps for existing facts."""
        cleaned_facts = []
        seen_facts: set[str] = set()
        for fact in facts:
            normalized_fact = self._normalize_fact(fact)
            if not normalized_fact or normalized_fact in seen_facts:
                continue
            seen_facts.add(normalized_fact)
            cleaned_facts.append(normalized_fact)

        if not cleaned_facts:
            return 0

        now = time.time()
        with sqlite3.connect(self.db_path) as connection:
            for fact in cleaned_facts:
                keywords = " ".join(sorted(self._tokenize(fact)))
                connection.execute(
                    """
                    INSERT INTO facts (fact, keywords, source_context, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(fact) DO UPDATE SET
                        keywords = excluded.keywords,
                        source_context = excluded.source_context,
                        updated_at = excluded.updated_at
                    """,
                    (fact, keywords, source_context[:2000], now, now),
                )
            connection.commit()

        return len(cleaned_facts)

    def _normalize_fact(self, fact: str) -> str:
        """Normalize fact strings before persisting them."""
        normalized_fact = re.sub(r"\s+", " ", fact).strip()
        if len(normalized_fact) < 12:
            return ""
        return normalized_fact[:400]

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for simple keyword matching."""
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", text.lower())
            if token not in {"that", "this", "with", "from", "into", "should", "would", "there", "their", "about"}
        }


class OllamaChatCLI:
    """A terminal chat application using Ollama."""

    def __init__(self, model_id: str = None):
        """Initialize the chat application."""
        self.model_id = model_id or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.temperature = self._get_float_env("OLLAMA_TEMPERATURE", 0.0)
        self.enable_exact_cache = os.environ.get("OLLAMA_EXACT_CACHE", "1") != "0"
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
        self.num_ctx = self._get_int_env("OLLAMA_NUM_CTX", 4096)
        self.history_token_budget = min(
            self._get_int_env("OLLAMA_HISTORY_TOKENS", max(1024, int(self.num_ctx * 0.6))),
            self.num_ctx,
        )

        if self.enable_exact_cache and self.temperature != 0:
            logger.warning(
                "Disabling exact response cache because OLLAMA_TEMPERATURE=%s is non-deterministic.",
                self.temperature,
            )
            self.enable_exact_cache = False

        self.response_cache = SessionResponseCache() if self.enable_exact_cache else None
        self.console = Console()
        self.memory = MemorySaver()
        self.chat_history: List[Tuple[str, str]] = []
        self.project_root = Path.cwd().resolve()
        self.current_dir = self.project_root
        self.project_memory = ProjectMemoryStore(self.project_root)

        # Keep the model resident and cap the dynamic suffix so Ollama can reuse
        # the same prompt prefix across turns.
        self.base_model = ChatOllama(
            model=self.model_id,
            temperature=self.temperature,
            cache=self.response_cache,
            keep_alive=self.keep_alive,
            num_ctx=self.num_ctx,
        )
        self.model = self.base_model.bind_tools([BashTool, ReadFileTool, WriteFileTool])

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Setup prompt toolkit session with custom completer
        self.session = PromptSession()
        self.completer = FileCompleter(self)
        self._setup_keybindings()

    def _get_int_env(self, env_name: str, default: int) -> int:
        """Parse integer environment variables with a safe fallback."""
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            return default

        try:
            parsed_value = int(raw_value)
        except ValueError:
            logger.warning(f"Invalid integer for {env_name}: {raw_value!r}. Using default {default}.")
            return default

        if parsed_value <= 0:
            logger.warning(f"Non-positive integer for {env_name}: {parsed_value}. Using default {default}.")
            return default

        return parsed_value

    def _get_float_env(self, env_name: str, default: float) -> float:
        """Parse float environment variables with a safe fallback."""
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            return default

        try:
            return float(raw_value)
        except ValueError:
            logger.warning(f"Invalid float for {env_name}: {raw_value!r}. Using default {default}.")
            return default

    def _trim_messages_for_context(self, messages: list[Any]) -> list[Any]:
        """Trim dynamic conversation history to preserve a stable cached prefix."""
        if len(messages) <= 1:
            return messages

        try:
            trimmed_messages = trim_messages(
                messages,
                max_tokens=self.history_token_budget,
                token_counter=self.base_model,
                strategy="last",
                start_on="human",
                include_system=False,
            )
        except Exception as e:
            logger.warning(f"Falling back to untrimmed message history: {e}")
            return messages

        logger.info(
            "Trimmed message history from %s to %s messages using a %s-token budget",
            len(messages),
            len(trimmed_messages),
            self.history_token_budget,
        )
        return trimmed_messages

    def _build_system_prompt(self) -> str:
        """Build a comprehensive system prompt for the agent."""
        cwd = Path.cwd()
        return f"""You are an expert AI coding assistant with deep knowledge of software development, architecture, and best practices.

## Your Capabilities
You have access to three powerful tools:
1. **BashTool**: Execute bash commands to explore the codebase, run tests, build projects, etc.
2. **ReadFileTool**: Read file contents to understand existing code and structure
3. **WriteFileTool**: Create new files, edit existing files by replacing unique strings, or append content

## Your Approach to Complex Tasks
When given a complex task (like building a React component, refactoring code, or implementing features):

1. **Understand the Context**: 
   - Use BashTool to explore the project structure (tree, ls, find commands)
   - Read relevant files to understand existing patterns, conventions, and dependencies
   - Identify the tech stack and coding standards

2. **Plan Your Approach**:
   - Think through the task step-by-step
   - Identify all files that need to be created or modified
   - Consider dependencies and how changes affect other parts of the codebase

3. **Execute Strategically**:
   - Make targeted, focused edits using unique string replacements
   - Create new files with proper structure and formatting
   - Test your changes when possible using BashTool

4. **Verify Your Work**:
   - Read back modified files to ensure correctness
   - Run relevant tests or linters if available
   - Provide a summary of changes made

## Best Practices
- Always read files before editing to understand context
- Use unique, multi-line strings when replacing code to avoid ambiguity
- Include proper imports, type hints, and documentation
- Follow the existing code style and conventions in the project
- When uncertain about exact formatting, read similar files first
- Make multiple targeted edits rather than one large replacement
- Provide clear explanations of what you're doing and why

## Current Working Directory
Your operations are restricted to: {cwd}

## Project Memory
- Before making decisions, consider any stored project facts that are included in the system prompt.
- Treat stored facts as durable project conventions unless the current repository state or the user explicitly contradicts them.
- When a stored fact says to prefer one tool or workflow, follow it consistently during the task.

## Important Notes
- You can only create/edit files within the current working directory
- Always verify file paths are correct before making changes
- Use the tools iteratively - read, plan, execute, verify
- For complex tasks, break them into smaller, manageable steps
- Communicate your progress and reasoning clearly"""

    def _build_memory_context(self, query: str) -> str:
        """Build the project-memory section for the current task."""
        relevant_facts = self.project_memory.search(query)
        if not relevant_facts:
            return "\n\n## Stored Project Facts\n- No stored project facts were found for this task."

        facts_text = "\n".join(f"- {fact}" for fact in relevant_facts)
        return f"\n\n## Stored Project Facts\n{facts_text}"

    def _extract_active_user_query(self, messages: list[Any]) -> str:
        """Return the most recent human message to drive memory retrieval."""
        for message in reversed(messages):
            if getattr(message, "type", None) == "human" and hasattr(message, "content"):
                return str(message.content)
        return ""

    def _extract_project_facts(self, transcript: str) -> list[str]:
        """Extract durable project facts from a finished turn."""
        extraction_prompt = (
            "Extract durable, project-specific facts from this completed coding task. "
            "Only keep facts that will help future tasks in the same repository, such as package managers, "
            "workspace tools, build commands, test commands, frameworks, important directories, or coding conventions. "
            "Ignore transient details. Return strict JSON with shape {\"facts\": [\"fact 1\", \"fact 2\"]}. "
            "Keep each fact under 140 characters and phrase it as an instruction or durable repository fact.\n\n"
            f"Transcript:\n{transcript[:12000]}"
        )

        try:
            extraction_response = self.base_model.invoke(extraction_prompt)
            content = getattr(extraction_response, "content", "")
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            payload = self._extract_json_object(str(content))
            facts = payload.get("facts", []) if isinstance(payload, dict) else []
            if isinstance(facts, list):
                normalized_facts = [str(fact).strip() for fact in facts if str(fact).strip()]
                if normalized_facts:
                    return normalized_facts
        except Exception as e:
            logger.warning(f"Project memory extraction failed, using heuristic fallback: {e}")

        return self._extract_project_facts_heuristic(transcript)

    def _extract_project_facts_heuristic(self, transcript: str) -> list[str]:
        """Fallback fact extraction for common repository conventions."""
        lowered = transcript.lower()
        heuristic_facts: list[str] = []

        if "yarn workspace" in lowered or "yarn workspaces" in lowered:
            heuristic_facts.append("This project uses Yarn workspaces; prefer yarn over npm for package scripts.")
        if "pnpm workspace" in lowered or "pnpm-workspace.yaml" in lowered:
            heuristic_facts.append("This project uses pnpm workspaces; prefer pnpm commands for package scripts.")
        if "package-lock.json" in lowered or "npm run" in lowered:
            heuristic_facts.append("Use npm commands for package scripts unless the repository shows another package manager.")
        if "poetry" in lowered or "poetry.lock" in lowered:
            heuristic_facts.append("This Python project uses Poetry for dependency and script management.")
        if "uv run" in lowered or "[tool.uv]" in lowered:
            heuristic_facts.append("This project uses uv for Python dependency management and command execution.")
        if "pytest" in lowered:
            heuristic_facts.append("Use pytest for this repository's Python test runs.")

        return heuristic_facts

    def _extract_json_object(self, content: str) -> dict[str, Any]:
        """Parse the first JSON object found in model output."""
        stripped_content = content.strip()
        if not stripped_content:
            return {}

        try:
            parsed = json.loads(stripped_content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped_content, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _queue_project_memory_update(self, user_input: str, new_messages: list[Any]) -> None:
        """Update the project memory in a background thread after a successful turn."""
        final_ai_messages = [
            message
            for message in new_messages
            if getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", None)
        ]
        if not final_ai_messages:
            return

        final_answer = getattr(final_ai_messages[-1], "content", "")
        if not str(final_answer).strip():
            return

        transcript_parts = [f"User request:\n{user_input}"]
        for message in new_messages:
            message_type = getattr(message, "type", type(message).__name__)
            message_content = getattr(message, "content", "")
            if not message_content:
                continue
            transcript_parts.append(f"{message_type.upper()}:\n{message_content}")

        transcript = "\n\n".join(transcript_parts)

        def _persist_memory() -> None:
            try:
                facts = self._extract_project_facts(transcript)
                stored_count = self.project_memory.upsert_facts(facts, transcript)
                logger.info("Stored %s project memory facts in %s", stored_count, self.project_memory.db_path)
            except Exception as e:
                logger.warning(f"Failed to persist project memory: {e}")

        threading.Thread(target=_persist_memory, daemon=True).start()

    def _setup_keybindings(self) -> None:
        """Setup custom key bindings for file suggestions."""
        kb = KeyBindings()

        @kb.add(Keys.Tab)
        def _(event):
            """Handle Tab for accepting file suggestions without submitting."""
            buffer = event.app.current_buffer
            # If there's a completion being shown, accept it
            if buffer.complete_state:
                # Get the current completion text
                current_completion = buffer.complete_state.current_completion
                if current_completion:
                    # Cancel the completion menu
                    buffer.cancel_completion()
                    # Insert the completion text
                    buffer.insert_text(current_completion.text)
            else:
                # No completion menu, start one
                buffer.start_completion(select_first=True)

        @kb.add(Keys.Escape)
        def _(event):
            """Handle Escape to cancel completions."""
            buffer = event.app.current_buffer
            buffer.cancel_completion()

        self.session.key_bindings = kb

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for chat."""
        workflow = StateGraph(MessagesState)

        # Define the chat node
        def chat_node(state: MessagesState):
            """Process user message and get model response."""
            from langchain_core.messages import HumanMessage, SystemMessage
            
            logger.info("=== CHAT NODE START ===")
            logger.info(f"State messages count: {len(state['messages'])}")
            
            # Extract file contents and prepend to message
            last_msg = state["messages"][-1]
            logger.info(f"Last message type: {type(last_msg)}")
            logger.info(f"Last message content: {last_msg.content[:100] if hasattr(last_msg, 'content') else 'N/A'}")
            
            processed_message = self._process_message_with_files(last_msg.content)
            logger.info(f"Processed message length: {len(processed_message)}")
            active_user_query = self._extract_active_user_query(state["messages"])
            effective_system_prompt = self._build_system_prompt() + self._build_memory_context(active_user_query)
            
            # Replace the last message with processed content
            # Only replace if it's a HumanMessage, otherwise keep as is
            if isinstance(last_msg, HumanMessage):
                messages = state["messages"][:-1] + [HumanMessage(content=processed_message)]
            else:
                # For ToolMessage or other types, keep as is
                messages = state["messages"]

            messages = self._trim_messages_for_context(messages)
            
            # Prepend system message if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=effective_system_prompt)] + messages
            else:
                messages = [SystemMessage(content=effective_system_prompt)] + messages[1:]
            
            logger.info(f"Messages prepared for model: {len(messages)}")
            
            # Invoke model with messages (system prompt is now in the messages)
            response = self.model.invoke(messages)
            logger.info(f"Model response type: {type(response)}")
            logger.info(f"Model response: {response}")
            if hasattr(response, "tool_calls"):
                logger.info(f"Tool calls in response: {response.tool_calls}")
            
            logger.info("=== CHAT NODE END ===")
            return {"messages": [response]}

        # Define the tool node
        def tool_node(state: MessagesState):
            """Execute tools requested by the model."""
            from langchain_core.messages import AIMessage, ToolMessage
            import uuid

            def _extract_tool_call_field(tool_call: Any, key: str, default: Any = None) -> Any:
                """Get a field from a tool call represented as dict or object."""
                if isinstance(tool_call, dict):
                    return tool_call.get(key, default)
                return getattr(tool_call, key, default)
            
            logger.info("=== TOOL NODE START ===")
            logger.info(f"State messages count: {len(state['messages'])}")
            
            messages = state["messages"]
            last_message = messages[-1]
            
            logger.info(f"Last message type: {type(last_message)}")
            logger.info(f"Is AIMessage: {isinstance(last_message, AIMessage)}")
            
            if hasattr(last_message, "tool_calls"):
                logger.info(f"Tool calls present: {last_message.tool_calls}")
                logger.info(f"Tool calls type: {type(last_message.tool_calls)}")
                logger.info(f"Tool calls length: {len(last_message.tool_calls)}")
                
                for i, tc in enumerate(last_message.tool_calls):
                    logger.info(f"Tool call {i}: {tc}")
                    logger.info(f"Tool call {i} type: {type(tc)}")
            else:
                logger.info("No tool_calls attribute")
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_results = []
                for tool_call in last_message.tool_calls:
                    logger.info(f"Processing tool call: {tool_call}")

                    tool_call_id = _extract_tool_call_field(tool_call, "id")
                    tool_name = _extract_tool_call_field(tool_call, "name")
                    args = _extract_tool_call_field(tool_call, "args", {})
                    logger.info(f"Extracted tool call - id: {tool_call_id}, name: {tool_name}, args: {args}")
                    
                    if tool_call_id is None:
                        logger.warning("tool_call_id is None, generating UUID")
                        tool_call_id = str(uuid.uuid4())
                    
                    logger.info(f"Final tool_call_id: {tool_call_id}")
                    logger.info(f"Final tool_name: {tool_name}")
                    
                    result = None
                    
                    # Handle BashTool
                    if tool_name == "BashTool":
                        logger.info("Executing BashTool")
                        command = args.get("command", "")
                        logger.info(f"Command: {command}")
                        
                        tool = BashTool(command=command)
                        result = tool.execute()
                        logger.info(f"Tool result: {result[:200]}")
                    
                    # Handle ReadFileTool
                    elif tool_name == "ReadFileTool":
                        logger.info("Executing ReadFileTool")
                        file_path = args.get("file_path", "")
                        logger.info(f"File path: {file_path}")
                        
                        tool = ReadFileTool(file_path=file_path)
                        result = tool.execute()
                        logger.info(f"Tool result: {result[:200]}")
                    
                    # Handle WriteFileTool
                    elif tool_name == "WriteFileTool":
                        logger.info("Executing WriteFileTool")
                        file_path = args.get("file_path", "")
                        old_string = args.get("old_string", "")
                        new_string = args.get("new_string", "")
                        append = args.get("append", False)
                        logger.info(f"File path: {file_path}")
                        logger.info(f"Old string length: {len(old_string)}")
                        logger.info(f"New string length: {len(new_string)}")
                        logger.info(f"Append mode: {append}")

                        tool = WriteFileTool(
                            file_path=file_path,
                            old_string=old_string,
                            new_string=new_string,
                            append=append,
                        )
                        result = tool.execute()
                        logger.info(f"Tool result: {result[:200]}")
                    
                    else:
                        logger.warning(f"Unknown tool: {tool_name}")
                        result = f"Unknown tool: {tool_name}"
                    
                    if result is not None:
                        tool_msg = ToolMessage(
                            content=result,
                            tool_call_id=tool_call_id,
                        )
                        logger.info("Created ToolMessage")
                        tool_results.append(tool_msg)
                
                logger.info(f"Tool results count: {len(tool_results)}")
                logger.info("=== TOOL NODE END ===")
                return {"messages": tool_results}
            
            logger.info("No tool calls to process")
            logger.info("=== TOOL NODE END ===")
            return {"messages": []}

        # Add the nodes
        workflow.add_node("chat", chat_node)
        workflow.add_node("tools", tool_node)

        # Define the entry point
        workflow.set_entry_point("chat")

        # Add conditional edges for tool calling
        def should_continue(state: MessagesState):
            """Determine if we should continue to tools or end."""
            messages = state["messages"]
            last_message = messages[-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return END

        workflow.add_conditional_edges(
            "chat",
            should_continue,
        )

        # Add edge from tools back to chat
        workflow.add_edge("tools", "chat")

        return workflow

    def _process_message_with_files(self, message: str) -> str:
        """Extract file paths from message and prepend their contents."""
        # Find all file paths in the message (paths starting with ./, ../, /, or @)
        # This pattern matches file paths that can appear anywhere in the message
        file_pattern = r'(?<!\w)([./@][^\s\'"]+|/[^\s\'"]+)'
        file_paths = re.findall(file_pattern, message)

        if not file_paths:
            return message

        # Process each file and extract its content
        file_contents = []
        for file_path in file_paths:
            # Clean up the path (remove @ prefix if present)
            clean_path = file_path.lstrip("@").strip("'\"")
            try:
                full_path = Path(clean_path).resolve()
                if full_path.exists() and full_path.is_file():
                    content = full_path.read_text()
                    file_contents.append(f"\n\nFile: {clean_path}\nContent:\n```\n{content}\n```")
            except Exception:
                continue

        if file_contents:
            return message + "".join(file_contents)
        return message

    def _get_file_suggestions(self, prefix: str) -> List[str]:
        """Get file suggestions based on current input prefix."""
        if not prefix.startswith("@"):
            return []

        # Get the search term after @
        search_term = prefix[1:].strip()
        current_dir = Path.cwd()

        # Find matching files
        suggestions = []
        for item in current_dir.iterdir():
            if item.is_file():
                name = item.name
                if search_term.lower() in name.lower():
                    suggestions.append(f"@{name}")

        return suggestions[:10]  # Limit to 10 suggestions

    def _display_suggestions(self, suggestions: List[str], index: int) -> None:
        """Display file suggestions in a nice format."""
        if not suggestions:
            return

        self.console.print("\n[dim]Select a file:[/dim]")
        for i, suggestion in enumerate(suggestions):
            if i == index:
                self.console.print(f"[bold cyan]> {suggestion}[/bold cyan]")
            else:
                self.console.print(f"  {suggestion}")

    def display_message(self, message, sender: str = "user") -> None:
        """Display a message with formatting."""
        if sender == "user":
            self.console.print(
                Panel(
                    str(message),
                    title="[bold green]You[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        elif sender == "assistant":
            if isinstance(message, str):
                self.console.print(
                    Panel(
                        Markdown(message),
                        title=f"[bold blue]{self.model_id}[/bold blue]",
                        title_align="left",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )
            else:
                # Handle tool messages
                self.console.print(
                    Panel(
                        str(message),
                        title="[bold yellow]Tool[/bold yellow]",
                        title_align="left",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )

    def get_user_input(self) -> str:
        """Get input from user with a nice prompt and file suggestions."""
        try:
            return self.session.prompt(
                "You ",
                completer=self.completer,
                complete_while_typing=True,
                style=Style.from_dict({
                    "prompt": "bold green",
                }),
            )
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "exit"

    def run(self) -> None:
        """Run the chat application loop."""
        # Print welcome message
        self.console.clear()
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold]Ollama Coder CLI[/bold]\n"
                    f"Model: [cyan]{self.model_id}[/cyan]\n"
                    f"Temperature: [cyan]{self.temperature}[/cyan]\n"
                    f"Context window: [cyan]{self.num_ctx}[/cyan] tokens\n"
                    f"Keep alive: [cyan]{self.keep_alive}[/cyan]\n"
                    f"Exact cache: [cyan]{'on' if self.enable_exact_cache else 'off'}[/cyan]\n"
                    f"Type 'exit' or 'quit' to leave\n"
                    f"Type 'clear' to reset conversation\n"
                    f"Use '@' + Tab to browse files in current directory\n"
                ),
                title="[bold]Welcome[/bold]",
                border_style="cyan",
            )
        )
        self.console.print()

        # Conversation state
        thread_id = "1"
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            try:
                # Get user input
                user_input = self.get_user_input().strip()

                # Handle exit commands
                if user_input.lower() in ("exit", "quit", "q"):
                    self.console.print(
                        Panel(
                            "[bold]Goodbye![/bold]",
                            title="[bold cyan]Session Ended[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    break

                # Handle clear command
                if user_input.lower() == "clear":
                    thread_id = str(int(thread_id) + 1)
                    config = {"configurable": {"thread_id": thread_id}}
                    self.console.clear()
                    self.console.print(
                        Panel(
                            "[bold]Conversation cleared.[/bold]",
                            title="[bold cyan]Reset[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    self.console.print()
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Add user message
                self.display_message(user_input, sender="user")

                # Get previous message count from graph state to avoid re-printing old messages
                previous_count = 0
                try:
                    state_snapshot = self.app.get_state(config)
                    existing_messages = state_snapshot.values.get("messages", [])
                    previous_count = len(existing_messages)
                except Exception as e:
                    logger.debug(f"Could not read previous state: {e}")

                # Get model response
                with self.console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                    response = self.app.invoke(
                        {"messages": [("user", user_input)]},
                        config=config,
                    )

                logger.info(f"Response from app.invoke: {response}")
                logger.info(f"Response messages count: {len(response['messages'])}")

                # Process only the new messages from this turn
                new_messages = response["messages"][previous_count:]
                for msg in new_messages:
                    logger.info(f"Processing message: type={type(msg)}, content_preview={str(msg)[:100]}")
                    
                    if msg.type == "ai":
                        # Check if it's a tool call response
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            logger.info(f"AI message with tool calls: {msg.tool_calls}")
                            # Show thinking/planning
                            if hasattr(msg, "content") and msg.content:
                                self.console.print(
                                    Panel(
                                        Markdown(msg.content),
                                        title="[bold yellow]Thinking[/bold yellow]",
                                        title_align="left",
                                        border_style="yellow",
                                        padding=(1, 2),
                                    )
                                )
                            # Show tool calls
                            for tool_call in msg.tool_calls:
                                if isinstance(tool_call, dict):
                                    tool_name = tool_call.get("name", "unknown")
                                    args = tool_call.get("args", {})
                                else:
                                    tool_name = getattr(tool_call, "name", "unknown")
                                    args = getattr(tool_call, "args", {})
                                self.console.print(
                                    Panel(
                                        f"[bold]{tool_name}[/bold]\n{json.dumps(args, indent=2)}",
                                        title="[bold cyan]Tool Call[/bold cyan]",
                                        title_align="left",
                                        border_style="cyan",
                                        padding=(1, 2),
                                    )
                                )
                        elif hasattr(msg, "content") and msg.content:
                            logger.info(f"AI message with content: {msg.content[:100]}")
                            self.display_message(msg.content, sender="assistant")
                    elif msg.type == "tool":
                        logger.info(f"Tool message: {msg.content[:100]}")
                        # Show tool results
                        result_preview = msg.content[:300] if len(msg.content) > 300 else msg.content
                        self.console.print(
                            Panel(
                                result_preview,
                                title="[bold green]Tool Result[/bold green]",
                                title_align="left",
                                border_style="green",
                                padding=(1, 2),
                            )
                        )

                self._queue_project_memory_update(user_input, new_messages)

            except KeyboardInterrupt:
                self.console.print("\n[bold]Use 'exit' or 'quit' to leave[/bold]")
            except Exception as e:
                logger.exception(f"Exception occurred: {e}")
                self.console.print(
                    Panel(
                        f"[bold red]Error:[/bold red] {str(e)}",
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                    )
                )


class FileCompleter(Completer):
    """Custom completer for file suggestions."""

    def __init__(self, cli: "OllamaChatCLI"):
        self.cli = cli

    def get_completions(self, document, complete_event):
        """Get file completions."""
        text = document.text_before_cursor
        
        # Find the last @ symbol in the text
        last_at_index = text.rfind("@")
        if last_at_index == -1:
            return
        
        # Get the search term after the last @
        search_term = text[last_at_index + 1:]
        current_dir = Path.cwd()

        # Find matching files
        for item in current_dir.iterdir():
            if item.is_file():
                name = item.name
                if search_term.lower() in name.lower():
                    # Return the full path with @ prefix
                    yield Completion(f"@{name}", start_position=-len(search_term))


def main():
    """Main entry point."""
    model_id = os.environ.get("OLLAMA_MODEL")
    if not model_id:
        print("Error: OLLAMA_MODEL environment variable not set")
        print("Example: export OLLAMA_MODEL=llama3.2 && python -m ollama_coder.cli")
        sys.exit(1)

    cli = OllamaChatCLI(model_id=model_id)
    cli.run()


if __name__ == "__main__":
    main()
