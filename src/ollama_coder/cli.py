"""Main CLI application for Ollama chat."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from .config.settings import load_settings
from .utils.logging import setup_logging
from .tools.registry import ToolRegistry
from .memory.store import ProjectMemoryStore
from .memory.extractor import ProjectMemoryExtractor
from .ui.renderer import MessageRenderer
from .ui.completer import FileCompleter
from .core.system_prompt import SystemPromptBuilder
from .core.context_builder import ContextBuilder
from .core.agent import AgentExecutor
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class OllamaChatCLI:
    """A terminal chat application using Ollama."""

    def __init__(self, model_id: str = None):
        """Initialize the chat application."""
        # Load settings
        self.ollama_settings, self.app_settings = load_settings(model_id)

        # Setup logging
        setup_logging(
            log_level=logging.DEBUG,
            log_file=self.app_settings.log_file,
        )

        # Initialize components
        self.console = MessageRenderer(model_id=self.ollama_settings.model_id)
        self.tool_registry = ToolRegistry()
        self.project_root = Path.cwd().resolve()
        self.project_memory = ProjectMemoryStore(self.project_root)
        self.memory_saver = MemorySaver()

        # Initialize base model
        self.base_model = ChatOllama(
            model=self.ollama_settings.model_id,
            temperature=self.ollama_settings.temperature,
            cache=None,  # Cache is handled separately
            keep_alive=self.ollama_settings.keep_alive,
            num_ctx=self.ollama_settings.num_ctx,
        )

        # Initialize system prompt builder
        self.system_prompt_builder = SystemPromptBuilder(self.project_root)

        # Initialize context builder
        self.context_builder = ContextBuilder(
            system_prompt_builder=self.system_prompt_builder,
            memory_store=self.project_memory,
            max_file_content_size=self.app_settings.max_file_content_size,
        )

        # Initialize agent executor
        self.agent = AgentExecutor(
            model=self.base_model,
            tool_registry=self.tool_registry,
            context_builder=self.context_builder,
            memory=self.memory_saver,
            project_memory=self.project_memory,
            project_memory_extractor=ProjectMemoryExtractor(self.base_model),
            history_token_budget=self.ollama_settings.history_token_budget,
        )

        # Setup prompt session
        self.session = PromptSession()
        self.completer = FileCompleter(self.project_root)
        self._setup_keybindings()

    def _setup_keybindings(self) -> None:
        """Setup custom key bindings for file suggestions."""
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys

        kb = KeyBindings()

        @kb.add(Keys.Tab)
        def _(event):
            """Handle Tab for accepting file suggestions."""
            buffer = event.app.current_buffer
            if buffer.complete_state:
                current_completion = buffer.complete_state.current_completion
                if current_completion:
                    buffer.cancel_completion()
                    buffer.insert_text(current_completion.text)
            else:
                buffer.start_completion(select_first=True)

        @kb.add(Keys.Escape)
        def _(event):
            """Handle Escape to cancel completions."""
            buffer = event.app.current_buffer
            buffer.cancel_completion()

        self.session.key_bindings = kb

    def run(self) -> None:
        """Run the chat application loop."""
        # Display welcome message
        settings_dict = {
            "model_id": self.ollama_settings.model_id,
            "temperature": self.ollama_settings.temperature,
            "num_ctx": self.ollama_settings.num_ctx,
            "keep_alive": self.ollama_settings.keep_alive,
            "enable_cache": self.ollama_settings.enable_cache,
        }
        self.console.render_welcome(settings_dict)

        # Conversation state
        thread_id = "1"

        while True:
            try:
                # Get user input
                user_input = self._get_user_input().strip()

                # Handle exit commands
                if user_input.lower() in ("exit", "quit", "q"):
                    self.console.render_exit()
                    break

                # Handle clear command
                if user_input.lower() == "clear":
                    thread_id = str(int(thread_id) + 1)
                    self.console.render_clear()
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Display user message
                self.console.render_user_message(user_input)

                # Get previous message count
                previous_count = self._get_previous_message_count(thread_id)

                # Stream model response in real-time
                self._stream_response(user_input, thread_id, previous_count)

                # Queue memory update
                # Get all messages for memory update
                try:
                    state_snapshot = self.agent.get_state(thread_id)
                    all_messages = state_snapshot.values.get("messages", [])
                    new_messages = all_messages[previous_count:]
                    self.agent.queue_memory_update(user_input, new_messages)
                except Exception as e:
                    logger.debug(f"Could not read state for memory update: {e}")

            except KeyboardInterrupt:
                self.console.console.print("\n[bold]Use 'exit' or 'quit' to leave[/bold]")
            except Exception as e:
                logger.exception(f"Exception occurred: {e}")
                self.console.render_error(str(e))

    def _stream_response(
        self, user_input: str, thread_id: str, previous_count: int
    ) -> None:
        """Stream the agent response in real-time.

        Args:
            user_input: The user input message.
            thread_id: The thread ID for conversation state.
            previous_count: The previous message count.
        """
        try:
            # Collect all messages from streaming
            all_messages = []
            rendered_messages = set()
            
            for chunk in self.agent.stream(user_input, thread_id=thread_id):
                messages = chunk.get("messages", [])
                all_messages.extend(messages)
                
                # Process each message
                for msg in messages:
                    msg_id = id(msg)
                    
                    if msg.type == "ai":
                        # Check if this is a thinking message or tool call message
                        if not getattr(msg, "tool_calls", None):
                            # This is a thinking/response message - stream it
                            if hasattr(msg, "content") and msg.content:
                                self.console.render_streaming_content(str(msg.content))
                        else:
                            # This is a tool call message - finalize thinking and render it
                            self.console.finalize_thinking()
                            self.console.render_assistant_message(msg)
                            rendered_messages.add(msg_id)
                            
                    elif msg.type == "tool":
                        # Render tool results
                        self.console.render_tool_message(msg)
            
            # Finalize any remaining streaming content
            self.console.finalize_thinking()
            
            # Render final AI messages that weren't rendered yet
            for msg in all_messages:
                msg_id = id(msg)
                if msg.type == "ai" and msg_id not in rendered_messages:
                    if not getattr(msg, "tool_calls", None):
                        # Only render if it has content
                        if hasattr(msg, "content") and msg.content:
                            self.console.render_assistant_message(msg)

        except Exception as e:
            logger.warning(f"Streaming failed: {e}, falling back to invoke")
            response = self.agent.invoke(user_input, thread_id=thread_id)
            new_messages = response["messages"][previous_count:]
            self._process_messages(new_messages)

    def _get_user_input(self) -> str:
        """Get input from user with a nice prompt and file suggestions."""
        try:
            return self.session.prompt(
                "You ",
                completer=self.completer,
                complete_while_typing=True,
                style=Style.from_dict({"prompt": "bold green"}),
            )
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "exit"

    def _get_previous_message_count(self, thread_id: str) -> int:
        """Get the previous message count from graph state."""
        try:
            state_snapshot = self.agent.get_state(thread_id)
            existing_messages = state_snapshot.values.get("messages", [])
            return len(existing_messages)
        except Exception as e:
            logger.debug(f"Could not read previous state: {e}")
            return 0

    def _process_messages(self, messages: List[Any]) -> None:
        """Process and render messages from the agent response.

        Args:
            messages: The list of messages to process.
        """
        for msg in messages:
            if msg.type == "ai":
                self.console.render_assistant_message(msg)
            elif msg.type == "tool":
                self.console.render_tool_message(msg)


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
