"""Message rendering for the CLI."""

import json
from typing import Any, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class MessageRenderer:
    """Renders messages to the console with proper formatting."""

    def __init__(self, model_id: str = "llama3.2"):
        """Initialize the renderer.

        Args:
            model_id: The ID of the model being used.
        """
        self.model_id = model_id
        self.console = Console()
        self._thinking_panel = None
        self._thinking_content = ""
        self._thinking_buffer = ""

    def render_user_message(self, message: str) -> None:
        """Render a user message.

        Args:
            message: The message content to render.
        """
        self.console.print(
            Panel(
                str(message),
                title="[bold green]You[/bold green]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )

    def render_streaming_content(self, content: str) -> None:
        """Render streaming content in real-time, similar to Gemini CLI.

        Args:
            content: The content to render.
        """
        # Accumulate content
        self._thinking_buffer += content

        # If we don't have a thinking panel yet, create one
        if self._thinking_panel is None:
            self._thinking_panel = self.console.status(
                f"[bold yellow]Thinking...[/bold yellow]",
                spinner="dots",
            )
            self._thinking_panel.start()

        # Update the panel with accumulated content
        if self._thinking_panel:
            self._thinking_panel.update(
                f"[bold yellow]Thinking...[/bold yellow]\n{self._thinking_buffer}",
                spinner="dots",
            )

    def finalize_thinking(self) -> None:
        """Finalize the thinking panel and display the full content."""
        if self._thinking_panel is not None:
            self._thinking_panel.stop()
            self._thinking_panel = None

            # Display the thinking content as a panel
            if self._thinking_buffer:
                self.console.print(
                    Panel(
                        Markdown(self._thinking_buffer),
                        title="[bold yellow]Thinking[/bold yellow]",
                        title_align="left",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
            self._thinking_buffer = ""

    def render_assistant_message(self, message: AIMessage) -> None:
        """Render an assistant message.

        Args:
            message: The AI message to render.
        """
        # Finalize any streaming thinking content first
        self.finalize_thinking()

        if hasattr(message, "tool_calls") and message.tool_calls:
            # Show thinking/planning content if present
            if hasattr(message, "content") and message.content:
                self.console.print(
                    Panel(
                        Markdown(str(message.content)),
                        title="[bold yellow]Thinking[/bold yellow]",
                        title_align="left",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )

            # Show tool calls
            for tool_call in message.tool_calls:
                tool_name, args = self._extract_tool_call(tool_call)
                self.console.print(
                    Panel(
                        f"[bold]{tool_name}[/bold]\n{json.dumps(args, indent=2)}",
                        title="[bold cyan]Tool Call[/bold cyan]",
                        title_align="left",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )

            # Show token status
            self._render_token_status(message)

        elif hasattr(message, "content") and message.content:
            self.console.print(
                Panel(
                    Markdown(str(message.content)),
                    title=f"[bold blue]{self.model_id}[/bold blue]",
                    title_align="left",
                    border_style="blue",
                    padding=(1, 2),
                )
            )
            self._render_token_status(message)

    def render_tool_message(self, message: ToolMessage) -> None:
        """Render a tool message.

        Args:
            message: The tool message to render.
        """
        result_preview = (
            message.content[:300] if len(message.content) > 300 else message.content
        )
        self.console.print(
            Panel(
                result_preview,
                title="[bold green]Tool Result[/bold green]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )

    def render_welcome(self, settings: dict) -> None:
        """Render the welcome message.

        Args:
            settings: Dictionary of settings to display.
        """
        self.console.clear()
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold]Ollama Coder CLI[/bold]\n"
                    f"Model: [cyan]{settings.get('model_id', 'N/A')}[/cyan]\n"
                    f"Temperature: [cyan]{settings.get('temperature', 'N/A')}[/cyan]\n"
                    f"Context window: [cyan]{settings.get('num_ctx', 'N/A')}[/cyan] tokens\n"
                    f"Keep alive: [cyan]{settings.get('keep_alive', 'N/A')}[/cyan]\n"
                    f"Exact cache: [cyan]{'on' if settings.get('enable_cache', False) else 'off'}[/cyan]\n"
                    f"Type 'exit' or 'quit' to leave\n"
                    f"Type 'clear' to reset conversation\n"
                    f"Use '@' + Tab to browse files in current directory\n"
                ),
                title="[bold]Welcome[/bold]",
                border_style="cyan",
            )
        )
        self.console.print()

    def render_status(self, message: str) -> None:
        """Render a status message.

        Args:
            message: The status message to render.
        """
        self.console.print(f"[dim]{message}[/dim]")

    def render_error(self, error: str) -> None:
        """Render an error message.

        Args:
            error: The error message to render.
        """
        self.console.print(
            Panel(
                f"[bold red]Error:[/bold red] {error}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )

    def render_clear(self) -> None:
        """Render the clear conversation message."""
        self.console.clear()
        self.console.print(
            Panel(
                "[bold]Conversation cleared.[/bold]",
                title="[bold cyan]Reset[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

    def render_exit(self) -> None:
        """Render the exit message."""
        self.console.print(
            Panel(
                "[bold]Goodbye![/bold]",
                title="[bold cyan]Session Ended[/bold cyan]",
                border_style="cyan",
            )
        )

    def _extract_tool_call(self, tool_call: Any) -> tuple[str, dict]:
        """Extract tool name and arguments from a tool call.

        Args:
            tool_call: The tool call object or dict.

        Returns:
            Tuple of (tool_name, arguments_dict).
        """
        if isinstance(tool_call, dict):
            return tool_call.get("name", "unknown"), tool_call.get("args", {})
        return (
            getattr(tool_call, "name", "unknown"),
            getattr(tool_call, "args", {}),
        )

    def _render_token_status(self, message: Any) -> None:
        """Render token usage information.

        Args:
            message: The message with token metadata.
        """
        status_line = self._format_token_status(message)
        if status_line:
            self.console.print(status_line)

    def _format_token_status(self, message: Any) -> str:
        """Format token usage information.

        Args:
            message: The message with token metadata.

        Returns:
            Formatted status string or empty string if no metadata.
        """
        response_metadata = getattr(message, "response_metadata", None) or {}
        usage_metadata = getattr(message, "usage_metadata", None) or {}

        prompt_eval_tokens = response_metadata.get("prompt_eval_count")
        if not isinstance(prompt_eval_tokens, int):
            prompt_eval_tokens = usage_metadata.get("input_tokens")

        output_tokens = response_metadata.get("eval_count")
        if not isinstance(output_tokens, int):
            output_tokens = usage_metadata.get("output_tokens")

        cached_prompt_tokens = response_metadata.get("cached_prompt_tokens")
        if not isinstance(cached_prompt_tokens, int):
            full_prompt_tokens = response_metadata.get("full_prompt_tokens")
            if isinstance(full_prompt_tokens, int) and isinstance(prompt_eval_tokens, int):
                cached_prompt_tokens = max(full_prompt_tokens - prompt_eval_tokens, 0)

        parts = []
        if isinstance(cached_prompt_tokens, int):
            parts.append(f"cached prompt tokens: {cached_prompt_tokens}")
        if isinstance(prompt_eval_tokens, int):
            parts.append(f"evaluated prompt tokens: {prompt_eval_tokens}")
        if isinstance(output_tokens, int):
            parts.append(f"output tokens: {output_tokens}")

        if not parts:
            return ""

        return f"[dim]Status: {' | '.join(parts)}[/dim]"
