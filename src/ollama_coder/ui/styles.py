"""UI styling configuration."""

from prompt_toolkit.styles import Style

# Default prompt style
PROMPT_STYLE = Style.from_dict({
    "prompt": "bold green",
})

# Panel styles
PANEL_STYLES = {
    "user": {
        "title": "[bold green]You[/bold green]",
        "border_style": "green",
    },
    "assistant": {
        "title": "[bold blue]{model_id}[/bold blue]",
        "border_style": "blue",
    },
    "tool": {
        "title": "[bold yellow]Tool[/bold yellow]",
        "border_style": "yellow",
    },
    "tool_result": {
        "title": "[bold green]Tool Result[/bold green]",
        "border_style": "green",
    },
    "thinking": {
        "title": "[bold yellow]Thinking[/bold yellow]",
        "border_style": "yellow",
    },
    "tool_call": {
        "title": "[bold cyan]Tool Call[/bold cyan]",
        "border_style": "cyan",
    },
    "error": {
        "title": "[bold red]Error[/bold red]",
        "border_style": "red",
    },
    "welcome": {
        "title": "[bold]Welcome[/bold]",
        "border_style": "cyan",
    },
    "reset": {
        "title": "[bold cyan]Reset[/bold cyan]",
        "border_style": "cyan",
    },
    "ended": {
        "title": "[bold cyan]Session Ended[/bold cyan]",
        "border_style": "cyan",
    },
}

# Status line style
STATUS_STYLE = "[dim]"

# Error style
ERROR_STYLE = "[bold red]"

# Success style
SUCCESS_STYLE = "[bold green]"

# Info style
INFO_STYLE = "[bold cyan]"


class UIStyles:
    """UI styling configuration class."""

    def __init__(self):
        """Initialize UI styles."""
        self.prompt = PROMPT_STYLE
        self.panels = PANEL_STYLES
        self.status = STATUS_STYLE
        self.error = ERROR_STYLE
        self.success = SUCCESS_STYLE
        self.info = INFO_STYLE
