"""File path completion for the CLI."""

from pathlib import Path
from typing import List

from prompt_toolkit.completion import Completer, Completion


class FileCompleter(Completer):
    """Custom completer for file suggestions."""

    def __init__(self, base_path: Path = None):
        """Initialize the completer.

        Args:
            base_path: The base path for file completion. Defaults to current directory.
        """
        self.base_path = base_path or Path.cwd()

    def get_completions(self, document, complete_event):
        """Get file completions based on the current input.

        Args:
            document: The current document being edited.
            complete_event: The completion event.

        Yields:
            Completion objects for matching files.
        """
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

    def get_file_suggestions(self, prefix: str, limit: int = 10) -> List[str]:
        """Get file suggestions based on a prefix.

        Args:
            prefix: The prefix to search for (should start with @).
            limit: Maximum number of suggestions to return.

        Returns:
            List of file suggestions.
        """
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

        return suggestions[:limit]
