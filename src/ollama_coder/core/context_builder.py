"""Context building for messages."""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds message context with file contents and memory."""

    def __init__(
        self,
        system_prompt_builder,
        memory_store=None,
        max_file_content_size: int = 100000,
    ):
        """Initialize the context builder.

        Args:
            system_prompt_builder: The system prompt builder instance.
            memory_store: The project memory store instance.
            max_file_content_size: Maximum file content size to include.
        """
        self.system_prompt_builder = system_prompt_builder
        self.memory_store = memory_store
        self.max_file_content_size = max_file_content_size

    def build_context(
        self,
        user_message: str,
        messages: List[Any],
        query: str = "",
    ) -> tuple[str, List[Any]]:
        """Build the complete message context.

        Args:
            user_message: The current user message.
            messages: The existing message history.
            query: The query for memory retrieval.

        Returns:
            Tuple of (system_prompt, processed_messages).
        """
        # Extract file contents and prepend to message
        processed_message = self._process_message_with_files(user_message)

        # Build memory context
        memory_context = self._build_memory_context(query)

        # Build system prompt
        system_prompt = self.system_prompt_builder.build(memory_context)

        # Process messages
        processed_messages = self._process_messages(
            messages, processed_message, system_prompt
        )

        return system_prompt, processed_messages

    def _process_message_with_files(self, message: str) -> str:
        """Extract file paths from message and prepend their contents.

        Args:
            message: The user message.

        Returns:
            Message with file contents prepended.
        """
        # Find all file paths in the message (paths starting with ./, ../, /, or @)
        # This regex pattern matches common file path patterns
        file_pattern = r"(?<!\w)([./@][^\s\'\"]+|/[^\s\'\"]+)"
        file_paths = re.findall(file_pattern, message)

        if not file_paths:
            return message

        # Process each file and extract its content
        file_contents: List[str] = []
        for file_path in file_paths:
            # Clean up the path (remove @ prefix if present)
            clean_path = file_path.lstrip("@").strip("'\"")
            try:
                full_path = Path(clean_path).resolve()
                
                # Verify the file exists and is a file (not directory)
                if not full_path.exists():
                    logger.debug(f"File not found: {full_path}")
                    continue
                    
                if not full_path.is_file():
                    logger.debug(f"Path is not a file: {full_path}")
                    continue
                
                # Read file content
                content = full_path.read_text()
                
                # Check file size
                if len(content) > self.max_file_content_size:
                    logger.warning(
                        f"File {full_path} exceeds max size ({len(content)} > {self.max_file_content_size}), skipping"
                    )
                    continue
                    
                file_contents.append(
                    f"\n\nFile: {clean_path}\nContent:\n```\n{content}\n```"
                )
            except (OSError, ValueError) as e:
                logger.debug(f"Error processing file {file_path}: {e}")
                continue

        if file_contents:
            return message + "".join(file_contents)
        return message

    def _build_memory_context(self, query: str) -> str:
        """Build the project-memory section for the current task.

        Args:
            query: The query for memory retrieval.

        Returns:
            The memory context string.
        """
        if not self.memory_store:
            return "\n\n## Stored Project Facts\n- No project memory configured."

        relevant_facts = self.memory_store.search(query)
        if not relevant_facts:
            return "\n\n## Stored Project Facts\n- No stored project facts were found for this task."

        facts_text = "\n".join(f"- {fact}" for fact in relevant_facts)
        return f"\n\n## Stored Project Facts\n{facts_text}"

    def _process_messages(
        self,
        messages: List[Any],
        processed_content: str,
        system_prompt: str,
    ) -> List[Any]:
        """Process messages with the new content and system prompt.

        Args:
            messages: The existing message history.
            processed_content: The processed user message content.
            system_prompt: The system prompt.

        Returns:
            The processed message list.
        """
        from langchain_core.messages import HumanMessage

        if not messages:
            return [SystemMessage(content=system_prompt)]

        # Replace the last message with processed content if it's a HumanMessage
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            messages = messages[:-1] + [HumanMessage(content=processed_content)]

        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            messages = [SystemMessage(content=system_prompt)] + messages[1:]

        return messages

    def extract_active_query(self, messages: List[Any]) -> str:
        """Extract the most recent human message for memory retrieval.

        Args:
            messages: The message history.

        Returns:
            The most recent human message content.
        """
        for message in reversed(messages):
            if getattr(message, "type", None) == "human" and hasattr(
                message, "content"
            ):
                return str(message.content)
        return ""
