"""File read and write tools."""

import logging
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from .base import BaseTool

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """Tool to read file contents."""

    description: ClassVar[str] = "Read the contents of a file."

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

    def validate(self) -> bool:
        """Validate the file path parameter."""
        if not self.file_path or not self.file_path.strip():
            return False
        return True

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "description": "Read the contents of a file. "
                          f"Current directory: {Path.cwd()}"
        }


class WriteFileTool(BaseTool):
    """Tool to write/edit file contents by replacing unique strings, creating new files, or appending."""

    description: ClassVar[str] = (
        "Edit an existing file by replacing a unique string, create a new file, "
        "or append to a file."
    )

    file_path: str = Field(
        description="The path to the file to edit, create, or append to. "
                   "Can be relative or absolute path. "
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
                return self._append_to_file(file_path)

            # Case 2: Create a new file
            elif not file_path.exists():
                return self._create_new_file(file_path)

            # Case 3: Edit an existing file
            else:
                return self._edit_existing_file(file_path)

        except PermissionError:
            logger.error(f"Permission denied writing to file: {self.file_path}")
            return f"Error: Permission denied writing to file: {self.file_path}"
        except Exception as e:
            logger.error(f"Error writing to file: {str(e)}")
            return f"Error writing to file: {str(e)}"

    def _append_to_file(self, file_path: Path) -> str:
        """Append content to an existing file."""
        if self.old_string:
            return (
                "Error: Cannot use old_string when appending. "
                "Leave old_string empty when append=True."
            )

        content = file_path.read_text()
        original_size = len(content)
        new_content = content + self.new_string
        file_path.write_text(new_content)

        logger.info(f"Successfully appended to file: {file_path}")
        return (
            f"Success: Text appended to {self.file_path}\n"
            f"Appended {len(self.new_string)} characters.\n"
            f"File size changed from {original_size} to {len(new_content)} bytes."
        )

    def _create_new_file(self, file_path: Path) -> str:
        """Create a new file with the given content."""
        if self.old_string:
            return (
                f"Error: Cannot use old_string when creating a new file. "
                f"File {self.file_path} does not exist. "
                f"Leave old_string empty to create a new file with the content in new_string."
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.new_string)

        logger.info(f"Successfully created file: {file_path}, size: {len(self.new_string)} bytes")
        return (
            f"Success: New file created at {self.file_path}\n"
            f"File size: {len(self.new_string)} bytes\n"
            f"Location: {file_path}"
        )

    def _edit_existing_file(self, file_path: Path) -> str:
        """Edit an existing file by replacing a unique string."""
        if not self.old_string:
            return (
                f"Error: old_string is required when editing an existing file. "
                f"File {self.file_path} already exists. "
                f"Provide the exact string to replace in old_string, or set append=True to append instead."
            )

        content = file_path.read_text()
        logger.info(f"File size: {len(content)} bytes")
        logger.info(f"Looking for string: {self.old_string[:100]}")

        if self.old_string not in content:
            logger.warning(f"String not found in file: {file_path}")
            return (
                f"Error: The string to replace was not found in {self.file_path}.\n"
                f"Please provide the exact string including proper indentation and context.\n"
                f"String to find:\n{repr(self.old_string)}"
            )

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

        new_content = content.replace(self.old_string, self.new_string)
        file_path.write_text(new_content)

        logger.info(f"Successfully wrote to file: {file_path}")
        return (
            f"Success: File {self.file_path} has been updated.\n"
            f"Replaced {len(self.old_string)} characters with {len(self.new_string)} characters.\n"
            f"File size changed from {len(content)} to {len(new_content)} bytes."
        )

    def validate(self) -> bool:
        """Validate the tool parameters."""
        if not self.file_path or not self.file_path.strip():
            return False
        if self.append and self.old_string:
            return False
        return True

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "description": "Edit an existing file by replacing a unique string, create a new file, or append to a file. "
                          "For editing: old_string must be unique in the file. "
                          "For creating: leave old_string empty and provide file content in new_string. "
                          "For appending: set append=True, leave old_string empty, and provide text to append in new_string. "
                          "Files must be created/edited within the current working directory. "
                          f"Current directory: {Path.cwd()}"
        }
