"""Bash command execution tool."""

import os
import subprocess
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from .base import BaseTool


class BashTool(BaseTool):
    """Tool to execute bash commands."""

    description: ClassVar[str] = "Execute a bash command in the current directory."

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

    def validate(self) -> bool:
        """Validate the command parameter."""
        if not self.command or not self.command.strip():
            return False
        return True

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "description": "Execute a bash command in the current directory. "
                          f"Current directory: {Path.cwd()}"
        }
