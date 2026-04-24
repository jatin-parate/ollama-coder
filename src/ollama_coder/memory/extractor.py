"""Project memory fact extraction."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ProjectMemoryExtractor:
    """Extract durable project facts from conversation transcripts."""

    def __init__(self, model: BaseChatModel):
        """Initialize the extractor with a language model.

        Args:
            model: The language model to use for fact extraction.
        """
        self.model = model

    def extract_facts(self, transcript: str) -> List[str]:
        """Extract durable project facts from a conversation transcript.

        Args:
            transcript: The conversation transcript to analyze.

        Returns:
            A list of extracted project facts.
        """
        extraction_prompt = (
            "Extract durable, project-specific facts from this completed coding task. "
            "Only keep facts that will help future tasks in the same repository, such as package managers, "
            "workspace tools, build commands, test commands, frameworks, important directories, or coding conventions. "
            "Ignore transient details. Return strict JSON with shape {\"facts\": [\"fact 1\", \"fact 2\"]}. "
            "Keep each fact under 140 characters and phrase it as an instruction or durable repository fact.\n\n"
            f"Transcript:\n{transcript[:12000]}"
        )

        try:
            extraction_response = self.model.invoke(extraction_prompt)
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

        return self._extract_facts_heuristic(transcript)

    def _extract_json_object(self, content: str) -> dict:
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

    def _extract_facts_heuristic(self, transcript: str) -> List[str]:
        """Fallback fact extraction for common repository conventions."""
        lowered = transcript.lower()
        heuristic_facts: List[str] = []

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

    def clear_cache(self) -> None:
        """Clear any cached extraction results."""
        pass  # No caching implemented yet
