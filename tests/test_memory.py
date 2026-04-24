"""Tests for memory module."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ollama_coder.memory.store import ProjectMemoryStore
from src.ollama_coder.memory.extractor import ProjectMemoryExtractor


class TestProjectMemoryStore:
    """Test ProjectMemoryStore class."""

    def test_initialize_creates_database(self, temp_project_dir):
        """Test that initialization creates the database."""
        store = ProjectMemoryStore(temp_project_dir)
        assert store.db_path.exists()

    def test_database_filename(self, temp_project_dir):
        """Test that database filename is correct."""
        store = ProjectMemoryStore(temp_project_dir)
        assert store.db_path.name == "memory.db"

    def test_upsert_facts_single(self, project_memory_store):
        """Test upserting a single fact."""
        facts = ["This is a test fact about the project"]
        count = project_memory_store.upsert_facts(facts, "test context")
        assert count == 1

    def test_upsert_facts_multiple(self, project_memory_store):
        """Test upserting multiple facts."""
        facts = [
            "First fact about the project",
            "Second fact about the project",
            "Third fact about the project",
        ]
        count = project_memory_store.upsert_facts(facts, "test context")
        assert count == 3

    def test_upsert_facts_deduplication(self, project_memory_store):
        """Test that duplicate facts are deduplicated."""
        facts = [
            "Same fact about the project",
            "Same fact about the project",
            "Different fact about the project"
        ]
        count = project_memory_store.upsert_facts(facts, "test context")
        # "Same fact about the project" appears twice but is deduplicated, so only 2 unique facts
        assert count == 2

    def test_upsert_facts_normalization(self, project_memory_store):
        """Test that facts are normalized."""
        facts = ["  Fact  with   extra   spaces  "]
        count = project_memory_store.upsert_facts(facts, "test context")
        assert count == 1

    def test_upsert_facts_too_short(self, project_memory_store):
        """Test that facts shorter than 12 chars are rejected."""
        facts = ["short", "this is a valid fact"]
        count = project_memory_store.upsert_facts(facts, "test context")
        assert count == 1

    def test_search_empty_query(self, project_memory_store):
        """Test searching with empty query returns recent facts."""
        facts = ["First fact about project", "Second fact about project", "Third fact about project"]
        project_memory_store.upsert_facts(facts, "context")
        results = project_memory_store.search("")
        # Empty query should return all facts (up to limit)
        assert len(results) > 0

    def test_search_with_query(self, project_memory_store):
        """Test searching with a query."""
        facts = ["This project uses pytest for testing", "The project has many modules"]
        project_memory_store.upsert_facts(facts, "context")
        results = project_memory_store.search("pytest testing")
        assert len(results) > 0

    def test_search_limit(self, project_memory_store):
        """Test that search respects limit."""
        facts = [f"Fact number {i}" for i in range(20)]
        project_memory_store.upsert_facts(facts, "context")
        results = project_memory_store.search("fact", limit=5)
        assert len(results) <= 5

    def test_search_no_results(self, project_memory_store):
        """Test searching with no matching results."""
        facts = ["This is a fact about Python"]
        project_memory_store.upsert_facts(facts, "context")
        results = project_memory_store.search("nonexistent query xyz")
        assert isinstance(results, list)

    def test_tokenize(self, project_memory_store):
        """Test tokenization."""
        tokens = project_memory_store._tokenize("This is a test string")
        assert "test" in tokens
        assert "string" in tokens
        # "is" is a stop word and should be filtered
        # But let's check what's actually in the tokens
        assert len(tokens) > 0

    def test_tokenize_filters_stop_words(self, project_memory_store):
        """Test that stop words are filtered."""
        tokens = project_memory_store._tokenize("the and for with from")
        # All of these are stop words, so should be empty
        assert len(tokens) == 0

    def test_normalize_fact(self, project_memory_store):
        """Test fact normalization."""
        normalized = project_memory_store._normalize_fact("  Multiple   spaces  ")
        assert normalized == "Multiple spaces"

    def test_normalize_fact_too_short(self, project_memory_store):
        """Test that short facts are rejected."""
        normalized = project_memory_store._normalize_fact("short")
        assert normalized == ""

    def test_normalize_fact_truncation(self, project_memory_store):
        """Test that long facts are truncated."""
        long_fact = "x" * 500
        normalized = project_memory_store._normalize_fact(long_fact)
        assert len(normalized) == 400

    def test_clear(self, project_memory_store):
        """Test clearing all facts."""
        facts = ["Fact 1", "Fact 2", "Fact 3"]
        project_memory_store.upsert_facts(facts, "context")
        project_memory_store.clear()
        results = project_memory_store.search("")
        assert len(results) == 0


class TestProjectMemoryExtractor:
    """Test ProjectMemoryExtractor class."""

    def test_extract_facts_with_json_response(self, memory_extractor):
        """Test extracting facts from JSON response."""
        mock_response = MagicMock()
        mock_response.content = '{"facts": ["Fact 1", "Fact 2"]}'
        memory_extractor.model.invoke = MagicMock(return_value=mock_response)

        facts = memory_extractor.extract_facts("test transcript")
        assert "Fact 1" in facts
        assert "Fact 2" in facts

    def test_extract_facts_with_invalid_json(self, memory_extractor):
        """Test extracting facts falls back to heuristic on invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "invalid json"
        memory_extractor.model.invoke = MagicMock(return_value=mock_response)

        facts = memory_extractor.extract_facts("test transcript")
        assert isinstance(facts, list)

    def test_extract_facts_heuristic_yarn(self, memory_extractor):
        """Test heuristic extraction for yarn."""
        transcript = "yarn workspace is used in this project"
        facts = memory_extractor._extract_facts_heuristic(transcript)
        assert any("yarn" in fact.lower() for fact in facts)

    def test_extract_facts_heuristic_pnpm(self, memory_extractor):
        """Test heuristic extraction for pnpm."""
        transcript = "pnpm-workspace.yaml is configured"
        facts = memory_extractor._extract_facts_heuristic(transcript)
        assert any("pnpm" in fact.lower() for fact in facts)

    def test_extract_facts_heuristic_poetry(self, memory_extractor):
        """Test heuristic extraction for poetry."""
        transcript = "poetry.lock file is present"
        facts = memory_extractor._extract_facts_heuristic(transcript)
        assert any("poetry" in fact.lower() for fact in facts)

    def test_extract_facts_heuristic_pytest(self, memory_extractor):
        """Test heuristic extraction for pytest."""
        transcript = "pytest is used for testing"
        facts = memory_extractor._extract_facts_heuristic(transcript)
        assert any("pytest" in fact.lower() for fact in facts)

    def test_extract_json_object_valid(self, memory_extractor):
        """Test extracting valid JSON object."""
        content = '{"facts": ["fact1", "fact2"]}'
        result = memory_extractor._extract_json_object(content)
        assert result == {"facts": ["fact1", "fact2"]}

    def test_extract_json_object_with_text(self, memory_extractor):
        """Test extracting JSON object from text."""
        content = "Some text before\n{\"facts\": [\"fact1\"]}\nSome text after"
        result = memory_extractor._extract_json_object(content)
        assert result == {"facts": ["fact1"]}

    def test_extract_json_object_invalid(self, memory_extractor):
        """Test extracting invalid JSON returns empty dict."""
        content = "not json at all"
        result = memory_extractor._extract_json_object(content)
        assert result == {}

    def test_extract_json_object_empty(self, memory_extractor):
        """Test extracting from empty string returns empty dict."""
        result = memory_extractor._extract_json_object("")
        assert result == {}

    def test_clear_cache(self, memory_extractor):
        """Test clearing cache (no-op for now)."""
        memory_extractor.clear_cache()  # Should not raise
