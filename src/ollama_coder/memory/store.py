"""Project memory storage using SQLite."""

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ProjectMemoryStore:
    """Persist and retrieve durable project facts from a local SQLite database."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.db_path = self.project_root / "memory.db"
        self._initialize()

    def _initialize(self) -> None:
        """Create the database schema if it does not exist."""
        with self._get_connection() as connection:
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

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def search(self, query: str, limit: int = 6) -> List[str]:
        """Return the most relevant stored facts for a query."""
        normalized_query = query.strip()

        with self._get_connection() as connection:
            rows = connection.execute(
                "SELECT fact, keywords, updated_at FROM facts ORDER BY updated_at DESC"
            ).fetchall()

        if not normalized_query:
            return [fact for fact, _, _ in rows[:limit]]

        query_terms = self._tokenize(normalized_query)

        ranked_rows: List[tuple[int, float, str]] = []
        recent_facts: List[str] = []
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

    def upsert_facts(self, facts: List[str], source_context: str) -> int:
        """Insert new facts or refresh timestamps for existing facts."""
        cleaned_facts = []
        seen_facts: Set[str] = set()
        for fact in facts:
            normalized_fact = self._normalize_fact(fact)
            if not normalized_fact or normalized_fact in seen_facts:
                continue
            seen_facts.add(normalized_fact)
            cleaned_facts.append(normalized_fact)

        if not cleaned_facts:
            return 0

        now = time.time()
        with self._get_connection() as connection:
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

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for simple keyword matching."""
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", text.lower())
            if token not in {
                "that", "this", "with", "from", "into", "should", "would",
                "there", "their", "about", "the", "and", "for", "are", "but",
                "not", "you", "your", "from", "into", "then", "there", "their"
            }
        }

    def clear(self) -> None:
        """Clear all stored facts."""
        with self._get_connection() as connection:
            connection.execute("DELETE FROM facts")
            connection.commit()
        logger.info("Cleared all project memory facts")
