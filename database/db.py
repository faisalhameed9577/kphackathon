"""SQLite connection handling and schema creation."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from utils.helpers import get_logger

log = get_logger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails (message is safe to show to users)."""


# Timestamps are stored as UTC text (YYYY-MM-DD HH:MM:SS).
SCHEMA = """
CREATE TABLE IF NOT EXISTS research_topic (
    topic_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    title_key   TEXT NOT NULL UNIQUE,
    keywords    TEXT NOT NULL DEFAULT '',
    region      TEXT NOT NULL DEFAULT '',
    year_from   INTEGER,
    year_to     INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS research_paper (
    paper_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT NOT NULL,
    title_key         TEXT NOT NULL,
    abstract          TEXT NOT NULL DEFAULT '',
    journal           TEXT NOT NULL DEFAULT '',
    pub_year          INTEGER,
    doi               TEXT UNIQUE,
    pmid              TEXT UNIQUE,
    pmcid             TEXT UNIQUE,
    europepmc_id      TEXT,
    europepmc_source  TEXT,
    sources           TEXT NOT NULL DEFAULT '',
    url               TEXT NOT NULL DEFAULT '',
    is_open_access    INTEGER NOT NULL DEFAULT 0 CHECK (is_open_access IN (0, 1)),
    fulltext_path     TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_paper_title_key ON research_paper (title_key);
CREATE INDEX IF NOT EXISTS idx_paper_year ON research_paper (pub_year);

-- Associative table: a paper can belong to several topics and vice versa.
CREATE TABLE IF NOT EXISTS topic_paper (
    topic_id  INTEGER NOT NULL REFERENCES research_topic (topic_id) ON DELETE CASCADE,
    paper_id  INTEGER NOT NULL REFERENCES research_paper (paper_id) ON DELETE CASCADE,
    added_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (topic_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_author (
    author_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      INTEGER NOT NULL REFERENCES research_paper (paper_id) ON DELETE CASCADE,
    author_order  INTEGER NOT NULL,
    author_name   TEXT NOT NULL,
    UNIQUE (paper_id, author_order)
);

-- Only chunk METADATA lives here; chunk text + vectors live in ChromaDB.
CREATE TABLE IF NOT EXISTS document_chunk (
    chunk_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid          TEXT NOT NULL UNIQUE,
    paper_id           INTEGER NOT NULL REFERENCES research_paper (paper_id) ON DELETE CASCADE,
    section            TEXT NOT NULL DEFAULT '',
    chunk_index        INTEGER NOT NULL DEFAULT 0,
    page_number        INTEGER,
    char_count         INTEGER NOT NULL DEFAULT 0,
    embedding_backend  TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunk_paper ON document_chunk (paper_id);

-- A search run ('search') or a question asked in the Research Assistant ('question').
CREATE TABLE IF NOT EXISTS research_query (
    query_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id            INTEGER REFERENCES research_topic (topic_id) ON DELETE SET NULL,
    query_type          TEXT NOT NULL CHECK (query_type IN ('search', 'question')),
    query_text          TEXT NOT NULL,
    filters_json        TEXT NOT NULL DEFAULT '{}',
    sources             TEXT NOT NULL DEFAULT '',
    result_count        INTEGER NOT NULL DEFAULT 0,
    answer_text         TEXT,
    sufficient_evidence INTEGER CHECK (sufficient_evidence IN (0, 1)),
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS research_analysis (
    analysis_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id       INTEGER NOT NULL REFERENCES research_topic (topic_id) ON DELETE CASCADE,
    paper_id       INTEGER REFERENCES research_paper (paper_id) ON DELETE CASCADE,
    analysis_type  TEXT NOT NULL CHECK (analysis_type IN ('summary', 'limitations', 'comparison')),
    content_json   TEXT NOT NULL,
    generated_by   TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_analysis_lookup ON research_analysis (topic_id, analysis_type, paper_id);

CREATE TABLE IF NOT EXISTS research_gap (
    gap_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id            INTEGER NOT NULL REFERENCES research_topic (topic_id) ON DELETE CASCADE,
    gap_type            TEXT NOT NULL DEFAULT 'other',
    description         TEXT NOT NULL,
    rationale           TEXT NOT NULL DEFAULT '',
    origin              TEXT NOT NULL DEFAULT 'coverage-analysis',
    needs_verification  INTEGER NOT NULL DEFAULT 1 CHECK (needs_verification IN (0, 1)),
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Associative table: papers that support (motivate) a gap.
CREATE TABLE IF NOT EXISTS gap_supporting_paper (
    gap_id    INTEGER NOT NULL REFERENCES research_gap (gap_id) ON DELETE CASCADE,
    paper_id  INTEGER NOT NULL REFERENCES research_paper (paper_id) ON DELETE CASCADE,
    PRIMARY KEY (gap_id, paper_id)
);

CREATE TABLE IF NOT EXISTS research_question (
    question_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id          INTEGER NOT NULL REFERENCES research_topic (topic_id) ON DELETE CASCADE,
    gap_id            INTEGER REFERENCES research_gap (gap_id) ON DELETE CASCADE,
    question_text     TEXT NOT NULL,
    rationale         TEXT NOT NULL DEFAULT '',
    suggested_design  TEXT NOT NULL DEFAULT '',
    origin            TEXT NOT NULL DEFAULT 'template',
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS generated_report (
    report_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id         INTEGER NOT NULL REFERENCES research_topic (topic_id) ON DELETE CASCADE,
    title            TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    file_size_bytes  INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Database:
    """Thin wrapper around sqlite3 that gives short-lived, safe connections."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, roll back and raise DatabaseError on failure."""
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            if conn is not None:
                conn.rollback()
            log.error("Database error: %s", exc)
            raise DatabaseError(f"Database error: {exc}") from exc
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                conn.close()

    def init_schema(self) -> None:
        """Create all tables if they do not exist yet (safe to call on every start)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(SCHEMA)
        log.info("Database ready at %s", self.path)


if __name__ == "__main__":  # `python -m database.db` initialises the database
    from config.settings import get_settings

    settings = get_settings()
    settings.ensure_directories()
    Database(settings.db_path).init_schema()
    print(f"Database initialised: {settings.db_path}")
