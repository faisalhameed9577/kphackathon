"""
Application settings.

All configuration comes from environment variables (optionally loaded from a
local ``.env`` file). Nothing secret is hard-coded, and the ``Settings`` object
never prints API keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# override=False: real environment variables win over the .env file.
load_dotenv(BASE_DIR / ".env", override=False)

APP_NAME = "ZETESIS AI"
APP_VERSION = "1.0.0"
TOOL_NAME = "ZetesisAI"  # sent to NCBI as the "tool" identifier

# These are only defaults. Override them in .env if Google renames models.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, trimmed; empty string if unset."""
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ConfigIssue:
    """A configuration message shown to the user. level: error | warning | info."""

    level: str
    message: str


@dataclass(frozen=True, repr=False)
class Settings:
    """Immutable application settings."""

    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    force_local_embeddings: bool = False
    data_dir: Path = BASE_DIR / "data"

    # Networking
    http_timeout: int = 25
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    # RAG
    embedding_dimension: int = 768
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # ------------------------------------------------------------------ paths
    @property
    def db_path(self) -> Path:
        return self.data_dir / "zetesis.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env("ZETESIS_DATA_DIR")
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            gemini_embedding_model=_env("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            ncbi_api_key=_env("NCBI_API_KEY"),
            ncbi_email=_env("NCBI_EMAIL"),
            force_local_embeddings=_env_bool("FORCE_LOCAL_EMBEDDINGS"),
            data_dir=Path(data_dir) if data_dir else BASE_DIR / "data",
        )

    @classmethod
    def for_testing(cls, data_dir: Path, **overrides) -> "Settings":
        """Settings for automated tests: no keys, no waiting between retries."""
        values = dict(data_dir=Path(data_dir), retry_backoff_seconds=0.0, max_retries=2)
        values.update(overrides)
        return cls(**values)

    # ---------------------------------------------------------------- helpers
    def ensure_directories(self) -> None:
        for folder in (self.data_dir, self.chroma_dir, self.papers_dir, self.reports_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[ConfigIssue]:
        """Return human-readable configuration messages (never includes secrets)."""
        issues: List[ConfigIssue] = []
        if not self.gemini_api_key:
            issues.append(ConfigIssue(
                "warning",
                "GEMINI_API_KEY is not set. The app still runs in limited mode: paper "
                "discovery, the library and PDF reports work, but AI summaries are "
                "replaced by simple extraction and search uses local keyword embeddings. "
                "Add the key to your .env file to enable Gemini.",
            ))
        if not self.ncbi_email:
            issues.append(ConfigIssue(
                "info",
                "NCBI_EMAIL is not set. NCBI asks API users to identify themselves with "
                "an email address; add it to .env (optional).",
            ))
        if not self.ncbi_api_key:
            issues.append(ConfigIssue(
                "info",
                "NCBI_API_KEY is not set. PubMed still works, but is limited to about "
                "3 requests per second (optional key raises this to 10).",
            ))
        try:
            self.ensure_directories()
        except OSError as exc:
            issues.append(ConfigIssue("error", f"Cannot create the data folder '{self.data_dir}': {exc}"))
        return issues

    def __repr__(self) -> str:  # never reveal keys
        return (
            f"Settings(gemini_key={'set' if self.gemini_api_key else 'not set'}, "
            f"ncbi_key={'set' if self.ncbi_api_key else 'not set'}, "
            f"model={self.gemini_model!r}, data_dir={str(self.data_dir)!r})"
        )

    @property
    def secrets(self) -> tuple:
        """Secret strings that must be scrubbed from any log or error text."""
        return tuple(s for s in (self.gemini_api_key, self.ncbi_api_key) if s)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the running app. Call get_settings.cache_clear() in tests."""
    return Settings.from_env()
