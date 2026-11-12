"""Small, dependency-free helper functions shared across the project."""
from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, List, Optional, Sequence

# --------------------------------------------------------------------- logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a simple, secret-free format."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logger


# -------------------------------------------------------------------- secrets

_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|key)=([^&\s'\"]+)", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
]


def redact_secrets(text: Any, secrets: Iterable[str] = ()) -> str:
    """Remove API keys from text before it is logged or shown to the user."""
    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    out = _SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)}=***", out)
    out = _SECRET_PATTERNS[1].sub("***", out)
    return out


# ---------------------------------------------------------------- text cleanup

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Strip HTML/XML tags and entities and collapse whitespace into single spaces."""
    if not text:
        return ""
    text = _TAG_RE.sub("", str(text))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def abstract_html_to_text(raw: Optional[str]) -> str:
    """
    Convert an HTML-ish abstract (Europe PMC style) to plain text.

    Section headings such as ``<h4>Background</h4>`` become ``Background: `` at the
    start of a new line, so structured abstracts stay structured.
    """
    if not raw:
        return ""
    text = str(raw)
    heading = lambda m: "\n" + clean_text(m.group(1)).rstrip(":") + ": "  # noqa: E731
    text = re.sub(r"<h\d[^>]*>(.*?)</h\d>", heading, text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<title[^>]*>(.*?)</title>", heading, text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</(p|sec|div)>|<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def truncate(text: str, limit: int = 200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def split_sentences(text: str) -> List[str]:
    """Very small sentence splitter, good enough for abstracts."""
    if not text:
        return []
    flat = _WS_RE.sub(" ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", flat)
    return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------- identifiers

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Lower-case a DOI and strip URL / 'doi:' prefixes. Returns None if empty."""
    if not doi:
        return None
    value = str(doi).strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value)
    value = value.strip().rstrip(".")
    return value or None


def normalize_title(title: Optional[str]) -> str:
    """Lower-case alphanumeric-only title used as a de-duplication key."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def parse_year(value: Any) -> Optional[int]:
    """Extract a 4-digit publication year (1800-2100) from a string or int."""
    if value is None:
        return None
    match = re.search(r"\b(1[89]\d{2}|20\d{2}|2100)\b", str(value))
    return int(match.group(1)) if match else None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return (slug or "item")[:max_len]


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


# ------------------------------------------------------------------ JSON

def parse_json_loose(text: str) -> Any:
    """
    Parse JSON returned by an LLM. Handles ```json fences and stray text around
    the JSON value. Raises ValueError if nothing parseable is found.
    """
    if text is None:
        raise ValueError("Empty response")
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = raw.find(open_ch), raw.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Response did not contain valid JSON")


# --------------------------------------------------------------- query terms

# Words that add no search value ("research", "studies", ...) are dropped so that
# "Diabetes research in rural Khyber Pakhtunkhwa" becomes diabetes AND rural AND ...
QUERY_STOPWORDS = {
    "a", "an", "the", "in", "of", "on", "for", "and", "or", "to", "with", "from", "by",
    "at", "about", "into", "among", "amongst", "within", "is", "are", "was", "were",
    "research", "researches", "study", "studies", "paper", "papers", "literature",
    "evidence", "article", "articles",
}

# Region names that are commonly written in several ways in the literature.
_KP_TERMS = [
    "Khyber Pakhtunkhwa", "Khyber Pakhtoonkhwa", "KPK", "North West Frontier Province",
    "NWFP", "Peshawar", "Abbottabad", "Mardan", "Swat", "Kohat", "Bannu", "Dera Ismail Khan",
]


def topic_to_terms(topic: str) -> List[str]:
    """Split a free-text topic into search terms, dropping generic filler words."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", topic or "")
    seen, terms = set(), []
    for word in words:
        key = word.lower()
        if key in QUERY_STOPWORDS or key in seen:
            continue
        seen.add(key)
        terms.append(word)
    return terms


def split_keywords(keywords: str) -> List[str]:
    """Split comma/semicolon separated keywords into clean phrases."""
    phrases = []
    for part in re.split(r"[;,\n]", keywords or ""):
        phrase = re.sub(r"[^A-Za-z0-9\- ']", " ", part)
        phrase = _WS_RE.sub(" ", phrase).strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def expand_region_terms(region: str) -> List[str]:
    """Expand a region into the spellings authors actually use (KP is expanded)."""
    region = _WS_RE.sub(" ", re.sub(r"[^A-Za-z0-9\- ']", " ", region or "")).strip()
    if not region:
        return []
    lowered = region.lower()
    if "khyber" in lowered or lowered in {"kp", "kpk", "nwfp"}:
        return list(_KP_TERMS)
    return [region]
