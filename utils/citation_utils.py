"""Citation helpers: reference formatting and checking of AI-written citations."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from database.models import PaperRecord

_GROUP_RE = re.compile(r"\[([^\[\]]*\bS\d+\b[^\[\]]*)\]")


def first_author_surname(paper: PaperRecord) -> str:
    """'Khan A' -> 'Khan'. Falls back to 'Anonymous'."""
    if not paper.authors:
        return "Anonymous"
    return paper.authors[0].split()[0].strip(",.") or "Anonymous"


def format_authors(authors: Sequence[str], max_authors: int = 3) -> str:
    if not authors:
        return "Anonymous"
    shown = list(authors[:max_authors])
    return ", ".join(shown) + (", et al." if len(authors) > max_authors else "")


def format_inline(paper: PaperRecord) -> str:
    """Short in-text form: 'Khan et al., 2021'."""
    year = paper.year or "n.d."
    if len(paper.authors) > 2:
        return f"{first_author_surname(paper)} et al., {year}"
    if len(paper.authors) == 2:
        return f"{first_author_surname(paper)} & {paper.authors[1].split()[0]}, {year}"
    return f"{first_author_surname(paper)}, {year}"


def format_reference(paper: PaperRecord, max_authors: int = 6, include_url: bool = True) -> str:
    """Vancouver-style reference built only from stored bibliographic data."""
    parts = [format_authors(paper.authors, max_authors) + "."]
    parts.append(paper.title.rstrip(".") + ".")
    if paper.journal:
        parts.append(paper.journal.rstrip(".") + ".")
    if paper.year:
        parts.append(f"{paper.year}.")
    if paper.doi:
        parts.append(f"doi:{paper.doi}")
    elif paper.pmid:
        parts.append(f"PMID:{paper.pmid}")
    link = paper.best_url
    if include_url and link and not paper.doi:
        parts.append(link)
    return " ".join(parts)


def paper_url(paper: PaperRecord) -> str:
    return paper.best_url


def extract_citation_numbers(text: str) -> List[int]:
    """Return the distinct [S#] numbers cited in ``text`` in order of first use."""
    seen: List[int] = []
    for match in _GROUP_RE.finditer(text or ""):
        for num in re.findall(r"S(\d+)", match.group(1)):
            value = int(num)
            if value not in seen:
                seen.append(value)
    return seen


def strip_invalid_citations(text: str, valid_numbers: Iterable[int]) -> str:
    """Remove [S#] references that do not correspond to a real retrieved passage."""
    valid = set(valid_numbers)

    def _rewrite(match: re.Match) -> str:
        nums = [int(n) for n in re.findall(r"S(\d+)", match.group(1))]
        keep = [n for n in nums if n in valid]
        return "[" + ", ".join(f"S{n}" for n in keep) + "]" if keep else ""

    cleaned = _GROUP_RE.sub(_rewrite, text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")


def parse_paper_label(label: object) -> Optional[int]:
    """'P12' -> 12. Returns None for anything that is not a valid paper label."""
    match = re.fullmatch(r"\s*P(\d+)\s*", str(label or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def humanize_labels(text: str, papers: Iterable[PaperRecord]) -> str:
    """Replace internal labels such as 'P12' with readable citations ('Khan et al., 2021')."""
    by_id = {p.paper_id: p for p in papers if p.paper_id is not None}

    def _swap(match: re.Match) -> str:
        paper = by_id.get(int(match.group(1)))
        return format_inline(paper) if paper else match.group(0)

    return re.sub(r"\bP(\d+)\b", _swap, text or "")


def filter_paper_labels(labels: Iterable[object], allowed_ids: Iterable[int]) -> List[int]:
    """Convert LLM-provided labels to paper ids, dropping any that were not supplied."""
    allowed = set(allowed_ids)
    result: List[int] = []
    for label in labels or []:
        pid = parse_paper_label(label)
        if pid is not None and pid in allowed and pid not in result:
            result.append(pid)
    return result
