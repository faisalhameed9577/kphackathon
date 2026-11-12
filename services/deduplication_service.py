"""
Duplicate detection across PubMed and Europe PMC.

Two records are the same paper if they share ANY reliable identifier:
DOI, PMID or PMCID. As a last resort a long, identical normalised title is used.
Matching records are merged (missing fields are filled in, sources are combined).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from database.models import PaperRecord
from utils.helpers import normalize_title

MIN_TITLE_KEY_LENGTH = 25  # short/generic titles ("Editorial") must not be merged by title


def identity_keys(paper: PaperRecord) -> List[str]:
    keys: List[str] = []
    if paper.doi:
        keys.append(f"doi:{paper.doi}")
    if paper.pmid:
        keys.append(f"pmid:{paper.pmid}")
    if paper.pmcid:
        keys.append(f"pmcid:{paper.pmcid}")
    title_key = normalize_title(paper.title)
    if len(title_key) >= MIN_TITLE_KEY_LENGTH:
        keys.append(f"title:{title_key}")
    return keys


def merge_records(primary: PaperRecord, other: PaperRecord) -> PaperRecord:
    """Return ``primary`` enriched with anything it lacks from ``other``."""
    merged = primary.model_copy(deep=True)
    if len(other.abstract) > len(merged.abstract):
        merged.abstract = other.abstract
    if not merged.authors and other.authors:
        merged.authors = list(other.authors)
    for attr in ("journal", "year", "doi", "pmid", "pmcid", "europepmc_id", "europepmc_source", "url"):
        if not getattr(merged, attr) and getattr(other, attr):
            setattr(merged, attr, getattr(other, attr))
    merged.is_open_access = merged.is_open_access or other.is_open_access
    merged.pub_types = list(dict.fromkeys([*merged.pub_types, *other.pub_types]))
    merged.sources = sorted(set(merged.sources) | set(other.sources),
                            key=lambda s: ["PubMed", "Europe PMC"].index(s) if s in ("PubMed", "Europe PMC") else 9)
    return merged


def deduplicate(papers: Sequence[PaperRecord]) -> Tuple[List[PaperRecord], int]:
    """Remove duplicates, keeping first-seen order. Returns ``(unique_papers, removed_count)``."""
    kept: List[PaperRecord] = []
    key_to_index: Dict[str, int] = {}
    removed = 0
    for paper in papers:
        hit: Set[int] = {key_to_index[k] for k in identity_keys(paper) if k in key_to_index}
        if hit:
            index = min(hit)
            kept[index] = merge_records(kept[index], paper)
            removed += 1
            target = kept[index]
        else:
            kept.append(paper)
            index = len(kept) - 1
            target = paper
        for key in identity_keys(target):  # new identifiers learnt from the merge
            key_to_index.setdefault(key, index)
    return kept, removed


def interleave(lists: Sequence[Sequence[PaperRecord]]) -> List[PaperRecord]:
    """Round-robin merge so each source contributes its best-ranked papers first."""
    merged: List[PaperRecord] = []
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank < len(lst):
                merged.append(lst[rank])
    return merged
