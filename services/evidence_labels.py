"""
Evidence badges for Idea Check.

Turns PubMed publication types into a short, honest label. This is a *label of the
study design the indexers recorded*, not a quality score: a retracted paper is flagged,
and a paper without a recorded type is simply "Study (type not indexed)".
"""
from __future__ import annotations

from typing import Iterable, Tuple

RETRACTED = "retracted publication"

# (substring of publication type, badge label) - first match wins, strongest design first.
_ORDER: Tuple[Tuple[str, str], ...] = (
    ("meta-analysis", "Meta-analysis"),
    ("systematic review", "Systematic review"),
    ("randomized controlled trial", "RCT"),
    ("clinical trial", "Clinical trial"),
    ("practice guideline", "Guideline"),
    ("guideline", "Guideline"),
    ("observational study", "Observational"),
    ("cohort", "Cohort"),
    ("case reports", "Case report"),
    ("review", "Review"),
    ("preprint", "Preprint"),
    ("editorial", "Editorial"),
    ("letter", "Letter"),
    ("comment", "Comment"),
)


def is_retracted(pub_types: Iterable[str]) -> bool:
    return any(RETRACTED in (t or "").lower() for t in pub_types)


def evidence_badge(pub_types: Iterable[str], is_preprint: bool = False) -> str:
    types = [(t or "").lower() for t in pub_types]
    for needle, label in _ORDER:
        if any(needle in t for t in types):
            return label
    if is_preprint:
        return "Preprint"
    return "Study (type not indexed)"
