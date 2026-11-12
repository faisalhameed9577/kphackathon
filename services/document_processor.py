"""
Document processing: clean text, split it into chunks and attach metadata.

Chunks are LangChain ``Document`` objects. Every chunk carries: paper ID, title,
authors, publication year, DOI, source, section and (when available) page number.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.models import PaperRecord
from utils.helpers import clean_text

MIN_CHUNK_CHARS = 40

_ABSTRACT_LABELS = {
    "background", "introduction", "objective", "objectives", "aim", "aims", "purpose", "methods",
    "method", "materials and methods", "design", "setting", "participants", "results", "findings",
    "conclusion", "conclusions", "interpretation", "discussion", "importance", "context",
    "significance", "limitations", "rationale", "study design", "patients and methods",
    "main outcome measures", "design setting and participants", "unlabelled",
}
_LABEL_LINE = re.compile(r"^([A-Za-z][A-Za-z ,/&\-]{1,45}):\s*(.*)$")

# Full-text sections that are not research content.
_SKIP_SECTIONS = re.compile(
    r"references?|acknowledg|funding|competing|conflict|author contribution|supplement|"
    r"abbreviation|data availability|declaration|disclosure|appendix", re.IGNORECASE)

_CANONICAL_SECTIONS = [
    (re.compile(r"intro|background", re.I), "Introduction"),
    (re.compile(r"method|material|design|participants|procedure", re.I), "Methods"),
    (re.compile(r"result|finding", re.I), "Results"),
    (re.compile(r"limitation", re.I), "Limitations"),
    (re.compile(r"discussion", re.I), "Discussion"),
    (re.compile(r"conclusion", re.I), "Conclusions"),
]


# Some sources flatten a structured abstract into one line ("... done. Methods: ...");
# a known label right after a sentence end starts a new section.
_INLINE_LABEL = re.compile(
    r"(?<=[.!?])\s+(?=(?:" + "|".join(sorted(map(re.escape, _ABSTRACT_LABELS), key=len, reverse=True)) + r"):\s)",
    re.IGNORECASE)


def split_abstract_sections(abstract: str) -> List[Tuple[Optional[str], str]]:
    """Split a structured abstract into (label, text) parts; unstructured -> [(None, text)]."""
    sections: List[Tuple[Optional[str], List[str]]] = []
    for line in _INLINE_LABEL.sub("\n", abstract or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LABEL_LINE.match(line)
        if match and match.group(1).strip().lower() in _ABSTRACT_LABELS:
            sections.append((match.group(1).strip().title(), [match.group(2)]))
        elif sections:
            sections[-1][1].append(line)
        else:
            sections.append((None, [line]))
    return [(label, " ".join(parts).strip()) for label, parts in sections if " ".join(parts).strip()]


def _canonical_section(title: str) -> str:
    for pattern, name in _CANONICAL_SECTIONS:
        if pattern.search(title):
            return name
    return clean_text(title)[:60] or "Full text"


def parse_jats_sections(xml_text: str) -> Dict[str, str]:
    """
    Extract body sections from JATS XML (Europe PMC full text).
    Returns ``{section name: text}`` in reading order; ``{}`` if there is no usable body.
    """
    if not xml_text:
        return {}
    cleaned = re.sub(r"^\s*<\?xml[^>]*\?>", "", xml_text)  # encoding declaration confuses str parsing
    try:
        root = ET.fromstring(cleaned.encode("utf-8"))
    except ET.ParseError:
        return {}
    for element in root.iter():  # drop XML namespaces from tag names
        if isinstance(element.tag, str):
            element.tag = element.tag.split("}")[-1]

    body = root.find(".//body")
    if body is None:
        return {}

    result: Dict[str, str] = {}

    def add(name: str, text: str) -> None:
        text = clean_text(text)
        if len(text) >= MIN_CHUNK_CHARS:
            result[name] = f"{result[name]} {text}" if name in result else text

    for sec in body.findall("sec"):
        title = clean_text("".join(sec.find("title").itertext())) if sec.find("title") is not None else ""
        if title and _SKIP_SECTIONS.search(title):
            continue
        paragraphs = [" ".join("".join(p.itertext()).split()) for p in sec.iter("p")]
        add(_canonical_section(title) if title else "Full text", " ".join(paragraphs))

    if not result:  # body without <sec>: use loose paragraphs
        add("Full text", " ".join(" ".join("".join(p.itertext()).split()) for p in body.iter("p")))
    return result


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""])

    def build_documents(self, paper: PaperRecord,
                        fulltext_sections: Optional[Dict[str, str]] = None) -> List[Document]:
        """Chunk the abstract (and open-access full text if given) with full metadata."""
        if paper.paper_id is None:
            raise ValueError("Paper must be saved (have a paper_id) before it can be chunked.")

        sections: List[Tuple[str, str, str]] = []  # (section name, text, origin)
        for label, text in split_abstract_sections(paper.abstract):
            sections.append((f"Abstract - {label}" if label else "Abstract", text, "abstract"))
        for name, text in (fulltext_sections or {}).items():
            sections.append((name, text, "full text"))

        authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
        docs: List[Document] = []
        for sec_no, (section, text, origin) in enumerate(sections):
            for chunk_no, piece in enumerate(self.splitter.split_text(text)):
                piece = piece.strip()
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                metadata = {
                    "chunk_id": f"p{paper.paper_id}-s{sec_no}-c{chunk_no}",
                    "paper_id": int(paper.paper_id),
                    "title": paper.title,
                    "authors": authors or "Unknown authors",
                    "year": paper.year if paper.year else 0,  # 0 = unknown (Chroma has no null)
                    "doi": paper.doi or "",
                    "source": paper.source_label,
                    "section": section,
                    "text_origin": origin,
                    "chunk_index": chunk_no,
                    "url": paper.best_url,
                }
                # Page numbers are not available from abstracts or Europe PMC XML, so the key is omitted.
                docs.append(Document(page_content=piece, metadata=metadata))
        return docs
