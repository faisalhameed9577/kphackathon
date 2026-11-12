"""
RAG service: question -> embedding -> retrieve chunks -> Gemini answer with citations.

Safety behaviour
* No relevant chunks            -> "insufficient evidence" (Gemini is NOT called).
* Gemini says INSUFFICIENT_EVIDENCE -> reported as insufficient.
* Citations like [S3] are checked against the retrieved passages; unknown ones are removed.
* Gemini unavailable            -> the most relevant passages are shown (extractive fallback).
* Requests for personal medical advice are declined.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from config.prompts import (CLINICAL_ADVICE_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE, RAG_PROMPT,
                            SYSTEM_INSTRUCTION, render)
from database.models import PaperRecord
from database.repositories import PaperRepository
from services.gemini_service import GeminiError, GeminiService
from services.vector_store import RetrievedChunk, VectorStore, VectorStoreError
from utils.citation_utils import extract_citation_numbers, format_inline, strip_invalid_citations
from utils.helpers import truncate

_ADVICE_PATTERNS = [
    r"\bshould i (take|use|stop|start|try|get|have)\b",
    r"\b(what|which) (medicine|medication|drug|dose|dosage|treatment|therapy)\b.{0,40}\b(should|do|can|must) i\b",
    r"\b(do|should|can) i (need|take|use|stop|start|try)\b.{0,40}\b(dose|dosage|mg|medicine|medication|drug|insulin|tablet)",
    r"\bwhat (is|are) my (dose|dosage|diagnosis|treatment)\b",
    r"\b(diagnose|diagnosis for) (me|my)\b",
    r"\bdo i have\b",
    r"\bmy (symptoms?|blood sugar|blood pressure|child|baby|son|daughter|mother|father|wife|husband)\b",
    r"\bhow (much|many) (mg|milligrams|tablets|pills)\b",
    r"\bprescribe\b",
]


def looks_like_clinical_advice_request(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(p, q) for p in _ADVICE_PATTERNS)


@dataclass
class SourceRef:
    label: str  # "S1"
    number: int
    paper_id: int
    title: str
    authors: str
    year: Optional[int]
    doi: str
    url: str
    section: str
    excerpt: str
    score: float
    text_origin: str = "abstract"
    inline_citation: str = ""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sufficient: bool
    sources: List[SourceRef] = field(default_factory=list)  # every passage retrieved
    cited_numbers: List[int] = field(default_factory=list)  # passages actually cited in the answer
    used_llm: bool = False
    notices: List[str] = field(default_factory=list)

    @property
    def cited_sources(self) -> List[SourceRef]:
        if not self.cited_numbers:
            return self.sources if not self.used_llm else []
        return [s for s in self.sources if s.number in self.cited_numbers]


class RagService:
    def __init__(self, vectors: VectorStore, gemini: GeminiService, papers: PaperRepository):
        self.vectors = vectors
        self.gemini = gemini
        self.papers = papers

    # ---------------------------------------------------------------- helpers
    def _build_sources(self, chunks: Sequence[RetrievedChunk]) -> List[SourceRef]:
        paper_ids = [int(c.metadata.get("paper_id", 0)) for c in chunks]
        by_id = {p.paper_id: p for p in self.papers.get_many(paper_ids)}
        sources: List[SourceRef] = []
        for n, chunk in enumerate(chunks, start=1):
            md = chunk.metadata
            paper: Optional[PaperRecord] = by_id.get(int(md.get("paper_id", 0)))
            sources.append(SourceRef(
                label=f"S{n}", number=n, paper_id=int(md.get("paper_id", 0)),
                title=paper.title if paper else md.get("title", "Unknown title"),
                authors=md.get("authors", ""), year=(md.get("year") or None),
                doi=md.get("doi", ""), url=paper.best_url if paper else md.get("url", ""),
                section=md.get("section", ""), excerpt=chunk.text, score=round(chunk.score, 3),
                text_origin=md.get("text_origin", "abstract"),
                inline_citation=format_inline(paper) if paper else md.get("title", "")))
        return sources

    @staticmethod
    def _context(sources: Sequence[SourceRef]) -> str:
        blocks = []
        for s in sources:
            blocks.append(f"[{s.label}] {s.inline_citation} | {s.title} | section: {s.section} "
                          f"({s.text_origin})\n{s.excerpt}")
        return "\n\n".join(blocks)

    # ----------------------------------------------------------------- answer
    def answer(self, question: str, paper_ids: Optional[Sequence[int]] = None, top_k: int = 6) -> RagAnswer:
        question = (question or "").strip()
        if len(question) < 5:
            return RagAnswer(question, "Please type a complete research question.", sufficient=False)
        if looks_like_clinical_advice_request(question):
            return RagAnswer(question, CLINICAL_ADVICE_MESSAGE, sufficient=False,
                             notices=["Medical-advice request declined."])

        try:
            chunks = self.vectors.query(question, k=top_k, paper_ids=paper_ids)
        except VectorStoreError as exc:
            return RagAnswer(question, f"Search of the evidence database failed: {exc}", sufficient=False)

        threshold = self.vectors.embedder.min_relevance
        chunks = [c for c in chunks if c.score >= threshold]
        if not chunks:
            return RagAnswer(question, INSUFFICIENT_EVIDENCE_MESSAGE, sufficient=False)

        sources = self._build_sources(chunks)
        notices: List[str] = []
        if self.vectors.embedder.backend == "local":
            notices.append("Retrieval used local keyword matching (no Gemini embeddings), so semantically "
                           "related passages with different wording may be missed.")
        if all(s.text_origin == "abstract" for s in sources):
            notices.append("All evidence passages come from abstracts, not full texts.")

        if not self.gemini.available:
            return self._extractive(question, sources, notices + ["Gemini is not configured, so no AI-written answer was produced."])

        prompt = render(RAG_PROMPT, CONTEXT=self._context(sources), QUESTION=question)
        try:
            raw = self.gemini.generate_text(prompt, SYSTEM_INSTRUCTION, temperature=0.2).strip()
        except GeminiError as exc:
            return self._extractive(question, sources, notices + [f"{exc} Showing the retrieved passages instead."])

        if raw.upper().startswith("INSUFFICIENT_EVIDENCE"):
            reason = raw.split(":", 1)[1].strip() if ":" in raw else ""
            return RagAnswer(question, f"{INSUFFICIENT_EVIDENCE_MESSAGE}\n\n{reason}".strip(),
                             sufficient=False, sources=sources, used_llm=True, notices=notices)

        valid = {s.number for s in sources}
        answer = strip_invalid_citations(raw, valid)
        cited = extract_citation_numbers(answer)
        if not cited:
            notices.append("The answer contains no citations; treat it as unverified and check the passages below.")
        return RagAnswer(question, answer, sufficient=True, sources=sources, cited_numbers=cited,
                         used_llm=True, notices=notices)

    @staticmethod
    def _extractive(question: str, sources: List[SourceRef], notices: List[str]) -> RagAnswer:
        lines = ["Most relevant passages found in the indexed papers:"]
        for s in sources[:4]:
            lines.append(f"- [{s.label}] {s.inline_citation}: {truncate(s.excerpt, 280)}")
        return RagAnswer(question, "\n".join(lines), sufficient=True, sources=sources,
                         cited_numbers=[s.number for s in sources[:4]], used_llm=False, notices=notices)
