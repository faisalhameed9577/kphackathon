"""
Paper service: the coordinator for discovery -> storage -> indexing.

search_papers()   : query PubMed + Europe PMC, merge, remove duplicates
save_and_index()  : save chosen papers to SQLite, fetch open-access full text (optional),
                    chunk, embed (only new chunks) and store in ChromaDB
get_paper_text()  : text used by the analysis modules (abstract + useful full-text parts)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from config.settings import Settings
from database.models import IndexReport, PaperRecord, SearchOutcome
from database.repositories import ChunkRepository, PaperRepository, QueryRepository, TopicRepository
from services.deduplication_service import deduplicate, interleave
from services.document_processor import DocumentProcessor, parse_jats_sections
from services.europe_pmc_service import EuropePMCService
from services.pubmed_service import PubMedService
from services.vector_store import VectorStore, VectorStoreError
from utils.helpers import get_logger, truncate
from utils.http import ResearchAPIError
from utils.validators import SearchRequest

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]  # (fraction 0-1, message)


class PaperService:
    def __init__(self, settings: Settings, pubmed: PubMedService, europepmc: EuropePMCService,
                 papers: PaperRepository, chunks: ChunkRepository, topics: TopicRepository,
                 queries: QueryRepository, processor: DocumentProcessor, vectors: VectorStore):
        self.settings = settings
        self.pubmed = pubmed
        self.europepmc = europepmc
        self.papers = papers
        self.chunks = chunks
        self.topics = topics
        self.queries = queries
        self.processor = processor
        self.vectors = vectors

    # ------------------------------------------------------------- discovery
    def search_papers(self, request: SearchRequest) -> SearchOutcome:
        """Search both sources, merge, de-duplicate and cap at ``max_papers``."""
        per_source = [self.pubmed.search(request), self.europepmc.search(request)]
        merged = interleave([r.papers for r in per_source])
        unique, removed = deduplicate(merged)
        return SearchOutcome(
            papers=unique[: request.max_papers], per_source=per_source,
            duplicates_removed=removed, retrieved_before_dedup=len(merged))

    def record_search(self, request: SearchRequest, outcome: SearchOutcome) -> int:
        """Store the topic + search run (methodology) and return the topic_id."""
        topic_id = self.topics.get_or_create(
            request.topic, request.keywords, request.region, request.year_from, request.year_to)
        filters = request.as_filters()
        filters["per_source"] = [
            {"source": r.source, "query": r.query, "total_found": r.total_found,
             "retrieved": len(r.papers), "error": r.error} for r in outcome.per_source]
        filters["duplicates_removed"] = outcome.duplicates_removed
        self.queries.add_search(topic_id, request.topic, filters,
                                [r.source for r in outcome.per_source if not r.error], len(outcome.papers))
        return topic_id

    # --------------------------------------------------------------- storage
    def _fulltext_file(self, paper_id: int) -> Path:
        return self.settings.papers_dir / f"paper_{paper_id}.json"

    def load_fulltext_sections(self, paper: PaperRecord) -> Dict[str, str]:
        path = Path(paper.fulltext_path) if paper.fulltext_path else None
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _fetch_and_cache_fulltext(self, paper: PaperRecord) -> Optional[Dict[str, str]]:
        """Download legally open-access full text from Europe PMC and cache the parsed sections."""
        if not (paper.is_open_access and paper.pmcid):
            return None
        xml_text = self.europepmc.fetch_fulltext_xml(paper.pmcid)
        sections = parse_jats_sections(xml_text or "")
        if not sections:
            return None
        path = self._fulltext_file(paper.paper_id)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")
        self.papers.set_fulltext_path(paper.paper_id, str(path))  # type: ignore[arg-type]
        paper.fulltext_path = str(path)
        return sections

    def save_and_index(self, papers: Sequence[PaperRecord], topic_id: Optional[int] = None,
                       fetch_fulltext: bool = True,
                       progress: Optional[ProgressCallback] = None) -> IndexReport:
        """Save papers, optionally fetch open-access full text, and index them in ChromaDB."""
        report = IndexReport(topic_id=topic_id)
        notify = progress or (lambda fraction, message: None)
        total = max(1, len(papers))
        saved: List[PaperRecord] = []
        all_docs = []
        sections_by_paper: Dict[int, Dict[str, str]] = {}

        # Phase 1: save metadata + full text (0% -> 60%)
        for i, paper in enumerate(papers):
            notify(0.6 * i / total, f"Saving: {truncate(paper.title, 70)}")
            paper_id = self.papers.upsert(paper)
            stored = self.papers.get(paper_id) or paper
            if topic_id is not None:
                self.papers.link_to_topic(topic_id, paper_id)
            if not stored.has_abstract:
                report.papers_without_abstract += 1

            sections = self.load_fulltext_sections(stored)
            if not sections and fetch_fulltext and stored.is_open_access and stored.pmcid:
                try:
                    sections = self._fetch_and_cache_fulltext(stored) or {}
                    report.fulltext_fetched += 1 if sections else 0
                except ResearchAPIError as exc:
                    report.warnings.append(f"Full text unavailable for '{truncate(stored.title, 50)}': {exc}")
            sections_by_paper[paper_id] = sections
            saved.append(stored)
            report.papers_saved += 1

            docs = self.processor.build_documents(stored, sections or None)
            if not docs:
                report.warnings.append(f"No text to index for '{truncate(stored.title, 60)}' (metadata saved only).")
            all_docs.extend(docs)

        # Phase 2: embed + store only the chunks that are not indexed yet (60% -> 95%)
        notify(0.65, f"Embedding {len(all_docs)} text chunks (already indexed chunks are reused)...")
        try:
            counts = self.vectors.add_documents(all_docs)
        except VectorStoreError as exc:
            report.warnings.append(str(exc))
            counts = {"new": 0, "reused": 0}
        report.chunks_new, report.chunks_reused = counts["new"], counts["reused"]
        report.chunks_total = len(all_docs)
        report.embedding_backend = self.vectors.embedder.backend
        if self.vectors.embedder.fallback_reason and self.vectors.embedder.backend == "local" and self.settings.has_gemini:
            report.warnings.append(f"Using local keyword embeddings. {self.vectors.embedder.fallback_reason}")

        # Phase 3: chunk metadata into SQLite (95% -> 100%)
        notify(0.95, "Recording chunk metadata...")
        by_paper: Dict[int, List[dict]] = {}
        for doc in all_docs:
            md = doc.metadata
            by_paper.setdefault(md["paper_id"], []).append({
                "chunk_uid": md["chunk_id"], "section": md["section"], "chunk_index": md["chunk_index"],
                "page_number": md.get("page"), "char_count": len(doc.page_content)})
        for paper_id, items in by_paper.items():
            self.chunks.upsert_many(paper_id, items, report.embedding_backend)

        if topic_id is not None:
            self.topics.touch(topic_id)
        notify(1.0, "Done")
        return report

    def reindex_paper(self, paper_id: int) -> IndexReport:
        paper = self.papers.get(paper_id)
        if paper is None:
            return IndexReport(warnings=["Paper not found."])
        return self.save_and_index([paper], topic_id=None, fetch_fulltext=False)

    # ------------------------------------------------------------- text access
    def get_paper_text(self, paper: PaperRecord, max_chars: int = 3500) -> str:
        """
        Text handed to the analysis modules: the abstract, plus (when cached) the Methods,
        Limitations, Discussion and Conclusions sections of open-access full text.
        """
        parts = [f"Title: {paper.title}"]
        if paper.abstract:
            parts.append(f"Abstract: {paper.abstract}")
        sections = self.load_fulltext_sections(paper)
        for name in ("Methods", "Results", "Limitations", "Discussion", "Conclusions"):
            if name in sections:
                parts.append(f"{name} (full text): {truncate(sections[name], 900)}")
        return truncate("\n".join(parts), max_chars)

    def has_fulltext(self, paper: PaperRecord) -> bool:
        return bool(self.load_fulltext_sections(paper))
