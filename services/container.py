"""
Service container: builds every service once and wires the dependencies together.

Frontend pages call ``get_services()`` (see frontend/state.py) and never create
services themselves, so each module stays small and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import Settings, get_settings
from database.db import Database
from database.repositories import (AnalysisRepository, ChunkRepository, GapRepository, PaperRepository,
                                   QueryRepository, ReportRepository, StatsRepository, TopicRepository)
from services.analysis_pipeline import AnalysisPipeline
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.europe_pmc_service import EuropePMCService
from services.gap_detection import GapDetectionService
from services.gemini_service import GeminiService
from services.idea_service import IdeaService
from services.paper_service import PaperService
from services.pdf_service import PdfReportService
from services.pubmed_service import PubMedService
from services.rag_service import RagService
from services.report_service import ReportService
from services.research_analysis import ResearchAnalysisService
from services.vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    db: Database
    topics: TopicRepository
    papers: PaperRepository
    chunks: ChunkRepository
    queries: QueryRepository
    analyses: AnalysisRepository
    gaps: GapRepository
    reports: ReportRepository
    stats: StatsRepository
    gemini: GeminiService
    embedder: EmbeddingService
    vectors: VectorStore
    paper_service: PaperService
    rag: RagService
    pipeline: AnalysisPipeline
    report_service: ReportService
    idea: IdeaService


def build_services(settings: Optional[Settings] = None, gemini_client=None, pubmed: Optional[PubMedService] = None,
                   europepmc: Optional[EuropePMCService] = None) -> Services:
    """Create all services. Arguments allow tests to inject fakes (no network needed)."""
    settings = settings or get_settings()
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.init_schema()

    topics, papers, chunks = TopicRepository(db), PaperRepository(db), ChunkRepository(db)
    queries, analyses, gaps = QueryRepository(db), AnalysisRepository(db), GapRepository(db)
    reports, stats = ReportRepository(db), StatsRepository(db)

    gemini = GeminiService(settings, client=gemini_client)
    embedder = EmbeddingService(settings, gemini)
    vectors = VectorStore(settings, embedder)
    processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
    paper_service = PaperService(
        settings, pubmed or PubMedService(settings), europepmc or EuropePMCService(settings),
        papers, chunks, topics, queries, processor, vectors)
    rag = RagService(vectors, gemini, papers)
    analysis = ResearchAnalysisService(gemini, paper_service)
    pipeline = AnalysisPipeline(analyses, gaps, papers, topics, paper_service, analysis, GapDetectionService(gemini))
    report_service = ReportService(settings, topics, queries, reports, pipeline, paper_service, PdfReportService())
    idea = IdeaService(gemini, paper_service, embedder)

    return Services(settings, db, topics, papers, chunks, queries, analyses, gaps, reports, stats, gemini,
                    embedder, vectors, paper_service, rag, pipeline, report_service, idea)
