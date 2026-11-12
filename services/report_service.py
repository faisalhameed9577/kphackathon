"""Report service: gathers everything for a topic, builds the PDF and records it in SQLite."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Tuple

from config.settings import Settings
from database.repositories import QueryRepository, ReportRepository, TopicRepository
from services.analysis_pipeline import AnalysisPipeline
from services.paper_service import PaperService
from services.pdf_service import PdfReportService, ReportData
from utils.helpers import now_iso, slugify, today_str

ProgressCallback = Callable[[float, str], None]


class ReportService:
    def __init__(self, settings: Settings, topics: TopicRepository, queries: QueryRepository,
                 reports: ReportRepository, pipeline: AnalysisPipeline, paper_service: PaperService,
                 pdf: PdfReportService):
        self.settings = settings
        self.topics = topics
        self.queries = queries
        self.reports = reports
        self.pipeline = pipeline
        self.paper_service = paper_service
        self.pdf = pdf

    def build_report_data(self, topic_id: int, progress: Optional[ProgressCallback] = None) -> ReportData:
        topic = self.topics.get(topic_id)
        if topic is None:
            raise ValueError("Topic not found.")
        bundle = self.pipeline.ensure_bundle_for_report(topic_id, progress)
        search = self.queries.latest_search(topic_id)
        filters = (search or {}).get("filters", {}) or {
            "year_from": topic.get("year_from"), "year_to": topic.get("year_to"),
            "keywords": topic.get("keywords", ""), "region": topic.get("region", ""), "max_papers": len(bundle.papers)}
        uses_ai = self.pipeline.analysis.gemini.available and any(
            s.generated_by == "gemini" for s in bundle.summaries.values())
        return ReportData(
            title=f"Evidence report: {topic['title']}", topic=topic["title"], region=topic.get("region", ""),
            search_date=((search or {}).get("created_at") or now_iso())[:10], generated_on=today_str(),
            filters=filters, per_source=filters.get("per_source", []),
            duplicates_removed=int(filters.get("duplicates_removed", 0)), papers=bundle.papers, bundle=bundle,
            fulltext_flags={p.paper_id: self.paper_service.has_fulltext(p) for p in bundle.papers},
            ai_model=self.settings.gemini_model if uses_ai else "")

    def create_report(self, topic_id: int, progress: Optional[ProgressCallback] = None) -> Tuple[Path, int]:
        """Generate the PDF; returns ``(file_path, report_id)``."""
        data = self.build_report_data(topic_id, progress)
        stamp = re.sub(r"\D", "", now_iso())
        path = self.settings.reports_dir / f"report_{topic_id}_{slugify(data.topic, 30)}_{stamp}.pdf"
        self.pdf.generate(data, path)
        report_id = self.reports.add(topic_id, data.title, str(path), path.stat().st_size)
        if progress:
            progress(1.0, "Report ready")
        return path, report_id
