"""
Analysis pipeline: runs the analysis modules in order and caches results in SQLite.

summaries -> limitations -> gaps -> research questions   (run_gap_analysis)
comparison of chosen papers                              (run_comparison)

Cached results are reused; pass ``force=True`` to recompute. Rule-based results are
upgraded automatically once Gemini becomes available.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from database.models import (AnalysisBundle, ComparisonResult, LimitationItem, PaperRecord, PaperSummary)
from database.repositories import AnalysisRepository, GapRepository, PaperRepository, TopicRepository
from services.gap_detection import GapDetectionService
from services.paper_service import PaperService
from services.research_analysis import ResearchAnalysisService
from utils.helpers import get_logger

log = get_logger(__name__)

MAX_ANALYSIS_PAPERS = 25
MAX_COMPARE_PAPERS = 12
ProgressCallback = Callable[[float, str], None]


def _sub(progress: Optional[ProgressCallback], start: float, end: float) -> Optional[ProgressCallback]:
    """Map a 0-1 sub-task progress onto the [start, end] part of the whole job."""
    if progress is None:
        return None
    return lambda fraction, message: progress(start + (end - start) * fraction, message)


class AnalysisPipeline:
    def __init__(self, analyses: AnalysisRepository, gaps: GapRepository, papers: PaperRepository,
                 topics: TopicRepository, paper_service: PaperService, analysis: ResearchAnalysisService,
                 gap_service: GapDetectionService):
        self.analyses = analyses
        self.gaps = gaps
        self.papers = papers
        self.topics = topics
        self.paper_service = paper_service
        self.analysis = analysis
        self.gap_service = gap_service

    # ------------------------------------------------------------------ loading
    def load_bundle(self, topic_id: int) -> AnalysisBundle:
        """Everything already stored for a topic (nothing is computed here)."""
        bundle = AnalysisBundle(topic_id=topic_id, papers=self.papers.list_for_topic(topic_id))
        for pid, row in self.analyses.get_for_papers(topic_id, "summary").items():
            bundle.summaries[pid] = PaperSummary(**row["content"])
        for pid, row in self.analyses.get_for_papers(topic_id, "limitations").items():
            bundle.limitations[pid] = [LimitationItem(**item) for item in row["content"]]
        stored = self.analyses.get(topic_id, "comparison")
        if stored:
            bundle.comparison = ComparisonResult(**stored["content"]["result"])
        bundle.gaps = self.gaps.list_gaps(topic_id)
        bundle.questions = self.gaps.list_questions(topic_id)
        return bundle

    def stored_comparison_paper_ids(self, topic_id: int) -> List[int]:
        stored = self.analyses.get(topic_id, "comparison")
        return list(stored["content"].get("paper_ids", [])) if stored else []

    # ------------------------------------------------------------- per-paper steps
    def ensure_summaries(self, topic_id: int, papers: Sequence[PaperRecord], force: bool = False,
                         notes: Optional[List[str]] = None,
                         progress: Optional[ProgressCallback] = None) -> Dict[int, PaperSummary]:
        cached = self.analyses.get_for_papers(topic_id, "summary")
        use_ai = self.analysis.gemini.available
        result: Dict[int, PaperSummary] = {}
        todo: List[PaperRecord] = []
        for paper in papers:
            row = cached.get(paper.paper_id)  # type: ignore[arg-type]
            stale = row is not None and row["generated_by"] == "extractive" and use_ai
            if row is None or force or stale:
                todo.append(paper)
            else:
                result[paper.paper_id] = PaperSummary(**row["content"])  # type: ignore[index]
        if todo:
            fresh = self.analysis.summarize_papers(todo, notes, progress)
            for pid, summary in fresh.items():
                self.analyses.save(topic_id, "summary", summary.model_dump(), summary.generated_by, paper_id=pid)
                result[pid] = summary
        return result

    def ensure_limitations(self, topic_id: int, papers: Sequence[PaperRecord], force: bool = False,
                           notes: Optional[List[str]] = None,
                           progress: Optional[ProgressCallback] = None) -> Dict[int, List[LimitationItem]]:
        cached = self.analyses.get_for_papers(topic_id, "limitations")
        use_ai = self.analysis.gemini.available
        result: Dict[int, List[LimitationItem]] = {}
        todo: List[PaperRecord] = []
        for paper in papers:
            row = cached.get(paper.paper_id)  # type: ignore[arg-type]
            stale = row is not None and row["generated_by"] == "keyword-match" and use_ai
            if row is None or force or stale:
                todo.append(paper)
            else:
                result[paper.paper_id] = [LimitationItem(**i) for i in row["content"]]  # type: ignore[index]
        if todo:
            notes = notes if notes is not None else []
            before = len(notes)
            fresh = self.analysis.extract_limitations(todo, notes, progress)
            failed = len(notes) > before  # the service adds a note when it had to fall back
            origin = "keyword-match" if (not use_ai or failed) else "gemini"
            for pid, items in fresh.items():
                self.analyses.save(topic_id, "limitations", [i.model_dump() for i in items], origin, paper_id=pid)
                result[pid] = items
        return result

    # --------------------------------------------------------------- big steps
    def run_gap_analysis(self, topic_id: int, force: bool = False,
                         progress: Optional[ProgressCallback] = None) -> AnalysisBundle:
        """Summaries + limitations + gaps + research questions for a topic's papers."""
        topic = self.topics.get(topic_id)
        if topic is None:
            raise ValueError("Topic not found.")
        all_papers = self.papers.list_for_topic(topic_id)
        if not all_papers:
            raise ValueError("This topic has no saved papers yet. Search and index papers first.")
        papers = all_papers[:MAX_ANALYSIS_PAPERS]
        notes: List[str] = []
        if len(all_papers) > MAX_ANALYSIS_PAPERS:
            notes.append(f"Analysis covers the {MAX_ANALYSIS_PAPERS} most recent of {len(all_papers)} papers "
                         "to keep it fast and within API limits.")

        summaries = self.ensure_summaries(topic_id, papers, force, notes, _sub(progress, 0.0, 0.4))
        limitations = self.ensure_limitations(topic_id, papers, force, notes, _sub(progress, 0.4, 0.7))

        if progress:
            progress(0.72, "Looking for possible research gaps...")
        flags = {p.paper_id: self.paper_service.has_fulltext(p) for p in papers}
        gaps = self.gap_service.detect_gaps(topic["title"], topic.get("region", ""), papers, summaries,
                                            limitations, flags, notes)
        if progress:
            progress(0.88, "Generating research questions...")
        questions = self.gap_service.generate_questions(topic["title"], topic.get("region", ""), gaps, notes=notes)
        self.gaps.replace_for_topic(topic_id, gaps, questions)
        if progress:
            progress(1.0, "Analysis complete")

        bundle = self.load_bundle(topic_id)
        bundle.papers = papers
        bundle.notes = notes
        return bundle

    def run_comparison(self, topic_id: int, paper_ids: Sequence[int], notes: Optional[List[str]] = None,
                       progress: Optional[ProgressCallback] = None) -> ComparisonResult:
        """Compare the chosen papers (needs at least two) and store the result."""
        notes = notes if notes is not None else []
        papers = self.papers.get_many(list(paper_ids))
        if len(papers) < 2:
            raise ValueError("Select at least two papers to compare.")
        if len(papers) > MAX_COMPARE_PAPERS:
            notes.append(f"Only the first {MAX_COMPARE_PAPERS} selected papers were compared.")
            papers = papers[:MAX_COMPARE_PAPERS]
        summaries = self.ensure_summaries(topic_id, papers, False, notes, _sub(progress, 0.0, 0.7))
        if progress:
            progress(0.75, "Comparing papers...")
        result = self.analysis.compare(papers, summaries, notes)
        self.analyses.save(topic_id, "comparison", {"paper_ids": [p.paper_id for p in papers],
                                                    "result": result.model_dump()}, result.generated_by)
        if progress:
            progress(1.0, "Comparison complete")
        return result

    def ensure_bundle_for_report(self, topic_id: int,
                                 progress: Optional[ProgressCallback] = None) -> AnalysisBundle:
        """Load stored analyses and compute whatever a complete report still needs."""
        bundle = self.load_bundle(topic_id)
        if not bundle.papers:
            raise ValueError("This topic has no saved papers yet. Search and index papers first.")
        if not (bundle.summaries and bundle.gaps):
            bundle = self.run_gap_analysis(topic_id, progress=_sub(progress, 0.0, 0.8))
        if bundle.comparison is None and len(bundle.papers) >= 2:
            ids = [p.paper_id for p in bundle.papers[:MAX_COMPARE_PAPERS]]
            bundle.comparison = self.run_comparison(topic_id, ids, bundle.notes, _sub(progress, 0.8, 1.0))  # type: ignore[arg-type]
        return bundle
