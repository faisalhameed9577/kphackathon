"""
Research gap discovery and research-question generation.

Two complementary methods, both producing *possible* gaps that need verification:

1. coverage_gaps()  - transparent rules over the retrieved papers (counts of regions,
                      designs, subgroups, years, full-text availability).
2. llm_gaps()       - Gemini reads the paper summaries + stated limitations and proposes gaps,
                      each tied to supporting papers (labels are validated).

IMPORTANT: a gap here only means "the retrieved set suggests this may be under-studied".
It is never evidence that a topic has not been researched.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from config.prompts import GAPS_PROMPT, QUESTIONS_PROMPT, SYSTEM_INSTRUCTION, render
from database.models import GapItem, LimitationItem, PaperRecord, PaperSummary, ResearchQuestionItem
from services.gemini_service import GeminiError, GeminiService
from utils.citation_utils import filter_paper_labels, format_inline
from utils.helpers import expand_region_terms

VALID_GAP_TYPES = {"population", "location", "methodology", "outcome", "temporal", "evidence_quality", "other"}
MAX_GAPS = 8

_SUBGROUPS = {
    "women": r"\bwom[ae]n\b|\bfemale|\bmaternal|\bpregnan",
    "children": r"\bchild|\bpaediatric|\bpediatric|\binfant|\bneonat",
    "adolescents": r"\badolescen|\bteen|\byouth",
    "older adults": r"\belderly|\bolder adult|\bgeriatric|\baged \d",
    "rural populations": r"\brural|\bvillage|\bremote",
    "urban populations": r"\burban|\bslum|\bcity",
    "refugees / displaced people": r"\brefugee|\bdisplaced|\bafghan",
}


def _text(p: PaperRecord) -> str:
    return f"{p.title} {p.abstract}".lower()


def _ids(papers: Sequence[PaperRecord]) -> List[int]:
    return [p.paper_id for p in papers if p.paper_id is not None]  # type: ignore[misc]


def coverage_gaps(papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary], topic: str, region: str = "",
                  fulltext_flags: Optional[Dict[int, bool]] = None) -> List[GapItem]:
    """Rule-based gaps with the numbers that justify them in the rationale."""
    gaps: List[GapItem] = []
    n = len(papers)
    if n == 0:
        return gaps
    fulltext_flags = fulltext_flags or {}
    caveat = " This may reflect the limits of the search rather than a true gap in the literature."

    # 1. Region coverage
    region_terms = [t.lower() for t in expand_region_terms(region)]
    if region_terms:
        matching = [p for p in papers if any(t in _text(p) for t in region_terms)]
        if len(matching) < max(1, round(0.3 * n)):
            gaps.append(GapItem(
                gap_type="location",
                description=f"Few retrieved papers explicitly mention {region} in their title or abstract.",
                rationale=f"{len(matching)} of {n} retrieved papers mention the region." + caveat,
                supporting_paper_ids=_ids(matching)[:5], origin="coverage-analysis"))

    # 2. Study designs
    designs = [(p, (summaries.get(p.paper_id).study_design if summaries.get(p.paper_id) else "").lower()) for p in papers]  # type: ignore[union-attr]
    if n >= 3:
        interventional = [p for p, d in designs if re.search(r"randomi|trial|quasi|intervention", d + " " + _text(p))]
        if not interventional:
            gaps.append(GapItem(
                gap_type="methodology",
                description="No randomized or interventional studies were identified among the retrieved papers.",
                rationale=f"None of the {n} retrieved papers mention a randomized, trial or intervention design." + caveat,
                supporting_paper_ids=_ids(papers)[:5], origin="coverage-analysis"))
        cross = [p for p, d in designs if "cross-sectional" in d or "cross-sectional" in _text(p)]
        if len(cross) / n > 0.5:
            gaps.append(GapItem(
                gap_type="methodology",
                description="Most retrieved studies are cross-sectional; longitudinal (cohort) evidence appears limited.",
                rationale=f"{len(cross)} of {n} retrieved papers describe a cross-sectional design, which cannot show "
                          "change over time or causation." + caveat,
                supporting_paper_ids=_ids(cross)[:5], origin="coverage-analysis"))

    # 3. Population subgroups never mentioned
    if n >= 3:
        missing = [name for name, pattern in _SUBGROUPS.items() if not any(re.search(pattern, _text(p)) for p in papers)]
        if missing:
            shown = missing[:4]
            gaps.append(GapItem(
                gap_type="population",
                description="Some population groups are not mentioned in any retrieved abstract: " + ", ".join(shown) + ".",
                rationale=f"Keyword check over {n} titles/abstracts found no mention of these groups." + caveat,
                supporting_paper_ids=[], origin="coverage-analysis"))

    # 4. Recency
    years = [p.year for p in papers if p.year]
    if len(years) >= 5:
        this_year = datetime.now().year
        recent = [p for p in papers if p.year and p.year >= this_year - 3]
        if len(recent) / len(years) < 0.15:
            gaps.append(GapItem(
                gap_type="temporal",
                description="Recent evidence (last 3 years) is sparse in the retrieved set.",
                rationale=f"Only {len(recent)} of {len(years)} papers with a known year were published since {this_year - 3}." + caveat,
                supporting_paper_ids=_ids(recent)[:5], origin="coverage-analysis"))

    # 5. Evidence depth
    abstract_only = [p for p in papers if not fulltext_flags.get(p.paper_id, False)]
    if len(abstract_only) / n > 0.7:
        gaps.append(GapItem(
            gap_type="evidence_quality",
            description="Most evidence in this analysis comes from abstracts, not full texts.",
            rationale=f"{len(abstract_only)} of {n} papers were analysed from the abstract only, so methods details and "
                      "author-stated limitations may be missing. Read the full papers to confirm any gap.",
            supporting_paper_ids=[], origin="coverage-analysis"))
    return gaps


class GapDetectionService:
    def __init__(self, gemini: GeminiService):
        self.gemini = gemini

    # ------------------------------------------------------------------ gaps
    def llm_gaps(self, topic: str, region: str, papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary],
                 limitations: Dict[int, List[LimitationItem]]) -> List[GapItem]:
        blocks = []
        for p in papers:
            s = summaries.get(p.paper_id)
            lims = "; ".join(l.text for l in limitations.get(p.paper_id, [])) or "none stated in available text"
            blocks.append(
                f"=== P{p.paper_id} | {format_inline(p)} ===\nDesign: {s.study_design if s else 'n/a'}\n"
                f"Population/setting: {s.population_setting if s else 'n/a'}\n"
                f"Findings: {' | '.join(s.key_findings) if s and s.key_findings else 'n/a'}\n"
                f"Author-stated limitations: {lims}")
        prompt = render(GAPS_PROMPT, TOPIC=topic, REGION=region or "not specified", PAPERS="\n\n".join(blocks))
        data = self.gemini.generate_json(prompt, SYSTEM_INSTRUCTION)
        allowed = _ids(papers)
        gaps: List[GapItem] = []
        for item in (data.get("gaps", []) if isinstance(data, dict) else [])[:6]:
            if not isinstance(item, dict) or not str(item.get("description", "")).strip():
                continue
            gap_type = str(item.get("gap_type", "other")).lower()
            gaps.append(GapItem(
                gap_type=gap_type if gap_type in VALID_GAP_TYPES else "other",
                description=str(item["description"]).strip(), rationale=str(item.get("rationale", "")).strip(),
                supporting_paper_ids=filter_paper_labels(item.get("supporting_papers", []), allowed),
                origin="gemini"))
        return gaps

    def detect_gaps(self, topic: str, region: str, papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary],
                    limitations: Dict[int, List[LimitationItem]], fulltext_flags: Optional[Dict[int, bool]] = None,
                    notes: Optional[List[str]] = None) -> List[GapItem]:
        """Combine rule-based and (if available) Gemini gaps, de-duplicated and capped."""
        notes = notes if notes is not None else []
        gaps = coverage_gaps(papers, summaries, topic, region, fulltext_flags)
        if self.gemini.available and papers:
            try:
                gaps = self.llm_gaps(topic, region, papers, summaries, limitations) + gaps
            except GeminiError as exc:
                notes.append(f"AI gap analysis unavailable: {exc} Only rule-based gaps are shown.")
        seen, unique = set(), []
        for gap in gaps:
            key = re.sub(r"\W+", "", gap.description.lower())[:60]
            if key not in seen:
                seen.add(key)
                unique.append(gap)
        return unique[:MAX_GAPS]

    # ------------------------------------------------------------- questions
    def generate_questions(self, topic: str, region: str, gaps: Sequence[GapItem], max_questions: int = 6,
                           notes: Optional[List[str]] = None) -> List[ResearchQuestionItem]:
        notes = notes if notes is not None else []
        if not gaps:
            return []
        if self.gemini.available:
            try:
                gap_text = "\n".join(f"{i}. [{g.gap_type}] {g.description}" for i, g in enumerate(gaps))
                prompt = render(QUESTIONS_PROMPT, TOPIC=topic, REGION=region or "not specified", GAPS=gap_text,
                                MAX_QUESTIONS=str(max_questions))
                data = self.gemini.generate_json(prompt, SYSTEM_INSTRUCTION)
                questions = []
                for item in (data.get("questions", []) if isinstance(data, dict) else [])[:max_questions]:
                    if not isinstance(item, dict) or not str(item.get("question", "")).strip():
                        continue
                    idx = item.get("gap_index")
                    questions.append(ResearchQuestionItem(
                        gap_index=idx if isinstance(idx, int) and 0 <= idx < len(gaps) else None,
                        question=str(item["question"]).strip(), rationale=str(item.get("rationale", "")).strip(),
                        suggested_design=str(item.get("suggested_design", "")).strip(), origin="gemini"))
                if questions:
                    return questions
            except GeminiError as exc:
                notes.append(f"AI question generation unavailable: {exc} Template questions were used instead.")
        return template_questions(topic, region, gaps)[:max_questions]


def template_questions(topic: str, region: str, gaps: Sequence[GapItem]) -> List[ResearchQuestionItem]:
    """Simple, clearly-labelled template questions derived from each gap type (no AI)."""
    where = f" in {region}" if region else ""
    templates = {
        "location": (f"What is the current evidence on {topic}{where}, based on locally collected data?",
                     "community-based cross-sectional study with local sampling"),
        "population": (f"How do outcomes related to {topic}{where} differ across under-represented population groups?",
                       "stratified survey or subgroup analysis"),
        "methodology": (f"Can a more rigorous design (cohort or interventional) clarify causal relationships in {topic}{where}?",
                        "prospective cohort or pragmatic trial"),
        "temporal": (f"Have findings on {topic}{where} changed in recent years?",
                     "repeated cross-sectional surveys or updated systematic review"),
        "evidence_quality": (f"What do full-text analyses of studies on {topic}{where} reveal that abstracts do not?",
                             "systematic review with full-text data extraction"),
        "outcome": (f"Which outcomes of {topic}{where} are rarely measured, and how should they be assessed?",
                    "outcome-focused cohort study"),
    }
    questions = []
    for i, gap in enumerate(gaps):
        question, design = templates.get(gap.gap_type, (f"What remains unanswered about {topic}{where}?", "scoping review"))
        questions.append(ResearchQuestionItem(
            gap_index=i, question=question, suggested_design=design, origin="template",
            rationale=f"Derived from the possible gap: {gap.description}"))
    return questions
