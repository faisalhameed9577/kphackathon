"""
Research analysis modules (plain Python functions, NOT autonomous agents):

A. summarize_papers()      - structured summary of each paper
B. compare()               - side-by-side comparison of several papers
C. extract_limitations()   - limitations the authors explicitly state

Each module calls Gemini when it is available and otherwise falls back to a transparent
extractive method (keyword/pattern matching on the abstract). Results always record how
they were produced ("gemini" or "extractive"). Labels returned by Gemini (P12) are checked
against the papers that were actually supplied, so invented papers are dropped.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

from config.prompts import (COMPARISON_PROMPT, LIMITATIONS_PROMPT, SUMMARY_PROMPT, SYSTEM_INSTRUCTION, render)
from database.models import (ComparisonResult, ComparisonRow, LimitationItem, PaperRecord, PaperSummary)
from services.document_processor import split_abstract_sections
from services.gemini_service import GeminiError, GeminiService
from services.paper_service import PaperService
from utils.citation_utils import filter_paper_labels, format_inline, parse_paper_label
from utils.helpers import chunked, split_sentences, truncate

NOT_REPORTED = "Not reported"
ProgressCallback = Callable[[float, str], None]

# ----------------------------------------------------------- pattern helpers

_DESIGN_PATTERNS = [
    ("systematic review / meta-analysis", r"systematic review|meta-?analys"),
    ("randomized controlled trial", r"randomi[sz]ed|\brcts?\b"),
    ("cohort study", r"\bcohort\b"),
    ("case-control study", r"case[- ]control"),
    ("cross-sectional study", r"cross[- ]sectional"),
    ("qualitative study", r"qualitative|focus group|in-depth interview"),
    ("survey / questionnaire", r"\bsurvey\b|questionnaire"),
    ("case report / series", r"case report|case series"),
    ("quasi-experimental / intervention study", r"quasi[- ]experiment|intervention study|before[- ]and[- ]after"),
    ("retrospective study", r"retrospective"),
    ("prospective study", r"prospective"),
    ("narrative / literature review", r"\breview\b"),
]

_LIMITATION_CUES = re.compile(
    r"\blimitations?\b|\blimited by\b|\bsmall sample\b|\bsample size was (small|limited)\b|"
    r"\bsingle[- ]cent(er|re)\b|\bselection bias\b|\brecall bias\b|\bnot generali[sz]able\b|"
    r"\bcannot be generali[sz]ed\b|\bshort follow-?up\b|\bself-reported\b|\bresidual confounding\b|"
    r"\bunable to (establish|determine|assess)\b", re.IGNORECASE)

_FINDING_CUES = re.compile(
    r"\bfound\b|\bshowed\b|\brevealed\b|\bsignificant|\bprevalence\b|\bassociated\b|\bincreas|\bdecreas|"
    r"\bhigher\b|\blower\b|\b\d+(\.\d+)?\s?%|\bp\s?[<=>]|\bodds ratio\b|\bOR\b", re.IGNORECASE)


def detect_study_design(text: str) -> str:
    hits = [name for name, pattern in _DESIGN_PATTERNS if re.search(pattern, text or "", re.IGNORECASE)]
    if "systematic review / meta-analysis" in hits and "narrative / literature review" in hits:
        hits.remove("narrative / literature review")
    return "; ".join(hits[:2]) if hits else NOT_REPORTED


def detect_sample_size(text: str) -> str:
    text = text or ""
    patterns = [
        (r"\b[nN]\s*=\s*(\d[\d,]*)", "n = {}"),
        (r"\b(\d[\d,]{1,8})\s+(patients|participants|children|women|men|adults|subjects|individuals|respondents|"
         r"cases|persons|people|households|students|mothers|studies|articles|papers|trials)\b", "{} {}"),
        (r"(?:sample (?:size )?of|total of|enrolled|recruited|analy[sz]ed)\s+(\d[\d,]*)", "n = {}"),
    ]
    for pattern, template in patterns:
        match = re.search(pattern, text)
        if match:
            return template.format(*match.groups())
    return NOT_REPORTED


def _first_label(by_label: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        if by_label.get(name):
            return by_label[name]
    return ""


_AIM_CUES = re.compile(r"\b(aim|aims|aimed|objective|objectives|purpose|we (sought|investigated|examined|assessed|"
                       r"evaluated|explored|compared|conducted|surveyed))\b", re.IGNORECASE)
_LABEL_PREFIX = re.compile(r"^[A-Za-z][A-Za-z ,/&\-]{1,45}:\s+")


def _aim_sentence(sentences: Sequence[str]) -> str:
    """First sentence that states an aim; otherwise the first sentence (without a 'Background:' prefix)."""
    if not sentences:
        return NOT_REPORTED
    for sentence in sentences:
        if _AIM_CUES.search(sentence):
            return _LABEL_PREFIX.sub("", sentence)
    return _LABEL_PREFIX.sub("", sentences[0])


def extractive_summary(paper: PaperRecord) -> PaperSummary:
    """Rule-based summary built only from the abstract text (no AI)."""
    summary = PaperSummary(paper_id=paper.paper_id, generated_by="extractive")
    if not paper.has_abstract:
        return summary
    sections = split_abstract_sections(paper.abstract)
    by_label = {(label or "").lower(): text for label, text in sections}
    sentences = split_sentences(paper.abstract)

    summary.objective = truncate(_first_label(by_label, ["objective", "objectives", "aim", "aims", "purpose"])
                                 or _aim_sentence(sentences), 400)
    summary.methods = truncate(_first_label(
        by_label, ["methods", "method", "materials and methods", "patients and methods", "design", "study design"])
        or NOT_REPORTED, 400)
    summary.population_setting = truncate(_first_label(by_label, ["participants", "setting", "design setting and participants"])
                                          or NOT_REPORTED, 300)
    if summary.population_setting == NOT_REPORTED:
        match = re.search(r"(?:among|in|of)\s+([^.;]{8,140}?(?:patients|children|women|men|adults|participants|"
                          r"population|students|residents|people|respondents)[^.;]{0,60})", paper.abstract, re.I)
        if match:
            summary.population_setting = truncate(match.group(1).strip(), 200)

    results = _first_label(by_label, ["results", "findings"])
    candidates = split_sentences(results) if results else [s for s in sentences[1:] if _FINDING_CUES.search(s)]
    summary.key_findings = [truncate(s, 300) for s in candidates[:3]]
    summary.conclusion = truncate(_first_label(by_label, ["conclusion", "conclusions", "interpretation"])
                                  or (sentences[-1] if len(sentences) >= 3 else NOT_REPORTED), 400)
    summary.study_design = detect_study_design(paper.abstract)
    summary.sample_size = detect_sample_size(paper.abstract)
    return summary


def extractive_limitations(paper: PaperRecord, text: str) -> List[LimitationItem]:
    """Sentences that contain explicit limitation wording (flagged as keyword matches)."""
    items: List[LimitationItem] = []
    for sentence in split_sentences(text):
        if _LIMITATION_CUES.search(sentence):
            items.append(LimitationItem(paper_id=paper.paper_id, text=truncate(sentence, 300),
                                        method="keyword-match"))
        if len(items) >= 4:
            break
    return items


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _as_list(value) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else []


# ------------------------------------------------------------------- service


class ResearchAnalysisService:
    def __init__(self, gemini: GeminiService, paper_service: PaperService, batch_size: int = 4):
        self.gemini = gemini
        self.paper_service = paper_service
        self.batch_size = batch_size

    def _papers_block(self, papers: Sequence[PaperRecord]) -> str:
        blocks = [f"=== P{p.paper_id} | {format_inline(p)} ===\n{self.paper_service.get_paper_text(p)}" for p in papers]
        return "\n\n".join(blocks)

    # -------------------------------------------------- A. summaries
    def summarize_papers(self, papers: Sequence[PaperRecord], notes: Optional[List[str]] = None,
                         progress: Optional[ProgressCallback] = None) -> Dict[int, PaperSummary]:
        notes = notes if notes is not None else []
        results: Dict[int, PaperSummary] = {}
        use_llm = self.gemini.available
        batches = list(chunked(list(papers), self.batch_size))
        for i, batch in enumerate(batches):
            if progress:
                progress(i / max(1, len(batches)), f"Summarising papers {i * self.batch_size + 1}-"
                         f"{i * self.batch_size + len(batch)} of {len(papers)}")
            parsed: Dict[int, PaperSummary] = {}
            if use_llm:
                try:
                    parsed = self._summarize_batch_llm(batch)
                except GeminiError as exc:
                    notes.append(f"AI summaries unavailable: {exc} Rule-based summaries were used instead.")
                    use_llm = False  # do not keep retrying after a failure
            for paper in batch:
                results[paper.paper_id] = parsed.get(paper.paper_id) or extractive_summary(paper)  # type: ignore[index]
        return results

    def _summarize_batch_llm(self, batch: Sequence[PaperRecord]) -> Dict[int, PaperSummary]:
        data = self.gemini.generate_json(render(SUMMARY_PROMPT, PAPERS=self._papers_block(batch)), SYSTEM_INSTRUCTION)
        allowed = {p.paper_id for p in batch}
        by_id = {p.paper_id: p for p in batch}
        out: Dict[int, PaperSummary] = {}
        for item in (data.get("papers", []) if isinstance(data, dict) else []):
            pid = parse_paper_label(item.get("label"))
            if pid not in allowed:
                continue  # label not supplied by us -> ignore (prevents invented papers)
            paper = by_id[pid]
            out[pid] = PaperSummary(
                paper_id=pid, generated_by="gemini",
                objective=str(item.get("objective") or NOT_REPORTED),
                study_design=str(item.get("study_design") or detect_study_design(paper.abstract)),
                population_setting=str(item.get("population_setting") or NOT_REPORTED),
                methods=str(item.get("methods") or NOT_REPORTED),
                key_findings=_as_list(item.get("key_findings"))[:4],
                conclusion=str(item.get("conclusion") or NOT_REPORTED),
                sample_size=detect_sample_size(paper.abstract))
        return out

    # -------------------------------------------------- C. limitations
    def extract_limitations(self, papers: Sequence[PaperRecord], notes: Optional[List[str]] = None,
                            progress: Optional[ProgressCallback] = None) -> Dict[int, List[LimitationItem]]:
        notes = notes if notes is not None else []
        results: Dict[int, List[LimitationItem]] = {}
        use_llm = self.gemini.available
        batches = list(chunked(list(papers), self.batch_size))
        for i, batch in enumerate(batches):
            if progress:
                progress(i / max(1, len(batches)), f"Extracting limitations ({i * self.batch_size + len(batch)}/{len(papers)})")
            parsed: Optional[Dict[int, List[LimitationItem]]] = None
            if use_llm:
                try:
                    parsed = self._limitations_batch_llm(batch)
                except GeminiError as exc:
                    notes.append(f"AI limitation extraction unavailable: {exc} Keyword matching was used instead.")
                    use_llm = False
            for paper in batch:
                if parsed is not None and paper.paper_id in parsed:
                    results[paper.paper_id] = parsed[paper.paper_id]  # type: ignore[index]
                else:
                    results[paper.paper_id] = extractive_limitations(  # type: ignore[index]
                        paper, self.paper_service.get_paper_text(paper, max_chars=6000))
        return results

    def _limitations_batch_llm(self, batch: Sequence[PaperRecord]) -> Dict[int, List[LimitationItem]]:
        data = self.gemini.generate_json(render(LIMITATIONS_PROMPT, PAPERS=self._papers_block(batch)), SYSTEM_INSTRUCTION)
        by_id = {p.paper_id: p for p in batch}
        out: Dict[int, List[LimitationItem]] = {}
        for item in (data.get("papers", []) if isinstance(data, dict) else []):
            pid = parse_paper_label(item.get("label"))
            if pid not in by_id:
                continue
            source_text = _norm(self.paper_service.get_paper_text(by_id[pid]))
            items: List[LimitationItem] = []
            for lim in item.get("limitations", []) or []:
                if not isinstance(lim, dict) or not str(lim.get("text", "")).strip():
                    continue
                evidence = str(lim.get("evidence") or "").strip()
                verified = bool(evidence) and _norm(evidence) in source_text
                items.append(LimitationItem(
                    paper_id=pid, text=str(lim["text"]).strip(),
                    evidence=evidence if verified else "",
                    method="gemini" if verified else "gemini-unverified"))
            out[pid] = items
        return out

    # -------------------------------------------------- B. comparison
    def compare(self, papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary],
                notes: Optional[List[str]] = None) -> ComparisonResult:
        notes = notes if notes is not None else []
        if len(papers) < 2:
            raise ValueError("Select at least two papers to compare.")
        if self.gemini.available:
            try:
                return self._compare_llm(papers, summaries)
            except GeminiError as exc:
                notes.append(f"AI comparison unavailable: {exc} A rule-based comparison was produced instead.")
        return extractive_comparison(papers, summaries)

    def _compare_llm(self, papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary]) -> ComparisonResult:
        blocks = []
        for p in papers:
            s = summaries.get(p.paper_id) or extractive_summary(p)  # type: ignore[arg-type]
            blocks.append(
                f"=== P{p.paper_id} | {format_inline(p)} | {p.year or 'n.d.'} ===\n"
                f"Objective: {s.objective}\nDesign: {s.study_design}\nPopulation/setting: {s.population_setting}\n"
                f"Sample size: {s.sample_size}\nMethods: {s.methods}\nKey findings: {' | '.join(s.key_findings) or NOT_REPORTED}\n"
                f"Conclusion: {s.conclusion}")
        data = self.gemini.generate_json(render(COMPARISON_PROMPT, SUMMARIES="\n\n".join(blocks)), SYSTEM_INSTRUCTION)
        if not isinstance(data, dict):
            raise GeminiError("Gemini returned an unexpected comparison format.")
        allowed = [p.paper_id for p in papers]
        rows_by_id: Dict[int, ComparisonRow] = {}
        for row in data.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            ids = filter_paper_labels([row.get("label")], allowed)
            if ids:
                rows_by_id[ids[0]] = ComparisonRow(
                    paper_id=ids[0], objective=str(row.get("objective") or NOT_REPORTED),
                    design=str(row.get("design") or NOT_REPORTED), population=str(row.get("population") or NOT_REPORTED),
                    sample_size=str(row.get("sample_size") or NOT_REPORTED),
                    key_findings=str(row.get("key_findings") or NOT_REPORTED),
                    conclusion=str(row.get("conclusion") or NOT_REPORTED))
        rows = [rows_by_id.get(p.paper_id) or _row_from_summary(p, summaries.get(p.paper_id)) for p in papers]  # type: ignore[arg-type]
        return ComparisonResult(
            rows=rows, similarities=_as_list(data.get("similarities")), differences=_as_list(data.get("differences")),
            overall_assessment=str(data.get("overall_assessment") or ""), generated_by="gemini")


def _row_from_summary(paper: PaperRecord, summary: Optional[PaperSummary]) -> ComparisonRow:
    s = summary or extractive_summary(paper)
    return ComparisonRow(paper_id=paper.paper_id, objective=s.objective, design=s.study_design,
                         population=s.population_setting, sample_size=s.sample_size,
                         key_findings="; ".join(s.key_findings) or NOT_REPORTED, conclusion=s.conclusion)


def extractive_comparison(papers: Sequence[PaperRecord], summaries: Dict[int, PaperSummary]) -> ComparisonResult:
    """Transparent, rule-based comparison used when Gemini is unavailable."""
    rows = [_row_from_summary(p, summaries.get(p.paper_id)) for p in papers]  # type: ignore[arg-type]
    similarities: List[str] = []
    differences: List[str] = []
    total = len(papers)

    # Designs
    design_groups: Dict[str, List[PaperRecord]] = {}
    for paper, row in zip(papers, rows):
        for design in (row.design.split("; ") if row.design != NOT_REPORTED else []):
            design_groups.setdefault(design, []).append(paper)
    for design, members in sorted(design_groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) >= 2:
            similarities.append(f"{len(members)} of {total} papers describe a {design}: "
                                + ", ".join(format_inline(p) for p in members) + ".")
    if len(design_groups) > 1:
        differences.append("Study designs differ across papers: " + ", ".join(sorted(design_groups)) + ".")
    elif not design_groups:
        differences.append("Study design could not be detected from the abstracts.")

    # Terms shared by most abstracts
    word_sets = []
    for p in papers:
        words = {w for w in re.findall(r"[a-z]{5,}", (p.title + " " + p.abstract).lower())}
        word_sets.append(words)
    counts: Counter = Counter(w for ws in word_sets for w in ws)
    shared = [w for w, c in counts.most_common(60) if c >= max(2, int(0.6 * total + 0.5)) and w not in _GENERIC_WORDS]
    if shared:
        similarities.append("Terms that appear in most abstracts: " + ", ".join(shared[:8]) + ".")

    years = [p.year for p in papers if p.year]
    if years and min(years) != max(years):
        differences.append(f"Publication years range from {min(years)} to {max(years)}.")
    sizes = [int(m.group(1).replace(",", "")) for r in rows for m in [re.search(r"(\d[\d,]*)", r.sample_size)] if m]
    if len(sizes) >= 2 and min(sizes) != max(sizes):
        differences.append(f"Reported sample sizes range from {min(sizes):,} to {max(sizes):,}.")
    missing = sum(1 for p in papers if not p.has_abstract)
    assessment = ("Rule-based comparison (no AI): it only reflects patterns found in titles and abstracts and cannot judge "
                  "whether findings agree. Read the individual summaries and original papers before drawing conclusions.")
    if missing:
        assessment += f" {missing} paper(s) have no abstract and could not be compared in detail."
    return ComparisonResult(rows=rows, similarities=similarities, differences=differences,
                            overall_assessment=assessment, generated_by="extractive")


_GENERIC_WORDS = {
    "study", "studies", "results", "methods", "conclusion", "conclusions", "background", "objective",
    "patients", "between", "among", "their", "which", "these", "there", "were", "after", "using",
    "based", "analysis", "significant", "associated", "total", "group", "groups", "should", "other",
}
