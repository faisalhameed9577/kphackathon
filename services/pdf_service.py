"""
PDF report generation with ReportLab (pure Python, works on Windows).

The report keeps four kinds of statements visibly apart:
  * findings supported by research      * limitations stated by authors
  * AI-inferred possible research gaps   * suggestions that need further verification
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from config.prompts import EVIDENCE_LABELS, MEDICAL_DISCLAIMER, SEARCH_LIMITATION_NOTE
from database.models import AnalysisBundle, PaperRecord
from utils.citation_utils import format_inline, format_reference, humanize_labels

TEAL = colors.HexColor("#0f766e")
DARK = colors.HexColor("#134e4a")
LIGHT = colors.HexColor("#ecfdf5")
AMBER = "#b45309"
GREY = colors.HexColor("#64748b")

_REPLACEMENTS = {"≥": ">=", "≤": "<=", "→": "->", "−": "-", " ": " ", " ": " ", "×": "x", "≈": "~",
                 "​": "", "α": "alpha", "β": "beta", "κ": "kappa", "±": "+/-"}


def pdf_safe_text(text: Any) -> str:
    """Helvetica supports Western-European characters only; map or replace anything else."""
    out = str(text or "")
    for src, dst in _REPLACEMENTS.items():
        out = out.replace(src, dst)
    return out.encode("cp1252", errors="replace").decode("cp1252")


def markup(text: Any) -> str:
    """Escape text for ReportLab Paragraph and support **bold**."""
    safe = escape(pdf_safe_text(text))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)


@dataclass
class ReportData:
    title: str
    topic: str
    region: str
    search_date: str
    generated_on: str
    filters: Dict[str, Any]
    per_source: List[Dict[str, Any]]
    duplicates_removed: int
    papers: List[PaperRecord]
    bundle: AnalysisBundle
    fulltext_flags: Dict[int, bool] = field(default_factory=dict)
    ai_model: str = ""  # Gemini model name if AI was used, else ""
    extra_notes: List[str] = field(default_factory=list)


class PdfReportService:
    def __init__(self) -> None:
        base = getSampleStyleSheet()
        self.s = {
            "title": ParagraphStyle("T", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
                                    textColor=DARK, alignment=TA_CENTER, spaceAfter=4),
            "subtitle": ParagraphStyle("ST", parent=base["Normal"], fontSize=11, textColor=GREY,
                                       alignment=TA_CENTER, spaceAfter=10),
            "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                                 textColor=TEAL, spaceBefore=14, spaceAfter=6),
            "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                                 textColor=DARK, spaceBefore=8, spaceAfter=3),
            "body": ParagraphStyle("B", parent=base["Normal"], fontSize=9.2, leading=12.8, spaceAfter=3),
            "small": ParagraphStyle("S", parent=base["Normal"], fontSize=7.8, leading=10, textColor=colors.HexColor("#334155")),
            "cell": ParagraphStyle("C", parent=base["Normal"], fontSize=7.8, leading=9.8),
            "cellhead": ParagraphStyle("CH", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
                                       leading=10, textColor=colors.white),
            "bullet": ParagraphStyle("BL", parent=base["Normal"], fontSize=9.2, leading=12.5, leftIndent=12,
                                     bulletIndent=2, spaceAfter=2),
            "ref": ParagraphStyle("R", parent=base["Normal"], fontSize=8.2, leading=11, leftIndent=18,
                                  firstLineIndent=-18, spaceAfter=3),
            "disclaimer": ParagraphStyle("D", parent=base["Normal"], fontSize=8.6, leading=11.5,
                                         textColor=colors.HexColor("#7c2d12"), backColor=colors.HexColor("#fff7ed"),
                                         borderPadding=6, spaceBefore=6, spaceAfter=10),
        }

    # ----------------------------------------------------------------- helpers
    def _p(self, text: Any, style: str = "body") -> Paragraph:
        return Paragraph(markup(text), self.s[style])

    def _raw(self, html: str, style: str = "body") -> Paragraph:
        return Paragraph(html, self.s[style])

    def _label(self, kind: str) -> str:
        color = {"finding": "#0f766e", "limitation": "#1d4ed8", "gap": AMBER, "verify": AMBER}[kind]
        return f'<font color="{color}" size="7.5"><b>[{escape(EVIDENCE_LABELS[kind])}]</b></font>'

    @staticmethod
    def _table(rows: List[List[Any]], widths: Sequence[float]) -> Table:
        table = Table(rows, colWidths=list(widths), repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return table

    @staticmethod
    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.2 * cm, "ZETESIS AI - research support tool, not medical advice")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    # ---------------------------------------------------------------- generate
    def generate(self, data: ReportData, output_path: Path) -> Path:
        """Build the PDF and return its path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=1.8 * cm, bottomMargin=2 * cm, title=pdf_safe_text(data.title),
                                author="ZETESIS AI")
        refs = {p.paper_id: i for i, p in enumerate(data.papers, start=1)}
        by_id = {p.paper_id: p for p in data.papers}

        def cite(ids: Sequence[int]) -> str:
            nums = sorted({refs[i] for i in ids if i in refs})
            return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

        story: List[Any] = []
        story += self._cover(data)
        story += self._methodology(data)
        story += self._papers_table(data, refs)
        story += self._summaries(data, refs)
        story += self._comparison(data, refs, by_id)
        story += self._limitations(data, refs)
        story += self._gaps(data, cite, by_id)
        story += self._questions(data)
        story += self._evidence_quality(data)
        story += self._references(data)
        doc.build(story, onFirstPage=self._footer, onLaterPages=self._footer)
        return output_path

    # ---------------------------------------------------------------- sections
    def _cover(self, data: ReportData) -> List[Any]:
        items: List[Any] = [
            Spacer(1, 0.3 * cm), self._p("ZETESIS AI", "title"),
            self._p("AI-Assisted Medical Research Evidence Report", "subtitle"),
            HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=8),
            self._p("1. Research topic", "h1"),
            self._p(f"**{data.topic}**" + (f" (region: {data.region})" if data.region else "")),
            self._p(f"Search date: {data.search_date}    |    Report generated: {data.generated_on}", "small"),
            Paragraph(markup("IMPORTANT: " + MEDICAL_DISCLAIMER + " Statements labelled as AI-inferred are hypotheses "
                             "that must be verified against the original papers."), self.s["disclaimer"]),
        ]
        return items

    def _methodology(self, data: ReportData) -> List[Any]:
        f = data.filters
        items: List[Any] = [self._p("2. Search sources, methodology and filters", "h1")]
        rows = [[Paragraph("Source", self.s["cellhead"]), Paragraph("Query used", self.s["cellhead"]),
                 Paragraph("Records found", self.s["cellhead"]), Paragraph("Retrieved", self.s["cellhead"]),
                 Paragraph("Status", self.s["cellhead"])]]
        for src in data.per_source:
            rows.append([self._p(src.get("source", ""), "cell"), self._p(src.get("query", ""), "cell"),
                         self._p(f"{src.get('total_found', 0):,}", "cell"), self._p(src.get("retrieved", 0), "cell"),
                         self._p("Error: " + src["error"] if src.get("error") else "OK", "cell")])
        if len(rows) > 1:
            items.append(self._table(rows, [2.2 * cm, 8 * cm, 2.2 * cm, 1.8 * cm, 2.8 * cm]))
        items.append(Spacer(1, 4))
        filters = [f"Years: {f.get('year_from', '?')}-{f.get('year_to', '?')}",
                   f"Maximum papers: {f.get('max_papers', '?')}",
                   f"Optional keywords: {f.get('keywords') or 'none'}",
                   f"Region filter: {f.get('region') or 'none'}"]
        for line in filters:
            items.append(Paragraph(markup(line), self.s["bullet"], bulletText="•"))
        items.append(self._p(f"Duplicate records removed across sources: {data.duplicates_removed}. Papers analysed: "
                             f"{len(data.papers)}. Identifiers used for de-duplication: DOI, PMID, PMCID, then title.", "small"))
        return items

    def _papers_table(self, data: ReportData, refs: Dict[int, int]) -> List[Any]:
        items: List[Any] = [self._p("3. Papers retrieved", "h1")]
        rows = [[Paragraph(h, self.s["cellhead"]) for h in ("Ref", "Title / journal", "Authors", "Year", "Identifier")]]
        for p in data.papers:
            ident = f"doi:{p.doi}" if p.doi else (f"PMID:{p.pmid}" if p.pmid else "-")
            flags = (" [preprint]" if p.is_preprint else "") + ("" if p.has_abstract else " [no abstract]")
            rows.append([self._p(refs[p.paper_id], "cell"),  # type: ignore[index]
                         self._p(f"**{p.title}** {p.journal}{flags}", "cell"),
                         self._p(", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""), "cell"),
                         self._p(p.year or "n.d.", "cell"), self._p(ident, "cell")])
        items.append(self._table(rows, [1 * cm, 7.4 * cm, 3.6 * cm, 1.2 * cm, 3.8 * cm]))
        return items

    def _summaries(self, data: ReportData, refs: Dict[int, int]) -> List[Any]:
        items: List[Any] = [self._p("4. Individual paper summaries", "h1"),
                            self._raw(self._label("finding") + " " + escape(
                                "Summaries restate what each paper reports; they are not independent verification."), "small")]
        for p in data.papers:
            s = data.bundle.summaries.get(p.paper_id)  # type: ignore[arg-type]
            items.append(self._p(f"[{refs[p.paper_id]}] {format_inline(p)}: {p.title}", "h2"))  # type: ignore[index]
            if not s:
                items.append(self._p("No summary available.", "small"))
                continue
            how = "AI-generated (Gemini)" if s.generated_by == "gemini" else "rule-based extraction from the abstract"
            for label, value in (("Objective", s.objective), ("Study design", s.study_design),
                                 ("Population / setting", s.population_setting), ("Sample size", s.sample_size)):
                items.append(self._p(f"**{label}:** {value}"))
            if s.key_findings:
                items.append(self._p("**Key findings:**"))
                items += [Paragraph(markup(k), self.s["bullet"], bulletText="•") for k in s.key_findings]
            items.append(self._p(f"**Conclusion:** {s.conclusion}"))
            items.append(self._p(f"Method: {how}.", "small"))
        return items

    def _comparison(self, data: ReportData, refs: Dict[int, int], by_id: Dict[Any, PaperRecord]) -> List[Any]:
        items: List[Any] = [self._p("5. Comparison of research findings", "h1")]
        comp = data.bundle.comparison
        if not comp or not comp.rows:
            return items + [self._p("No comparison was generated (at least two papers are needed).", "small")]
        rows = [[Paragraph(h, self.s["cellhead"]) for h in ("Ref", "Objective", "Design / population", "Key findings", "Conclusion")]]
        for r in comp.rows:
            context = "; ".join(v for v in (r.design, r.population, r.sample_size) if v and v != "Not reported")
            rows.append([self._p(refs.get(r.paper_id, "?"), "cell"), self._p(r.objective, "cell"),
                         self._p(context or "Not reported", "cell"),
                         self._p(r.key_findings, "cell"), self._p(r.conclusion, "cell")])
        items.append(self._table(rows, [0.9 * cm, 3.6 * cm, 3.6 * cm, 4.6 * cm, 4.3 * cm]))
        humanize = lambda text: humanize_labels(text, by_id.values())  # noqa: E731
        for title, values in (("Similarities", comp.similarities), ("Differences", comp.differences)):
            if values:
                items.append(self._p(title, "h2"))
                items += [Paragraph(markup(humanize(v)), self.s["bullet"], bulletText="•") for v in values]
        if comp.overall_assessment:
            items += [self._p("Overall assessment", "h2"), self._p(humanize(comp.overall_assessment)),
                      self._raw(self._label("verify"), "small")]
        how = "AI-generated (Gemini)" if comp.generated_by == "gemini" else "rule-based (no AI)"
        items.append(self._p(f"Comparison method: {how}.", "small"))
        return items

    def _limitations(self, data: ReportData, refs: Dict[int, int]) -> List[Any]:
        items: List[Any] = [self._p("6. Identified limitations", "h1"),
                            self._raw(self._label("limitation") + " " + escape(
                                "Only limitations the authors themselves state in the available text are listed."), "small")]
        none_stated = []
        for p in data.papers:
            lims = data.bundle.limitations.get(p.paper_id, [])  # type: ignore[arg-type]
            if not lims:
                none_stated.append(refs[p.paper_id])  # type: ignore[index]
                continue
            items.append(self._p(f"[{refs[p.paper_id]}] {format_inline(p)}", "h2"))  # type: ignore[index]
            for lim in lims:
                tag = {"keyword-match": " (keyword match - verify)", "gemini-unverified": " (AI-reported, quote not verified)"}.get(lim.method, "")
                text = lim.text + (f' - evidence: "{lim.evidence}"' if lim.evidence else "") + tag
                items.append(Paragraph(markup(text), self.s["bullet"], bulletText="•"))
        if none_stated:
            items.append(self._p("No explicit limitations were found in the available text for papers: "
                                 + ", ".join(f"[{n}]" for n in none_stated) + ". This does not mean the studies have none.", "small"))
        return items

    def _gaps(self, data: ReportData, cite, by_id) -> List[Any]:
        items: List[Any] = [self._p("7. Possible research gaps", "h1"),
                            self._raw(self._label("gap") + " " + escape(
                                "These are hypotheses suggested by the retrieved papers. They do not prove that a topic is "
                                "unstudied."), "small")]
        if not data.bundle.gaps:
            return items + [self._p("No gaps were identified from the available papers.", "small")]
        for i, g in enumerate(data.bundle.gaps, start=1):
            items.append(self._p(f"Gap {i} ({g.gap_type.replace('_', ' ')}): {humanize_labels(g.description, by_id.values())}", "h2"))
            items.append(self._p(f"**Why identified:** {humanize_labels(g.rationale, by_id.values())}"))
            support = cite(g.supporting_paper_ids)
            items.append(self._p(f"**Supporting references:** {support or 'none (derived from absence in the retrieved set)'}"
                                 f"    Origin: {'Gemini' if g.origin == 'gemini' else 'coverage analysis (rules)'}", "small"))
        return items

    def _questions(self, data: ReportData) -> List[Any]:
        items: List[Any] = [self._p("8. Suggested research questions", "h1"),
                            self._raw(self._label("verify") + " " + escape(
                                "Suggestions for researchers; they need expert review and ethical approval before use."), "small")]
        if not data.bundle.questions:
            return items + [self._p("No research questions were generated.", "small")]
        for i, q in enumerate(data.bundle.questions, start=1):
            items.append(self._p(f"Q{i}. {q.question}", "h2"))
            if q.rationale:
                items.append(self._p(f"**Rationale:** {q.rationale}"))
            if q.suggested_design:
                items.append(self._p(f"**Possible design:** {q.suggested_design}"))
        return items

    def _evidence_quality(self, data: ReportData) -> List[Any]:
        n = len(data.papers)
        abstract_only = sum(1 for p in data.papers if not data.fulltext_flags.get(p.paper_id, False))
        preprints = sum(1 for p in data.papers if p.is_preprint)
        no_abstract = sum(1 for p in data.papers if not p.has_abstract)
        notes = [
            f"{abstract_only} of {n} papers were analysed from abstracts only; {n - abstract_only} had open-access full text.",
            f"{preprints} paper(s) are preprints (not peer reviewed)." if preprints else "No preprints among the papers.",
            f"{no_abstract} paper(s) had no abstract and could be assessed from metadata only." if no_abstract else
            "All papers had an abstract.",
            "Analysis method: " + (f"AI-assisted ({data.ai_model}) with rule-based checks." if data.ai_model
                                   else "rule-based extraction only (Gemini was not used)."),
            SEARCH_LIMITATION_NOTE,
            "Evidence quality of the individual studies (risk of bias) was NOT formally assessed.",
        ] + list(data.bundle.notes) + list(data.extra_notes)
        items: List[Any] = [self._p("9. Evidence quality and search limitations", "h1")]
        items += [Paragraph(markup(t), self.s["bullet"], bulletText="•") for t in notes]
        return items

    def _references(self, data: ReportData) -> List[Any]:
        items: List[Any] = [PageBreak(), self._p("10. References", "h1"),
                            self._p("References are generated only from bibliographic records retrieved from PubMed and "
                                    "Europe PMC.", "small")]
        for i, p in enumerate(data.papers, start=1):
            link = p.best_url
            text = f"[{i}] " + markup(format_reference(p, include_url=False))  # the clickable link is added below
            if link:
                text += f' <a href="{escape(link)}" color="#1d4ed8">{escape(pdf_safe_text(link))}</a>'
            items.append(Paragraph(text, self.s["ref"]))
        return items
