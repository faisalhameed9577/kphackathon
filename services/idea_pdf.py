"""PDF export for an Idea Check report (a one-to-three page "research idea evidence note")."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, SimpleDocTemplate, Spacer, Table, TableStyle

from config.prompts import MEDICAL_DISCLAIMER
from services.evidence_labels import evidence_badge, is_retracted
from services.idea_service import GO_MAX_PAPERS, LEAVE_MIN_PAPERS, IdeaReport
from services.pdf_service import DARK, GREY, TEAL, PdfReportService, pdf_safe_text

_VERDICT_COLOURS = {"GO": "#15803d", "MODIFY": "#b45309", "LEAVE": "#b91c1c", "INCONCLUSIVE": "#475569"}
_VERDICT_MEANING = {
    "GO": "Few papers match this exact idea. There is room for new work.",
    "MODIFY": "The idea has been studied. Change the thin part to make it new.",
    "LEAVE": "The idea is already well covered. Change it instead of repeating it.",
    "INCONCLUSIVE": "The search found nothing to judge. Rephrase the idea.",
}


def _n(value: Any) -> str:
    return "n/a" if value is None else f"{value:,}"


class IdeaPdfService(PdfReportService):
    """Re-uses the styles and helpers of the main report so both PDFs look alike."""

    def generate_idea(self, report: IdeaReport, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="ZETESIS AI - Idea Check",
                                author="ZETESIS AI")
        story: List[Any] = []
        story += self._cover(report)
        story += self._numbers(report)
        story += self._advice(report)
        story += self._closest(report)
        story += self._method(report)
        story.append(self._p(MEDICAL_DISCLAIMER, "disclaimer"))
        doc.build(story, onFirstPage=self._footer, onLaterPages=self._footer)
        return output_path

    # ------------------------------------------------------------------ parts
    def _cover(self, r: IdeaReport) -> List[Any]:
        colour = _VERDICT_COLOURS.get(r.verdict, "#475569")
        out: List[Any] = [
            self._p("ZETESIS AI - Idea Check", "title"),
            self._p(f"Research idea evidence note - {r.created}", "subtitle"),
            HRFlowable(width="100%", color=TEAL, thickness=1.2, spaceAfter=8),
            self._p("Your idea", "h2"), self._p(r.idea),
            self._raw(f'<font color="{colour}" size="15"><b>{pdf_safe_text(r.verdict)}</b></font>', "h1"),
            self._p(r.headline, "h2"),
            self._p(_VERDICT_MEANING.get(r.verdict, "")),
        ]
        if r.explanation:
            out.append(self._p(r.explanation))
            out.append(self._p("Explanation written by Gemini from the numbers in this note."
                               if r.explanation_by == "gemini" else
                               "Rule-based explanation built from the numbers in this note.", "small"))
        rows = [["Part of the idea", "What was searched"]]
        for facet in r.facets.facets.values():
            rows.append([self._p(facet.name.title(), "cell"), self._p(facet.label, "cell")])
        if r.facets.design:
            rows.append([self._p("Design", "cell"), self._p(r.facets.design, "cell")])
        if len(rows) > 1:
            out += [Spacer(1, 6), self._table(rows, [4 * cm, 13 * cm])]
        return out

    def _numbers(self, r: IdeaReport) -> List[Any]:
        m = r.metrics
        out: List[Any] = [self._p("The numbers behind the verdict", "h1")]
        rows = [["Check", "Papers"],
                [self._p("Papers on your whole idea", "cell"), _n(m.get("whole_idea"))],
                [self._p(f"Papers since {r.recent_from}", "cell"), _n(m.get("recent"))],
                [self._p(f"Reviews since {r.recent_from}", "cell"), _n(m.get("reviews_recent"))],
                [self._p("Topic alone", "cell"), _n(m.get("topic_only"))]]
        if m.get("region_world") is not None and r.facets.region_label:
            rows += [[self._p(f"Mentioning {r.facets.region_label}", "cell"), _n(m.get("region_local"))],
                     [self._p(r.facets.country or "Country", "cell"), _n(m.get("region_country"))],
                     [self._p("Worldwide", "cell"), _n(m.get("region_world"))]]
        out.append(self._table(rows, [12 * cm, 5 * cm]))
        for reason in r.reasons:
            out.append(self._p(f"- {reason}", "bullet"))
        if r.designs:
            out.append(self._p("Study designs on your idea", "h2"))
            out.append(self._table([["Design", "Papers"]] + [[self._p(d.label, "cell"), _n(d.count)] for d in r.designs],
                                   [12 * cm, 5 * cm]))
        return out

    def _advice(self, r: IdeaReport) -> List[Any]:
        out: List[Any] = [self._p("What to do next", "h1")]
        out += [self._p(f"- {line}", "bullet") for line in r.advice]
        if r.angles:
            out.append(self._p("Untouched or thin angles", "h2"))
            out += [self._p(f"[{a.kind}] {a.text}", "bullet") for a in r.angles]
        return out

    def _closest(self, r: IdeaReport) -> List[Any]:
        if not r.nearest:
            return []
        rows: List[List[Any]] = [["Study", "Year", "Type", "Similarity"]]
        for item in r.nearest:
            p = item.paper
            kind = evidence_badge(p.pub_types, p.is_preprint)
            if is_retracted(p.pub_types):
                kind = "RETRACTED"
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            rows.append([self._p(f"**{p.title}** {authors} {p.journal}".strip(), "cell"),
                         str(p.year or "n.d."), self._p(kind, "cell"), f"{item.similarity:.2f}"])
        return [self._p("Closest existing studies", "h1"), self._table(rows, [9.2 * cm, 1.6 * cm, 3.7 * cm, 2.5 * cm]),
                self._p("Similarity is only used to rank these papers. Read them before deciding that your idea is covered.",
                        "small")]

    def _method(self, r: IdeaReport) -> List[Any]:
        out: List[Any] = [self._p("Methodology and limits", "h1"),
                          self._p(f"Verdict rules: GO when the whole idea has {GO_MAX_PAPERS} or fewer papers; LEAVE when it "
                                  f"has {LEAVE_MIN_PAPERS} or more and a recent review or many recent papers exist; "
                                  "otherwise MODIFY. The verdict comes from these rules, not from an AI model.")]
        for note in r.notes:
            out.append(self._p(f"- {note}", "bullet"))
        out.append(self._p("PubMed queries used", "h2"))
        for row in r.counts:
            out.append(self._p(f"**{row.label}** ({_n(row.count)} papers)", "small"))
            out.append(self._p(row.query or "(empty)", "small"))
        return out
