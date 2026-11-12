"""PDF report generation."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from services.pdf_service import markup, pdf_safe_text
from tests.conftest import make_paper

SEARCH_FILTERS = {
    "year_from": 2010, "year_to": 2025, "max_papers": 20, "region": "Fictionland", "keywords": "",
    "per_source": [
        {"source": "PubMed", "query": "(glycaemic AND control)", "total_found": 5, "retrieved": 5, "error": None},
        {"source": "Europe PMC", "query": "(glycaemic AND control) AND (PUB_YEAR:[2010 TO 2025])", "total_found": 0,
         "retrieved": 0, "error": "Europe PMC is not responding (HTTP 503) after 3 attempts."}],
    "duplicates_removed": 1}


def pdf_text(path) -> str:
    """Extract text if a PDF text tool is available; otherwise return '' (content checks are skipped)."""
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    try:
        from pypdf import PdfReader  # optional

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except ImportError:
        return ""


def test_pdf_safe_text_and_markup():
    assert pdf_safe_text("HbA1c ≥ 7% → worse; β-cell") == "HbA1c >= 7% -> worse; beta-cell"
    assert pdf_safe_text("Привет") == "??????"  # unsupported scripts are replaced, never crash
    assert pdf_safe_text(None) == ""
    assert markup("<script>alert(1)</script> & **bold**") == "&lt;script&gt;alert(1)&lt;/script&gt; &amp; <b>bold</b>"


def test_report_end_to_end(services, saved_topic):
    topic_id, papers = saved_topic
    services.queries.add_search(topic_id, "Fictional glycaemic outcomes", SEARCH_FILTERS, ["PubMed"], 5)
    steps = []
    path, report_id = services.report_service.create_report(topic_id, progress=lambda f, m: steps.append(f))

    assert path.exists() and path.suffix == ".pdf" and path.parent == services.settings.reports_dir
    data = path.read_bytes()
    assert data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:] and len(data) > 6000
    assert steps[-1] == 1.0
    listed = services.reports.list()[0]
    assert listed["report_id"] == report_id and listed["file_size_bytes"] == len(data)

    text = pdf_text(path)
    flat = " ".join(text.split())  # PDF text extraction wraps lines; compare on single-spaced text
    if text:
        for heading in ("Research topic", "Search sources, methodology and filters", "Papers retrieved",
                        "Individual paper summaries", "Comparison of research findings", "Identified limitations",
                        "Possible research gaps", "Suggested research questions", "Evidence quality and search limitations",
                        "References"):
            assert heading in text, heading
        assert "does not provide medical" in flat and "diagnosis or treatment advice" in flat
        assert "AI-inferred possible gap" in flat and "Limitation stated by authors" in flat
        assert "Europe PMC is not responding" in flat  # failed source is disclosed, not hidden
        assert "doi:10.0000/test.001" in flat.replace(" ", "")
        # The only mention is the safety statement that few results do NOT prove a topic is unstudied.
        assert flat.lower().count("never been studied") == 1 and "does not prove a topic has never been studied" in flat


def test_report_survives_special_characters_and_minimal_data(services):
    topic_id = services.topics.get_or_create("Tricky <topic> & β-blockers ≥ 7%")
    pid = services.papers.upsert(make_paper(1, paper_id=None, title="Effect of β-blockers & <b>markup</b> on HbA1c ≥ 7%",
                                            authors=["Müller K", "O'Brien P"], journal="J & <Test>"))
    services.papers.link_to_topic(topic_id, pid)  # a single paper: no comparison possible
    path, _ = services.report_service.create_report(topic_id)
    assert path.read_bytes().startswith(b"%PDF-")


def test_report_needs_papers(services):
    topic_id = services.topics.get_or_create("Nothing saved")
    with pytest.raises(ValueError, match="no saved papers"):
        services.report_service.create_report(topic_id)
    with pytest.raises(ValueError, match="Topic not found"):
        services.report_service.create_report(31337)
