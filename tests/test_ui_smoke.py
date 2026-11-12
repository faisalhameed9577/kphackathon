"""
UI smoke tests: run every Streamlit page headlessly with Streamlit's AppTest and
check that nothing crashes. Network calls are replaced by the mock fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("chromadb")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import utils.http as http_module  # noqa: E402
from config.settings import get_settings  # noqa: E402
from services.container import build_services  # noqa: E402
from services.europe_pmc_service import EuropePMCService  # noqa: E402
from services.pubmed_service import PubMedService  # noqa: E402
from tests.conftest import FIXTURES, FakeResponse, FakeSession, make_paper  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def ui(tmp_path, monkeypatch):
    """Fresh data folder, no API keys, fake research APIs. Returns the Services used by the app."""
    monkeypatch.setenv("ZETESIS_DATA_DIR", str(tmp_path / "ui_data"))
    for var in ("GEMINI_API_KEY", "NCBI_API_KEY", "NCBI_EMAIL"):
        monkeypatch.setenv(var, "")
    monkeypatch.setattr(http_module.time, "sleep", lambda s: None)
    monkeypatch.setattr("services.pubmed_service.time.sleep", lambda s: None)

    pubmed_xml = (FIXTURES / "pubmed_sample.xml").read_text(encoding="utf-8")
    europepmc = json.loads((FIXTURES / "europepmc_sample.json").read_text(encoding="utf-8"))

    def pm(url, params):
        if url.endswith("esearch.fcgi"):
            return FakeResponse(json_data={"esearchresult": {"count": "3", "idlist": ["90000001", "90000002", "90000003"]}})
        return FakeResponse(text=pubmed_xml)

    def ep(url, params):
        return FakeResponse(status_code=404) if url.endswith("fullTextXML") else FakeResponse(json_data=europepmc)

    def fake_build(settings):
        return build_services(settings, pubmed=PubMedService(settings, FakeSession(pm)),
                              europepmc=EuropePMCService(settings, FakeSession(ep)))

    monkeypatch.setattr("frontend.state.build_services", fake_build)
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield build_services(get_settings())  # same files as the app's own Services object
    get_settings.cache_clear()
    st.cache_resource.clear()


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["nav_page"] = page
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def texts(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.info] + [e.value for e in at.warning]
    parts += [e.value for e in at.success] + [e.value for e in at.error] + [e.value for e in at.caption]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", ["Dashboard", "Research Discovery", "Research Library", "Research Assistant",
                                  "Research Comparison", "Research Gap Analysis", "Reports"])
def test_every_page_renders_on_an_empty_database(ui, page):
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "does not provide medical" in texts(at)  # safety disclaimer is on every page


def test_empty_pages_guide_the_user(ui):
    assert "library is empty" in texts(open_page("Research Library"))
    assert "No papers are indexed yet" in texts(open_page("Research Assistant"))
    assert "No research topics with saved papers" in texts(open_page("Reports"))
    assert len(open_page("Dashboard").metric) == 5


def test_missing_gemini_key_is_explained_not_fatal(ui):
    assert any("GEMINI_API_KEY is not set" in w.value for w in open_page("Dashboard").warning)


def test_discovery_validation_error(ui):
    at = open_page("Research Discovery")
    at.text_input(key="disc_topic").set_value("ab")
    at = click(at, "Search PubMed + Europe PMC")
    assert any("at least 4 characters" in e.value for e in at.error)


def test_discovery_search_save_and_index(ui):
    at = open_page("Research Discovery")
    at.text_input(key="disc_topic").set_value("Glycaemic control fictional district")
    at = click(at, "Search PubMed + Europe PMC")
    assert not at.exception, [e.value for e in at.exception]
    assert "5 unique papers" in texts(at)
    at = click(at, "Save and index selected papers")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Saved 5 papers" in s.value for s in at.success)
    assert ui.papers.count() == 5 and ui.chunks.count() > 0 and ui.queries.count("search") == 1


def test_dashboard_quick_search_opens_discovery(ui):
    at = open_page("Dashboard")
    at.text_input(key="dash_quick_topic").set_value("Glycaemic control fictional district")
    at = click(at, "Search PubMed + Europe PMC")
    assert at.session_state["nav_page"] == "Research Discovery"
    assert "5 unique papers" in texts(at)


def seed(ui):
    topic_id = ui.topics.get_or_create("Fictional glycaemic outcomes", region="Fictionland", year_from=2010, year_to=2025)
    papers = [make_paper(i, paper_id=None, pmid=str(93000000 + i), doi=f"10.0000/ui.{i}") for i in range(1, 5)]
    ui.paper_service.save_and_index(papers, topic_id, fetch_fulltext=False)
    return topic_id


def test_library_assistant_comparison_gaps_and_reports_flow(ui):
    seed(ui)

    lib = open_page("Research Library")
    assert not lib.exception and "4 paper(s) match" in texts(lib)

    asst = open_page("Research Assistant")
    asst.text_area(key="asst_question").set_value("What was the sample size of the cross-sectional survey?")
    asst = click(asst, "Ask")
    assert not asst.exception, [e.value for e in asst.exception]
    assert "Evidence found" in " ".join(s.value for s in asst.success)

    cmp_page = open_page("Research Comparison")
    assert not cmp_page.exception
    cmp_page = click(cmp_page, "Compare selected papers")
    assert not cmp_page.exception, [e.value for e in cmp_page.exception]
    assert "Similarities" in texts(cmp_page)

    gaps = open_page("Research Gap Analysis")
    gaps = click(gaps, "Run analysis")
    assert not gaps.exception, [e.value for e in gaps.exception]
    assert len(gaps.tabs) == 3 and "Possible research gaps" in [t.label for t in gaps.tabs]

    rep = open_page("Reports")
    rep = click(rep, "Generate PDF report")
    assert not rep.exception, [e.value for e in rep.exception]
    assert any("Report created" in s.value for s in rep.success)
    assert ui.reports.count() == 1 and Path(ui.reports.list()[0]["file_path"]).exists()


def test_medical_advice_is_declined_in_the_ui(ui):
    seed(ui)
    asst = open_page("Research Assistant")
    asst.text_area(key="asst_question").set_value("Should I take insulin for my symptoms?")
    asst = click(asst, "Ask")
    assert any("Insufficient evidence" in w.value for w in asst.warning)
    assert "cannot provide that" in texts(asst)
