"""Discovery pipeline: search both sources, merge duplicates, store, fetch open-access text."""
from __future__ import annotations

import pytest

import utils.http as http_module
from services.container import build_services
from services.europe_pmc_service import EuropePMCService
from services.pubmed_service import PubMedService
from tests.conftest import FakeResponse, FakeSession
from utils.validators import SearchRequest

pytest.importorskip("chromadb")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_module.time, "sleep", lambda s: None)
    monkeypatch.setattr("services.pubmed_service.time.sleep", lambda s: None)


def build(settings, pubmed_xml, europepmc_json, jats_xml=None, pubmed_down=False, epmc_down=False):
    def pubmed_handler(url, params):
        if pubmed_down:
            return FakeResponse(status_code=503)
        if url.endswith("esearch.fcgi"):
            return FakeResponse(json_data={"esearchresult": {"count": "3", "idlist": ["90000001", "90000002", "90000003"]}})
        return FakeResponse(text=pubmed_xml)

    def epmc_handler(url, params):
        if epmc_down:
            return FakeResponse(status_code=503)
        if url.endswith("fullTextXML"):
            return FakeResponse(text=jats_xml) if jats_xml and "PMC9000001" in url else FakeResponse(status_code=404)
        return FakeResponse(json_data=europepmc_json)

    return build_services(settings, pubmed=PubMedService(settings, FakeSession(pubmed_handler)),
                          europepmc=EuropePMCService(settings, FakeSession(epmc_handler)))


REQUEST = dict(topic="Glycaemic control fictional district", year_from=2010, year_to=2025, max_papers=20)


def test_search_merges_sources_and_removes_duplicates(settings, pubmed_xml, europepmc_json):
    services = build(settings, pubmed_xml, europepmc_json)
    outcome = services.paper_service.search_papers(SearchRequest(**REQUEST))
    assert outcome.retrieved_before_dedup == 7 and outcome.duplicates_removed == 2 and len(outcome.papers) == 5
    assert [r.source for r in outcome.per_source] == ["PubMed", "Europe PMC"]
    both = [p for p in outcome.papers if len(p.sources) == 2]
    assert len(both) == 2


def test_max_papers_caps_the_final_list(settings, pubmed_xml, europepmc_json):
    services = build(settings, pubmed_xml, europepmc_json)
    outcome = services.paper_service.search_papers(SearchRequest(**{**REQUEST, "max_papers": 3}))
    assert len(outcome.papers) == 3


def test_one_source_failing_does_not_break_the_search(settings, pubmed_xml, europepmc_json):
    services = build(settings, pubmed_xml, europepmc_json, pubmed_down=True)
    outcome = services.paper_service.search_papers(SearchRequest(**REQUEST))
    assert len(outcome.papers) == 4 and outcome.per_source[0].error and outcome.per_source[1].error is None

    both_down = build(settings, pubmed_xml, europepmc_json, pubmed_down=True, epmc_down=True)
    outcome = both_down.paper_service.search_papers(SearchRequest(**REQUEST))
    assert outcome.papers == [] and all(r.error for r in outcome.per_source)


def test_search_is_recorded_with_methodology(settings, pubmed_xml, europepmc_json):
    services = build(settings, pubmed_xml, europepmc_json)
    request = SearchRequest(**REQUEST)
    outcome = services.paper_service.search_papers(request)
    topic_id = services.paper_service.record_search(request, outcome)
    latest = services.queries.latest_search(topic_id)
    assert latest["result_count"] == 5 and latest["filters"]["duplicates_removed"] == 2
    assert latest["filters"]["per_source"][0]["source"] == "PubMed" and latest["filters"]["per_source"][0]["query"]
    assert services.queries.count("search") == 1


def test_save_and_index_with_open_access_full_text(settings, pubmed_xml, europepmc_json, jats_xml):
    services = build(settings, pubmed_xml, europepmc_json, jats_xml=jats_xml)
    request = SearchRequest(**REQUEST)
    outcome = services.paper_service.search_papers(request)
    topic_id = services.paper_service.record_search(request, outcome)
    steps = []
    report = services.paper_service.save_and_index(outcome.papers, topic_id, fetch_fulltext=True,
                                                   progress=lambda f, m: steps.append(f))
    assert report.papers_saved == 5 and report.fulltext_fetched == 1  # only PMC9000001 has open-access text
    assert report.papers_without_abstract == 1 and any("No text to index" in w for w in report.warnings)
    assert steps[-1] == 1.0 and steps == sorted(steps)  # progress never goes backwards

    stored = {p.pmid: p for p in services.papers.list_for_topic(topic_id) if p.pmid}
    oa = stored["90000001"]
    assert services.paper_service.has_fulltext(oa) and oa.fulltext_path
    hits = services.vectors.query("small sample size and single-centre design limitation", k=5, paper_ids=[oa.paper_id])
    assert any(h.metadata["text_origin"] == "full text" for h in hits)
    assert "Discussion (full text)" in services.paper_service.get_paper_text(oa, max_chars=6000)

    # Saving the same papers again must not duplicate rows or chunks.
    before = (services.papers.count(), services.chunks.count())
    services.paper_service.save_and_index(outcome.papers, topic_id, fetch_fulltext=True)
    assert (services.papers.count(), services.chunks.count()) == before
