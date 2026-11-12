"""PubMed and Europe PMC retrieval, using mock HTTP responses only."""
from __future__ import annotations

import pytest

import utils.http as http_module
from config.settings import Settings
from services.europe_pmc_service import (EuropePMCService, build_europepmc_query, parse_europepmc_result)
from services.pubmed_service import PubMedService, build_pubmed_query, parse_pubmed_xml
from tests.conftest import FakeResponse, FakeSession
from utils.http import ResearchAPIError
from utils.validators import SearchRequest, validate_search_input


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Never actually wait during retry/back-off tests."""
    monkeypatch.setattr(http_module.time, "sleep", lambda s: None)
    monkeypatch.setattr("services.pubmed_service.time.sleep", lambda s: None)


def make_request(**kw) -> SearchRequest:
    values = dict(topic="Diabetes research in rural Khyber Pakhtunkhwa", region="Khyber Pakhtunkhwa",
                  year_from=2015, year_to=2024, max_papers=10)
    values.update(kw)
    return SearchRequest(**values)


# ------------------------------------------------------------------ queries
def test_pubmed_query_drops_filler_words_and_expands_region():
    query = build_pubmed_query(make_request())
    assert "(Diabetes AND rural AND Khyber AND Pakhtunkhwa)" in query
    assert "research" not in query.lower().split("(")[1]  # generic word removed from topic terms
    assert '"Khyber Pakhtunkhwa"[tiab]' in query and '"Peshawar"[tiab]' in query


def test_pubmed_query_keywords_are_or_combined():
    query = build_pubmed_query(make_request(region="", keywords="HbA1c, insulin resistance"))
    assert '(HbA1c OR "insulin resistance")' in query


def test_europepmc_query_has_year_filter_and_field_restricted_region():
    query = build_europepmc_query(make_request())
    assert "PUB_YEAR:[2015 TO 2024]" in query
    assert 'TITLE:"Khyber Pakhtunkhwa" OR ABSTRACT:"Khyber Pakhtunkhwa"' in query


# ------------------------------------------------------------------ PubMed parsing
def test_parse_pubmed_xml(pubmed_xml):
    papers = parse_pubmed_xml(pubmed_xml)
    assert [p.pmid for p in papers] == ["90000001", "90000002", "90000003"]

    first = papers[0]
    assert first.doi == "10.0000/test.001" and first.pmcid == "PMC9000001"
    assert first.year == 2021 and first.journal == "Journal of Synthetic Testing"
    assert first.authors[:2] == ["Testauthor A", "Fixture B"]
    assert "synthetic cross-sectional study" in first.title  # <i> tag text kept, tags removed
    assert "<i>" not in first.title
    assert first.abstract.splitlines()[0].startswith("Background: ")  # structured labels kept
    assert "Methods: " in first.abstract and "Conclusions: " in first.abstract
    assert first.sources == ["PubMed"]
    assert first.pub_types == ["Journal Article", "Randomized Controlled Trial"]
    assert first.best_url == "https://doi.org/10.0000/test.001"

    missing = papers[1]
    assert missing.pub_types == []
    assert missing.abstract == "" and not missing.has_abstract  # missing abstract handled
    assert missing.year == 2019  # parsed from MedlineDate "2019 Jan-Feb"
    assert missing.doi is None
    assert missing.best_url == "https://pubmed.ncbi.nlm.nih.gov/90000002/"

    third = papers[2]
    assert third.doi == "10.0000/test.003"  # from ELocationID
    assert third.authors == ["Synthetic Study Group"]  # collective author


def test_parse_pubmed_xml_rejects_garbage():
    with pytest.raises(ResearchAPIError):
        parse_pubmed_xml("this is not xml")


# ------------------------------------------------------------------ PubMed search
def pubmed_handler(pubmed_xml):
    def handler(url, params):
        if url.endswith("esearch.fcgi"):
            return FakeResponse(json_data={"esearchresult": {"count": "42", "idlist": ["90000001", "90000002", "90000003"]}})
        if url.endswith("efetch.fcgi"):
            return FakeResponse(text=pubmed_xml)
        raise AssertionError(url)
    return handler


def test_pubmed_search_end_to_end(keyed_settings, pubmed_xml):
    session = FakeSession(pubmed_handler(pubmed_xml))
    result = PubMedService(keyed_settings, session).search(make_request())
    assert result.error is None and result.total_found == 42 and len(result.papers) == 3

    esearch_params = session.calls[0][1]
    assert esearch_params["mindate"] == 2015 and esearch_params["maxdate"] == 2024
    assert esearch_params["retmax"] == 10 and esearch_params["db"] == "pubmed"
    assert esearch_params["api_key"] == "ncbi-secret-key" and esearch_params["email"] == "test@example.org"
    assert esearch_params["tool"] == "ZetesisAI"


def test_pubmed_works_without_optional_keys(settings, pubmed_xml):
    session = FakeSession(pubmed_handler(pubmed_xml))
    PubMedService(settings, session).search(make_request())
    assert "api_key" not in session.calls[0][1] and "email" not in session.calls[0][1]


def test_pubmed_empty_results(settings):
    session = FakeSession(lambda u, p: FakeResponse(json_data={"esearchresult": {"count": "0", "idlist": []}}))
    result = PubMedService(settings, session).search(make_request())
    assert result.papers == [] and result.error is None and result.total_found == 0
    assert len(session.calls) == 1  # efetch is skipped when nothing was found


def test_pubmed_retries_rate_limit_then_succeeds(settings, pubmed_xml):
    state = {"n": 0}
    inner = pubmed_handler(pubmed_xml)

    def handler(url, params):
        state["n"] += 1
        return FakeResponse(status_code=429) if state["n"] == 1 else inner(url, params)

    result = PubMedService(settings, FakeSession(handler)).search(make_request())
    assert result.error is None and len(result.papers) == 3 and state["n"] == 3  # 429, esearch, efetch


def test_pubmed_gives_up_after_retries_without_leaking_key(keyed_settings):
    session = FakeSession(lambda u, p: FakeResponse(status_code=503))
    result = PubMedService(keyed_settings, session).search(make_request())
    assert result.papers == [] and "not responding" in result.error
    assert "ncbi-secret-key" not in result.error
    assert len(session.calls) == keyed_settings.max_retries


def test_pubmed_timeout_is_reported_not_raised(settings):
    import requests

    def handler(url, params):
        raise requests.Timeout("timed out")

    result = PubMedService(settings, FakeSession(handler)).search(make_request())
    assert "not responding" in result.error


def test_pubmed_client_error_is_not_retried(settings):
    session = FakeSession(lambda u, p: FakeResponse(status_code=400))
    result = PubMedService(settings, session).search(make_request())
    assert "HTTP 400" in result.error and len(session.calls) == 1


# ------------------------------------------------------------------ Europe PMC
def test_parse_europepmc_results(europepmc_json):
    papers = [parse_europepmc_result(r) for r in europepmc_json["resultList"]["result"]]
    first = papers[0]
    assert first.doi == "10.0000/test.001"  # normalised to lower case
    assert first.is_open_access and first.pmcid == "PMC9000001" and first.sources == ["Europe PMC"]
    assert first.abstract.startswith("Background: Synthetic background")  # html headings -> labels
    assert "<h4>" not in first.abstract and "<p>" not in first.abstract
    assert papers[3].is_preprint and papers[3].pmid is None
    assert papers[3].best_url == "https://doi.org/10.0000/test.preprint.005"


def test_europepmc_search_end_to_end(settings, europepmc_json):
    session = FakeSession(lambda u, p: FakeResponse(json_data=europepmc_json))
    result = EuropePMCService(settings, session).search(make_request(max_papers=25))
    assert result.total_found == 4 and len(result.papers) == 4 and result.error is None
    url, params = session.calls[0]
    assert url.endswith("/search") and params["format"] == "json" and params["resultType"] == "core"
    assert params["pageSize"] == 25 and "PUB_YEAR" in params["query"]


def test_europepmc_error_is_reported(settings):
    result = EuropePMCService(settings, FakeSession(lambda u, p: FakeResponse(status_code=500))).search(make_request())
    assert result.papers == [] and "not responding" in result.error


def test_europepmc_fulltext_available_and_missing(settings, jats_xml):
    ok = EuropePMCService(settings, FakeSession(lambda u, p: FakeResponse(text=jats_xml)))
    assert ok.fetch_fulltext_xml("pmc9000001") == jats_xml
    missing = EuropePMCService(settings, FakeSession(lambda u, p: FakeResponse(status_code=404)))
    assert missing.fetch_fulltext_xml("PMC9000001") is None  # not open access -> no full text, no error


# ------------------------------------------------------------------ validation
def test_validation_messages():
    assert validate_search_input("ab")[1].startswith("Please enter a research topic")
    assert validate_search_input("12345 678")[1] is not None
    assert "generic words" in validate_search_input("research studies")[1]
    assert validate_search_input("diabetes", year_from=2020, year_to=2010)[1] == "'From year' cannot be later than 'To year'."
    assert validate_search_input("diabetes", max_papers=0)[1] is not None
    request, error = validate_search_input("diabetes in Peshawar", max_papers=15)
    assert error is None and request.max_papers == 15 and request.year_to >= 2024
