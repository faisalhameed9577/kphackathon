"""Idea Check tests: query building, verdict rules, the full run with fake PubMed counts, AI parsing."""
from __future__ import annotations

import pytest

from database.models import SearchOutcome
from services.embedding_service import EmbeddingService
from services.gemini_service import GeminiService
from services.idea_service import (DESIGNS, Facet, IdeaError, IdeaFacets, IdeaService, compose_query,
                                   decide_verdict)
from services.pubmed_service import PubMedService
from tests.conftest import FakeGeminiClient, FakeResponse, FakeSession, make_paper
from utils.http import ResearchAPIError

IDEA = "Effect of school-based exercise programs on childhood obesity in Peshawar"


class FakePubMed:
    """Stands in for PubMedService: ``rule(query, year_from)`` returns the count (or raises)."""

    def __init__(self, rule):
        self.rule = rule
        self.queries = []

    def count(self, query, year_from, year_to):
        self.queries.append((query, year_from))
        value = self.rule(query, year_from)
        if isinstance(value, Exception):
            raise value
        return value


class FakePaperService:
    def __init__(self, pubmed, papers=None):
        self.pubmed = pubmed
        self.papers = papers or []
        self.requests = []

    def search_papers(self, request):
        self.requests.append(request)
        return SearchOutcome(papers=self.papers, per_source=[], duplicates_removed=0,
                             retrieved_before_dedup=len(self.papers))


def make_service(settings, rule, papers=None, gemini=None):
    gemini = gemini or GeminiService(settings)
    pubmed = FakePubMed(rule)
    return IdeaService(gemini, FakePaperService(pubmed, papers), EmbeddingService(settings, gemini)), pubmed


# ------------------------------------------------------------------ pure rules
@pytest.mark.parametrize("full, topic, recent, reviews, expected", [
    (0, 0, 0, 0, "INCONCLUSIVE"),   # even the topic matches nothing
    (0, 500, 0, 0, "GO"),
    (2, 500, 1, 0, "GO"),
    (3, 500, 2, 0, "MODIFY"),
    (19, 900, 12, 3, "MODIFY"),     # just under the LEAVE threshold
    (20, 900, 12, 0, "LEAVE"),      # many papers and many recent ones
    (50, 900, 2, 1, "LEAVE"),       # many papers and a recent review
    (50, 900, 2, 0, "MODIFY"),      # many papers, but old and no recent review
])
def test_decide_verdict(full, topic, recent, reviews, expected):
    assert decide_verdict(full, topic, recent, reviews) == expected


def test_compose_query_joins_parts_and_design_filter():
    facets = IdeaFacets("x", "x", facets={
        "topic": Facet("topic", "obesity", ["obesity", "overweight"]),
        "population": Facet("population", "children", ["children"]),
        "region": Facet("region", "Peshawar", ["Peshawar"], fields=("tiab", "ad")),
    })
    query = compose_query(facets, ["topic", "population", "region"], "rct")
    assert '("obesity"[tiab] OR "overweight"[tiab])' in query
    assert '("children"[tiab])' in query
    assert '(("Peshawar"[tiab] OR "Peshawar"[ad]))' in query
    assert query.endswith(DESIGNS["rct"][1]) and query.count(" AND ") == 3
    # dropping a part removes only that part
    assert "children" not in compose_query(facets, ["topic", "region"])
    assert compose_query(facets, ["topic"], "") == '("obesity"[tiab] OR "overweight"[tiab])'


def test_pubmed_count_uses_retmax_zero(settings):
    session = FakeSession(lambda url, params: FakeResponse(json_data={"esearchresult": {"count": "123", "idlist": []}}))
    assert PubMedService(settings, session).count("diabetes", 2000, 2025) == 123
    assert session.calls[0][1]["retmax"] == 0


# ------------------------------------------------------------- full run (rules)
def local_rule(query, year_from):
    """World: 60 papers on the idea, only 1 with Peshawar; designs differ; 2 recent reviews."""
    with_region = "Peshawar" in query
    if "Systematic Review" in query:
        return 2
    if "Randomized Controlled Trial" in query:
        return 0 if with_region else 4
    if "Cross-Sectional" in query or "Cohort" in query or "Qualitative" in query:
        return 1 if with_region else 20
    if '"children"' not in query and '"obesity"' not in query and "exercise" not in query:
        return 900
    return 1 if with_region else 60


def test_run_go_with_region_angle_and_nearest_papers(settings):
    papers = [make_paper(1, title="School-based exercise program reduces childhood obesity in children",
                         abstract="Exercise in school children and obesity outcomes in Peshawar."),
              make_paper(2, title="Dengue vaccine serotype distribution", abstract="Serotype survey of dengue virus.")]
    service, pubmed = make_service(settings, local_rule, papers)
    report = service.run(IDEA)

    assert report.facets.parsed_by == "rules"
    assert report.facets.region_label == "Peshawar" and report.facets.country == "Pakistan"
    assert report.verdict == "GO" and report.headline.startswith("GO")
    assert report.metrics["whole_idea"] == 1 and report.metrics["region_world"] == 60
    assert any(a.kind == "region" and "60 papers" in a.text for a in report.angles)
    assert report.designs and {d.label for d in report.designs} >= {"randomized controlled trial"}
    assert report.nearest and "exercise" in report.nearest[0].paper.title.lower()
    assert report.nearest[0].in_region is True
    assert report.explanation_by == "rules" and "Only 1 paper" in report.explanation
    assert any("poorly indexed" in n for n in report.notes)
    # every PubMed query is recorded for the methodology tab
    assert len(report.counts) == len({(r.query, r.year_from) for r in report.counts}) > 8


def test_run_leave_when_well_covered(settings):
    def rule(query, year_from):
        if "Systematic Review" in query:
            return 3
        return 15 if year_from > 1990 else 80

    report = make_service(settings, rule)[0].run("Prevalence of hypertension in adults in Pakistan")
    assert report.verdict == "LEAVE" and "well covered" in report.explanation
    assert report.advice[0].startswith("Repeating this adds little")


def test_run_modify_when_studied_but_thin(settings):
    def rule(query, year_from):
        if "Randomized Controlled Trial" in query:
            return 0
        return 8

    report = make_service(settings, rule)[0].run("Hypertension screening in adults in Peshawar")
    assert report.verdict == "MODIFY"
    assert any(a.kind == "design" and "randomized controlled trial" in a.text for a in report.angles)


def test_run_inconclusive_when_topic_matches_nothing(settings):
    report = make_service(settings, lambda q, y: 0)[0].run("Hypertension screening in adults in Peshawar")
    assert report.verdict == "INCONCLUSIVE" and report.designs == []


def test_run_fails_clearly_when_pubmed_is_down(settings):
    service, _ = make_service(settings, lambda q, y: ResearchAPIError("PubMed is not reachable."))
    with pytest.raises(IdeaError, match="PubMed is not reachable"):
        service.run(IDEA)


def test_failed_secondary_counts_are_reported_not_fatal(settings):
    def rule(query, year_from):
        if "Systematic Review" in query:
            return ResearchAPIError("temporary failure")
        return 30

    report = make_service(settings, rule)[0].run("Hypertension screening in adults in Pakistan")
    assert report.verdict in {"LEAVE", "MODIFY"}
    assert any("count request(s) failed" in n for n in report.notes)
    assert report.metrics["reviews_recent"] is None


def test_idea_validation_and_urdu_without_gemini(settings):
    service, _ = make_service(settings, lambda q, y: 5)
    with pytest.raises(IdeaError, match="at least one full sentence"):
        service.run("obesity")
    with pytest.raises(IdeaError, match="maximum 600"):
        service.run("obesity " * 100)
    with pytest.raises(IdeaError, match="Urdu"):
        service.run("پشاور میں بچوں میں موٹاپے پر تحقیق")


# --------------------------------------------------------------- full run (AI)
URDU_PARSE = {"language": "ur", "english_idea": "Effect of school exercise programmes on obesity in children in Peshawar",
              "topic": {"label": "childhood obesity", "terms": ["obesity", "overweight"]},
              "population": {"label": "children", "terms": ["children", "child", "paediatric"]},
              "intervention": {"label": "school exercise programme", "terms": ["exercise", "physical activity"]},
              "outcome": {"label": "", "terms": []},
              "region": {"label": "Peshawar", "country": "Pakistan"}, "design": "rct"}


def keyed_service(keyed_settings, replies, rule=local_rule):
    client = FakeGeminiClient(replies, embed_dim=8)
    gemini = GeminiService(keyed_settings, client=client)
    return make_service(keyed_settings, rule, [make_paper(1, title="School exercise and obesity in children")], gemini)[0], client


def test_ai_parse_urdu_idea_and_explain_in_urdu(keyed_settings):
    explanation = "یہ آئیڈیا مقامی سطح پر کم تحقیق شدہ لگتا ہے۔ کم تعداد کا مطلب ناقص انڈیکسنگ بھی ہو سکتا ہے۔"
    service, client = keyed_service(keyed_settings, [URDU_PARSE, explanation])
    report = service.run("پشاور کے بچوں میں اسکول ورزش اور موٹاپا")

    assert report.facets.parsed_by == "ai" and report.facets.language == "ur"
    assert report.facets.design == "rct" and report.facets.country == "Pakistan"
    assert set(report.facets.facets) == {"topic", "population", "intervention", "region"}
    assert report.explanation == explanation and report.explanation_by == "gemini"
    assert "Urdu (Arabic script)" in client.generate_calls[-1]
    # the user's own design (RCT) is applied to the whole-idea query
    assert DESIGNS["rct"][1] in report.counts[0].query


def test_ai_explanation_with_invented_numbers_is_discarded(keyed_settings):
    service, _ = keyed_service(keyed_settings, [URDU_PARSE, "There are 99999 papers on this, so it is a clear GO."])
    report = service.run("Exercise and obesity in children in Peshawar")
    assert report.explanation_by == "rules" and "99999" not in report.explanation
    assert any("discarded" in n for n in report.notes)


def test_ai_failure_falls_back_to_rules_for_english(keyed_settings):
    service, _ = keyed_service(keyed_settings, [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"),
                                                RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    report = service.run(IDEA)
    assert report.facets.parsed_by == "rules" and report.verdict in {"GO", "MODIFY", "LEAVE"}
    assert any("AI parsing unavailable" in n for n in report.notes)
